"""
SubGEN AI — ESP32 Hardware Validator + Software MFCC Fallback.

Implements:
  - compute_mfcc_software(): Python-side MFCC identical to ESP32 C firmware algorithm
  - find_esp32_port(): auto-detect ESP32 USB serial port
  - send_audio_to_esp32(): PCM transfer and JSON response parsing
  - get_fingerprint(): unified entry point; tries HW first, falls back to SW
"""
import json
import struct
from typing import Optional

import numpy as np
from scipy.fft import dct

try:
    import serial
    import serial.tools.list_ports
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False

# MFCC Parameters (must match ESP32 firmware)
N_MFCC      = 12
N_MELS      = 26
SAMPLE_RATE = 16000
N_FFT       = 512
HOP_LENGTH  = 160    # 10 ms
WIN_LENGTH  = 400    # 25 ms
FMIN        = 0.0
FMAX        = 8000.0

# Serial Protocol
BAUD_RATE = 460800
HEADER    = bytes([0xAA, 0x55])
TIMEOUT_S = 3.0

# Cached Mel Filterbank
_MEL_FILTERBANK: Optional[np.ndarray] = None


def hz_to_mel(hz: float) -> float:
    """Convert Hz to mel scale."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    """Convert mel scale to Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(n_mels: int, n_fft: int,
                          sr: int, fmin: float, fmax: float) -> np.ndarray:
    """Build (n_mels, n_fft//2 + 1) triangular filterbank on mel scale."""
    mel_min    = hz_to_mel(fmin)
    mel_max    = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points  = np.array([mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_left, f_center, f_right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(f_left, f_center):
            if f_center != f_left:
                filterbank[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right != f_center:
                filterbank[m - 1, k] = (f_right - k) / (f_right - f_center)
    return filterbank


def get_filterbank() -> np.ndarray:
    """Return cached mel filterbank (built on first call)."""
    global _MEL_FILTERBANK
    if _MEL_FILTERBANK is None:
        _MEL_FILTERBANK = build_mel_filterbank(N_MELS, N_FFT, SAMPLE_RATE, FMIN, FMAX)
    return _MEL_FILTERBANK


def compute_mfcc_software(audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """
    Compute MFCC fingerprint using the same algorithm as the ESP32 firmware.

    Steps: Hanning window → 512-pt FFT → power spectrum → 26-band mel filterbank
           → log compression → DCT-II → 12 MFCC coefficients per frame
           → mean and variance across all frames.

    Args:
        audio: float32 array, normalised [-1, 1] or int16 range.
        sr: sample rate (default 16000 Hz).

    Returns:
        dict with keys: ok (bool), hw (bool), mfcc_mean (list[float]),
                        mfcc_var (list[float]), rms (float), frames (int).
    """
    if len(audio) == 0:
        return {"ok": False, "hw": False,
                "mfcc_mean": [0.0] * N_MFCC, "mfcc_var": [0.0] * N_MFCC,
                "rms": 0.0, "frames": 0}

    # Normalise: detect int16 range
    audio = audio.astype(np.float32)
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / 32768.0

    rms        = float(np.sqrt(np.mean(audio ** 2)))
    filterbank = get_filterbank()
    coefficients_per_frame: list = []

    for start in range(0, len(audio) - WIN_LENGTH, HOP_LENGTH):
        frame = audio[start:start + WIN_LENGTH]

        # Hanning window (matches ESP32 firmware)
        window          = np.hanning(len(frame))
        frame_windowed  = frame * window

        # Zero-pad to N_FFT
        padded          = np.zeros(N_FFT, dtype=np.float32)
        padded[:len(frame_windowed)] = frame_windowed

        # FFT power spectrum
        spectrum = np.fft.rfft(padded)
        power    = (np.abs(spectrum) ** 2) / N_FFT

        # Mel filterbank energies
        mel_energies = np.dot(filterbank, power)

        # Log compression
        log_mel = np.log10(mel_energies + 1e-9)

        # DCT-II → first 12 coefficients
        cepstrum = dct(log_mel, type=2, norm='ortho')
        coefficients_per_frame.append(cepstrum[:N_MFCC])

    if not coefficients_per_frame:
        return {"ok": False, "hw": False,
                "mfcc_mean": [0.0] * N_MFCC, "mfcc_var": [0.0] * N_MFCC,
                "rms": rms, "frames": 0}

    frames_arr = np.array(coefficients_per_frame)
    mfcc_mean  = frames_arr.mean(axis=0).tolist()
    mfcc_var   = frames_arr.var(axis=0).tolist()

    return {
        "ok": True, "hw": False,
        "mfcc_mean": mfcc_mean, "mfcc_var": mfcc_var,
        "rms": rms, "frames": len(coefficients_per_frame)
    }


# USB VID:PID pairs commonly used by ESP32 dev boards
_ESP32_VIDS = {
    0x10C4,  # Silicon Labs CP2102/CP2104  ← COM9 on this machine
    0x1A86,  # QinHeng CH340/CH341
    0x0403,  # FTDI
    0x303A,  # Espressif native USB
}

_ESP32_KEYWORDS = ["cp210", "ch340", "ch341", "esp32", "uart", "ftdi", "usb serial", "usb-serial"]

# HWID prefixes that are definitively NOT USB-serial (skip them for speed).
# On Windows, Bluetooth ports have HWIDs starting with "BTHENUM"
# and Intel AMT SOL ports start with "PCI".
_SKIP_HWID_PREFIXES = ("bthenum", "pci\\", "acpi\\")


def _hwid_is_usb(hwid: str) -> bool:
    """Fast pre-check: return False if the HWID is definitely not a USB serial port."""
    h = (hwid or "").lower()
    return not any(h.startswith(p) for p in _SKIP_HWID_PREFIXES)


def _port_is_esp32(port) -> bool:
    """Return True if a port entry looks like an ESP32 / common USB-serial adapter."""
    # Fast: skip Bluetooth & PCI ports entirely
    if not _hwid_is_usb(port.hwid or ""):
        return False
    # Check USB VID first (fastest path)
    if port.vid in _ESP32_VIDS:
        return True
    # Then keyword scan on description/manufacturer/hwid strings
    desc  = (port.description  or "").lower()
    manuf = (port.manufacturer or "").lower()
    hwid  = (port.hwid         or "").lower()
    return any(k in desc or k in manuf or k in hwid for k in _ESP32_KEYWORDS)


def _fast_scan_esp32_ports() -> list:
    """
    Fast ESP32 port discovery — instant on Windows, falls back to pyserial elsewhere.

    On Windows, reads HKLM\\SYSTEM\\CurrentControlSet\\Enum\\USB from the
    registry to enumerate USB devices and their COM port assignments WITHOUT
    calling win32 SetupAPI (which blocks for 60+ seconds when Bluetooth COM
    ports are present).

    Returns a list of dicts with keys: device, description, vid, pid, hwid.
    Only returns ports whose VID matches _ESP32_VIDS.
    """
    import platform
    if platform.system() != "Windows":
        # Non-Windows: use pyserial normally (no Bluetooth issue)
        if not _SERIAL_AVAILABLE:
            return []
        return [
            {
                "device":      p.device,
                "description": p.description or "",
                "hwid":        p.hwid or "",
                "vid":         p.vid,
                "pid":         p.pid,
            }
            for p in serial.tools.list_ports.comports()
            if _port_is_esp32(p)
        ]

    # Windows fast path: read registry
    try:
        import winreg, re
        results = []
        usb_key_path = r"SYSTEM\CurrentControlSet\Enum\USB"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, usb_key_path) as usb_key:
            n_vid = winreg.QueryInfoKey(usb_key)[0]  # number of subkeys
            for i in range(n_vid):
                try:
                    vid_pid_name = winreg.EnumKey(usb_key, i)  # e.g. "VID_10C4&PID_EA60"
                    # Quick VID check before going deeper
                    m = re.match(r"VID_([0-9A-Fa-f]{4})&PID_([0-9A-Fa-f]{4})", vid_pid_name)
                    if not m:
                        continue
                    vid = int(m.group(1), 16)
                    pid = int(m.group(2), 16)
                    if vid not in _ESP32_VIDS:
                        continue
                    # Drill down: vid_pid_name -> serial_number -> Device Parameters -> PortName
                    with winreg.OpenKey(usb_key, vid_pid_name) as vp_key:
                        n_serial = winreg.QueryInfoKey(vp_key)[0]
                        for j in range(n_serial):
                            try:
                                serial_name = winreg.EnumKey(vp_key, j)
                                dp_path = f"{serial_name}\\Device Parameters"
                                try:
                                    with winreg.OpenKey(vp_key, dp_path) as dp_key:
                                        port_name, _ = winreg.QueryValueEx(dp_key, "PortName")
                                        results.append({
                                            "device":      port_name,  # e.g. "COM9"
                                            "description": f"USB VID_{m.group(1)}&PID_{m.group(2)}",
                                            "hwid":        f"USB VID:PID={m.group(1)}:{m.group(2)} SER={serial_name}",
                                            "vid":         vid,
                                            "pid":         pid,
                                        })
                                except FileNotFoundError:
                                    pass
                            except OSError:
                                pass
                except OSError:
                    pass
        return results
    except Exception:
        # Registry access failed — fall back to pyserial with a generous timeout
        if not _SERIAL_AVAILABLE:
            return []
        import threading
        found: list = []
        ev = threading.Event()
        def _worker():
            try:
                found.extend(
                    {
                        "device":      p.device,
                        "description": p.description or "",
                        "hwid":        p.hwid or "",
                        "vid":         p.vid,
                        "pid":         p.pid,
                    }
                    for p in serial.tools.list_ports.comports()
                    if _port_is_esp32(p)
                )
            finally:
                ev.set()
        threading.Thread(target=_worker, daemon=True).start()
        ev.wait(timeout=10.0)
        return found


def _scan_all_ports_pyserial() -> list:
    """
    Return ALL serial ports using pyserial (slow on Windows if BT ports present).
    Used only by list_all_ports() which is called from the diagnostics expander,
    not the hot path.
    """
    if not _SERIAL_AVAILABLE:
        return []
    import threading
    result: list = []
    ev = threading.Event()
    def _worker():
        try:
            result.extend(serial.tools.list_ports.comports())
        finally:
            ev.set()
    threading.Thread(target=_worker, daemon=True).start()
    ev.wait(timeout=12.0)
    return result


def find_esp32_port() -> Optional[str]:
    """
    Auto-detect ESP32 by checking for CP2102/CH340/FTDI USB descriptors.
    Returns the device path (e.g. 'COM9' or '/dev/ttyUSB0') or None.
    Uses the fast Windows registry path; Bluetooth ports are never probed.
    """
    if not _SERIAL_AVAILABLE:
        return None
    ports = _fast_scan_esp32_ports()
    return ports[0]["device"] if ports else None


def find_all_esp32_ports() -> list:
    """
    Return a list of ALL detected ESP32-compatible port device strings.
    Uses the fast Windows registry path — completes in milliseconds.
    """
    if not _SERIAL_AVAILABLE:
        return []
    return [p["device"] for p in _fast_scan_esp32_ports()]


def list_all_ports() -> list:
    """
    Return info about every serial port found (for diagnostics / manual selection).
    Each entry is a dict: device, description, hwid, vid, pid.
    Uses pyserial with a timeout (may be slow on Windows due to Bluetooth ports).
    """
    if not _SERIAL_AVAILABLE:
        return []
    result = []
    for p in _scan_all_ports_pyserial():
        result.append({
            "device":      p.device,
            "description": p.description or "",
            "hwid":        p.hwid        or "",
            "vid":         p.vid,
            "pid":         p.pid,
        })
    return result


def send_audio_to_esp32(audio_int16: np.ndarray, port: str) -> dict:
    """
    Send PCM int16 audio to ESP32 over USB serial.

    Protocol: [0xAA][0x55][N_high][N_low][PCM int16 LE bytes...]
    ESP32 responds with a single JSON line containing the MFCC fingerprint.

    Caps input at 32000 samples (2 seconds at 16 kHz).

    Returns parsed fingerprint dict or an error dict with ok=False.
    """
    if not _SERIAL_AVAILABLE:
        return {"ok": False, "hw": False, "error": "pyserial not installed"}

    n = len(audio_int16)
    if n > 32000:
        audio_int16 = audio_int16[:32000]
        n = 32000

    payload = HEADER + struct.pack(">H", n) + audio_int16.tobytes()

    try:
        with serial.Serial(port, BAUD_RATE, timeout=TIMEOUT_S) as ser:
            ser.reset_input_buffer()
            ser.write(payload)
            response_bytes = ser.readline()   # ESP32 sends one JSON line
        response_str = response_bytes.decode("utf-8", errors="replace").strip()
        result = json.loads(response_str)
        result["hw"] = True
        return result
    except Exception as e:
        return {"ok": False, "hw": False, "error": str(e)}


def get_fingerprint(audio: np.ndarray, sr: int,
                    esp32_port: Optional[str] = None) -> dict:
    """
    Obtain MFCC fingerprint for an audio clip.

    Tries ESP32 hardware first (if port provided), falls back to software.

    Args:
        audio:      float32 array normalised [-1, 1] at sr Hz.
        sr:         sample rate.
        esp32_port: serial device path or None for software-only mode.

    Returns:
        Standard fingerprint dict: ok, hw, mfcc_mean, mfcc_var, rms, frames.
    """
    if esp32_port is not None:
        audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        result = send_audio_to_esp32(audio_int16, esp32_port)
        if result.get("ok"):
            return result

    return compute_mfcc_software(audio, sr)
