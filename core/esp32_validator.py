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


# USB VID:PID pairs for known ESP32 / USB-serial bridge chips.
# Used as a fast-accept path — devices not in this set are still checked
# by keyword matching against their FriendlyName / DeviceDesc strings.
_ESP32_VIDS = {
    0x10C4,  # Silicon Labs CP2102/CP2104
    0x1A86,  # QinHeng CH340/CH341/CH9102
    0x0403,  # FTDI FT232
    0x303A,  # Espressif native USB (ESP32-S2/S3/C3)
    0x2341,  # Arduino SA (some ESP32 dev-board variants)
    0x239A,  # Adafruit (ESP32 Feather etc.)
    0x1B4F,  # SparkFun
}

# Keywords matched against FriendlyName / DeviceDesc when VID is unknown.
_ESP32_KEYWORDS = [
    "cp210", "ch340", "ch341", "ch9102",
    "esp32", "esp8266",
    "ftdi", "ft232",
    "usb serial", "usb-serial", "usb uart",
    "silabs", "silicon labs",
    "nodemcu", "wemos", "lolin",
]

# Registry key paths under HKLM that hold USB device info.
_USB_ENUM_KEY   = r"SYSTEM\CurrentControlSet\Enum\USB"
_USBSER_ENUM_KEY = r"SYSTEM\CurrentControlSet\Enum\USB\*"   # placeholder, iterated below


def _registry_scan_all_usb_com_ports() -> list:
    """
    Scan HKLM\\SYSTEM\\CurrentControlSet\\Enum\\USB for every device that has
    a COM port assignment (Device Parameters\\PortName).

    For each found port, also read FriendlyName and DeviceDesc from the
    device's registry key so we can do keyword-based matching without needing
    pyserial or SetupAPI (both of which block on Windows when Bluetooth COM
    ports are present).

    Returns a list of dicts:
        device      — "COM8"
        description — FriendlyName or DeviceDesc string (may be empty)
        hwid        — "USB VID:PID=10C4:EA60 SER=..."
        vid         — int or None
        pid         — int or None
    """
    import winreg
    import re

    results = []
    vid_pid_re = re.compile(r"VID_([0-9A-Fa-f]{4})&PID_([0-9A-Fa-f]{4})", re.I)

    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _USB_ENUM_KEY) as usb_root:
            n_entries = winreg.QueryInfoKey(usb_root)[0]
            for i in range(n_entries):
                try:
                    vid_pid_name = winreg.EnumKey(usb_root, i)
                except OSError:
                    continue

                m = vid_pid_re.match(vid_pid_name)
                vid = int(m.group(1), 16) if m else None
                pid = int(m.group(2), 16) if m else None

                try:
                    with winreg.OpenKey(usb_root, vid_pid_name) as vp_key:
                        n_serials = winreg.QueryInfoKey(vp_key)[0]
                        for j in range(n_serials):
                            try:
                                serial_name = winreg.EnumKey(vp_key, j)
                            except OSError:
                                continue

                            # Read FriendlyName / DeviceDesc from the device key
                            description = ""
                            try:
                                with winreg.OpenKey(vp_key, serial_name) as dev_key:
                                    for val_name in ("FriendlyName", "DeviceDesc"):
                                        try:
                                            description, _ = winreg.QueryValueEx(dev_key, val_name)
                                            break
                                        except FileNotFoundError:
                                            pass
                            except OSError:
                                pass

                            # Read COM port name from Device Parameters
                            try:
                                with winreg.OpenKey(
                                    vp_key, serial_name + "\\Device Parameters"
                                ) as dp_key:
                                    port_name, _ = winreg.QueryValueEx(dp_key, "PortName")
                            except (FileNotFoundError, OSError):
                                continue  # no COM port assignment for this device

                            vid_str = m.group(1).upper() if m else "????"
                            pid_str = m.group(2).upper() if m else "????"
                            results.append({
                                "device":      port_name,
                                "description": description,
                                "hwid":        f"USB VID:PID={vid_str}:{pid_str} SER={serial_name}",
                                "vid":         vid,
                                "pid":         pid,
                            })
                except OSError:
                    continue
    except Exception:
        pass  # registry unavailable — caller falls back to pyserial

    return results


def _is_esp32_port(entry: dict) -> bool:
    """
    Return True if a port registry entry looks like an ESP32 / USB-serial adapter.

    Decision order (fastest first):
      1. VID in known-ESP32 set  → accept
      2. Keyword in description  → accept
      3. Otherwise               → reject
    """
    vid = entry.get("vid")
    if vid is not None and vid in _ESP32_VIDS:
        return True

    desc = (entry.get("description") or "").lower()
    hwid = (entry.get("hwid") or "").lower()
    combined = desc + " " + hwid
    return any(kw in combined for kw in _ESP32_KEYWORDS)


def _fast_scan_esp32_ports() -> list:
    """
    Return ESP32-compatible COM ports.

    Windows path: reads the registry directly — completes in milliseconds,
    never calls SetupAPI, never blocks on Bluetooth COM ports.
    Matches by VID (fast-accept) OR by FriendlyName/DeviceDesc keyword.

    Non-Windows path: uses pyserial with keyword+VID filtering.
    """
    import platform

    if platform.system() != "Windows":
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
            if (
                (p.vid in _ESP32_VIDS)
                or any(
                    kw in (p.description or "").lower()
                    or kw in (p.manufacturer or "").lower()
                    or kw in (p.hwid or "").lower()
                    for kw in _ESP32_KEYWORDS
                )
            )
        ]

    # Windows — fast registry scan
    all_ports = _registry_scan_all_usb_com_ports()
    if all_ports:
        return [p for p in all_ports if _is_esp32_port(p)]

    # Registry scan returned nothing — fall back to pyserial with timeout
    if not _SERIAL_AVAILABLE:
        return []
    import threading
    found: list = []
    ev = threading.Event()

    def _worker() -> None:
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
                if (
                    (p.vid in _ESP32_VIDS)
                    or any(
                        kw in (p.description or "").lower()
                        or kw in (p.manufacturer or "").lower()
                        or kw in (p.hwid or "").lower()
                        for kw in _ESP32_KEYWORDS
                    )
                )
            )
        finally:
            ev.set()

    threading.Thread(target=_worker, daemon=True).start()
    ev.wait(timeout=10.0)
    return found


def _scan_all_ports_registry() -> list:
    """
    Return ALL USB COM ports from the registry (for diagnostics).
    Instant on Windows — no pyserial, no SetupAPI, no Bluetooth blocking.
    """
    import platform
    if platform.system() != "Windows":
        if not _SERIAL_AVAILABLE:
            return []
        import threading
        result: list = []
        ev = threading.Event()
        def _worker() -> None:
            try:
                result.extend(serial.tools.list_ports.comports())
            finally:
                ev.set()
        threading.Thread(target=_worker, daemon=True).start()
        ev.wait(timeout=12.0)
        return [
            {"device": p.device, "description": p.description or "",
             "hwid": p.hwid or "", "vid": p.vid, "pid": p.pid}
            for p in result
        ]
    return _registry_scan_all_usb_com_ports()


def _port_is_alive(port: str) -> bool:
    """
    Return True if the COM port is physically present on the system.

    Strategy: attempt a non-blocking open with pyserial.
      - Opens fine          → device is connected
      - "Access denied"     → device is connected but in use by another process
      - "No such file" etc. → device is disconnected / ghost registry entry
    """
    if not _SERIAL_AVAILABLE:
        return False
    try:
        s = serial.Serial(port, timeout=0)
        s.close()
        return True
    except serial.SerialException as exc:
        # Port exists but is held by another process → device is present
        msg = str(exc).lower()
        if "access" in msg or "denied" in msg or "permission" in msg:
            return True
        return False


def find_esp32_port() -> Optional[str]:
    """
    Auto-detect ESP32. Returns first live port string or None.
    Registry scan finds candidates; _port_is_alive() confirms physical presence.
    """
    if not _SERIAL_AVAILABLE:
        return None
    for p in _fast_scan_esp32_ports():
        if _port_is_alive(p["device"]):
            return p["device"]
    return None


def find_all_esp32_ports() -> list:
    """
    Return all live ESP32-compatible port strings.
    Registry scan for candidates + pyserial open-probe for liveness.
    """
    if not _SERIAL_AVAILABLE:
        return []
    return [
        p["device"]
        for p in _fast_scan_esp32_ports()
        if _port_is_alive(p["device"])
    ]


def list_all_ports() -> list:
    """
    Return info about every USB COM port (for diagnostics / manual selection).
    Each entry is a dict: device, description, hwid, vid, pid.
    Uses the registry on Windows — instant, never blocks on Bluetooth ports.
    """
    return _scan_all_ports_registry()


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
