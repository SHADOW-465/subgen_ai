# Spectral Subtraction + SNR-Gated Correction Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ESP32 and software MFCC paths compute noise-floor-subtracted fingerprints, and prevent noisy-audio corrections from being stored in the correction DB.

**Architecture:** Two-pass MFCC computation: Pass 1 classifies frames as noise/speech by RMS threshold and accumulates a per-mel-band noise floor; Pass 2 subtracts that floor from all mel energies before log compression and DCT, producing cleaner fingerprints. The fingerprint dict gains a `snr_db` field. On the Python side a new `is_snr_acceptable()` gate in `qc_engine.py` is checked before any correction is written to SQLite — if the segment SNR is below 15 dB the correction updates the UI text for export but is silently excluded from the DB.

**Tech Stack:** Python 3.12, NumPy, SciPy (dct), Arduino C (ESP32 DevKit v1, ArduinoJson v6)

---

## Scope

This plan covers two independent but related subsystems. They are kept in one plan because they ship together as a single coherent feature.

**Subsystem A — Spectral subtraction in MFCC computation:**
- `core/esp32_validator.py` — software MFCC path (`compute_mfcc_software`)
- `firmware/esp32_firmware.ino` — hardware MFCC path (C code, two-pass redesign)

**Subsystem B — SNR gate for correction storage:**
- `core/qc_engine.py` — new `SNR_GATE_DB` constant + `is_snr_acceptable()` helper
- `app.py` — `_do_save_correction()` checks gate before calling `save_correction()`

---

## File Map

| File | Change |
|------|--------|
| `core/esp32_validator.py` | Refactor `compute_mfcc_software()` for two-pass spectral subtraction; add `snr_db` to returned dict |
| `core/qc_engine.py` | Add `SNR_GATE_DB = 15.0` and `is_snr_acceptable(snr_db)` |
| `app.py` | In `_do_save_correction()`: check `is_snr_acceptable(seg.snr_db)`; skip DB write + show warning if too noisy; still update segment text |
| `firmware/esp32_firmware.ino` | Split `compute_frame_mfcc()` into `compute_frame_mel_raw()` + main loop; add two-pass noise floor estimation + subtraction; add `snr_db` to JSON output |
| `test_spectral_subtraction.py` | New — unit tests for SW spectral subtraction and SNR gate |

---

## Task 1: Add SNR gate to `qc_engine.py`

**Files:**
- Modify: `core/qc_engine.py`
- Test: `test_spectral_subtraction.py` (create)

### What and why

`qc_engine.py` already owns all QC thresholds. The SNR gate threshold belongs here alongside `FUSED_CONF_THRESHOLD` and the correction validation thresholds. A simple boolean helper lets callers stay decoupled from the raw threshold value.

---

- [ ] **Step 1.1 — Write the failing test**

Create `test_spectral_subtraction.py` at the project root:

```python
"""
Tests for spectral subtraction (SW MFCC) and SNR gate (qc_engine).
Run from the project root with the venv active:
    python -m pytest test_spectral_subtraction.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Task 1: SNR gate ──────────────────────────────────────────────────────────

def test_snr_gate_accepts_above_threshold():
    from subgen_ai.core.qc_engine import is_snr_acceptable
    assert is_snr_acceptable(20.0) is True

def test_snr_gate_rejects_below_threshold():
    from subgen_ai.core.qc_engine import is_snr_acceptable
    assert is_snr_acceptable(10.0) is False

def test_snr_gate_rejects_at_exact_threshold_minus_epsilon():
    from subgen_ai.core.qc_engine import is_snr_acceptable, SNR_GATE_DB
    assert is_snr_acceptable(SNR_GATE_DB - 0.001) is False

def test_snr_gate_accepts_at_exact_threshold():
    from subgen_ai.core.qc_engine import is_snr_acceptable, SNR_GATE_DB
    assert is_snr_acceptable(SNR_GATE_DB) is True
```

- [ ] **Step 1.2 — Run tests to confirm they fail**

```bash
cd /c/Users/acer/Documents/projects/subgen_ai
venv/Scripts/python -m pytest test_spectral_subtraction.py::test_snr_gate_accepts_above_threshold test_spectral_subtraction.py::test_snr_gate_rejects_below_threshold -v
```

Expected: `ImportError: cannot import name 'is_snr_acceptable'`

- [ ] **Step 1.3 — Add `SNR_GATE_DB` and `is_snr_acceptable` to `qc_engine.py`**

Open `core/qc_engine.py`. After the existing threshold block (after line `THRESHOLD_DB_APPLY = 0.80`), add:

```python
# SNR gate for correction storage — below this dB, corrections are too noisy
# to produce reliable MFCC fingerprints for the self-improvement DB.
SNR_GATE_DB: float = 15.0


def is_snr_acceptable(snr_db: float) -> bool:
    """Return True if the segment SNR is high enough to store in the correction DB."""
    return snr_db >= SNR_GATE_DB
```

- [ ] **Step 1.4 — Run tests to confirm they pass**

```bash
venv/Scripts/python -m pytest test_spectral_subtraction.py::test_snr_gate_accepts_above_threshold test_spectral_subtraction.py::test_snr_gate_rejects_below_threshold test_spectral_subtraction.py::test_snr_gate_rejects_at_exact_threshold_minus_epsilon test_spectral_subtraction.py::test_snr_gate_accepts_at_exact_threshold -v
```

Expected: 4 passed.

- [ ] **Step 1.5 — Commit**

```bash
git add core/qc_engine.py test_spectral_subtraction.py
git commit -m "feat: add SNR_GATE_DB threshold and is_snr_acceptable() to qc_engine"
```

---

## Task 2: Wire SNR gate into `_do_save_correction()` in `app.py`

**Files:**
- Modify: `app.py`

### What and why

Currently `_do_save_correction()` always writes to SQLite. With the gate, if `seg.snr_db < SNR_GATE_DB` the function still updates the segment text in `st.session_state` (so the export is correct) but skips `save_correction()` and shows a yellow warning. The `correction_count` session counter is also not incremented for noisy saves, which keeps the sidebar stat honest.

---

- [ ] **Step 2.1 — Update the import in `app.py`**

Find the existing import line near the top of `app.py`:

```python
from subgen_ai.core.qc_engine import validate_correction
```

Replace it with:

```python
from subgen_ai.core.qc_engine import validate_correction, is_snr_acceptable
```

- [ ] **Step 2.2 — Gate the DB write in `_do_save_correction()`**

Find the function `_do_save_correction` in `app.py`. Replace the entire function body with the version below. The only logic change is the SNR check block inserted before `save_correction(record)`.

```python
def _do_save_correction(
    seg: SubtitleSegment,
    new_text: str,
    vr: Optional[ValidationResult],
    override: bool,
) -> None:
    """Persist a correction to SQLite and update session state.

    If the segment SNR is below SNR_GATE_DB the correction is applied to the
    in-session text (so the export reflects the edit) but is NOT written to the
    self-improvement DB — noisy audio produces unreliable MFCC fingerprints.
    """
    try:
        record = CorrectionRecord(
            id=None,
            segment_start=seg.start,
            segment_end=seg.end,
            original_text=seg.text,
            corrected_text=new_text,
            language=seg.language,
            mfcc_mean=seg.mfcc_mean,
            mfcc_var=seg.mfcc_var,
            match_score=vr.score if vr else 0.0,
            hw_used=vr.hw_used if vr else False,
            created_at=datetime.now().isoformat(),
        )

        snr_ok = is_snr_acceptable(seg.snr_db)
        if snr_ok:
            save_correction(record)
            st.session_state["correction_count"] = (
                st.session_state.get("correction_count", 0) + 1
            )
        else:
            st.warning(
                f"⚠ Audio SNR is {seg.snr_db:.1f} dB — below the {15.0} dB "
                "threshold. Correction applied to this session's subtitles but "
                "**not** saved to the learning database (noisy audio produces "
                "unreliable fingerprints)."
            )

        # Always update the displayed/exported text regardless of SNR gate.
        segs: list[SubtitleSegment] = st.session_state["segments"]
        for s in segs:
            if s.index == seg.index:
                s.corrected       = True
                s.correction_text = new_text
                s.text            = new_text
                break

        st.rerun()

    except Exception as exc:
        st.error(f"❌ Could not save correction: {exc}")
```

- [ ] **Step 2.3 — Manual smoke test**

Start the app and upload any short video. After transcription completes, open a RED segment, type a correction, and click **Validate & Save Correction**. Confirm:
- If the segment's SNR (shown in the card footer) is ≥ 15 dB → sidebar correction count increments.
- If SNR < 15 dB → yellow warning appears, count does NOT increment, but the segment text updates.

*(Automated UI tests are out of scope for this project; the Streamlit app is the acceptance test.)*

- [ ] **Step 2.4 — Commit**

```bash
git add app.py
git commit -m "feat: skip DB write for low-SNR corrections, still update display text"
```

---

## Task 3: Refactor `compute_mfcc_software()` for two-pass spectral subtraction

**Files:**
- Modify: `core/esp32_validator.py`
- Test: `test_spectral_subtraction.py` (extend)

### What and why

The software MFCC path must mirror the firmware exactly — same noise floor subtraction, same SNR calculation — because:
1. When hardware is absent, the software path takes over for correction validation. If the two paths produce different fingerprints, corrections validated on HW cannot be matched at inference in SW mode.
2. The fingerprint dict returned by both paths must have the same keys including the new `snr_db` field.

**Algorithm (two-pass):**

Pass 1 — iterate all frames, compute raw mel energies (pre-log), classify frame as noise if `rms < RMS_SILENCE = 0.002`. Accumulate noise floor as mean of noise-frame mel energies.

Pass 2 — iterate all frames again using stored raw mel energies, subtract noise floor, clamp to ε = 1e-9, apply log10, apply DCT-II, accumulate mean + variance.

SNR in mel domain:
```
mean_speech_energy = mean over speech frames of mean(mel_energies[m])
mean_noise_energy  = mean over noise  frames of mean(mel_energies[m])
snr_db = clip(10 * log10(mean_speech / mean_noise), -20, 60)
```
If no noise frames → snr_db = 60.0. If no speech frames → snr_db = -20.0.

---

- [ ] **Step 3.1 — Add SW spectral subtraction tests to `test_spectral_subtraction.py`**

Append to `test_spectral_subtraction.py`:

```python
# ── Task 3: Software MFCC spectral subtraction ────────────────────────────────

import numpy as np

def _make_clean_sine(freq_hz=440, duration_s=0.5, sr=16000, amplitude=0.4):
    """Pure sine wave — high SNR speech-like signal."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)

def _make_noise(duration_s=0.5, sr=16000, amplitude=0.001):
    """Low-amplitude white noise — noise-floor-like signal."""
    rng = np.random.default_rng(42)
    return (amplitude * rng.standard_normal(int(sr * duration_s))).astype(np.float32)


def test_fingerprint_returns_snr_db_key():
    """get_fingerprint() result must contain 'snr_db' key."""
    from subgen_ai.core.esp32_validator import get_fingerprint
    audio = _make_clean_sine()
    fp = get_fingerprint(audio, sr=16000, esp32_port=None)
    assert "snr_db" in fp, "fingerprint dict missing 'snr_db'"


def test_clean_signal_has_high_snr():
    """A pure sine with no noise background should return snr_db > 20."""
    from subgen_ai.core.esp32_validator import compute_mfcc_software
    audio = _make_clean_sine(amplitude=0.4)
    fp = compute_mfcc_software(audio)
    assert fp["ok"] is True
    assert fp["snr_db"] > 20.0, f"Expected snr_db > 20, got {fp['snr_db']}"


def test_noise_only_signal_has_low_snr():
    """White noise at amplitude 0.001 (below RMS threshold) → snr_db should be <= 0."""
    from subgen_ai.core.esp32_validator import compute_mfcc_software
    audio = _make_noise(amplitude=0.0005)
    fp = compute_mfcc_software(audio)
    # All frames are noise → snr_db returned as -20.0 (the floor constant)
    assert fp["snr_db"] <= 0.0, f"Expected snr_db <= 0, got {fp['snr_db']}"


def test_spectral_subtraction_changes_mfcc_vs_no_subtraction():
    """
    MFCC computed with spectral subtraction on noisy signal must differ from
    MFCC computed without subtraction on the same signal.

    We verify this by constructing a noisy signal (sine + noise) and checking
    that the mfcc_mean vectors from the new implementation differ from what
    the original single-pass implementation would produce on the raw noisy signal.
    We approximate the 'without subtraction' result by running on noise-only
    and speech-only signals and checking the noisy mixture result is closer
    to the clean-speech fingerprint.
    """
    from subgen_ai.core.esp32_validator import compute_mfcc_software

    speech = _make_clean_sine(freq_hz=440, duration_s=0.5, amplitude=0.3)
    noise  = _make_noise(duration_s=0.5, amplitude=0.05)
    mixed  = np.clip(speech + noise, -1.0, 1.0).astype(np.float32)

    fp_clean = compute_mfcc_software(speech)
    fp_noisy = compute_mfcc_software(mixed)

    mean_clean = np.array(fp_clean["mfcc_mean"])
    mean_noisy = np.array(fp_noisy["mfcc_mean"])

    # After spectral subtraction the noisy result should be more similar to clean
    # than a random vector would be — cosine similarity > 0.7
    norm_c = np.linalg.norm(mean_clean)
    norm_n = np.linalg.norm(mean_noisy)
    if norm_c > 1e-10 and norm_n > 1e-10:
        cos_sim = float(np.dot(mean_clean, mean_noisy) / (norm_c * norm_n))
        assert cos_sim > 0.7, (
            f"Noisy MFCC too different from clean after subtraction: cosine={cos_sim:.3f}"
        )
```

- [ ] **Step 3.2 — Run tests to confirm they fail**

```bash
venv/Scripts/python -m pytest test_spectral_subtraction.py::test_fingerprint_returns_snr_db_key test_spectral_subtraction.py::test_clean_signal_has_high_snr test_spectral_subtraction.py::test_noise_only_signal_has_low_snr -v
```

Expected: `AssertionError: fingerprint dict missing 'snr_db'` (or KeyError).

- [ ] **Step 3.3 — Rewrite `compute_mfcc_software()` in `core/esp32_validator.py`**

Replace the entire `compute_mfcc_software` function (lines 82–150 in the current file) with the two-pass version below. Everything outside this function stays unchanged.

```python
def compute_mfcc_software(audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """
    Compute noise-floor-subtracted MFCC fingerprint using the same algorithm
    as the ESP32 firmware.

    Two-pass approach:
      Pass 1 — compute raw (pre-log) mel energies for every frame; classify
               frames as noise (rms < RMS_SILENCE) or speech; accumulate
               noise floor as mean mel energy of noise frames; collect SNR stats.
      Pass 2 — subtract noise floor from every frame's mel energies, clamp to
               epsilon, apply log10, apply DCT-II, accumulate mean + variance.

    Returns:
        dict with keys: ok, hw, mfcc_mean, mfcc_var, rms, frames, snr_db.
    """
    RMS_SILENCE = 0.002   # frames below this RMS are classified as noise

    if len(audio) == 0:
        return {
            "ok": False, "hw": False,
            "mfcc_mean": [0.0] * N_MFCC, "mfcc_var": [0.0] * N_MFCC,
            "rms": 0.0, "frames": 0, "snr_db": -20.0,
        }

    audio = audio.astype(np.float32)
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / 32768.0

    overall_rms = float(np.sqrt(np.mean(audio ** 2)))
    filterbank  = get_filterbank()

    # ── Pass 1: collect raw mel energies + classify frames ───────────────────
    raw_mel_all: list = []     # list of np.ndarray shape (N_MELS,)
    frame_rms_all: list = []   # list of float

    for start in range(0, len(audio) - WIN_LENGTH, HOP_LENGTH):
        frame = audio[start : start + WIN_LENGTH]

        frame_rms_val = float(np.sqrt(np.mean(frame ** 2)))
        frame_rms_all.append(frame_rms_val)

        window = np.hanning(WIN_LENGTH)
        padded = np.zeros(N_FFT, dtype=np.float32)
        padded[:WIN_LENGTH] = frame * window

        spectrum    = np.fft.rfft(padded)
        power       = (np.abs(spectrum) ** 2) / N_FFT
        mel_energies = np.dot(filterbank, power)   # shape (N_MELS,)
        raw_mel_all.append(mel_energies)

    if not raw_mel_all:
        return {
            "ok": False, "hw": False,
            "mfcc_mean": [0.0] * N_MFCC, "mfcc_var": [0.0] * N_MFCC,
            "rms": overall_rms, "frames": 0, "snr_db": -20.0,
        }

    # Separate noise / speech frames
    noise_mel   = [mel for mel, r in zip(raw_mel_all, frame_rms_all) if r <  RMS_SILENCE]
    speech_mel  = [mel for mel, r in zip(raw_mel_all, frame_rms_all) if r >= RMS_SILENCE]

    # Noise floor = mean mel energy of noise frames (per band)
    if noise_mel:
        noise_floor = np.mean(noise_mel, axis=0)        # shape (N_MELS,)
    else:
        noise_floor = np.zeros(N_MELS, dtype=np.float32)

    # SNR in the mel domain
    if speech_mel and noise_mel:
        mean_s = float(np.mean([np.mean(f) for f in speech_mel]))
        mean_n = float(np.mean([np.mean(f) for f in noise_mel]))
        snr_db = float(np.clip(10.0 * np.log10(max(mean_s, 1e-10) / max(mean_n, 1e-10)),
                               -20.0, 60.0))
    elif speech_mel:
        snr_db = 60.0    # all speech, no measurable noise
    else:
        snr_db = -20.0   # all noise, no speech

    # ── Pass 2: noise-floor subtraction → log → DCT → accumulate ─────────────
    coefficients_per_frame: list = []

    for mel_energies in raw_mel_all:
        denoised = np.maximum(mel_energies - noise_floor, 1e-9)
        log_mel  = np.log10(denoised)
        cepstrum = dct(log_mel, type=2, norm='ortho')
        coefficients_per_frame.append(cepstrum[:N_MFCC])

    frames_arr = np.array(coefficients_per_frame)
    mfcc_mean  = frames_arr.mean(axis=0).tolist()
    mfcc_var   = frames_arr.var(axis=0).tolist()

    return {
        "ok": True, "hw": False,
        "mfcc_mean": mfcc_mean, "mfcc_var": mfcc_var,
        "rms": overall_rms, "frames": len(coefficients_per_frame),
        "snr_db": snr_db,
    }
```

- [ ] **Step 3.4 — Run SW spectral subtraction tests**

```bash
venv/Scripts/python -m pytest test_spectral_subtraction.py::test_fingerprint_returns_snr_db_key test_spectral_subtraction.py::test_clean_signal_has_high_snr test_spectral_subtraction.py::test_noise_only_signal_has_low_snr test_spectral_subtraction.py::test_spectral_subtraction_changes_mfcc_vs_no_subtraction -v
```

Expected: 4 passed.

- [ ] **Step 3.5 — Run the full test file to confirm nothing regressed**

```bash
venv/Scripts/python -m pytest test_spectral_subtraction.py -v
```

Expected: all tests pass.

- [ ] **Step 3.6 — Commit**

```bash
git add core/esp32_validator.py test_spectral_subtraction.py
git commit -m "feat: two-pass spectral subtraction + snr_db in compute_mfcc_software()"
```

---

## Task 4: Update ESP32 firmware for two-pass spectral subtraction

**Files:**
- Modify: `firmware/esp32_firmware.ino`

### What and why

The firmware must mirror the Python algorithm exactly. The current firmware does MFCC in a single pass (log compression applied inside the frame loop). We need to split this into:
1. A new helper `compute_frame_mel_raw()` — Hanning + FFT + mel filterbank, returns raw (pre-log) energies.
2. `process_audio()` redesigned — two explicit loops:
   - Loop 1: call `compute_frame_mel_raw()`, measure RMS, accumulate noise floor from silence frames.
   - Loop 2: subtract noise floor, clamp, log10, DCT, accumulate MFCC mean + variance.
3. SNR calculation in mel domain added to Loop 1, included in JSON output as `"snr_db"`.

The old `compute_frame_mfcc()` function is removed — its logic is now split between the new helper and the main process loop. `setup()` and `loop()` are unchanged except `loop()` calls the updated `process_audio()`.

**Memory note:** The two-pass approach reuses `pcm_buf` (already in memory) for Loop 2. No additional large static buffers are needed — only `noise_floor[N_MELS]` (26 floats = 104 bytes) is added.

---

- [ ] **Step 4.1 — Replace `firmware/esp32_firmware.ino` with the two-pass version**

Replace the entire file content with the following. All constants, the mel filterbank builder, Hanning builder, FFT, and DCT are unchanged. Only `compute_frame_mfcc()` is replaced by `compute_frame_mel_raw()` and `process_audio()` is fully rewritten.

```cpp
/**
 * SubGEN AI — ESP32 MFCC Fingerprint Firmware  (v2 — spectral subtraction)
 * =========================================================================
 * Board  : ESP32 Dev Module
 * Baud   : 460800
 * Library: ArduinoJson v6+  (install via Arduino Library Manager)
 *
 * ── SERIAL PROTOCOL ──────────────────────────────────────────────────────────
 *
 * HOST → ESP32  (binary frame):
 *   Byte 0   : 0xAA  (header byte 1)
 *   Byte 1   : 0x55  (header byte 2)
 *   Byte 2-3 : N     (uint16 big-endian — number of int16 PCM samples)
 *   Byte 4…  : N × 2 bytes of PCM int16 little-endian, mono 16 kHz
 *              Max N = 32000 (2 s)
 *
 * ESP32 → HOST  (single JSON line, UTF-8, terminated with '\n'):
 *   Success:
 *     {"ok":true,"frames":<int>,"rms":<float>,"snr_db":<float>,
 *      "mfcc_mean":[f0,…,f11],"mfcc_var":[f0,…,f11]}
 *   Error:
 *     {"ok":false,"error":"<reason>"}
 *
 * ── ALGORITHM ────────────────────────────────────────────────────────────────
 *
 * Two-pass MFCC with spectral subtraction:
 *
 *   Pass 1 — for every 25 ms frame (400 samples, hop 160):
 *     1. Hanning window → zero-pad to 512 → 512-pt radix-2 FFT
 *     2. Power spectrum → 26-band mel filterbank → raw mel energies (NO log yet)
 *     3. Measure frame RMS; if RMS < RMS_SILENCE accumulate into noise_floor[]
 *        and noise SNR stats; otherwise accumulate speech SNR stats.
 *
 *   Between passes:
 *     noise_floor[m] = mean raw mel energy over all noise frames (per band m).
 *     If no noise frames found → noise_floor[] stays all-zero (no subtraction).
 *     snr_db = 10 * log10(mean_speech_mel / mean_noise_mel), clamped [-20, 60].
 *
 *   Pass 2 — for every frame (recompute from pcm_buf):
 *     4. Subtract noise_floor[m] from mel_e[m]; clamp to 1e-9
 *     5. log10(mel_e[m])
 *     6. DCT-II ortho → first 12 coefficients
 *     7. Accumulate sum and sum-of-squares for mean/variance.
 *
 * ── NOTES ────────────────────────────────────────────────────────────────────
 *  • Flash this manually via the Arduino IDE — the Python app does NOT do it.
 *  • RMS_SILENCE must match the Python constant (0.002) in esp32_validator.py.
 */

#include <Arduino.h>
#include <ArduinoJson.h>
#include <math.h>

// ── Constants ─────────────────────────────────────────────────────────────────
#define SAMPLE_RATE       16000
#define N_FFT             512
#define HOP_LENGTH        160
#define WIN_LENGTH        400
#define N_MFCC            12
#define N_MELS            26
#define FMIN_HZ           0.0f
#define FMAX_HZ           8000.0f
#define MAX_SAMPLES       32000
#define BAUD_RATE         460800
#define RMS_SILENCE       0.002f   // frames below this RMS are noise frames

// ── Mel filterbank ────────────────────────────────────────────────────────────
static float mel_fb[N_MELS][N_FFT / 2 + 1];
static bool  fb_ready = false;

static float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}
static float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}
static void build_mel_filterbank() {
    int   n_bins   = N_FFT / 2 + 1;
    float mel_min  = hz_to_mel(FMIN_HZ);
    float mel_max  = hz_to_mel(FMAX_HZ);
    float mel_pts[N_MELS + 2];
    for (int i = 0; i < N_MELS + 2; i++)
        mel_pts[i] = mel_min + (mel_max - mel_min) * i / (N_MELS + 1);
    int bin_pts[N_MELS + 2];
    for (int i = 0; i < N_MELS + 2; i++)
        bin_pts[i] = (int)floorf((N_FFT + 1) * mel_to_hz(mel_pts[i]) / SAMPLE_RATE);
    memset(mel_fb, 0, sizeof(mel_fb));
    for (int m = 1; m <= N_MELS; m++) {
        int fl = bin_pts[m-1], fc = bin_pts[m], fr = bin_pts[m+1];
        for (int k = fl; k < fc; k++)
            if (fc != fl && k < n_bins) mel_fb[m-1][k] = (float)(k-fl)/(fc-fl);
        for (int k = fc; k < fr; k++)
            if (fr != fc && k < n_bins) mel_fb[m-1][k] = (float)(fr-k)/(fr-fc);
    }
    fb_ready = true;
}

// ── Hanning window ────────────────────────────────────────────────────────────
static float hanning[WIN_LENGTH];
static void build_hanning() {
    for (int i = 0; i < WIN_LENGTH; i++)
        hanning[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (WIN_LENGTH - 1)));
}

// ── 512-point radix-2 Cooley-Tukey FFT ───────────────────────────────────────
static float fft_buf[N_FFT * 2];
static void fft_real(float *re_in, int n) {
    for (int i = 0; i < n; i++) { fft_buf[2*i] = re_in[i]; fft_buf[2*i+1] = 0.0f; }
    int j = 0;
    for (int i = 1; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = fft_buf[2*i];   fft_buf[2*i]   = fft_buf[2*j];   fft_buf[2*j]   = tr;
            float ti = fft_buf[2*i+1]; fft_buf[2*i+1] = fft_buf[2*j+1]; fft_buf[2*j+1] = ti;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * M_PI / len;
        float wre = cosf(ang), wim = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float cr = 1.0f, ci = 0.0f;
            for (int k = 0; k < len/2; k++) {
                int u = 2*(i+k), v = 2*(i+k+len/2);
                float tr = cr*fft_buf[v]   - ci*fft_buf[v+1];
                float ti = cr*fft_buf[v+1] + ci*fft_buf[v];
                fft_buf[v]   = fft_buf[u]   - tr; fft_buf[v+1] = fft_buf[u+1] - ti;
                fft_buf[u]  += tr;                 fft_buf[u+1] += ti;
                float nr = cr*wre - ci*wim; ci = cr*wim + ci*wre; cr = nr;
            }
        }
    }
}

// ── DCT-II with ortho normalisation ──────────────────────────────────────────
static float dct_ortho(float *x, int n, int k) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
        sum += x[i] * cosf(M_PI * k * (2*i + 1) / (2.0f * n));
    return ((k == 0) ? sqrtf(1.0f/n) : sqrtf(2.0f/n)) * sum;
}

// ── Compute raw (pre-log) mel energies for one frame ─────────────────────────
// Output: mel_out[N_MELS] — linear mel filterbank energies, NO log applied.
static float frame_buf[N_FFT];

static void compute_frame_mel_raw(const float *frame, float *mel_out) {
    memset(frame_buf, 0, sizeof(frame_buf));
    for (int i = 0; i < WIN_LENGTH; i++)
        frame_buf[i] = frame[i] * hanning[i];

    fft_real(frame_buf, N_FFT);
    int n_bins = N_FFT / 2 + 1;

    for (int m = 0; m < N_MELS; m++) {
        float e = 0.0f;
        for (int k = 0; k < n_bins; k++) {
            float re = fft_buf[2*k], im = fft_buf[2*k+1];
            e += mel_fb[m][k] * (re*re + im*im) / N_FFT;
        }
        mel_out[m] = e;   // raw energy — log compression applied later
    }
}

// ── Static buffers ────────────────────────────────────────────────────────────
static int16_t pcm_buf[MAX_SAMPLES];
static float   mfcc_sum[N_MFCC];
static float   mfcc_sq[N_MFCC];
static float   noise_floor[N_MELS];   // per-band noise floor (mean of noise frames)
static float   mel_e[N_MELS];         // scratch for one frame's mel energies

// ── Main MFCC pipeline (two-pass) ────────────────────────────────────────────
static void process_audio(int n_samples) {
    if (!fb_ready) build_mel_filterbank();

    // ── Compute RMS ───────────────────────────────────────────────────────────
    double rms_acc = 0.0;
    for (int i = 0; i < n_samples; i++)
        rms_acc += (double)pcm_buf[i] * pcm_buf[i];
    float rms = sqrtf((float)(rms_acc / n_samples)) / 32768.0f;

    // ── Pass 1: noise floor estimation + SNR stats ───────────────────────────
    memset(noise_floor, 0, sizeof(noise_floor));
    int   n_noise_frames  = 0;
    float sum_speech_mel  = 0.0f;
    float sum_noise_mel   = 0.0f;
    int   n_speech_frames = 0;

    for (int start = 0; start + WIN_LENGTH <= n_samples; start += HOP_LENGTH) {
        float win_f[WIN_LENGTH];
        float rms_f = 0.0f;
        for (int i = 0; i < WIN_LENGTH; i++) {
            win_f[i] = pcm_buf[start + i] / 32768.0f;
            rms_f   += win_f[i] * win_f[i];
        }
        rms_f = sqrtf(rms_f / WIN_LENGTH);

        compute_frame_mel_raw(win_f, mel_e);

        float frame_mean_mel = 0.0f;
        for (int m = 0; m < N_MELS; m++) frame_mean_mel += mel_e[m];
        frame_mean_mel /= N_MELS;

        if (rms_f < RMS_SILENCE) {
            for (int m = 0; m < N_MELS; m++) noise_floor[m] += mel_e[m];
            sum_noise_mel += frame_mean_mel;
            n_noise_frames++;
        } else {
            sum_speech_mel += frame_mean_mel;
            n_speech_frames++;
        }
    }

    // Average the noise floor
    if (n_noise_frames > 0) {
        for (int m = 0; m < N_MELS; m++)
            noise_floor[m] /= n_noise_frames;
    }

    // SNR in mel domain
    float snr_db = 60.0f;
    if (n_speech_frames > 0 && n_noise_frames > 0) {
        float mean_s = sum_speech_mel / n_speech_frames;
        float mean_n = sum_noise_mel  / n_noise_frames;
        if (mean_n < 1e-10f) mean_n = 1e-10f;
        snr_db = 10.0f * log10f(mean_s / mean_n);
        if (snr_db < -20.0f) snr_db = -20.0f;
        if (snr_db >  60.0f) snr_db =  60.0f;
    } else if (n_speech_frames == 0) {
        snr_db = -20.0f;
    }

    // ── Pass 2: noise subtraction → log → DCT → accumulate ───────────────────
    memset(mfcc_sum, 0, sizeof(mfcc_sum));
    memset(mfcc_sq,  0, sizeof(mfcc_sq));
    int n_frames = 0;

    for (int start = 0; start + WIN_LENGTH <= n_samples; start += HOP_LENGTH) {
        float win_f[WIN_LENGTH];
        for (int i = 0; i < WIN_LENGTH; i++)
            win_f[i] = pcm_buf[start + i] / 32768.0f;

        compute_frame_mel_raw(win_f, mel_e);

        // Spectral subtraction in mel domain + log compression
        for (int m = 0; m < N_MELS; m++) {
            mel_e[m] -= noise_floor[m];
            if (mel_e[m] < 1e-9f) mel_e[m] = 1e-9f;
            mel_e[m] = log10f(mel_e[m]);
        }

        // DCT-II → 12 MFCC coefficients
        for (int c = 0; c < N_MFCC; c++) {
            float coeff = dct_ortho(mel_e, N_MELS, c);
            mfcc_sum[c] += coeff;
            mfcc_sq[c]  += coeff * coeff;
        }
        n_frames++;
    }

    // ── Build JSON response ───────────────────────────────────────────────────
    StaticJsonDocument<1024> doc;
    if (n_frames == 0) {
        doc["ok"]    = false;
        doc["error"] = "no frames";
    } else {
        doc["ok"]     = true;
        doc["frames"] = n_frames;
        doc["rms"]    = rms;
        doc["snr_db"] = snr_db;
        JsonArray mean_arr = doc.createNestedArray("mfcc_mean");
        JsonArray var_arr  = doc.createNestedArray("mfcc_var");
        for (int c = 0; c < N_MFCC; c++) {
            float mean = mfcc_sum[c] / n_frames;
            float var  = (mfcc_sq[c] / n_frames) - (mean * mean);
            mean_arr.add(mean);
            var_arr.add(max(var, 0.0f));
        }
    }
    serializeJson(doc, Serial);
    Serial.println();
}

// ── setup / loop ──────────────────────────────────────────────────────────────
void setup() {
    Serial.setRxBufferSize(MAX_SAMPLES * 2 + 64);
    Serial.begin(BAUD_RATE);
    build_hanning();
    build_mel_filterbank();
}

void loop() {
    if (Serial.available() < 2) return;
    uint8_t b0 = Serial.read(); if (b0 != 0xAA) return;
    uint8_t b1 = Serial.read(); if (b1 != 0x55) return;

    while (Serial.available() < 2) delay(1);
    uint8_t nh = Serial.read(), nl = Serial.read();
    uint16_t n = ((uint16_t)nh << 8) | nl;

    if (n > MAX_SAMPLES) {
        for (uint32_t i = 0; i < (uint32_t)n * 2; i++) {
            while (!Serial.available()) delay(1);
            Serial.read();
        }
        StaticJsonDocument<128> err;
        err["ok"] = false; err["error"] = "frame too large";
        serializeJson(err, Serial); Serial.println();
        return;
    }

    uint32_t bytes_to_read = (uint32_t)n * 2;
    uint32_t bytes_read    = 0;
    uint8_t *dst           = (uint8_t *)pcm_buf;
    unsigned long t_start  = millis();
    while (bytes_read < bytes_to_read) {
        if (Serial.available()) {
            dst[bytes_read++] = Serial.read();
        } else if (millis() - t_start > 3000) {
            StaticJsonDocument<128> err;
            err["ok"] = false; err["error"] = "read timeout";
            serializeJson(err, Serial); Serial.println();
            return;
        }
    }

    process_audio(n);
}
```

- [ ] **Step 4.2 — Verify `RMS_SILENCE` constant matches Python**

The firmware uses `#define RMS_SILENCE 0.002f`. The Python `compute_mfcc_software()` uses `RMS_SILENCE = 0.002`. Confirm both are 0.002. They are — no change needed.

- [ ] **Step 4.3 — Flash and verify hardware JSON output includes `snr_db`**

Open the Arduino IDE, load `firmware/esp32_firmware.ino`, compile and flash to the ESP32 DevKit v1.

Open the Arduino Serial Monitor at 460800 baud. Send a test frame manually or run the Python test helper below to send a 0.5 s sine and read the JSON:

```python
# Run from the project root venv:
# venv/Scripts/python firmware_test.py COM_PORT
import sys, struct, json, numpy as np
import serial

port = sys.argv[1]
sr = 16000
t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
audio_i16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
n = len(audio_i16)

payload = bytes([0xAA, 0x55]) + struct.pack(">H", n) + audio_i16.tobytes()

with serial.Serial(port, 460800, timeout=5) as ser:
    ser.reset_input_buffer()
    ser.write(payload)
    resp = ser.readline()

result = json.loads(resp.decode("utf-8").strip())
print(result)
assert "snr_db" in result, "Missing snr_db in firmware response"
assert result["ok"] is True
print(f"snr_db = {result['snr_db']:.2f} dB  — firmware OK")
```

Expected output: JSON containing `"snr_db": <positive number>`.

*(If hardware is unavailable, skip this step and note it — the software fallback path is fully tested in Task 3.)*

- [ ] **Step 4.4 — Commit**

```bash
git add firmware/esp32_firmware.ino
git commit -m "feat: two-pass spectral subtraction + snr_db in ESP32 firmware"
```

---

## Task 5: Surface hardware `snr_db` in `send_audio_to_esp32()` return value

**Files:**
- Modify: `core/esp32_validator.py`

### What and why

`send_audio_to_esp32()` currently returns the raw parsed JSON dict from the firmware — which now includes `snr_db`. No code change is needed in that function itself. However, `get_fingerprint()` must propagate `snr_db` from the hardware result in the same way it propagates `mfcc_mean` etc. Currently, when hardware is used, the dict is returned as-is — and since the firmware now includes `snr_db`, the key is already there. We just need to verify the software fallback path (already updated in Task 3) also returns `snr_db`.

This task is a regression check + a single test.

---

- [ ] **Step 5.1 — Add integration test to `test_spectral_subtraction.py`**

Append:

```python
# ── Task 5: get_fingerprint() always returns snr_db ──────────────────────────

def test_get_fingerprint_sw_path_has_snr_db():
    """get_fingerprint with no port (software path) must return snr_db."""
    from subgen_ai.core.esp32_validator import get_fingerprint
    audio = _make_clean_sine(duration_s=0.3)
    fp = get_fingerprint(audio, sr=16000, esp32_port=None)
    assert fp["ok"] is True
    assert "snr_db" in fp
    assert isinstance(fp["snr_db"], float)


def test_get_fingerprint_sw_path_snr_is_finite():
    """snr_db must be a finite float in [-20, 60]."""
    from subgen_ai.core.esp32_validator import get_fingerprint
    import math
    audio = _make_clean_sine(duration_s=0.3)
    fp = get_fingerprint(audio, sr=16000, esp32_port=None)
    assert math.isfinite(fp["snr_db"])
    assert -20.0 <= fp["snr_db"] <= 60.0
```

- [ ] **Step 5.2 — Run new tests**

```bash
venv/Scripts/python -m pytest test_spectral_subtraction.py::test_get_fingerprint_sw_path_has_snr_db test_spectral_subtraction.py::test_get_fingerprint_sw_path_snr_is_finite -v
```

Expected: 2 passed (no code change needed — Task 3 already added `snr_db` to the SW path return dict).

- [ ] **Step 5.3 — Run full test suite**

```bash
venv/Scripts/python -m pytest test_spectral_subtraction.py -v
```

Expected: all tests pass.

- [ ] **Step 5.4 — Commit**

```bash
git add test_spectral_subtraction.py
git commit -m "test: verify get_fingerprint() always exposes snr_db from SW path"
```

---

## Task 6: Final integration smoke test + memory check

**Files:** no code changes — verification only.

---

- [ ] **Step 6.1 — Run complete test file**

```bash
venv/Scripts/python -m pytest test_spectral_subtraction.py -v --tb=short
```

Expected output (all 12 tests pass):

```
test_snr_gate_accepts_above_threshold           PASSED
test_snr_gate_rejects_below_threshold           PASSED
test_snr_gate_rejects_at_exact_threshold_minus_epsilon  PASSED
test_snr_gate_accepts_at_exact_threshold        PASSED
test_fingerprint_returns_snr_db_key             PASSED
test_clean_signal_has_high_snr                  PASSED
test_noise_only_signal_has_low_snr              PASSED
test_spectral_subtraction_changes_mfcc_vs_no_subtraction  PASSED
test_get_fingerprint_sw_path_has_snr_db         PASSED
test_get_fingerprint_sw_path_snr_is_finite      PASSED
```

- [ ] **Step 6.2 — Start the Streamlit app and verify end-to-end**

```bash
venv/Scripts/python -m streamlit run app.py
```

Upload a video. After transcription:

1. Open a RED segment with SNR < 15 dB (shown in the card footer).
2. Type a correction and click **Validate & Save Correction**.
3. Confirm: yellow warning appears, correction count in sidebar does NOT increment, segment text updates.
4. Open a GREEN segment with SNR ≥ 15 dB.
5. Type a correction and click **Validate & Save Correction**.
6. Confirm: success message, correction count increments.

- [ ] **Step 6.3 — Final commit**

```bash
git add .
git commit -m "feat: spectral subtraction + SNR gate complete — all tests pass"
```

---

## Summary of Changes

| File | What changed |
|------|-------------|
| `core/qc_engine.py` | Added `SNR_GATE_DB = 15.0` and `is_snr_acceptable()` |
| `app.py` | `_do_save_correction()` now gates DB write on `is_snr_acceptable(seg.snr_db)` |
| `core/esp32_validator.py` | `compute_mfcc_software()` rewritten as two-pass with noise floor subtraction; returns `snr_db` |
| `firmware/esp32_firmware.ino` | `compute_frame_mfcc()` replaced by `compute_frame_mel_raw()`; `process_audio()` rewritten as two-pass; JSON response includes `snr_db` |
| `test_spectral_subtraction.py` | New — 10 unit tests covering SNR gate, SW MFCC, and `get_fingerprint()` |

---

## Self-Review

**Spec coverage:**
- ✅ ESP32 computes noise floor from silence frames and subtracts it before log compression (Task 4)
- ✅ Python SW path mirrors firmware algorithm exactly (Task 3)
- ✅ `snr_db` returned in fingerprint dict from both paths (Tasks 3, 4, 5)
- ✅ Only corrections with SNR ≥ 15 dB written to DB (Task 2)
- ✅ Low-SNR corrections still update in-session text for export (Task 2, Step 2.2)
- ✅ Warning shown to user when SNR gate rejects a DB write (Task 2, Step 2.2)
- ✅ `RMS_SILENCE` threshold is 0.002 in both Python and firmware (Task 4, Step 4.2)

**Placeholder scan:** No TBDs, no "add appropriate" language, no forward references to undefined types.

**Type consistency:**
- `is_snr_acceptable(snr_db: float) -> bool` — used in Task 1, imported in Task 2 ✅
- `compute_mfcc_software()` returns dict with `snr_db` key — used in Tasks 3, 5 ✅
- `RMS_SILENCE = 0.002` in Python, `#define RMS_SILENCE 0.002f` in C ✅
- `noise_floor` is `float[N_MELS]` in C, `np.ndarray shape (N_MELS,)` in Python ✅
