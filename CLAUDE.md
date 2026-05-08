# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SubGEN AI** is an offline, hardware-aware subtitle generator for Indic languages (Tamil, Telugu, Hindi, Kannada, Malayalam). It combines:
- **Faster-Whisper ASR** for speech-to-text transcription
- **ESP32 MFCC co-processor** (with software fallback) for audio fingerprinting
- **Two-pass spectral subtraction + SNR gating** for audio quality control
- **Self-improving correction database** (SQLite) using MFCC fingerprint matching

**Key insight**: Audio quality determines whether corrections are persisted. Noisy segments (<15 dB SNR) update in-session text but skip database storage to prevent fingerprint corruption.

---

## Architecture & Data Flow

### High-Level Pipeline

1. **Input** → Video/audio file upload via Streamlit UI
2. **Audio Extraction** → ffmpeg extracts mono 16 kHz WAV
3. **Transcription** → Faster-Whisper ASR (standard or Indic-specific model)
4. **Per-Segment QC**:
   - Extract audio sub-clip with padding (±100 ms)
   - Compute MFCC fingerprint (hardware ESP32 or software fallback)
   - Compute SNR via two-pass spectral subtraction
   - Compute fused confidence (0.6×ASR + 0.3×SNR_term + 0.1×speaker_stability)
   - Assign GREEN (≥0.75) or RED (<0.75) label
   - Lookup nearest correction in DB (cosine+Euclidean similarity)
   - Auto-apply correction if SNR ≥ 15 dB
5. **User Editing** → Review & Edit tab (corrections validated)
6. **Export** → SRT, VTT, JSON, or burn-in to video

### Module Responsibilities

#### Core Modules

- **`core/transcriber.py`**: Transcription pipeline orchestration
  - Loads Faster-Whisper (cached, supports standard/Indic/HF repo IDs)
  - Extracts audio, runs ASR, spawns per-segment QC tasks
  - Returns `List[SubtitleSegment]` with all QC metadata

- **`core/qc_engine.py`**: Signal-informed quality control
  - `compute_asr_conf()` → Whisper logprob → [0,1] confidence
  - `compute_snr_penalty()` → 16-window energy analysis → [0,1] penalty
  - `compute_fused_conf()` → Weighted combination of ASR, SNR, speaker
  - `label_segment()` → RED/GREEN threshold (0.75)
  - `validate_correction()` → Compare fingerprints, return tier (HIGH/MEDIUM/MISMATCH)
  - **SNR gate**: `is_snr_acceptable(snr_db)` returns true if ≥ 15 dB

- **`core/esp32_validator.py`**: MFCC fingerprinting (dual-mode)
  - `get_fingerprint()` → Try ESP32 via serial; fallback to software
  - `compute_mfcc_software()` → Two-pass spectral subtraction pipeline
    - Pass 1: Estimate mel-domain noise floor from silence frames (RMS < 0.002)
    - Pass 2: Subtract noise, apply log, DCT-II
    - Returns dict with `mfcc_mean`, `mfcc_var`, `snr_db`, `ok` flag
  - Serial protocol: 2 Mbps baud, ~0.3 s latency for 64 KB audio

#### UI & Export

- **`app.py`**: Streamlit single-page app (3 tabs)
  - **Tab 1 (Transcribe)**: File upload → ASR pipeline → live progress
  - **Tab 2 (Review)**: Segment browser, RED/GREEN filter, inline correction validation
  - **Tab 3 (Export)**: SRT/VTT/JSON/burn-in downloads
  - **Sidebar**: Hardware status, model selection (standard or custom HF ID), DB stats
  - All state in `st.session_state` (no module-level globals for Streamlit compat)

- **`export/formatters.py`**: Output format converters
  - `to_srt()`, `to_vtt()`, `to_json()` → string output
  - `to_burn_in()` → ffmpeg overlay → video bytes

#### Persistence

- **`db/correction_store.py`**: SQLite auto-improvement DB
  - Location: `~/.subgen_ai/corrections.db`
  - Schema: `id, segment_start, segment_end, original_text, corrected_text, language, mfcc_mean, mfcc_var, match_score, hw_used, created_at`
  - Indexed on `language` for fast lookup
  - `find_nearest_correction()` → cosine similarity search with 0.80 threshold

#### Data Models

- **`core/models.py`**: Three dataclasses
  - `SubtitleSegment` → Result of transcription (16 fields: text, timing, ASR conf, SNR, MFCC, correction state)
  - `CorrectionRecord` → Persisted in DB (includes fingerprint + match score)
  - `ValidationResult` → Result of `validate_correction()` (score, tier, message)

#### Hardware & Testing

- **`firmware/esp32_firmware.ino`**: C firmware for ESP32 DevKit v1
  - Processes 16 kHz float32 audio in 512-sample frames
  - Computes MFCC (12 coefficients: `N_MELS=26`, `N_FFT=512`, `DCT-II`)
  - Two-pass: spectral subtraction, SNR measurement
  - Sends JSON response via serial (2 Mbps)
  - Requires ArduinoJson v6 library

- **`check_ports.py`**, **`test_detect.py`**, **`test_winreg.py`**: Windows serial port discovery
  - Detects CH340/CH341 USB bridge or CP2102 chips
  - Returns list of available COM ports

---

## Key Design Decisions

### Two-Pass Spectral Subtraction (mel domain, pre-log)

**Why**: Logarithm is non-linear; subtracting after log (e.g., Wiener filtering in log-space) does NOT equal subtracting before. To preserve speech while removing noise, subtraction must happen on raw mel energies.

**Implementation**:
```
Pass 1: For each frame, compute raw (pre-log) mel energies
        Classify frame as silence (RMS < 0.002) or speech
        Accumulate per-band noise floor from silence frames
        
Pass 2: For each frame, subtract noise floor from raw mel
        Apply log10(max(mel - noise_floor, 1e-9))
        Apply DCT-II → 12 coefficients (MFCC)
```

**Identical in**:
- `esp32_firmware.ino` (`process_audio()`, lines ~200–280)
- `esp32_validator.py` (`compute_mfcc_software()`, lines ~150–250)

### SNR Gate (15 dB threshold)

**Why**: At low SNR, mel spectrograms are dominated by noise floor, not speech. Storing these fingerprints corrupts the nearest-neighbor lookup for future segments.

**Behavior**:
- Segments with SNR ≥ 15 dB: correction stored in DB + applied in-session
- Segments with SNR < 15 dB: correction applied in-session ONLY, DB skipped
- User sees warning: "Applied but not saved (SNR {snr:.1f} dB < 15 dB)"

### MFCC Fingerprint (12 coefficients, 16 kHz sample rate)

**Parameters** (fixed across firmware + software):
- `N_MELS=26` (Mel bands)
- `N_FFT=512` (window size)
- `WIN_LENGTH=400` (400 samples = 25 ms @ 16 kHz)
- `HOP_LENGTH=160` (160 samples = 10 ms, 60% overlap)

**Why these values**:
- Matches Whisper's default preprocessing
- Low-power on ESP32 (26 bands → 12 DCT coefficients)
- 10 ms hop → 100 frames per second

**Mean vector** (12 floats): Average MFCC across all frames
**Variance vector** (12 floats): Per-coefficient variance (for future speaker-adaptive matching)

### Fused Confidence Formula

```
fused = 0.6 × ASR_conf + 0.3 × (1 - SNR_penalty) + 0.1 × speaker_stability

where:
  ASR_conf        = exp(Whisper.avg_logprob) ∈ [0, 1]
  SNR_penalty     = clip((20 - SNR_dB) / 15, 0, 1)  [0=no penalty, 1=max]
  speaker_stability = 1.0 (placeholder, not yet implemented)
  
Result ≥ 0.75 → GREEN (accept), < 0.75 → RED (review)
```

### Correction Validation Tiers

```
HIGH (≥ 0.72)    → Accept immediately, store with confidence
MEDIUM (≥ 0.55)  → Accept but flag for review (visual badge in UI)
MISMATCH (< 0.55) → Reject by default, allow override
```

Match score = `0.7 × cosine_similarity + 0.3 × euclidean_similarity`

### Model Loading (Flexible Format Support)

The `load_model()` function in `core/transcriber.py` accepts:
1. **Standard sizes**: `"tiny"`, `"base"`, `"small"`, `"medium"`, `"large-v2"`, `"large-v3"`
2. **HuggingFace repo IDs**: `"sathish-93/whisper-tamil-medium-ct2-int8"` (CTranslate2 format only)
3. **Local paths**: `"C:/models/faster-whisper-tamil"` (must contain `model.bin`, not `pytorch_model.bin`)

**Critical constraint**: Faster-Whisper requires **CTranslate2 format** (contains `model.bin`). PyTorch models (contain `pytorch_model.bin`) must be converted:
```bash
pip install ctranslate2 transformers
ct2-transformers-converter \
    --model vasista22/whisper-tamil-medium \
    --output_dir ./faster-whisper-tamil-medium \
    --quantization int8
```

**Verified Indic models** (in `INDIC_MODELS` dict):
- Tamil: `sathish-93/whisper-tamil-medium-ct2-int8`
- Hindi: `collabora/faster-whisper-{small,medium,large-v2}-hindi`
- Malayalam: `BettySara/betty-whisper-large-v3-malayalam-ct2`
- Kannada: `elprofessor67/faster-whisper-kannada-tiny` (low accuracy warning)
- Telugu: `cvas-544/autotinglishsub-whisper-telugu-ct2` (test first)
- Multilingual: `Superleap/faster_indic_whisper_nodcil`

---

## Development Commands

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# System prerequisites (outside pip)
ffmpeg --version  # Confirm ffmpeg on PATH
```

### Run the App
```bash
# From project root
streamlit run subgen_ai/app.py
```

### Flash ESP32 Firmware
```bash
# 1. Install Arduino IDE + ArduinoJson v6 library
# 2. Open firmware/esp32_firmware.ino in Arduino IDE
# 3. Select board: "ESP32 Dev Module"
# 4. Set baud rate: 115200 (for upload; runs at 2 Mbps)
# 5. Upload to connected ESP32
```

### Testing
```bash
# Windows serial port discovery test
python subgen_ai/check_ports.py
python subgen_ai/test_detect.py

# Smoke test: verify all Indic models are importable
python -c "from subgen_ai.core.transcriber import INDIC_MODELS, load_model; \
           print(f'Found {len(INDIC_MODELS)} models'); \
           model = load_model('large-v3'); print('✓ Model loaded')"
```

---

## Common Workflows

### Adding a New Indic Language Model

1. **Find a verified CTranslate2 model** on HuggingFace:
   - Must be public (no auth required)
   - Must contain `model.bin` file (not `pytorch_model.bin`)
   - Test via `huggingface-hub` API or manual download

2. **Add entry to `INDIC_MODELS` dict** in `core/transcriber.py`:
   ```python
   INDIC_MODELS: dict = {
       # ... existing models ...
       "Your Language · model-size  (author)": "huggingface/repo-id",
   }
   ```

3. **Test in UI**:
   - Streamlit sidebar → Indic / Custom Model dropdown
   - Select from curated list or paste HF repo ID
   - Transcribe a test file
   - Confirm model loads and produces output

4. **Document constraints**:
   - If model requires conversion (PyTorch → CTranslate2), add comment with conversion command
   - Note accuracy caveats (e.g., "Kannada: tiny size, test first")

### Debugging Transcription Failures

**Problem**: "Repository Not Found" or "401 Client Error"
- **Cause**: Model repo ID doesn't exist, is private, or wrong format
- **Fix**:
  1. Verify repo exists: `https://huggingface.co/<repo-id>` in browser
  2. Confirm `model.bin` exists (CTranslate2 format)
  3. Check if PyTorch-only (requires local conversion)
  4. If using custom model, ensure full path or valid HF ID

**Problem**: "CUDA out of memory" or slow transcription
- **Cause**: Model too large for CPU or GPU VRAM
- **Fix**:
  1. Try smaller model (e.g., `"small"` instead of `"large-v3"`)
  2. Reduce `beam_size` in `transcriber.py` from 5 to 1
  3. Batch shorter videos (< 5 min) to reduce memory peak

**Problem**: ESP32 not detected, software fallback slow
- **Cause**: Serial port unavailable or timeout
- **Fix**:
  1. Run `python subgen_ai/check_ports.py` to list available COM ports
  2. Confirm ESP32 firmware is flashed (not factory default)
  3. Check baud rate: firmware expects 2 Mbps, UI defaults to that
  4. Increase `TIMEOUT_S` in `esp32_validator.py` if frames are large

### Updating the Correction Database

**Inspect current corrections**:
```python
from subgen_ai.db.correction_store import init_db
import sqlite3

conn = init_db()
rows = conn.execute("SELECT language, COUNT(*) FROM corrections GROUP BY language").fetchall()
for lang, count in rows:
    print(f"{lang}: {count} corrections")
conn.close()
```

**Delete corrupted entries** (low SNR):
```python
# This would require adding a new helper function, but can be done via raw SQL:
conn = init_db()
conn.execute("DELETE FROM corrections WHERE created_at < '2026-04-01'")
conn.commit()
conn.close()
```

---

## Important Constraints & Gotchas

### Streamlit Session State Isolation
- **All mutable state lives in `st.session_state`** — no module-level globals
- Reason: Streamlit re-runs the entire script on every interaction; globals would be reset
- Implication: When modifying state, always use `st.session_state["key"] = value`

### Audio Sample Rate
- **Hardcoded to 16 kHz** throughout:
  - ffmpeg extraction: `-ar 16000`
  - Whisper input: expects 16 kHz
  - MFCC computation: assumes 16 kHz (hop_length=160 samples = 10 ms)
  - ESP32 firmware: hardcoded frame size (512 samples = 32 ms)
- **Do NOT change without updating all three locations**

### MFCC Fingerprint Stability
- Fingerprints are sensitive to noise and speaker variation
- High SNR (clean audio) → stable fingerprints → reliable corrections
- Low SNR (noisy audio) → unstable fingerprints → gate prevents DB pollution
- **If you see HIGH/MEDIUM/MISMATCH mismatches despite visually similar audio, likely causes**:
  1. Speaker changed (different person)
  2. Background noise changed significantly
  3. Microphone position/gain changed
  4. Software MFCC differs from firmware (rare; both should match formula exactly)

### Baud Rate & USB Bridge Chip
- **Current**: 2 Mbps (0.3 s per 64 KB audio segment)
- **Requires**: CH340 or CH341 USB-to-UART bridge chip
- **Will NOT work with**: CP2102 (max 921600 baud) — would need fallback to slower rate or software-only
- **Upload baud rate**: 115200 (fixed by Arduino IDE, separate from runtime)

### Database Corruption
- SQLite in `.../corrections.db` is thread-safe but each operation opens/closes connection
- **No concurrent access issues**, but:
  - If you manually edit DB while app runs, restart app to reload
  - If DB file gets corrupted (rare), delete and restart — app recreates it

---

## File Structure Summary

```
subgen_ai/
├── app.py                    # Streamlit UI entry point
├── requirements.txt          # Python dependencies
├── firmware/
│   └── esp32_firmware.ino    # ESP32 C firmware (MFCC co-processor)
├── core/
│   ├── transcriber.py        # ASR pipeline + model loading
│   ├── qc_engine.py          # Confidence scoring + validation
│   ├── esp32_validator.py    # MFCC (HW/SW) + serial protocol
│   ├── models.py             # Dataclasses (SubtitleSegment, etc.)
│   └── __init__.py
├── db/
│   ├── correction_store.py   # SQLite persistence
│   └── __init__.py
├── export/
│   ├── formatters.py         # SRT/VTT/JSON/burn-in
│   └── __init__.py
├── components/
│   ├── video_player.py       # Streamlit video widget
│   └── __init__.py
├── docs/
│   └── plans/
│       └── 2026-04-08-spectral-subtraction-snr-gate.md
└── CLAUDE.md                 # This file
```

---

## Performance Notes

### Typical Latencies (Per Video)
- **Audio extraction (ffmpeg)**: 0.5–2 s (depends on video codec/length)
- **Transcription (ASR)**: ~0.3–0.5 × audio_length (0.3x for `"small"`, 0.5x for `"large-v3"`)
- **Per-segment MFCC (ESP32 HW)**: 0.3 s + 16 × (12 ms MFCC time)
- **Per-segment MFCC (software)**: ~50–100 ms per segment
- **Total for 5 min video**: ~30–60 seconds on modern CPU

### Memory
- **Whisper model**: 400–700 MB resident (depends on size)
- **Audio buffer**: ~2 MB per minute (16 kHz float32)
- **Correction DB**: <10 MB (even with 10k corrections)

### Disk
- MFCC fingerprints: 12 × 4 bytes (mean) + 12 × 4 bytes (var) = 96 bytes per segment
- DB grows at ~500 bytes per correction

---

## Future Improvements (Not Yet Implemented)

- **Speaker-adaptive matching**: Use `mfcc_var` vector + speaker ID to improve fingerprint matching
- **Multi-language batch processing**: Process multiple languages in parallel (currently sequential)
- **Active learning**: Prioritize which corrections to store based on impact on ASR accuracy
- **Real-time streaming**: Transcode live video feeds instead of requiring pre-recorded files
- **Attention visualization**: Show which mel bands contribute most to MFCC distance (debug mismatches)

