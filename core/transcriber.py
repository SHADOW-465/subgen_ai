"""
SubGEN AI — Transcription Pipeline.

Ties together:
  - ffmpeg audio extraction
  - Faster-Whisper ASR
  - QC engine (fused confidence, SNR)
  - ESP32/software MFCC fingerprinting
  - Correction DB auto-apply

Usage:
    segments = transcribe("video.mp4", model_size="small", language="ta")
"""
import os
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from faster_whisper import WhisperModel

from subgen_ai.core.models import SubtitleSegment
from subgen_ai.core.qc_engine import (
    compute_asr_conf, compute_snr_penalty,
    compute_fused_conf, label_segment
)
from subgen_ai.core.esp32_validator import get_fingerprint
from subgen_ai.db.correction_store import find_nearest_correction

# Module-level model cache — prevents reloading Whisper on every Streamlit re-run.
_MODEL_CACHE: dict = {}

SUPPORTED_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
DEFAULT_MODEL    = "small"
CLIP_PADDING_S   = 0.1   # 100 ms padding on each side for fingerprinting


def load_model(model_size: str = DEFAULT_MODEL) -> WhisperModel:
    """
    Load (or return cached) Faster-Whisper model.
    CPU-only, INT8 quantisation for minimal RAM usage.
    """
    if model_size not in _MODEL_CACHE:
        _MODEL_CACHE[model_size] = WhisperModel(
            model_size, device="cpu", compute_type="int8"
        )
    return _MODEL_CACHE[model_size]


def extract_audio(video_path) -> tuple:
    """
    Extract mono 16 kHz float32 audio from any video/audio file using ffmpeg.

    Uses subprocess.run() directly for portability.

    Args:
        video_path: Path or str to the source media file.

    Returns:
        (audio_array: np.ndarray[float32], sample_rate: int)

    Raises:
        RuntimeError: if ffmpeg exits with a non-zero return code.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-ar", "16000", "-ac", "1", "-f", "wav", tmp_path,
            "-loglevel", "error"
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed: {result.stderr.decode(errors='replace')}"
            )

        with wave.open(tmp_path, "rb") as wf:
            sr       = wf.getframerate()
            n_frames = wf.getnframes()
            raw      = wf.readframes(n_frames)

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, sr
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def extract_clip(audio: np.ndarray, sr: int,
                 start_s: float, end_s: float) -> np.ndarray:
    """
    Extract an audio sub-clip with padding, clamped to array bounds.

    Args:
        audio:   Full audio array (float32).
        sr:      Sample rate.
        start_s: Clip start time in seconds.
        end_s:   Clip end time in seconds.

    Returns:
        Sub-array of audio with CLIP_PADDING_S on each side.
    """
    start_idx = max(0, int((start_s - CLIP_PADDING_S) * sr))
    end_idx   = min(len(audio), int((end_s + CLIP_PADDING_S) * sr))
    return audio[start_idx:end_idx]


def transcribe(
    video_path=None,
    model_size: str = DEFAULT_MODEL,
    language: Optional[str] = None,
    task: str = "transcribe",
    esp32_port: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
    audio: Optional[np.ndarray] = None,
    sr: int = 16000,
) -> List[SubtitleSegment]:
    """
    Main transcription pipeline.

    Steps:
      1. Extract audio (if not pre-supplied).
      2. Run Faster-Whisper lazily — segments are streamed one at a time so
         the caller sees real-time progress instead of one blocking call.
      3. For each segment: MFCC fingerprint (HW or SW), SNR, fused confidence,
         RED/GREEN label, correction DB lookup.

    Args:
        video_path:        Path to input media (required if audio not supplied).
        model_size:        Whisper model variant (default "small").
        language:          ISO 639-1 code or None for auto-detect.
        task:              "transcribe" or "translate".
        esp32_port:        Serial port for ESP32, or None for SW mode.
        progress_callback: Optional callable(seg_index, total_duration_s, segment).
                           seg_index      — 1-based count of segments done.
                           total_duration_s — total audio length in seconds.
                           segment        — the completed SubtitleSegment.
        audio:             Pre-extracted float32 audio array (skip ffmpeg step).
        sr:                Sample rate for pre-supplied audio.

    Returns:
        List of SubtitleSegment, one per Whisper segment.
    """
    if audio is None:
        audio, sr = extract_audio(video_path)

    total_duration = len(audio) / sr
    model = load_model(model_size)

    segments_raw, info = model.transcribe(
        audio,
        language=language,
        task=task,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        without_timestamps=False,
    )

    detected_lang = info.language if language is None else language
    results: List[SubtitleSegment] = []

    # Iterate the generator lazily so each segment fires progress_callback
    # as soon as Whisper finishes it — no upfront list() materialisation.
    for i, seg in enumerate(segments_raw):
        clip        = extract_clip(audio, sr, seg.start, seg.end)
        fp          = get_fingerprint(clip, sr, esp32_port)
        snr_penalty = compute_snr_penalty(clip, sr)
        asr_conf    = compute_asr_conf(seg.avg_logprob)
        fused       = compute_fused_conf(asr_conf, snr_penalty)
        label       = label_segment(fused)

        clip_sq     = clip ** 2
        speech_mask = np.abs(clip) >= 0.002
        noise_mask  = ~speech_mask
        mean_speech = float(np.mean(clip_sq[speech_mask])) if speech_mask.any() else 1e-10
        mean_noise  = float(np.mean(clip_sq[noise_mask]))  if noise_mask.any()  else 1e-10
        snr_db_val  = float(np.clip(10 * np.log10(max(mean_speech, 1e-10) / max(mean_noise, 1e-10)), -20.0, 60.0))

        text           = seg.text.strip()
        auto_corrected = False
        if fp.get("ok") and fp.get("mfcc_mean"):
            correction = find_nearest_correction(fp["mfcc_mean"], detected_lang)
            if correction:
                text           = correction.corrected_text
                auto_corrected = True

        result_seg = SubtitleSegment(
            index=i,
            start=seg.start,
            end=seg.end,
            text=text,
            language=detected_lang,
            asr_conf=asr_conf,
            snr_db=snr_db_val,
            fused_conf=fused,
            label=label,
            mfcc_mean=fp.get("mfcc_mean", []),
            mfcc_var=fp.get("mfcc_var", []),
            hw_fingerprint=fp.get("hw", False),
            corrected=auto_corrected,
            correction_text=text if auto_corrected else "",
        )
        results.append(result_seg)

        if progress_callback:
            progress_callback(i + 1, total_duration, result_seg)

    return results
