"""
SubGEN AI — Signal-Informed QC Engine.

Implements fused confidence scoring, SNR estimation, RED/GREEN labelling,
and cosine-similarity-based correction validation.
All functions are pure (no I/O) and use exact formulas from the spec.
"""
import numpy as np
from subgen_ai.core.models import ValidationResult

# QC Thresholds
FUSED_CONF_THRESHOLD = 0.75   # >= GREEN, < RED
WEIGHTS = (0.6, 0.3, 0.1)     # ASR_conf, SNR_term, speaker_stability

# Correction-Validation Thresholds
THRESHOLD_HIGH       = 0.72
THRESHOLD_MEDIUM     = 0.55
THRESHOLD_DB_APPLY   = 0.80   # auto-apply correction at inference


def compute_asr_conf(avg_logprob: float) -> float:
    """Convert Whisper avg_logprob to [0,1] linear confidence via exp()."""
    return float(np.exp(avg_logprob))


def compute_snr_penalty(audio_clip: np.ndarray, sr: int = 16000) -> float:
    """
    Estimate SNR from audio clip using sub-window energy method.

    Split clip into 16 equal sub-windows.
    Windows with mean energy > 0.002 = speech windows.
    Windows with mean energy <= 0.002 = noise windows.
    SNR_dB = 10 * log10(mean_speech_energy / mean_noise_energy)
    SNR_penalty = clip((20 - SNR_dB) / 15, 0, 1)

    Returns 1.0 if all silence (maximum penalty).
    Returns 0.0 if all speech, no detectable noise.
    """
    n_windows = 16
    window_size = max(1, len(audio_clip) // n_windows)
    windows = [
        audio_clip[i * window_size:(i + 1) * window_size]
        for i in range(n_windows)
        if len(audio_clip[i * window_size:(i + 1) * window_size]) > 0
    ]

    energies = [float(np.mean(w ** 2)) for w in windows]
    VAD_THRESHOLD = 0.002

    speech_energies = [e for e in energies if e > VAD_THRESHOLD]
    noise_energies  = [e for e in energies if e <= VAD_THRESHOLD]

    if not speech_energies:
        return 1.0   # all noise → maximum penalty
    if not noise_energies:
        return 0.0   # all speech, no noise → no penalty

    mean_speech = np.mean(speech_energies)
    mean_noise  = np.mean(noise_energies) + 1e-10

    snr_db = float(10 * np.log10(mean_speech / mean_noise))
    snr_db = float(np.clip(snr_db, 0, 40))

    penalty = float(np.clip((20 - snr_db) / 15, 0, 1))
    return penalty


def compute_fused_conf(asr_conf: float, snr_penalty: float,
                       speaker_stability: float = 1.0) -> float:
    """
    Combine ASR confidence, SNR penalty, and speaker stability into a
    single fused confidence score.

    Weights: ASR=0.6, SNR_term=0.3, speaker_stability=0.1
    """
    w_asr, w_snr, w_spk = WEIGHTS
    return (w_asr * asr_conf
            + w_snr * (1 - snr_penalty)
            + w_spk * speaker_stability)


def label_segment(fused_conf: float) -> str:
    """Return 'GREEN' if fused_conf >= threshold, else 'RED'."""
    return "GREEN" if fused_conf >= FUSED_CONF_THRESHOLD else "RED"


def cosine_similarity(v1: list, v2: list) -> float:
    """
    Cosine similarity mapped from [-1, 1] to [0, 1].
    Returns 0.0 if either vector is near-zero.
    """
    a = np.array(v1, dtype=np.float64)
    b = np.array(v2, dtype=np.float64)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    raw = float(np.dot(a, b) / (norm_a * norm_b))
    return (raw + 1.0) / 2.0   # map [-1,1] → [0,1]


def euclidean_similarity(v1: list, v2: list) -> float:
    """1 / (1 + dist/10) — returns 1.0 for identical vectors, decays towards 0."""
    dist = float(np.linalg.norm(np.array(v1) - np.array(v2)))
    return 1.0 / (1.0 + dist / 10.0)


def compute_match_score(fp1_mean: list, fp2_mean: list) -> float:
    """
    Weighted combination of cosine (0.7) and euclidean (0.3) similarities.
    Used to compare two MFCC fingerprints.
    """
    return (0.7 * cosine_similarity(fp1_mean, fp2_mean)
            + 0.3 * euclidean_similarity(fp1_mean, fp2_mean))


def validate_correction(original_fp: dict, new_fp: dict) -> ValidationResult:
    """
    Compare the audio fingerprint from original transcription
    vs the fingerprint computed when the user proposes a correction.

    Returns a ValidationResult with tier HIGH / MEDIUM / MISMATCH.
    If either fingerprint is unavailable, correction is accepted without validation.
    """
    if not original_fp.get("ok") or not new_fp.get("ok"):
        return ValidationResult(
            score=0.0, tier="UNKNOWN", accepted=True, hw_used=False,
            message="⚠ Fingerprint unavailable — correction saved without validation."
        )

    score = compute_match_score(original_fp["mfcc_mean"], new_fp["mfcc_mean"])
    hw_used = original_fp.get("hw", False) or new_fp.get("hw", False)

    if score >= THRESHOLD_HIGH:
        return ValidationResult(
            score=score, tier="HIGH", accepted=True, hw_used=hw_used,
            message=f"✅ HIGH confidence match ({score:.3f}) — correction validated and saved."
        )
    elif score >= THRESHOLD_MEDIUM:
        return ValidationResult(
            score=score, tier="MEDIUM", accepted=True, hw_used=hw_used,
            message=f"⚠ MEDIUM confidence ({score:.3f}) — accepted but flagged for review."
        )
    else:
        return ValidationResult(
            score=score, tier="MISMATCH", accepted=False, hw_used=hw_used,
            message=f"❌ Audio mismatch ({score:.3f}) — correction may not match audio. Override to save anyway."
        )
