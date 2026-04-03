"""
SubGEN AI — Core data models.
All dataclasses used throughout the application.
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class SubtitleSegment:
    """A single subtitle segment produced by the transcription pipeline."""

    index: int
    start: float          # seconds
    end: float            # seconds
    text: str             # raw Whisper output (may be overridden by embedding correction)
    language: str         # ISO 639-1 code e.g. "ta", "te", "hi", "en"
    asr_conf: float       # exp(avg_logprob) → [0, 1]
    snr_db: float         # estimated from audio sub-window energy
    fused_conf: float     # QC formula result
    label: str            # "GREEN" or "RED"
    mfcc_mean: List[float] = field(default_factory=list)  # 12 coefficients
    mfcc_var: List[float] = field(default_factory=list)   # 12 coefficients
    hw_fingerprint: bool = False   # True if ESP32 computed MFCC, False if software fallback
    corrected: bool = False        # True if user has corrected this segment
    correction_text: str = ""      # User's corrected text


@dataclass
class CorrectionRecord:
    """A user-validated correction stored in SQLite for future auto-apply."""

    id: Optional[int]
    segment_start: float
    segment_end: float
    original_text: str
    corrected_text: str
    language: str
    mfcc_mean: List[float]   # 12 floats
    mfcc_var: List[float]    # 12 floats
    match_score: float       # cosine similarity at time of validation
    hw_used: bool
    created_at: str          # ISO 8601 datetime string


@dataclass
class ValidationResult:
    """Result of validating a user correction against the stored audio fingerprint."""

    score: float
    tier: str              # "HIGH" | "MEDIUM" | "MISMATCH"
    accepted: bool
    hw_used: bool
    message: str
