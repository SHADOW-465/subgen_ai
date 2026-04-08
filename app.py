"""
SubGEN AI — Streamlit Entry Point.

Single-page, 3-tab layout:
  Tab 1: Upload & Transcribe
  Tab 2: Review & Edit
  Tab 3: Export

All mutable state lives in st.session_state — no module-level globals.
"""

import sys
import os
from pathlib import Path

# Ensure the parent of subgen_ai/ is on sys.path when run as
#   streamlit run subgen_ai/app.py
_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

import streamlit as st

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="SubGEN AI",
    page_icon="🎬",
    initial_sidebar_state="expanded",
)

# ── Local imports (after sys.path fix) ──────────────────────────────────────
from subgen_ai.core.models import SubtitleSegment, CorrectionRecord, ValidationResult
from subgen_ai.core.qc_engine import validate_correction
from subgen_ai.core.esp32_validator import (
    find_esp32_port, find_all_esp32_ports, list_all_ports, get_fingerprint
)
from subgen_ai.core.transcriber import transcribe, SUPPORTED_MODELS, DEFAULT_MODEL
from subgen_ai.db.correction_store import (
    save_correction, get_db_stats, delete_correction, init_db
)
from subgen_ai.export.formatters import to_srt, to_vtt, to_json, to_burn_in

from datetime import datetime
from typing import Optional
import tempfile

# ── Optional custom CSS ──────────────────────────────────────────────────────
_CSS_PATH = Path(__file__).parent / "assets" / "style.css"
if _CSS_PATH.exists():
    st.markdown(f"<style>{_CSS_PATH.read_text()}</style>", unsafe_allow_html=True)

# ── Session-state defaults ───────────────────────────────────────────────────

def _init_state() -> None:
    """Initialise all session-state keys to safe defaults (idempotent)."""
    defaults = {
        "segments": [],           # list[SubtitleSegment]
        "audio": None,            # np.ndarray | None
        "sr": 16000,              # int — sample rate
        "port": None,             # str | None — selected ESP32 port
        "done": False,            # bool — transcription complete
        "filename": "",           # str — uploaded file name
        "esp32_ports": [],        # list[str] — auto-detected ports
        "hw_status": "software",  # "connected" | "software" | "error"
        "correction_count": 0,    # int — corrections saved this session
        "filter_red_only": False, # bool — review tab filter
        "video_bytes": None,   # bytes | None — raw uploaded file for the player
        "video_ext":   "",     # str — e.g. ".mp4", ".avi"
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ── Helper: format seconds → HH:MM:SS ───────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    """Format float seconds to HH:MM:SS string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ── Helper: extract audio clip from session audio ───────────────────────────

def _get_audio_clip(seg: SubtitleSegment) -> Optional["np.ndarray"]:
    """Return the audio sub-array for a segment, or None if audio unavailable."""
    import numpy as np
    audio = st.session_state.get("audio")
    sr = st.session_state.get("sr", 16000)
    if audio is None:
        return None
    CLIP_PADDING_S = 0.1
    start_idx = max(0, int((seg.start - CLIP_PADDING_S) * sr))
    end_idx   = min(len(audio), int((seg.end + CLIP_PADDING_S) * sr))
    return audio[start_idx:end_idx]


def _get_video_mime() -> str:
    """Derive a MIME type from the stored video extension."""
    import mimetypes
    ext = st.session_state.get("video_ext", ".mp4")
    mime, _ = mimetypes.guess_type(f"file{ext}")
    return mime or "video/mp4"


# ────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────────────────────────────────

@st.fragment(run_every=2)
def _hw_status_fragment() -> None:
    """
    Live hardware-status widget — auto-reruns every 2 s so plug/unplug is
    reflected immediately without touching the rest of the page.
    """
    ports = find_all_esp32_ports()
    if ports:
        st.session_state["port"]      = ports[0]
        st.session_state["hw_status"] = "connected"
        st.success(f"Connected — {ports[0]}")
    else:
        st.session_state["port"]      = None
        st.session_state["hw_status"] = "software"
        st.warning("Not detected — software mode")

    if st.button("↺ Reconnect", use_container_width=True):
        st.rerun(scope="fragment")


def render_sidebar() -> None:
    """Render the full left sidebar: hardware, AI settings, DB stats."""
    with st.sidebar:
        st.markdown("## 🎬 SubGEN AI")
        st.caption("Hardware-Aware Subtitle Generator")
        st.divider()

        # ── Hardware section ────────────────────────────────────────────────
        st.markdown("### ⚙ Hardware")
        _hw_status_fragment()

        st.divider()

        # ── AI Settings ─────────────────────────────────────────────────────
        st.markdown("### 🧠 AI Settings")

        st.session_state["model_size"] = st.selectbox(
            "Whisper Model",
            options=SUPPORTED_MODELS,
            index=SUPPORTED_MODELS.index(DEFAULT_MODEL),
            help="Larger = more accurate but slower. 'small' is recommended.",
        )

        task_choice = st.radio(
            "Task",
            options=["Transcribe", "Translate to English"],
            horizontal=True,
            help="Translate converts any language to English subtitles.",
        )
        st.session_state["task"] = "translate" if task_choice == "Translate to English" else "transcribe"

        _LANG_DISPLAY = {
            "af": "Afrikaans",       "am": "Amharic",          "ar": "Arabic",
            "as": "Assamese",        "az": "Azerbaijani",      "ba": "Bashkir",
            "be": "Belarusian",      "bg": "Bulgarian",        "bn": "Bengali",
            "bo": "Tibetan",         "br": "Breton",           "bs": "Bosnian",
            "ca": "Catalan",         "cs": "Czech",            "cy": "Welsh",
            "da": "Danish",          "de": "German",           "el": "Greek",
            "en": "English",         "es": "Spanish",          "et": "Estonian",
            "eu": "Basque",          "fa": "Persian",          "fi": "Finnish",
            "fo": "Faroese",         "fr": "French",           "gl": "Galician",
            "gu": "Gujarati",        "ha": "Hausa",            "haw": "Hawaiian",
            "he": "Hebrew",          "hi": "Hindi",            "hr": "Croatian",
            "ht": "Haitian Creole",  "hu": "Hungarian",        "hy": "Armenian",
            "id": "Indonesian",      "is": "Icelandic",        "it": "Italian",
            "ja": "Japanese",        "jw": "Javanese",         "ka": "Georgian",
            "kk": "Kazakh",          "km": "Khmer",            "kn": "Kannada",
            "ko": "Korean",          "la": "Latin",            "lb": "Luxembourgish",
            "ln": "Lingala",         "lo": "Lao",              "lt": "Lithuanian",
            "lv": "Latvian",         "mg": "Malagasy",         "mi": "Maori",
            "mk": "Macedonian",      "ml": "Malayalam",        "mn": "Mongolian",
            "mr": "Marathi",         "ms": "Malay",            "mt": "Maltese",
            "my": "Burmese",         "ne": "Nepali",           "nl": "Dutch",
            "nn": "Nynorsk",         "no": "Norwegian",        "oc": "Occitan",
            "pa": "Punjabi",         "pl": "Polish",           "ps": "Pashto",
            "pt": "Portuguese",      "ro": "Romanian",         "ru": "Russian",
            "sa": "Sanskrit",        "sd": "Sindhi",           "si": "Sinhala",
            "sk": "Slovak",          "sl": "Slovenian",        "sn": "Shona",
            "so": "Somali",          "sq": "Albanian",         "sr": "Serbian",
            "su": "Sundanese",       "sv": "Swedish",          "sw": "Swahili",
            "ta": "Tamil",           "te": "Telugu",           "tg": "Tajik",
            "th": "Thai",            "tk": "Turkmen",          "tl": "Filipino",
            "tr": "Turkish",         "tt": "Tatar",            "uk": "Ukrainian",
            "ur": "Urdu",            "uz": "Uzbek",            "vi": "Vietnamese",
            "yi": "Yiddish",         "yo": "Yoruba",           "yue": "Cantonese",
            "zh": "Chinese",
        }
        _LANG_LABELS = ["Auto-detect"] + [f"{v} ({k})" for k, v in _LANG_DISPLAY.items()]
        _LANG_CODES  = [None]          + list(_LANG_DISPLAY.keys())

        lang_idx = st.selectbox(
            "Language",
            options=range(len(_LANG_LABELS)),
            format_func=lambda i: _LANG_LABELS[i],
            index=0,
            help="Type to search. Auto-detect works well for most languages.",
        )
        st.session_state["language"] = _LANG_CODES[lang_idx]

        st.divider()

        # ── Correction DB ────────────────────────────────────────────────────
        st.markdown("### 📊 Correction DB")
        try:
            stats = get_db_stats()
            st.metric("Total corrections", stats["total"])
            if stats["by_language"]:
                import pandas as pd
                df = pd.DataFrame(
                    stats["by_language"].items(),
                    columns=["Language", "Count"],
                )
                st.dataframe(df, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.caption(f"DB unavailable: {exc}")

        if st.button("🗑 Clear all corrections", use_container_width=True):
            _clear_all_corrections()

        st.divider()
        st.caption("SubGEN AI v1.0 | PSVPEC ECE | Batch A25")



def _clear_all_corrections() -> None:
    """Delete all correction records from SQLite after confirmation."""
    import sqlite3
    from subgen_ai.db.correction_store import DB_PATH, init_db
    try:
        conn = init_db()
        count = conn.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]
        conn.execute("DELETE FROM corrections")
        conn.commit()
        conn.close()
        st.sidebar.success(f"✅ Deleted {count} correction(s).")
    except Exception as exc:
        st.sidebar.error(f"❌ Could not clear DB: {exc}")


# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — Upload & Transcribe
# ────────────────────────────────────────────────────────────────────────────

def render_tab_upload() -> None:
    """Upload a media file and run transcription."""
    st.header("📁 Upload & Transcribe")

    uploaded = st.file_uploader(
        "Choose a video or audio file",
        type=["mp4", "avi", "mov", "mkv", "wav", "mp3", "m4a"],
        help="Supports MP4, AVI, MOV, MKV, WAV, MP3, M4A",
    )

    if uploaded is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📄 **{uploaded.name}**  ({uploaded.size / 1024:.1f} KB)")
        with col2:
            if st.session_state.get("done"):
                segs = st.session_state["segments"]
                lang = segs[0].language if segs else "?"
                st.success(f"✅ Transcription done — {len(segs)} segments — lang: `{lang}`")

        st.markdown("---")

        if st.button("▶ Generate Subtitles", type="primary", use_container_width=True):
            _run_transcription(uploaded)

    else:
        st.markdown(
            "<div style='text-align:center;padding:3rem;color:#888;'>"
            "⬆ Drop a file above to begin."
            "</div>",
            unsafe_allow_html=True,
        )


def _run_transcription(uploaded_file) -> None:
    """Save upload to temp file, run transcription pipeline, update session state."""
    import time
    import numpy as np
    from subgen_ai.core.transcriber import extract_audio as _extract_audio

    raw_bytes = uploaded_file.getvalue()
    suffix    = Path(uploaded_file.name).suffix or ".mp4"
    st.session_state["video_bytes"] = raw_bytes
    st.session_state["video_ext"]   = suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    # ── UI placeholders ──────────────────────────────────────────────────────
    step_status  = st.empty()
    progress_bar = st.progress(0)
    metrics_row  = st.empty()
    seg_preview  = st.empty()

    try:
        # ── Step 1: extract audio ────────────────────────────────────────────
        step_status.info("Step 1 / 2 — Extracting audio from file…")
        audio_arr, sr = _extract_audio(tmp_path)
        total_dur = len(audio_arr) / sr

        def _fmt_dur(s: float) -> str:
            m, sec = divmod(int(s), 60)
            h, m   = divmod(m, 60)
            return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"

        # ── Step 2: transcribe ───────────────────────────────────────────────
        step_status.info("Step 2 / 2 — Generating subtitles…")
        t_start    = time.time()
        seg_count  = [0]

        def _progress(i: int, total_duration: float, seg: SubtitleSegment) -> None:
            seg_count[0] = i
            elapsed  = time.time() - t_start
            pct      = min(seg.end / max(total_duration, 1), 1.0)
            eta_str  = (
                f"~{_fmt_dur(elapsed / pct * (1.0 - pct))}"
                if pct > 0.02 else "calculating…"
            )
            progress_bar.progress(pct)
            metrics_row.markdown(
                f"**{pct*100:.1f}%** complete &nbsp;|&nbsp; "
                f"Segments: **{i}** &nbsp;|&nbsp; "
                f"Elapsed: **{_fmt_dur(elapsed)}** &nbsp;|&nbsp; "
                f"ETA: **{eta_str}** &nbsp;|&nbsp; "
                f"Position: **{_fmt_dur(seg.end)}** / {_fmt_dur(total_duration)}"
            )
            label_icon = "🟢" if seg.label == "GREEN" else "🔴"
            seg_preview.caption(
                f"{label_icon} [{_fmt_time(seg.start)} → {_fmt_time(seg.end)}]  "
                f"{seg.text[:80]}{'…' if len(seg.text) > 80 else ''}"
            )

        segments = transcribe(
            model_size=st.session_state.get("model_size", DEFAULT_MODEL),
            language=st.session_state.get("language"),
            task=st.session_state.get("task", "transcribe"),
            esp32_port=st.session_state.get("port"),
            progress_callback=_progress,
            audio=audio_arr,
            sr=sr,
        )

        st.session_state["audio"]            = audio_arr
        st.session_state["sr"]               = sr
        st.session_state["segments"]         = segments
        st.session_state["filename"]         = uploaded_file.name
        st.session_state["done"]             = True
        st.session_state["correction_count"] = 0

        elapsed_total = time.time() - t_start
        step_status.success(
            f"Done — {seg_count[0]} segments in {_fmt_dur(elapsed_total)}"
        )
        progress_bar.progress(1.0)
        metrics_row.empty()
        seg_preview.empty()
        st.rerun()

    except Exception as exc:
        step_status.error(f"Transcription failed: {exc}")
        progress_bar.empty()
        metrics_row.empty()
        seg_preview.empty()

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — Review & Edit
# ────────────────────────────────────────────────────────────────────────────

def render_tab_review() -> None:
    """Review segments, edit RED ones, validate & save corrections."""
    if not st.session_state.get("done"):
        st.info("ℹ Transcription not yet run. Go to **Upload & Transcribe** first.")
        return

    segments: list[SubtitleSegment] = st.session_state["segments"]
    red_segs   = [s for s in segments if s.label == "RED"]
    green_segs = [s for s in segments if s.label == "GREEN"]

    # ── Video player (top of tab, before header) ─────────────────────────────
    vb = st.session_state.get("video_bytes")
    if vb:
        from subgen_ai.components.video_player import render_video_player
        render_video_player(vb, _get_video_mime(), to_vtt(segments))
        st.markdown("---")

    # ── Summary row ─────────────────────────────────────────────────────────
    st.header("✏ Review & Edit")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total segments",        len(segments))
    c2.metric("🔴 RED (review needed)", len(red_segs))
    c3.metric("🟢 GREEN (confident)",   len(green_segs))
    c4.metric("✅ Corrections saved",   st.session_state.get("correction_count", 0))

    # ── Filter toggle ────────────────────────────────────────────────────────
    st.session_state["filter_red_only"] = st.toggle(
        "🔴 Show RED segments only",
        value=st.session_state.get("filter_red_only", False),
    )

    display_segs = red_segs if st.session_state["filter_red_only"] else segments
    if not display_segs:
        st.success("🎉 No segments to show with current filter.")
        return

    st.markdown("---")

    # ── Per-segment cards ────────────────────────────────────────────────────
    for seg in display_segs:
        _render_segment_card(seg)


def _render_segment_card(seg: SubtitleSegment) -> None:
    """Render a single segment — all labels get an editable text area."""
    label_icon = "🟢" if seg.label == "GREEN" else "🔴"
    hw_badge   = "🔧 HW-MFCC" if seg.hw_fingerprint else "💻 SW-MFCC"
    auto_badge = " 🔄 Auto-corrected from DB" if seg.corrected else ""

    header = (
        f"{label_icon}  [{_fmt_time(seg.start)} → {_fmt_time(seg.end)}]"
        f"  fused_conf: {seg.fused_conf:.3f}{auto_badge}"
    )

    with st.expander(header, expanded=(seg.label == "RED")):
        # ── Editable transcript — ALL segments ───────────────────────────────
        edit_key = f"edit_{seg.index}"
        if edit_key not in st.session_state:
            st.session_state[edit_key] = seg.text
        edited_text = st.text_area("✏ Edit transcript", key=edit_key, height=80)

        # ── Footer metrics ────────────────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        m1.caption(f"ASR conf: **{seg.asr_conf:.3f}**")
        m2.caption(f"SNR: **{seg.snr_db:.1f} dB**")
        m3.caption(hw_badge)

        # ── Save button — ALL segments (same flow as before) ─────────────────
        col_btn, col_msg = st.columns([2, 3])
        with col_btn:
            validate_clicked = st.button(
                "💾 Validate & Save Correction",
                key=f"validate_{seg.index}",
                use_container_width=True,
            )
        with col_msg:
            vr_key = f"vr_{seg.index}"
            if vr_key in st.session_state:
                vr: ValidationResult = st.session_state[vr_key]
                _show_validation_result(vr)
                if vr.tier == "MISMATCH":
                    if st.button(
                        "💾 Save anyway (override)",
                        key=f"override_{seg.index}",
                    ):
                        _do_save_correction(seg, edited_text, vr, override=True)

        if validate_clicked:
            _handle_validate_correction(seg, edited_text)


def _handle_validate_correction(seg: SubtitleSegment, new_text: str) -> None:
    """
    Compute new MFCC fingerprint, validate against original, save if accepted.
    """
    import numpy as np
    port = st.session_state.get("port")
    sr   = st.session_state.get("sr", 16000)

    clip = _get_audio_clip(seg)
    if clip is None or len(clip) == 0:
        st.warning("⚠ Audio unavailable — saving correction without fingerprint validation.")
        _do_save_correction(seg, new_text, None, override=True)
        return

    # Build original fingerprint dict from stored MFCC
    original_fp = {
        "ok": bool(seg.mfcc_mean),
        "hw": seg.hw_fingerprint,
        "mfcc_mean": seg.mfcc_mean,
        "mfcc_var":  seg.mfcc_var,
    }

    # Compute new fingerprint on the same audio
    try:
        new_fp = get_fingerprint(clip, sr, esp32_port=port)
    except Exception as exc:
        st.error(f"❌ Fingerprint error: {exc}")
        new_fp = {"ok": False, "hw": False}

    vr: ValidationResult = validate_correction(original_fp, new_fp)
    st.session_state[f"vr_{seg.index}"] = vr

    if vr.accepted:
        _do_save_correction(seg, new_text, vr, override=False)
    else:
        st.rerun()


def _do_save_correction(
    seg: SubtitleSegment,
    new_text: str,
    vr: Optional[ValidationResult],
    override: bool,
) -> None:
    """Persist a correction to SQLite and update session state."""
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
        save_correction(record)

        # Update segment in session state
        segs: list[SubtitleSegment] = st.session_state["segments"]
        for s in segs:
            if s.index == seg.index:
                s.corrected       = True
                s.correction_text = new_text
                s.text            = new_text
                break

        st.session_state["correction_count"] = (
            st.session_state.get("correction_count", 0) + 1
        )
        st.rerun()

    except Exception as exc:
        st.error(f"❌ Could not save correction: {exc}")


def _show_validation_result(vr: ValidationResult) -> None:
    """Display a ValidationResult using the appropriate Streamlit call."""
    if vr.tier in ("HIGH", "UNKNOWN"):
        st.success(vr.message)
    elif vr.tier == "MEDIUM":
        st.warning(vr.message)
    else:  # MISMATCH
        st.error(vr.message)


# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — Export
# ────────────────────────────────────────────────────────────────────────────

def render_tab_export() -> None:
    """Render download buttons for SRT, VTT, JSON exports."""
    if not st.session_state.get("done"):
        st.info("ℹ Transcription not yet run. Go to **Upload & Transcribe** first.")
        return

    segments: list[SubtitleSegment] = st.session_state["segments"]
    base_name = Path(st.session_state.get("filename", "subtitles")).stem

    st.header("📥 Export")

    srt_str  = to_srt(segments)
    vtt_str  = to_vtt(segments)
    json_str = to_json(segments)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        video_bytes = st.session_state.get("video_bytes")
        video_ext   = st.session_state.get("video_ext", "")
        is_video    = video_ext in (".mp4", ".avi", ".mov", ".mkv")
        if not video_bytes or not is_video:
            st.info("🔥 Burn-in requires a video file (not audio-only).")
        else:
            if st.button("🔥 Generate Burn-in .mp4", use_container_width=True):
                try:
                    with st.spinner("🔥 Encoding burn-in subtitles…"):
                        burned = to_burn_in(video_bytes, segments, ext=video_ext)
                    st.download_button(
                        label="⬇ Download burned-in .mp4",
                        data=burned,
                        file_name=f"{base_name}_burned.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )
                except RuntimeError as exc:
                    st.error(f"❌ Burn-in failed: {exc}")

    with col2:
        st.download_button(
            label="⬇ Download .srt",
            data=srt_str.encode("utf-8"),
            file_name=f"{base_name}.srt",
            mime="text/plain",
            use_container_width=True,
        )

    with col3:
        st.download_button(
            label="⬇ Download .vtt",
            data=vtt_str.encode("utf-8"),
            file_name=f"{base_name}.vtt",
            mime="text/vtt",
            use_container_width=True,
        )

    with col4:
        st.download_button(
            label="⬇ Download .json (with QC metadata)",
            data=json_str.encode("utf-8"),
            file_name=f"{base_name}_qc.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("SRT Preview")
    st.code(srt_str[:3000] + ("…" if len(srt_str) > 3000 else ""), language="text")


# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point: render sidebar then tabs."""
    render_sidebar()

    tab1, tab2, tab3 = st.tabs([
        "📁 Upload & Transcribe",
        "✏ Review & Edit",
        "📥 Export",
    ])

    with tab1:
        render_tab_upload()

    with tab2:
        render_tab_review()

    with tab3:
        render_tab_export()


if __name__ == "__main__":
    main()
