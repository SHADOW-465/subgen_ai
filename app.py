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
        "port_scan_done": False,  # bool — initial port scan completed
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

def render_sidebar() -> None:
    """Render the full left sidebar: hardware, AI settings, DB stats."""
    with st.sidebar:
        st.markdown("## 🎬 SubGEN AI")
        st.caption("Hardware-Aware Subtitle Generator")
        st.divider()

        # ── Hardware section ────────────────────────────────────────────────
        st.markdown("### ⚙ Hardware")

        # --- Rescan button (always allow re-detection) ---
        col_scan, col_manual = st.columns([1, 1])
        with col_scan:
            do_scan = st.button("🔍 Rescan Ports", use_container_width=True)
        with col_manual:
            manual_entry = st.toggle("✏ Enter manually",
                                     value=st.session_state.get("manual_port_mode", False),
                                     key="manual_port_mode")

        # Auto-scan once on first load; only re-scan when button is clicked.
        # This avoids the Bluetooth COM port slowdown blocking every re-render.
        needs_scan = do_scan or not st.session_state.get("port_scan_done", False)
        if needs_scan:
            with st.spinner("🔎 Scanning serial ports…"):
                detected_ports = find_all_esp32_ports()
            st.session_state["esp32_ports"] = detected_ports
            st.session_state["port_scan_done"] = True
            if detected_ports:
                # Only auto-set port if not already set to a valid detected port
                if st.session_state.get("port") not in detected_ports:
                    st.session_state["port"] = detected_ports[0]
                st.session_state["hw_status"] = "connected"
            else:
                st.session_state["hw_status"] = "software"

        # Manual port text input
        if manual_entry:
            manual_port = st.text_input(
                "COM Port (e.g. COM3, /dev/ttyUSB0)",
                value=st.session_state.get("port") or "",
                placeholder="COM3",
                key="manual_port_input",
            ).strip()
            if manual_port:
                st.session_state["port"] = manual_port
                st.session_state["hw_status"] = "connected"
            else:
                st.session_state["port"] = None
                st.session_state["hw_status"] = "software"
        else:
            # Selectbox from auto-detected ports
            port_options = st.session_state["esp32_ports"] + ["Software only"]
            current_port = st.session_state.get("port") or "Software only"
            default_idx = (
                port_options.index(current_port)
                if current_port in port_options
                else len(port_options) - 1
            )
            selected_port = st.selectbox(
                "ESP32 Port",
                options=port_options,
                index=default_idx,
                key="port_select",
            )
            st.session_state["port"] = (
                None if selected_port == "Software only" else selected_port
            )

        # Status indicator
        hw_status = st.session_state["hw_status"]
        active_port = st.session_state.get("port")
        if active_port is not None:
            st.markdown(f"🟢 **Connected** ({active_port}) — Hardware MFCC active")
        elif hw_status == "error":
            st.markdown("🔴 **Not found** — check USB cable / driver")
        else:
            st.markdown("🔵 **Software mode** — SW-MFCC fallback")

        # Diagnostic expander — shows all available serial ports
        with st.expander("🔎 All serial ports (diagnostics)", expanded=False):
            all_ports = list_all_ports()
            if not all_ports:
                st.caption("No serial ports found on this machine.")
            else:
                for p in all_ports:
                    vid_str = f"VID:{p['vid']:04X}" if p["vid"] else "VID:---"
                    pid_str = f"PID:{p['pid']:04X}" if p["pid"] else "PID:---"
                    st.code(
                        f"{p['device']}  |  {p['description']}\n"
                        f"{vid_str} {pid_str}  |  {p['hwid']}",
                        language="text",
                    )

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

        language_options = [
            "Auto-detect", "ta", "te", "hi", "ml", "kn", "mr",
            "en", "zh", "ja", "ko", "fr", "de", "es", "pt", "ar",
        ]
        lang_sel = st.selectbox(
            "Language",
            options=language_options,
            index=0,
            help="Select the spoken language, or Auto-detect.",
        )
        st.session_state["language"] = None if lang_sel == "Auto-detect" else lang_sel

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
    import numpy as np

    # Capture raw bytes via getvalue() — works regardless of stream position.
    # Done BEFORE the try block so bytes are saved even if transcription fails.
    raw_bytes = uploaded_file.getvalue()
    suffix    = Path(uploaded_file.name).suffix or ".mp4"
    st.session_state["video_bytes"] = raw_bytes
    st.session_state["video_ext"]   = suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw_bytes)          # use raw_bytes, NOT uploaded_file.read()
        tmp_path = tmp.name

    progress_bar = st.progress(0, text="⏳ Starting…")
    status_text  = st.empty()

    segments_acc: list[SubtitleSegment] = []

    def _progress(i: int, total: int, seg: SubtitleSegment) -> None:
        pct = int(i / max(total, 1) * 100)
        progress_bar.progress(pct, text=f"🔄 Processing segment {i}/{total}…")
        status_text.caption(
            f"[{_fmt_time(seg.start)} → {_fmt_time(seg.end)}]  {seg.text[:60]}"
        )
        segments_acc.append(seg)

    try:
        with st.spinner("Extracting audio and running ASR…"):
            segments = transcribe(
                video_path=tmp_path,
                model_size=st.session_state.get("model_size", DEFAULT_MODEL),
                language=st.session_state.get("language"),
                task=st.session_state.get("task", "transcribe"),
                esp32_port=st.session_state.get("port"),
                progress_callback=_progress,
            )

        # Also store the raw audio array for on-demand clip extraction
        from subgen_ai.core.transcriber import extract_audio
        audio_arr, sr = extract_audio(tmp_path)
        st.session_state["audio"]    = audio_arr
        st.session_state["sr"]       = sr
        st.session_state["segments"] = segments
        st.session_state["filename"] = uploaded_file.name
        st.session_state["done"]     = True
        st.session_state["correction_count"] = 0

        progress_bar.progress(100, text="✅ Done!")
        status_text.empty()
        st.rerun()

    except Exception as exc:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Transcription failed: {exc}")

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
