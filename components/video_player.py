# subgen_ai/components/video_player.py
"""
SubGEN AI — Browser Video Player Component.

Renders a native HTML <video> element with a WebVTT subtitle <track>
injected via st.components.v1.html().

Strategy:
  - Files ≤ 200 MB: embedded as base64 data URIs so the subtitle <track>
    works natively in the browser without a file server.
  - Files > 200 MB: fall back to st.video() (no inline subtitle overlay,
    but playback always works regardless of file size).
"""
import base64

import streamlit as st
import streamlit.components.v1 as components

# Threshold for switching from inline-base64 to st.video() fallback.
# Base64 encoding adds ~33% overhead; 200 MB source → ~267 MB encoded string.
# Most modern browsers handle this without issue, and it keeps subtitles working.
_B64_THRESHOLD = 200 * 1024 * 1024  # 200 MB

_ALLOWED_MIME = frozenset({
    "video/mp4", "video/webm", "video/ogg",
    "video/x-msvideo", "video/quicktime", "video/x-matroska",
})


def render_video_player(
    video_bytes: bytes,
    mime: str,
    vtt_str: str,
    height: int = 360,
) -> None:
    """
    Render an HTML5 video player with embedded subtitle track.

    No hard size limit — large files (>200 MB) fall back to st.video()
    without the subtitle overlay, but playback is always available.

    Args:
        video_bytes: Raw video file bytes.
        mime:        MIME type string, e.g. "video/mp4".
        vtt_str:     Full WebVTT string (must start with "WEBVTT").
        height:      Player height in pixels (default 360).
    """
    if mime not in _ALLOWED_MIME:
        st.error(f"⚠ Unsupported video format: {mime}")
        return

    if len(video_bytes) == 0:
        st.warning("⚠ No video data to display.")
        return

    if not vtt_str.startswith("WEBVTT"):
        st.warning("⚠ Invalid WebVTT string — subtitles may not display.")
        # Don't return — still show the player, just warn

    # ── Large file path: use Streamlit's native player (no size limit) ────────
    if len(video_bytes) > _B64_THRESHOLD:
        size_mb = len(video_bytes) / (1024 * 1024)
        st.info(
            f"ℹ File is {size_mb:.0f} MB — using native player. "
            "Subtitle overlay is only available for files ≤ 200 MB, "
            "but you can still download subtitles below."
        )
        st.video(video_bytes)
        return

    # ── Standard path: inline base64 with WebVTT subtitle track ──────────────
    video_b64 = base64.b64encode(video_bytes).decode()
    vtt_b64   = base64.b64encode(vtt_str.encode("utf-8")).decode()

    html = f"""
<video controls width="100%"
       style="max-height:{height}px; background:#000; border-radius:8px;">
  <source src="data:{mime};base64,{video_b64}" type="{mime}">
  <track default kind="subtitles" srclang="en" label="Subtitles"
         src="data:text/vtt;base64,{vtt_b64}">
  Your browser does not support the video tag.
</video>
"""
    components.html(html, height=height + 20)
