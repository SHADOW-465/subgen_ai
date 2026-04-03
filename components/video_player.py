# subgen_ai/components/video_player.py
"""
SubGEN AI — Browser Video Player Component.

Renders a native HTML <video> element with a WebVTT subtitle <track>
injected via st.components.v1.html(). Both video and VTT are embedded
as base64 data URIs — no file server required.
"""
import base64

import streamlit as st
import streamlit.components.v1 as components

MAX_PLAYER_BYTES = 80 * 1024 * 1024  # 80 MB — browser limit for data URIs

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

    if len(video_bytes) > MAX_PLAYER_BYTES:
        st.warning(
            "⚠ File too large for in-browser player (>80 MB). "
            "Download subtitles below."
        )
        return

    if not vtt_str.startswith("WEBVTT"):
        st.warning("⚠ Invalid WebVTT string — subtitles may not display.")
        # Don't return — still show the player, just warn

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
