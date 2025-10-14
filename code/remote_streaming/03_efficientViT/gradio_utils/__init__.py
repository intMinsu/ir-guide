from .webrtc_viewer import build_webrtc_embed, launch_minimal_webrtc_viewer

# Utilities for embedding/controlling WebRTC streams in Gradio
def parse_webrtc_output(url: str):
    """
    Parse "webrtc://<host>:<port>/<name>" into (host_or_none, port, name).
    If host is '@' or empty, return None to use window.location.hostname in the UI.
    """
    if not url or not url.lower().startswith("webrtc://"):
        return None, None, None
    rest = url[len("webrtc://") :]
    if "/" in rest:
        hostport, name = rest.split("/", 1)
    else:
        hostport, name = rest, "output"
    host, port = None, 8554
    if ":" in hostport:
        h, p = hostport.split(":", 1)
        if h and h != "@":
            host = h
        try:
            port = int(p)
        except Exception:
            port = 8554
    else:
        if hostport and hostport != "@":
            host = hostport
    return host, port, name