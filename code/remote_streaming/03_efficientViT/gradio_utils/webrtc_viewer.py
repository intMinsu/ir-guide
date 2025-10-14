"""
Reusable helpers for embedding jetson-utils' WebRTC stream into Gradio.

- launch_minimal_webrtc_viewer(): start a lightweight HTTP page that plays the stream
- build_webrtc_embed(): return HTML snippet used inside Gradio (iframe + 'raw' link)
"""

from __future__ import annotations
from string import Template
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading


# Minimal viewer HTML (no controls/logs, just video)
_MINIMAL_VIEWER_HTML = Template(r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <style>html,body{height:100%;margin:0;background:#000}video{width:100%;height:100%;object-fit:contain;background:#000;display:block}</style>
  <script src="https://webrtc.github.io/adapter/adapter-latest.js"></script>
</head>
<body>
  <video id="v" autoplay playsinline muted></video>
  <script>
  (function(){
    const WS_URL = (location.protocol==='https:'?'wss://':'ws://') + '$HOST:$PORT$PATH';
    const pc = new RTCPeerConnection({iceServers:[{urls:'stun:stun.l.google.com:19302'}]});
    const v = document.getElementById('v');

    pc.ontrack = (e)=>{ 
      v.srcObject = e.streams[0];
      v.play().catch(()=>{});
    };

    const ws = new WebSocket(WS_URL);

    ws.onmessage = async (ev)=>{
      let msg; try{ msg = JSON.parse(ev.data); }catch(e){return;}
      if(msg.type === 'sdp'){
        await pc.setRemoteDescription(msg.data);
        const answer = await pc.createAnswer();
        await pc.setLocalDescription(answer);
        ws.send(JSON.stringify({ type:'sdp', data: pc.localDescription }));
      }else if(msg.type === 'ice'){
        try{ await pc.addIceCandidate(msg.data); }catch(e){}
      }
    };

    pc.onicecandidate = (ev)=>{
      if(ev.candidate){
        ws.send(JSON.stringify({ type:'ice', data: ev.candidate }));
      }
    };
  })();
  </script>
</body>
</html>""")


def launch_minimal_webrtc_viewer(host: str, port: int, path: str, page_port_candidates=None):
    """
    Start a tiny HTTP server that serves a single HTML page with a minimal
    WebRTC viewer (no logs/stats). Returns (bind_host, bind_port, http_url).
    """
    if page_port_candidates is None:
        page_port_candidates = [port + 1, 8560, 9000, 9070, 9080]

    html = _MINIMAL_VIEWER_HTML.substitute(
        HOST=host,
        PORT=str(port),
        PATH=path if path.startswith("/") else ("/" + path),
    )
    html_bytes = html.encode("utf-8")

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.end_headers()
            self.wfile.write(html_bytes)
        def log_message(self, *args, **kwargs):
            pass  # silence

    bind_host = "0.0.0.0"
    srv = None
    for p in page_port_candidates:
        try:
            srv = ThreadingHTTPServer((bind_host, p), _Handler)
            page_port = p
            break
        except OSError:
            continue
    if srv is None:
        srv = ThreadingHTTPServer((bind_host, 0), _Handler)
        page_port = srv.server_address[1]

    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return bind_host, page_port, f"http://{host}:{page_port}/"


def build_webrtc_embed(host_hint: str | None, webrtc_port: int, stream_name: str) -> str:
    """
    Return the HTML snippet for Gradio:
    - If host_hint is provided, launch the minimal viewer and iframe it.
    - Also add a link to the jetson-utils 'raw' page.
    - If no host_hint, show guidance to include host.
    """
    if host_hint:
        _, _, mini_url = launch_minimal_webrtc_viewer(
            host=host_hint, port=webrtc_port, path=f"/{stream_name}"
        )
        raw_url = f"http://{host_hint}:{webrtc_port}/?stream=/{stream_name}"
        return (
            "<div style='display:flex;gap:16px;flex-wrap:wrap'>"
            "  <div style='flex:1;min-width:360px'>"
            f"    <iframe src='{mini_url}' "
            "            style='width:100%;aspect-ratio:16/9;border:0;border-radius:12px' "
            "            allow='autoplay; fullscreen'></iframe>"
            "  </div>"
            "</div>"
            "<div style='margin-top:8px'>"
            f"  <a href='{raw_url}' target='_blank' "
            "     style='text-decoration:none;padding:6px 10px;border-radius:8px;"
            "            background:#eef;border:1px solid #ccd;color:#223;font-weight:600;'>"
            "     See raw WebRTC page</a>"
            "</div>"
        )
    else:
        return (
            "<div style='padding:8px;border:1px dashed #aaa;border-radius:8px'>"
            "<div style='font-weight:600;margin-bottom:6px'>WebRTC viewer</div>"
            "Output URL didn’t include a host. Re-run with an explicit host, e.g.:"
            "<pre>--output webrtc://&lt;JETSON-IP&gt;:8554/sam_points</pre>"
            "</div>"
        )
