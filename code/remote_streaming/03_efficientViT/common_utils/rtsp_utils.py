"""
RTSP viewer gating with startup grace + optional idle auto-exit + ultra-low keepalive.

Recommended defaults (practical on Jetson)
------------------------------------------
- startup_grace = 3.0
    Short prime window so the RTSP pipeline spins up and clients can attach quickly.
- idle_timeout  = 300.0
    Generous 5-minute inactivity timeout for long/variable pipelines.
- keepalive_hz  = 0.05
    Essentially disabled (one frame ~every 20s). Acts as a last-resort
    edge-trigger for picky clients, while avoiding GPU memory spill.

Why
---
RTSP pipelines often need a handful of initial buffers before clients can attach.
If you gate rendering from frame 0, the server may shut down before a viewer connects.
This helper:
  • Renders unconditionally for `startup_grace` seconds to prime the stream
  • Afterwards, renders only while a viewer is connected (ESTABLISHED TCP)
  • Optionally times out after prolonged inactivity (`idle_timeout`)
  • Optionally trickle-renders at a very low rate (`keepalive_hz`) to keep the
    pipeline alive without significant memory pressure

Usage
-----
from common_utils.rtsp_utils import RTSPGate, is_rtsp_uri

gate = RTSPGate("rtsp://localhost:8554/stream",
                startup_grace=3.0,
                idle_timeout=300.0,
                keepalive_hz=0.05)

while True:
    gate.update()
    if gate.should_render():
        out.Render(img)
    if gate.idle_timed_out():
        break
"""

import time
from urllib.parse import urlparse


def is_rtsp_uri(uri: str) -> bool:
    """Return True if the URI uses the rtsp:// scheme."""
    try:
        return (urlparse(uri).scheme or "").lower() == "rtsp"
    except Exception:
        return False


def _parse_port_from_rtsp(uri: str, default=8554) -> int:
    """Extract port from rtsp:// URI (returns default if missing/invalid)."""
    try:
        p = urlparse(uri)
        if (p.scheme or "").lower() != "rtsp":
            return -1
        return p.port or default
    except Exception:
        return -1


def _tcp_has_established_connections(port: int) -> bool:
    """Return True if there's at least one ESTABLISHED (01) TCP connection to `port`."""
    ACTIVE = {"01"}  # 01=ESTABLISHED

    def _scan(path):
        try:
            with open(path, "r") as f:
                lines = f.read().splitlines()[1:]  # skip header
        except FileNotFoundError:
            return False

        hex_port = f"{int(port):04X}"
        for ln in lines:
            cols = ln.split()
            if len(cols) < 4:
                continue
            local = cols[1]
            state = cols[3]  # 01=ESTABLISHED, 0A=LISTEN, 06=TIME_WAIT, ...
            try:
                lp = local.split(":")[1].upper()
            except Exception:
                continue
            if lp == hex_port and state in ACTIVE:
                return True
        return False

    return _scan("/proc/net/tcp") or _scan("/proc/net/tcp6")


def rtsp_viewer_connected(rtsp_uri: str) -> bool:
    """Instantaneous check for at least one ESTABLISHED TCP connection to the RTSP port."""
    port = _parse_port_from_rtsp(rtsp_uri)
    return port > 0 and _tcp_has_established_connections(port)


class RTSPGate:
    """
    Minimal RTSP viewer gate.

    Args:
        rtsp_uri:       e.g., "rtsp://localhost:8554/stream"
        startup_grace:  seconds to always allow rendering after start (default 3.0)
        idle_timeout:   seconds with no viewer after which idle_timed_out() becomes True
                        (default 300.0; None disables auto-exit)
        keepalive_hz:   while no viewer & after grace, render at this low rate to keep
                        pipeline alive (0 disables). Default 0.05 (~1 frame / 20s).
    """
    def __init__(self,
                 rtsp_uri: str,
                 startup_grace: float = 3.0,
                 idle_timeout: float = 300.0,
                 keepalive_hz: float = 0.05):
        self.rtsp_uri = rtsp_uri
        self.startup_grace = float(startup_grace)
        self.idle_timeout = None if idle_timeout is None else float(idle_timeout)
        self.keepalive_hz = float(keepalive_hz or 0.0)

        self._enabled = is_rtsp_uri(rtsp_uri)
        self._started_at = time.time()
        self._last_seen_connected = None
        self._connected_now = False
        self._next_keepalive = self._started_at

    def update(self) -> bool:
        """Poll current connection state and update internal timestamps. Returns raw connected bool."""
        if not self._enabled:
            # Not RTSP? Always allow rendering.
            self._connected_now = True
            if self._last_seen_connected is None:
                self._last_seen_connected = time.time()
            return True

        connected = rtsp_viewer_connected(self.rtsp_uri)
        self._connected_now = connected
        if connected:
            self._last_seen_connected = time.time()
        return connected

    def should_render(self) -> bool:
        """
        True if we should Render() this frame:
          • Always True during startup_grace
          • After that, True only while a viewer is connected
          • If keepalive_hz>0, allow a trickle Render() when disconnected to keep pipeline alive
        """
        if not self._enabled:
            return True

        now = time.time()

        # prime the stream so clients can connect
        if (now - self._started_at) <= self.startup_grace:
            return True

        if self._connected_now:
            return True

        # Optional trickle keepalive (very low rate by default)
        if self.keepalive_hz > 0.0:
            if now >= self._next_keepalive:
                self._next_keepalive = now + (1.0 / self.keepalive_hz)
                return True

        return False

    def idle_timed_out(self) -> bool:
        """
        True if we've seen no viewer for idle_timeout seconds (after startup_grace).
        Returns False when idle_timeout is None or gating is disabled.
        """
        if not self._enabled or self.idle_timeout is None:
            return False

        now = time.time()
        if (now - self._started_at) <= self.startup_grace:
            return False

        last = self._last_seen_connected or self._started_at
        return (now - last) >= self.idle_timeout
