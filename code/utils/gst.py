"""
GStreamer pipeline builders for Jetson RTSP labs (USB/CSI → OpenCV → RTSP).

English
-------
Utilities to assemble GStreamer strings for:
  • USB cameras via v4l2src → (OpenCV appsink)
  • CSI cameras via nvarguscamerasrc → (OpenCV appsink)
  • RTSP writer via appsrc → NVENC (nvv4l2h26xenc) → rtspclientsink
They add low-latency defaults (is-live, do-timestamp, leaky queue, config-interval).

한국어
------
다음 GStreamer 문자열을 손쉽게 만드는 유틸입니다:
  • USB 카메라(v4l2src) → OpenCV appsink
  • CSI 카메라(nvarguscamerasrc) → OpenCV appsink
  • RTSP 송출(appsrc → NVENC → rtspclientsink)
기본적으로 지연시간을 줄이는 옵션을 포함합니다(is-live, do-timestamp, leaky queue 등).
"""
from __future__ import annotations
from typing import Optional

def parse_fps(fps_str: str) -> float:
    """
    Parse 'N/D' style framerate to float (e.g., '30/1' -> 30.0).
    """
    if isinstance(fps_str, (int, float)):
        return float(fps_str)
    if "/" in fps_str:
        n, d = fps_str.split("/", 1)
        return float(n) / float(d)
    return float(fps_str)

def get_rtsp_url(host: str = "localhost", port: int = 8554, name: str = "mystream") -> str:
    """
    Build RTSP URL.
    """
    return f"rtsp://{host}:{port}/{name}"

def build_usb_caps(
    device: str,
    src_format: str,
    w: int,
    h: int,
    fps: str,
    flip: int = 0,
) -> str:
    """
    Build a USB-camera capture pipeline (to OpenCV appsink).

    Parameters
    ----------
    device      : e.g. '/dev/video0'
    src_format  : 'MJPG' or 'YUY2'
    w, h        : resolution
    fps         : e.g. '30/1'
    flip        : nvvidconv flip-method (0~7)

    Returns
    -------
    gst string ending in appsink (BGR)
    """
    if src_format.upper() == "YUY2":
        # CPU format convert to BGR (OpenCV), NVMM path breaks after videoconvert
        return (
            f"v4l2src device={device} "
            f"! video/x-raw,width={w},height={h},framerate={fps},format=YUY2 "
            f"! nvvidconv "
            f"! video/x-raw,format=BGRx "
            f"! videoconvert ! video/x-raw,format=BGR "
            f"! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream "
            f"! appsink drop=true max-buffers=1 sync=false"
        )
    elif src_format.upper() == "MJPG":
        # HW decode MJPEG → NVMM → convert to BGR for OpenCV appsink
        return (
            f"v4l2src device={device} io-mode=2 "
            f"! image/jpeg,width={w},height={h},framerate={fps} "
            f"! nvv4l2decoder mjpeg=1 "
            f"! nvvidconv "
            f"! video/x-raw,format=BGRx "
            f"! videoconvert ! video/x-raw,format=BGR "
            f"! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream "
            f"! appsink drop=true max-buffers=1 sync=false"
        )
    else:
        raise NotImplementedError("Supported src_format: 'YUY2' or 'MJPG'")

def build_csi_caps(
    sensor_id: int = 0,
    w: int = 1920,
    h: int = 1080,
    fps: str = "30/1",
    flip: int = 0,
) -> str:
    """
    Build a CSI-camera capture pipeline (to OpenCV appsink).
    Not used by default, but handy for IMX sensors.

    Notes
    -----
    The output to OpenCV is BGR (CPU space) after videoconvert.
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} tnr-mode=2 wbmode=3 "
        f"! video/x-raw(memory:NVMM),width={w},height={h},framerate={fps},format=NV12 "
        f"! nvvidconv flip-method={flip} "
        f"! video/x-raw,format=BGRx "
        f"! videoconvert ! video/x-raw,format=BGR "
        f"! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream "
        f"! appsink drop=true max-buffers=1 sync=false"
    )

def build_writer_caps(
    codec: str,
    flip: int,
    bitrate: int,
    host: str,
    port: int,
    name: str,
    w: Optional[int] = None,
    h: Optional[int] = None,
    fps: Optional[str] = None,
) -> str:
    """
    Build an RTSP writer pipeline (OpenCV VideoWriter → RTSP).

    Parameters
    ----------
    codec  : 'h264' or 'h265'
    flip   : nvvidconv flip-method
    bitrate: target bitrate in bps (e.g., 8_000_000)
    host   : RTSP server host
    port   : RTSP server port
    name   : stream name
    w, h   : (optional) caps on appsrc if known
    fps    : (optional) framerate 'N/D' caps on appsrc

    Returns
    -------
    gst string starting with appsrc → nvv4l2h26xenc → rtspclientsink

    Implementation details
    ----------------------
    • is-live/ do-timestamp/ format=time : real-time timestamps
    • leaky queue: avoid back-pressure/latency build-up
    • config-interval=1: periodically send VPS/SPS/PPS (faster client lock)
    • latency=0 on rtspclientsink: remove publisher-side extra buffer
    """
    enc = "nvv4l2h265enc" if codec.lower() == "h265" else "nvv4l2h264enc"
    parse = "h265parse" if codec.lower() == "h265" else "h264parse"

    caps = "video/x-raw,format=BGR"
    if w is not None and h is not None:
        caps += f",width={w},height={h}"
    if fps is not None:
        caps += f",framerate={fps}"

    return (
        f"appsrc is-live=true do-timestamp=true format=time "
        f"! {caps} "
        f"! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream "
        f"! videoconvert ! video/x-raw,format=BGRx "
        f"! nvvidconv flip-method={flip} "
        f"! video/x-raw(memory:NVMM),format=NV12 "
        f"! {enc} maxperf-enable=1 control-rate=1 bitrate={bitrate} iframeinterval=60 "
        f"! {parse} config-interval=1 "
        f"! queue "
        f"! rtspclientsink location={get_rtsp_url(host, port, name)} latency=0"
    )
