import sys
import cv2
from datetime import datetime
from utils.camera import val_camera_configuration

# [0. Camera w/h/fps setting]
# check available combination : camera_formats.json
fps = '30/1'
w = 1920
h = 1080
src_format = 'MJPG' # YUY2 or MJPG for Logitech C920
flip = 0

if not val_camera_configuration(src_format, w, h, fps):
    raise ValueError("Unsupported camera format. Check available combination in camera_formats.json")


# [1. USB Camera]
# [1-1. v4l2src configuration (USB Camera)] YUYV = YUY2
if src_format == 'YUY2':
    gst_cap = 'v4l2src device=/dev/video0' + \
        ' ! video/x-raw, width='+str(w)+',height='+str(h)+',framerate='+fps+',format=YUY2' + \
        ' ! nvvidconv' + \
        ' ! video/x-raw, format=BGRx' + \
        ' ! videoconvert' + \
        ' ! video/x-raw, format=BGR' + \
        ' ! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream' + \
        ' ! appsink drop=true max-buffers=1 sync=false'

elif src_format == 'MJPG':
    gst_cap = 'v4l2src device=/dev/video0 io-mode=2' + \
        ' ! image/jpeg, width='+str(w)+',height='+str(h)+',framerate='+fps + \
        ' ! nvv4l2decoder mjpeg=1' + \
        ' ! nvvidconv' + \
        ' ! video/x-raw,format=BGRx' + \
        ' ! videoconvert' + \
        ' ! video/x-raw, format=BGR' + \
        ' ! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream' + \
        ' ! appsink drop=true max-buffers=1 sync=false'

else:
    raise NotImplementedError("Supported format : YUY2/MJPG")

# [1-2. nvv4l2camerasrc configuration (Not working)]
# https://docs.nvidia.com/jetson/archives/r35.4.1/DeveloperGuide/text/SD/Multimedia/AcceleratedGstreamer.html#progressive-capture-using-nvv4l2camerasrc
# https://github.com/dusty-nv/jetson-inference/issues/1521
# gst_cap = 'nvv4l2camerasrc device=/dev/video0' + \
#     ' ! video/x-raw(memory:NVMM), width=1280, height=720, framerate=10/1, format=UYVY, interlace-mode=progressive' + \
#     ' ! nvvidconv flip-method=' + str(flip) + ' ! video/x-raw, format=BGRx' + \
#     ' ! videoconvert ! video/x-raw, format=BGR' + \
#     ' ! videobalance contrast=1.5 brightness=-.2' + \
#     ' ! queue ! appsink drop=false'


# [2. CSI Camera]
# https://www.youtube.com/watch?v=R_d_McJ2stg
# https://velog.io/@tmddn833/Gstreamer-%EC%A0%95%EB%A6%AC
# gst_cap = 'nvarguscamerasrc sensor-id=0 tnr-mode=2 wbmode=3' + \
#         ' ! video/x-raw(memory:NVMM), width=3264, height=2464, framerate=21/1,format=NV12' + \
#         ' ! nvvidconv flip-method=0' + \
#         ' ! video/x-raw, width=1920, height=1080, format=BGRx' + \
#         ' ! videoconvert ! video/x-raw, format=BGR' + \
#         ' ! videobalance contrast=1.5 saturation=0 brightness=-.2' + \
#         ' ! appsink'


# [3. To send the videostream via rtsp tunnel]
# [3-1. x264enc + rtspclientsink]
# gst_writer = 'appsrc ! videoconvert' + \
#             ' ! video/x-raw,format=I420' + \
#             ' ! x264enc speed-preset=ultrafast bitrate=2048000 key-int-max=' + str(int(fps) * 2) + \
#             ' ! video/x-h264,profile=baseline' + \
#             ' ! rtspclientsink location=rtsp://localhost:8554/mystream'

# [3-2. nvv4l2h265enc + rtspclientsink]
# This one shows much more robust streaming.
# https://forums.developer.nvidia.com/t/h264-vs-h265/115260/4
gst_writer = 'appsrc' + \
            ' ! videoconvert' + \
            ' ! video/x-raw, format=BGRx' +\
            ' ! nvvidconv flip-method=' + str(flip) + \
            ' ! video/x-raw(memory:NVMM), format=NV12' + \
            ' ! nvv4l2h265enc maxperf-enable=1 bitrate=14000000 control-rate=1 vbv-size=450000' + \
            ' ! h265parse' + \
            ' ! queue' + \
            ' ! rtspclientsink location=rtsp://localhost:8554/mystream'

# [4. To send the videostream X11 SSH tunneling]
# Do not use X11 SSH tunneling to use remote display as it is not recommended by NVIDIA
# https://forums.developer.nvidia.com/t/export-video-stream-from-orin-nano-to-a-remote-display/252813/13
# gst_writer = 'appsrc ! videoconvert ! video/x-raw, format=BGRx' + \
#             ' ! nvvidconv flip-method=' + str(flip) + ' ! video/x-raw(memory:NVMM), format=NV12' + \
#             ' ! queue ! nv3dsink'

# [5. VideoCapture & VideoWriter]
cap = cv2.VideoCapture(gst_cap ,cv2.CAP_GSTREAMER)
print(f'Src opened, {w}x{h} @ {fps} fps')

# [6. Frame w/h/fps setting]
frame_fps = float(eval(fps))
frame_w = int(w)
frame_h = int(h)

out = cv2.VideoWriter(gst_writer, 
                    cv2.CAP_GSTREAMER, # cv2.CAP_FFMPEG or cv2.CAP_GSTREAMER
                    0, # fourcc code is useless when using gstreamer backend
                    frame_fps, 
                    (frame_w, frame_h), 
                    True) # isColor

if not out.isOpened():
    print("Failed to open output")
    exit()


try:
    if cap.isOpened():
        while True:
            ret_val, img = cap.read()
            if not ret_val:
                break

            # ADD EDIT HERE 
            # [Example 1. Output grayscale video]
            # gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR) # Convert grayscale back to BGR format to match the pipeline's expectations
 
            out.write(img)
            print("%s frame written to the server" % datetime.now())
    else:
        print ("pipeline open failed")
        # TODO: It does not terminate the process
        exit()
except KeyboardInterrupt:
    print("\nProcess interrupted by user. Exiting...")
finally:
    cap.release()
    out.release()  
    cv2.destroyAllWindows()

print("successfully exit")