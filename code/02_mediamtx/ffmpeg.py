import cv2

# Set up the camera input
camera_index = 0  # Typically 0 is /dev/video0 on Linux
cap = cv2.VideoCapture(camera_index)

# Set camera properties (if needed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

# Define the RTSP stream address
rtsp_address = "rtsp://localhost:8554/mystream"  # Replace with $RTSP_PORT and $MTX_PATH as needed

# Define the codec and create VideoWriter object
# We will use 'ffmpeg' backend and settings similar to your command
# The codec used is 'H264', which ffmpeg will convert to yuv420p pixel format

fps = 30.0
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# FFmpeg options to use yuv420p pixel format and ultrafast preset
fourcc = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter(
    f"appsrc ! videoconvert ! x264enc tune=zerolatency speed-preset=ultrafast bitrate=600 ! rtspclientsink location={rtsp_address} sync=false",
    cv2.CAP_GSTREAMER,  # Backend
    fourcc,  # Codec
    fps,  # Frames per second
    (frame_width, frame_height),  # Frame size
    True  # Is color
)

if not out.isOpened():
    print("Error: Could not open the output video stream.")
    cap.release()
    exit()

# Capture and stream video frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Write the frame to the RTSP stream
    out.write(frame)

    # Display the frame (optional, can remove if headless)
    cv2.imshow('Frame', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
