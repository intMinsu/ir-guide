# Most of code is borrowed from jetson_inference repo.
# https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md#python

import sys
import argparse
import jetson_utils
from jetson_utils import videoSource, videoOutput

parser = argparse.ArgumentParser()

# [0. Camera w/h/fps setting]
# Please check 02_mediamtx/camera_formats.json
w = 1920
h = 1080

# V4L2 cameras will be created using the camera format with the highest framerate that most closely matches the desired resolution.
# See https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md#command-line-arguments
videosource_dict = {'width': w,
                    'height': h,
                    'codec': 'mjpeg', # For logitech C920, use mjpeg
                    'encoder': 'v4l2',
                    'save': './input.mp4', # Remove if you don't save
                    }

videooutput_dict = {'codec': 'h265',
                    'encoder': 'v4l2',
                    'bitrate': 8000000,
                    'save': './output.mp4', # Remove if you don't save
                    }
                    # H265 bitrate 720p@30fps : 4Mbps / 1080p@30fps : 8Mbps
                    # https://support.google.com/youtube/answer/2853702?hl=ko

# [1. Input Streams]
# https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md#input-streams 
parser.add_argument("--input", type=str, default="v4l2:///dev/video0", help="URI of the input stream")

# [2. Output Streams]
# https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md#output-streams
parser.add_argument("--output", type=str, default="rtsp://localhost:8554/mystream", nargs='?', help="URI of the output stream")
args = parser.parse_args()

# create video sources & outputs with videoSource and videoOutput
# [3. videoSource]
# https://github.com/dusty-nv/jetson-utils/blob/master/video/videoSource.h 
input = videoSource(args.input, options=videosource_dict)

# [4. videoOutput]                                    
# https://github.com/dusty-nv/jetson-utils/blob/master/video/videoOutput.h
output = videoOutput(args.output, options=videooutput_dict)
                            

# [5. Capture frames]
# For OpenCV-based implementation, refer to https://www.youtube.com/watch?v=mB025B7KpeE
# capture frames until end-of-stream (or the user exits)
while True:
    # format can be: rgb8, rgba8, rgb32f, rgba32f (rgb8 is the default)
    # timeout can be: -1 for infinite timeout (blocking), 0 to return immediately, >0 in milliseconds (default is 1000ms)
    imgCuda = input.Capture(format='rgb8', timeout=1000)  
	
    if imgCuda is None:  # if a timeout occurred
        continue
    
    # [6. See imgCuda's properties]
    # You can access properties about the image like imgCuda.width
    # https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-image.md#image-capsules-in-python
    # print(imgCuda) be like:
    #     <cudaImage object>
    #    -- ptr:      0x203cea000
    #    -- size:     6220800
    #    -- width:    1920
    #    -- height:   1080
    #    -- channels: 3
    #    -- format:   rgb8
    #    -- mapped:   true
    #    -- freeOnDelete: false
    #    -- timestamp:    6.318738

    # [Example 1. cuda-to-pytorch]
    # https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-image.md#cuda-array-interface
    # map to torch tensor using __cuda_array_interface__
    # tensor = torch.as_tensor(imgCuda, device='cuda')

    # [Example 2. color conversion from rgb8 to rgba32f]
    # https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-image.md#color-conversion
    # allocate the output as rgba32f, with the same width/height as the input
    # imgCuda_rgba32f = jetson_utils.cudaAllocMapped(width=imgCuda.width, height=imgCuda.height, format='rgba32f')

    # convert from rgb8 to rgba32f (the formats used for the conversion are taken from the image capsules)
    # jetson_utils.cudaConvertColor(imgCuda, imgCuda_rgba32f)
    
    output.Render(imgCuda)

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break