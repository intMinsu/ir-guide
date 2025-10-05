# Most of code is borrowed from jetson_inference repo.
# https://github.com/dusty-nv/jetson-inference/blob/master/python/examples/detectnet.py
# https://github.com/dusty-nv/jetson-inference/blob/master/python/examples/detectnet-snap.py

import sys
import argparse
from datetime import datetime
import math
import os

from jetson_inference import detectNet
from jetson_utils import (videoSource, videoOutput, saveImage, Log,
                          cudaAllocMapped, cudaCrop, cudaDeviceSynchronize)


# [0. Camera w/h/fps setting]
w = 640
h = 480

videosource_dict = {'width': w,
                    'height': h,
                    'codec': 'mjpeg', # For logitech C920, use mjpeg
                    'encoder': 'v4l2'
                    }

videooutput_dict = {'codec': 'h265',
                    'encoder': 'v4l2',
                    'bitrate': 8000000}

parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

# [1. Input Streams]
parser.add_argument("input", type=str, default="v4l2:///dev/video0", nargs='?', help="URI of the input stream")

# [2. Output Streams]
parser.add_argument("output", type=str, default="rtsp://localhost:8554/mystream", nargs='?', help="URI of the output stream")

# [3. Network]
# for more information, refer to https://github.com/dusty-nv/jetson-inference/blob/master/docs/detectnet-console-2.md
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 
parser.add_argument("--alpha", type=int, default=120, help="alpha blending value used during overlay")

parser.add_argument("--save-snapshot", dest='save_snapshot', action='store_true')
parser.add_argument("--dont-save-snapshot", dest='save_snapshot', action='store_false')
parser.add_argument("--snapshot-dir", type=str, default="image/detection", help="output directory of detection snapshots")
parser.add_argument("--timestamp", type=str, default="%Y%m%d-%H%M%S", help="timestamp format used in snapshot filenames")


try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

if args.save_snapshot:
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

net = detectNet(args.network, sys.argv, args.threshold)

# [4. videoSource]
input = videoSource(args.input, options=videosource_dict)

# [5. videoOutput]  
output = videoOutput(args.output, options=videooutput_dict)

# [6. Capture frames]
# For OpenCV-based implementation, refer to https://www.youtube.com/watch?v=mB025B7KpeE
# capture frames until end-of-stream (or the user exits)
while True:
    # capture the next image
    imgCuda = input.Capture()

    if imgCuda is None: # timeout
        continue  
    
    # [7. Inference with network]
    # detect objects in the image (with overlay)
    detections = net.Detect(imgCuda, overlay=args.overlay)

    # print the detections
    print("detected {:d} objects in image".format(len(detections)))

    # [8. Snapshot detected parts]
    # https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-image.md#image-allocation
    
    if args.save_snapshot:
        timestamp = datetime.now().strftime(args.timestamp)
        for idx, detection in enumerate(detections):
            print(detection)
            roi = (int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom))
            
            # To allocate empty GPU memory for storing intermediate/output images (i.e. working memory during processing).
            # Memory allocated by cudaAllocMapped() resides in a shared CPU/GPU memory space.
            snapshot = cudaAllocMapped(width=roi[2]-roi[0], 
                                        height=roi[3]-roi[1], 
                                        format=imgCuda.format)
            
            cudaCrop(imgCuda, snapshot, roi)
            
            # wait for the GPU to finish processing(cudaCrop)
            cudaDeviceSynchronize()

            saveImage(os.path.join(args.snapshot_dir, f"{timestamp}-{idx}.jpg"), 
                                    snapshot,
                                    100 # Quality (1~100)
                                    )
            del snapshot
    else:
        for detection in detections:
            print(detection)

    output.Render(imgCuda)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))
    # print out performance info
    net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break