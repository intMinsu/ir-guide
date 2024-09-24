import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

camera = jetson.utils.gstCamera(1280, 720, "/dev/video0")	# USB camera ONLY
# camera = jetson.utils.gstCamera(1280, 720, "/dev/video1")	# CSI camera + USB camera
#camera = jetson.utils.videoSource("csi://0")			# CSI camera ONLY
display = jetson.utils.glDisplay()				# USB camera ONLY
#display = jetson.utils.videoOutput("display://0")		# CSI camera ONLY

### USB camera ###
while display.IsOpen():
	img, width, height = camera.CaptureRGBA()
	detections = net.Detect(img, width, height)
	display.RenderOnce(img, width, height)
	display.SetTitle("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

### CSI camera ###
#while display.IsStreaming():
#	img = camera.Capture()
#	detections = net.Detect(img)
#	display.Render(img)
#	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))