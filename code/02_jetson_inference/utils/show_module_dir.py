import jetson_inference
import jetson_utils

def show_dir(module):
    print(f"List of functions, classes, and variables defined in {module}")
    print(dir(module))
    print("")

show_dir(jetson_inference)
# ['VERSION', 'actionNet', 'backgroundNet', 'depthNet', 
# 'detectNet', 'imageNet', 'jetson_utils', 'poseNet', 
# 'segNet', 'tensorNet']

show_dir(jetson_utils)
# ['Log', 'VERSION', 'adaptFontSize', 'cudaAllocMapped', 
# 'cudaConvertColor', 'cudaCrop', 'cudaDeviceSynchronize', 'cudaDrawCircle', 
# 'cudaDrawLine', 'cudaDrawRect', 'cudaEventCreate', 'cudaEventDestroy', 
# 'cudaEventElapsedTime', 'cudaEventQuery', 'cudaEventRecord', 'cudaEventSynchronize', 
# 'cudaFont', 'cudaFromNumpy', 'cudaImage', 'cudaMalloc', 
# 'cudaMallocMapped', 'cudaMemcpy', 'cudaMemory', 'cudaNormalize', 
# 'cudaOverlay', 'cudaResize', 'cudaStreamCreate', 'cudaStreamDestroy', 
# 'cudaStreamSynchronize', 'cudaStreamWaitEvent', 'cudaToNumpy', 'glDisplay', 
# 'gstCamera', 'loadImage', 'loadImageRGBA', 'logUsage', 
# 'saveImage', 'saveImageRGBA', 'videoOutput', 'videoSource']

show_dir(jetson_utils.videoSource)
# ['Capture', 'Close', 'GetFrameCount', 'GetFrameRate', 
# 'GetHeight', 'GetOptions', 'GetWidth', 'IsStreaming', 
# 'Open', 'Usage']

show_dir(jetson_utils.videoOutput)
# ['Close', 'GetFrameRate', 'GetHeight', 'GetOptions', 
# 'GetWidth', 'IsStreaming', 'Open', 'Render', 
# 'SetStatus', 'Usage']

show_dir(jetson_inference.detectNet)
# ['Detect', 'Detection', 'EnableDebug', 'EnableLayerProfiler', 
# 'GetClassDesc', 'GetClassLabel', 'GetClassSynset', 
# 'GetClusteringThreshold', 'GetConfidenceThreshold', 'GetLineWidth', 
# 'GetModelPath', 'GetModelType', 'GetNetworkFPS', 'GetNetworkTime', 
# 'GetNumClasses', 'GetOverlayAlpha', 'GetPrototxtPath', 'GetThreshold', 
# 'GetTrackerType', 'GetTrackingParams', 'IsTrackingEnabled', 'Overlay', 
# 'PrintProfilerTimes', 'SetClusteringThreshold', 'SetConfidenceThreshold', 'SetLineWidth', 
# 'SetOverlayAlpha', 'SetThreshold', 'SetTrackerType', 'SetTrackingEnabled', 
# 'SetTrackingParams', 'Usage']