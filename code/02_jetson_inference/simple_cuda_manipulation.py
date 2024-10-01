# Most of code is borrowed from jetson_inference repo.
# https://github.com/dusty-nv/jetson-utils/blob/master/python/examples/cuda-examples.py
import numpy as np
from jetson_utils import loadImage, cudaAllocMapped, cudaConvertColor, saveImage

# load the input image (default format is rgb8)
imgInput = loadImage('image/example/Example.jpg', format='rgb8') # default format is 'rgb8', but can also be 'rgba8', 'rgb32f', 'rgba32f'

# copy the image
# allocate the output as rgba32f, with the same width/height as the input
imgOutput = cudaAllocMapped(width=imgInput.width, height=imgInput.height, format='rgba32f')

# convert from rgb8 to rgba32f (the formats used for the conversion are taken from the image capsules)
cudaConvertColor(imgInput, imgOutput)

saveImage('image/example/Example_converted.jpg', 
            imgOutput,
            100, # Quality (1~100)
            )