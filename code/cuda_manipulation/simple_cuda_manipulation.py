# Most of code is borrowed from jetson_inference repo.
# https://github.com/dusty-nv/jetson-utils/blob/master/python/examples/cuda-examples.py
import torch
import os
import torchvision.transforms as transforms
import torchvision.utils as vutils

from jetson_utils import loadImage, cudaAllocMapped, cudaConvertColor, saveImage, cudaCrop

script_path = os.path.abspath(__file__)
directory_path = os.path.dirname(script_path)

# load the input image (default format is rgb8)
imgInput = loadImage(directory_path + '/image/example/Example.jpg', format='rgb8') # 'rgb8', 'rgba8', 'rgb32f', 'rgba32f'

print('input image:')
print(imgInput)

# Subscripting is available but not useful; use numpy or pytorch
# for y in range(imgInput.shape[0]):
#     for x in range(imgInput.shape[1]):
#         print(f"imgInput[{y}, {x}] = {imgInput[y,x]}")

# center crop an image
crop_factor = (0.5, 0.5)

crop_border = (
                round((1.0 - crop_factor[0]) * 0.5 * imgInput.width),
				round((1.0 - crop_factor[1]) * 0.5 * imgInput.height)
                )

crop_roi = (
            crop_border[0], 
            crop_border[1], 
            round(imgInput.width - crop_border[0]), 
            round(imgInput.height - crop_border[1])
            )

imgCrop = cudaAllocMapped(width=round(imgInput.width * crop_factor[0]),
                            height=round(imgInput.height * crop_factor[1]),
                            format=imgInput.format)

cudaCrop(imgInput, imgCrop, crop_roi)

saveImage(directory_path + '/image/example/Example-cropped.jpg', 
            imgCrop,
            100, # Quality (1~100)
            )

# map to torch tensor using __cuda_array_interface__
# convert to float & permute & normalize
imgTensor = torch.as_tensor(imgCrop, device='cuda').float()
# (H, W, C) to (C, H, W)
imgTensor = imgTensor.permute(2, 0, 1)
if imgTensor.max() > 1.0:
    imgTensor = imgTensor / 255.0

print("\nPyTorch tensor:\n")
print(type(imgTensor))
print(f"    -- ptr:   {hex(imgTensor.data_ptr())}")
print(f"    -- type:  {imgTensor.dtype}")
print(f"    -- shape: {imgTensor.shape}")
print(f"    -- device: {imgTensor.device}\n")
# print(imgTensor)

device = imgTensor.device

# Function to add diffusion-style noise to an image tensor
def diffusion_style_noise(img_tensor, t, noise_schedule="linear", device='cpu'):
    """
    Adds progressive Gaussian noise to the image in the style of diffusion models.
    
    Parameters:
        img_tensor: The input image tensor (C, H, W), normalized between 0 and 1.
        t: Number of timesteps for noise addition.
        noise_schedule: Schedule for the noise variance ("linear", "cosine").
        device: The device on which to perform the calculations ('cpu' or 'cuda').
    """
    
    # Ensure the input tensor is on the correct device
    img_tensor = img_tensor.to(device)

    # Define a linear noise schedule (betas) over t steps
    if noise_schedule == "linear":
        betas = torch.linspace(0.0001, 0.02, t, device=device)  # Linear increase
    elif noise_schedule == "cosine":
        betas = (1 - torch.cos(torch.linspace(0, np.pi / 2, t, device=device))) * 0.02
    
    # Start with the original image
    noisy_img = img_tensor.clone()
    
    # Progressively add noise at each step
    for step in range(t):
        # Calculate the variance (beta) for this step
        beta = betas[step]
        # Generate Gaussian noise on the same device
        noise = torch.randn_like(noisy_img, device=device) * beta.sqrt()
        # Add the noise to the image
        noisy_img = (1 - beta).sqrt() * noisy_img + noise
    
    # Clip the noisy image to keep values between 0 and 1
    noisy_img = torch.clamp(noisy_img, 0.0, 1.0)
    
    return noisy_img

# modify PyTorch tensor
print("\nAdding gaussian noise to the tensor...\n")

# Apply diffusion-style noise
for t in range(0, 4):
    imgTensor_noised = diffusion_style_noise(imgTensor, t, noise_schedule="linear", device=device)
    vutils.save_image(imgTensor_noised, directory_path + '/image/example/Example-cropped-noised-' + str(t) + '.jpg')
