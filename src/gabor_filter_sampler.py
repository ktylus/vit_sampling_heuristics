import torch
import cv2
import numpy as np

from elastic_vit_example.patch_sampler import PatchSampler

class Gabor(PatchSampler):
    def __init__(self, *args, patches_num=64, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_patches = patches_num

    def __call__(self, img):
        return split_image_by_gabor(img, self.num_patches)


def split_image_by_gabor(image_tensor, num_patches):
    # Convert the PyTorch tensor to a NumPy array and convert from CHW to HWC format
    image_np = image_tensor.permute(1, 2, 0).mul(255).byte().cpu().numpy()

    # Convert the image to BGR format for OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Define Gabor filter parameters
    ksize = 31  # Size of Gabor kernel
    sigma = 5.0  # Standard deviation of the Gaussian envelope
    theta = np.pi / 4  # Orientation of the Gabor kernel
    lambd = 5  # Wavelength of the sinusoidal factor
    gamma = 0.3  # Spatial aspect ratio

    # Create Gabor filter
    gabor = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma)

    # Apply Gabor filter to the image
    filtered_image = cv2.filter2D(gray, cv2.CV_8UC3, gabor)

    # Threshold the filtered image to get binary regions
    _, thresh = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours from the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the number of patches to create
    num_contours = min(len(contours), num_patches)

    patches = []
    patch_coords = []
    for i in range(len(contours)):
        # Find bounding box of each contour
        x, y, w, h = cv2.boundingRect(contours[i])

        # Get Gabor response for the patch
        patch_gray = gray[y:y + h, x:x + w]
        patch_response = np.sum(cv2.filter2D(patch_gray, cv2.CV_64F, gabor))

        patches.append((x, y, w, h, patch_response))

    # Sort patches based on response value (descending order)
    patches.sort(key=lambda x: x[4], reverse=True)

    tensor_patches = []
    for i in range(num_contours):
        x, y, w, h, _ = patches[i]

        # Normalize coordinates to range (-1 to 1)
        height, width = gray.shape[:2]
        x1 = 2.0 * x / width - 1.0
        y1 = 2.0 * y / height - 1.0
        x2 = 2.0 * (x + w) / width - 1.0
        y2 = 2.0 * (y + h) / height - 1.0

        # Extract patches based on contour bounding box
        patch = image_bgr[y:y + h, x:x + w]

        # Resize the patch to 16x16
        patch = cv2.resize(patch, (16, 16))

        # Convert BGR to RGB
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

        # Convert to PyTorch tensor
        tensor_patch = torch.tensor(patch_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0

        tensor_patches.append(tensor_patch)
        patch_coords.append((x1, y1, x2, y2))  # Store top-left and bottom-right coordinates

    # Stack the tensor patches into a single tensor
    tensor_patches = torch.stack(tensor_patches)

    # Convert patch coordinates to PyTorch tensor
    patch_coords_tensor = torch.tensor(patch_coords, dtype=torch.float32)

    return tensor_patches, patch_coords_tensor