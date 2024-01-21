from abc import ABC, abstractmethod

import numpy as np
import torch
import cv2

from elastic_vit_example.patch_sampler import PatchSampler


class Gabor(PatchSampler):
    def __init__(self, *args, patches_num=64, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_patches = patches_num

    def __call__(self, img):
        return split_image_by_gabor2(img, self.num_patches)


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


def split_image_by_gabor2(image_tensor, num_patches):
    def calculate_gabor(patch):
        gabor_kernels = []

        # Define Gabor filter parameters
        thetas = [0, np.pi / 2, np.pi / 4, 3 * np.pi / 4]  # Vertical, horizontal, and two oblique orientations
        sigmas = [0, 4]
        lambdas = [2]
        gammas = [0.5]

        for theta in thetas:
            for sigma in sigmas:
                for lambd in lambdas:
                    for gamma in gammas:
                        kernel = cv2.getGaborKernel((3, 3), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
                        gabor_kernels.append(cv2.filter2D(patch, cv2.CV_8UC3, kernel))

        # Calculate Gabor value for each filter
        gabor_values = [np.mean(g) for g in gabor_kernels]

        # Normalize by patch size (dividing by sqrt of log of patch size)
        patch_size = patch.shape[0] * patch.shape[1]
        normalized_gabor_values = [g * np.log2(patch_size) for g in gabor_values]

        return np.mean(normalized_gabor_values)

    image = image_tensor.permute(1, 2, 0).mul(255).byte().cpu().numpy()
    patches = [image]
    patch_coords = [((-1, -1), (image.shape[1], image.shape[0]))]

    while len(patches) < num_patches:
        max_gabor_value = float('-inf')
        max_patch_index = None

        for i, patch in enumerate(patches):
            if patch.shape[0] < 4 or patch.shape[1] < 4:
                continue  # Skip patches smaller than 4x4

            gabor_value = calculate_gabor(patch)

            if gabor_value > max_gabor_value:
                max_gabor_value = gabor_value
                max_patch_index = i

        if max_patch_index is None:
            break  # No suitable patches left

        max_patch = patches.pop(max_patch_index)
        max_patch_coords = patch_coords.pop(max_patch_index)
        h, w = max_patch.shape[:2]

        split_size = min(h // 2, w // 2)
        sub_patches = []
        sub_patch_coords = []
        for i in range(0, h, split_size):
            for j in range(0, w, split_size):
                sub_patch = max_patch[i:i + split_size, j:j + split_size]
                sub_patches.append(sub_patch)

                top_left_x = max_patch_coords[0][0] + j
                top_left_y = max_patch_coords[0][1] + i
                bottom_right_x = top_left_x + split_size
                bottom_right_y = top_left_y + split_size

                sub_patch_coords.append((
                    (top_left_x, top_left_y),
                    (bottom_right_x, bottom_right_y)
                ))

        patches.extend(sub_patches)
        patch_coords.extend(sub_patch_coords)
    normalized_coords = []
    for ((top_left_x, top_left_y), (bottom_right_x, bottom_right_y)) in patch_coords:
        normalized_top_left_x = (2 * top_left_x / image.shape[1]) - 1
        normalized_top_left_y = (2 * top_left_y / image.shape[0]) - 1
        normalized_bottom_right_x = (2 * bottom_right_x / image.shape[1]) - 1
        normalized_bottom_right_y = (2 * bottom_right_y / image.shape[0]) - 1
        normalized_top_left_x = np.ceil(normalized_top_left_x * 100) / 100
        normalized_top_left_y = np.ceil(normalized_top_left_y * 100) / 100
        normalized_bottom_right_x = np.ceil(normalized_bottom_right_x * 100) / 100
        normalized_bottom_right_y = np.ceil(normalized_bottom_right_y * 100) / 100
        normalized_coords.append(
            (normalized_top_left_x, normalized_top_left_y, normalized_bottom_right_x, normalized_bottom_right_y))
    tensor_patches = []
    for patch in patches:
        resized_patch = cv2.resize(patch, (16, 16), interpolation=cv2.INTER_AREA)
        # Convert BGR to RGB
        patch_rgb = cv2.cvtColor(resized_patch, cv2.COLOR_BGR2RGB)

        # Convert to PyTorch tensor
        tensor_patch = torch.tensor(patch_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0

        tensor_patches.append(tensor_patch)
    # Stack the tensor patches into a single tensor
    tensor_patches = torch.stack(tensor_patches)

    # Convert patch coordinates to PyTorch tensor
    patch_coords_tensor = torch.tensor(normalized_coords, dtype=torch.float32)
    return tensor_patches, patch_coords_tensor


def split_image_by_gabor3(image_tensor, num_patches):
    # Convert the PyTorch tensor to a NumPy array and convert from CHW to HWC format
    image_np = image_tensor.permute(1, 2, 0).mul(255).byte().cpu().numpy()

    # Convert the image to BGR format for OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Define Gabor filter parameters
    ksize = 5  # Size of Gabor kernel
    sigma = 1.0  # Standard deviation of the Gaussian envelope
    theta = np.pi / 4  # Orientation of the Gabor kernel
    lambd = 2  # Wavelength of the sinusoidal factor
    gamma = 0.5  # Spatial aspect ratio

    ksize_values = [3, 7]
    sigma_values = [0.0, 4.0]
    theta_values = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, 2 * np.pi / 3, -2 * np.pi / 3]
    all_patches = []
    all_contours = []
    all_boundings = []

    # Create patches for different theta values
    all_patches = []

    for ksize in ksize_values:
        for sigma in sigma_values:
            for theta in theta_values:
                # Create Gabor filter
                gabor = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma)

                # Apply Gabor filter to the image
                filtered_image = cv2.filter2D(gray, cv2.CV_8UC3, gabor)

                # Threshold the filtered image to get binary regions
                _, thresh = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_OTSU)

                # Find contours from the thresholded image
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Calculate the number of patches to create
                num_contours = min(len(contours), num_patches)

                # patches = []
                # patch_coords = []
                patch_responses = []
                for i in range(len(contours)):
                    # Find bounding box of each contour
                    x, y, w, h = cv2.boundingRect(contours[i])
                    if w < 3 or h < 3:
                        continue

                    # Get Gabor response for the patch
                    patch_gray = gray[y:y + h, x:x + w]
                    patch_response = np.abs(np.sum(cv2.filter2D(patch_gray, cv2.CV_64F, gabor)))
                    if ((x, y, w, h)) in all_boundings:
                        idx = all_boundings.index((x, y, w, h))
                        if (all_patches[idx][0] < patch_response):
                            all_patches[idx] = (patch_response, all_patches[idx][1])

                        continue

                    all_boundings.append((x, y, w, h))

                    patch_responses.append((patch_response, len(all_contours) + i))
                all_patches.extend(patch_responses)
                all_contours.extend(contours)

        patch_responses = all_patches
        patch_responses.sort(reverse=True)

        contours = all_contours

        patch_coords = []

        tensor_patches = []
        for i in range(num_contours):
            if i > num_patches:
                break
            if patch_responses[i][0] < 250:
                break
            idx = patch_responses[i][1]

            # Find bounding box of the selected contour
            x, y, w, h = cv2.boundingRect(contours[idx])

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
