import torch
from torchvision import datasets, transforms
from torchvision.transforms.functional import resized_crop
from skimage import feature


img_size = 224
patch_size = 32


def edge_detector_point_sampling(image, n_patches, sigma):
    gray_image = transforms.Grayscale().forward(image)
    edge_detector_result = torch.from_numpy(feature.canny(gray_image.squeeze().numpy(), sigma=sigma))
    indices = torch.stack(torch.where(edge_detector_result), dim=1)
    choice = torch.multinomial(torch.ones(len(indices)), n_patches)
    sampled_points_indices = torch.index_select(indices, dim=0, index=choice)
    patches, coords = generate_patches(image, sampled_points_indices)
    return patches, coords


def generate_patches(image, indices):
    upper_left = torch.maximum(indices - patch_size / 2, torch.zeros_like(indices))
    lower_right = torch.minimum(indices + patch_size / 2, torch.full_like(indices, img_size))
    coords = torch.cat((upper_left, lower_right), dim=-1)
    patches_list = []
    for x1, y1, x2, y2 in coords:
        patches_list.append(
            resized_crop(image, x1.int(), y1.int(), (x2 - x1).int(), (y2 - y1).int(), [patch_size, patch_size]))
    patches = torch.stack(patches_list, dim=0)
    return patches, coords

