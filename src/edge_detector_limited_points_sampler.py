import torch
from torchvision import datasets, transforms
from torchvision.transforms.functional import resized_crop
from skimage import feature

from elastic_vit_example.patch_sampler import PatchSampler


def edge_detector_point_sampling(image, n_patches, patch_size, sigma):
    gray_image = transforms.Grayscale().forward(image)
    edge_detector_result = torch.from_numpy(feature.canny(gray_image.squeeze().numpy(), sigma=sigma))
    indices = torch.stack(torch.where(edge_detector_result), dim=1)
    choice = torch.multinomial(torch.ones(len(indices)), n_patches)
    sampled_points_indices = torch.index_select(indices, dim=0, index=choice)
    patches, coords = generate_patches(image, sampled_points_indices, patch_size)
    return patches, coords


def generate_patches(image, indices, patch_size):
    upper_left = torch.maximum(indices - patch_size / 2, torch.zeros_like(indices))
    lower_right = torch.minimum(indices + patch_size / 2, torch.full_like(indices, image.shape[-1]))
    coords = torch.cat((upper_left, lower_right), dim=-1)
    patches_list = []
    for x1, y1, x2, y2 in coords:
        patches_list.append(
            resized_crop(image, x1.int(), y1.int(), (x2 - x1).int(), (y2 - y1).int(), [patch_size, patch_size]))
    patches = torch.stack(patches_list, dim=0)
    return patches, coords


class EdgeDetectorLimitedPointsSampler(PatchSampler):
    def __init__(self, n_patches=100, canny_detector_sigma=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_patches = n_patches
        self.canny_detector_sigma = canny_detector_sigma

    def __call__(self, img):
        return edge_detector_point_sampling(img, self.n_patches, self.patch_size, self.canny_detector_sigma)
