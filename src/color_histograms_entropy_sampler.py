import torch
from torchvision import datasets, transforms

from elastic_vit_example.patch_sampler import GridSamplerV2, PatchSampler
from elastic_vit_example.custom_dataset import CustomDataset
from src.utils import batch_histogram


def sample_patches(image, n_patches_in_stages: list, patch_size: tuple):
    grid_sampler = GridSamplerV2(patches_num_yx=(image.shape[-2] // patch_size[0], image.shape[-1] // patch_size[1]))
    patches, coords = grid_sampler(image)
    for k in n_patches_in_stages:
        top_entropy_patches_indices = find_top_k_entropy_patches(patches, k)
        patches, coords = divide_patches_in_four(patches, coords, top_entropy_patches_indices)
    return patches, coords


def find_top_k_entropy_patches(patches, k):
    patch_entropy = calculate_patches_entropy(patches)
    sorted_entropy, patch_indices = torch.sort(patch_entropy, descending=True, dim=-1)
    return patch_indices[:k]


def calculate_patches_entropy(patches):
    histograms = batch_histogram(patches.flatten(-2) * 255, 16)
    histograms = torch.where(histograms == 0, 1, histograms)
    histograms = torch.nn.functional.normalize(histograms, p=1, dim=-1)
    patches_entropy = -torch.sum((histograms * torch.log2(histograms)), dim=-1)
    patches_entropy = torch.sum(patches_entropy, dim=-1)
    return patches_entropy


def divide_patch_coords_in_four(coords):
    patch_sizes = torch.cat((
        (coords[:, 2] - coords[:, 0]).unsqueeze(1),
        (coords[:, 3] - coords[:, 1]).unsqueeze(1)
    ), dim=-1)

    upper_left = torch.cat((
        coords[:, :2],
        coords[:, 2:] - patch_sizes / 2
    ), dim=-1)
    upper_right = torch.cat((
        coords[:, 0].unsqueeze(1),
        (coords[:, 1] + patch_sizes[:, 1] / 2).unsqueeze(1),
        (coords[:, 2] - patch_sizes[:, 0] / 2).unsqueeze(1),
        coords[:, 3].unsqueeze(1)
    ), dim=-1)
    lower_left = torch.cat((
        (coords[:, 0] + patch_sizes[:, 0] / 2).unsqueeze(1),
        coords[:, 1].unsqueeze(1),
        coords[:, 2].unsqueeze(1),
        (coords[:, 3] - patch_sizes[:, 1] / 2).unsqueeze(1)
    ), dim=-1)
    lower_right = torch.cat((
        coords[:, :2] + patch_sizes / 2,
        coords[:, 2:]
    ), dim=-1)

    coords_after_division = torch.cat((
        upper_left, upper_right,
        lower_left, lower_right
    ), dim=0)
    return coords_after_division


def divide_patches_pixels_in_four(patches):
    patch_size = patches.shape[-1]
    upper_left = patches[:, :, :(patch_size // 2), :(patch_size // 2)]
    upper_right = patches[:, :, :(patch_size // 2), (patch_size // 2):]
    lower_left = patches[:, :, (patch_size // 2):, :(patch_size // 2)]
    lower_right = patches[:, :, (patch_size // 2):, (patch_size // 2):]

    resize = transforms.Resize((patch_size, patch_size), antialias=True)
    upper_left = resize.forward(upper_left)
    upper_right = resize.forward(upper_right)
    lower_left = resize.forward(lower_left)
    lower_right = resize.forward(lower_right)

    patches_after_division = torch.cat((
        upper_left, upper_right,
        lower_left, lower_right
    ), dim=0)
    return patches_after_division


# patch_indices is [batch_size, n_patches] - indices of patches to divide
def divide_patches_in_four(patches, coords, patch_indices):
    coords_to_divide = torch.index_select(coords, dim=0, index=patch_indices)
    patches_to_divide = torch.index_select(patches, dim=0, index=patch_indices)
    divided_coords = divide_patch_coords_in_four(coords_to_divide)
    divided_patches = divide_patches_pixels_in_four(patches_to_divide)

    undivided_patches_filter = torch.ones(patches.shape[0], dtype=torch.bool)
    undivided_patches_filter[patch_indices] = False
    undivided_coords = coords[undivided_patches_filter]
    undivided_patches = patches[undivided_patches_filter]

    all_coords = torch.cat((undivided_coords, divided_coords), dim=0)
    all_patches = torch.cat((undivided_patches, divided_patches), dim=0)
    return all_patches, all_coords


class ColorHistogramsEntropySampler(PatchSampler):
    def __init__(self, n_patches_in_stages=[10, 7], *args,  **kwargs):
        super().__init__(*args, **kwargs)
        self.n_patches_in_stages = n_patches_in_stages

    def __call__(self, img):
        return sample_patches(img, self.n_patches_in_stages, self.patch_size)
