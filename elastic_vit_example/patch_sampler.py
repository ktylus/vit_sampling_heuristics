from abc import ABC, abstractmethod
from heapq import heapify, heappop, heappush
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from math import sqrt
from skimage import feature
import cv2
# from debug import DEBUG

class PatchModifier:
    def __init__(self, modify_patches, patch_size, patch_zoom=0, patch_shake=0, patch_dropout=0, patch_dropout_blocks=1, patch_select=0, patch_select_mode="drop"):
        assert not ((patch_dropout != 0) and (patch_select != 0)), (patch_dropout, patch_select)
        assert patch_shake >= 0 and patch_shake <= min(patch_size) // 2
        assert patch_select_mode in ["drop", "resize"]
        self.modify_patches, self.patch_size, self.patch_zoom, self.patch_shake, self.patch_dropout, self.patch_dropout_blocks, self.patch_select, self.patch_select_mode = modify_patches, patch_size, patch_zoom, patch_shake, patch_dropout, patch_dropout_blocks, patch_select, patch_select_mode

    def _patch_shake(self, y_x_s, img_size):
        N = y_x_s.shape[0]
        directions_x = np.random.randint(0, 2, N)
        directions_y = 1 - directions_x
        flip = np.random.randint(0, 2, N)
        directions_x[flip == 1] *= -1
        directions_y[flip == 1] *= -1

        shake_x, shake_y = self.patch_shake, self.patch_shake

        shake_x *= directions_x
        shake_y *= directions_y
        y_x_s[:, 0] += shake_y
        y_x_s[:, 1] += shake_x
        y_x_s[:, 0] = y_x_s[:, 0].clip(0, img_size[0]-y_x_s[:, 2])
        y_x_s[:, 1] = y_x_s[:, 1].clip(0, img_size[1]-y_x_s[:, 2])
        return y_x_s

    def _patch_dropout(self, n):
        n_dropouts = self.patch_dropout
        indices = np.arange(0, n // self.patch_dropout_blocks)
        np.random.shuffle(indices)
        dropout_indices = indices[:n_dropouts]
        
        mask = np.ones(indices.shape[0], dtype="bool")
        mask[dropout_indices] = 0
        if self.patch_dropout_blocks != 1:
            k = int(sqrt(mask.shape[0]))
            l = int(sqrt(self.patch_dropout_blocks))
            mask = mask.reshape(k, k, *mask.shape[1:])
            mask = mask.repeat(l, axis=0).repeat(l, axis=1)
            mask = mask.reshape(n, *mask.shape[2:])
        
        return mask
    
    def _patch_zoom(self, y_x_s, img_size):
        y_x_s[:, 2] -= self.patch_zoom
        space_left = np.stack([img_size[0]-y_x_s[:, 0], img_size[1]-y_x_s[:, 1]])
        y_x_s[:, 2] = y_x_s[:, 2].clip(0, space_left.min(axis=0))
        return y_x_s

    def _patch_select(self, y_x_s):
        n = len(y_x_s)
        sq_n = int(sqrt(n))
        assert sq_n**2 == n
        
        block_size = 4
        assert n%block_size == 0
        
        block_len = 2
        assert sq_n%block_len==0
        
        n_select = self.patch_select
        n_blocks_row = sq_n // block_len
        n_blocks_col = sq_n // block_len
        n_blocks = n_blocks_row * n_blocks_col

        selected_blocks = np.arange(0,n_blocks)
        np.random.shuffle(selected_blocks)
        selected_blocks = selected_blocks[:n_select]

        if self.patch_select_mode == "drop":
            ids = np.random.randint(0, block_size, size=n_select)
        elif self.patch_select_mode == "resize":
            ids = np.zeros(shape=n_select, dtype="int")
        else:
            assert False

        base_mask = np.ones(shape=(n_blocks, block_size), dtype="bool")
        base_mask[selected_blocks, :] = 0

        selected_mask = np.zeros(shape=(n_blocks, block_size), dtype="bool")
        selected_mask[selected_blocks, :] = 0
        selected_mask[selected_blocks, ids] = 1
        
        def reshape_mask(mask):
            mask = mask.reshape(n_blocks, block_len, block_len) 
            mask = list(mask)
            mask = np.concatenate(
                [
                    np.concatenate(mask[r*n_blocks_row:(r+1)*n_blocks_row])
                    for r in range(0, n_blocks_col)
                ], axis=1)
            mask = mask.flatten()
            return mask

        base_mask = reshape_mask(base_mask)
        selected_mask = reshape_mask(selected_mask)

        if self.patch_select_mode == "resize":
            y_x_s[selected_mask, 2] *= block_len

        return y_x_s, base_mask + selected_mask

    def __call__(self, img_size, y_x_s):
        y_x_s = np.array(y_x_s)
        y_x_s = self._patch_zoom(y_x_s, img_size)
        y_x_s = self._patch_shake(y_x_s, img_size)
        assert not((self.patch_dropout != 0) and (self.patch_select != 0))
        if self.patch_select != 0:
            y_x_s, mask = self._patch_select(y_x_s)
        elif self.patch_dropout != 0:
            mask = self._patch_dropout(len(y_x_s))
        else:
            mask = None
        return y_x_s, mask
        
class PatchSampler(ABC):
    def __init__(self, patch_size=(16, 16), seed=1000000009):
        self.patch_size = patch_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset_state(self):
        self.rng = np.random.default_rng(self.seed)

    @abstractmethod
    def __call__(self, img):
        raise NotImplemented()

class SimplePatchSampler(PatchSampler):
    def _get_coordinates(self, img_size, img):
        raise NotImplemented()

    def _build_patches_coords(self, y_x_s, img):
        patches = []
        coords = []
        for y, x, s in y_x_s:
            patch = img[:, y:y + s, x:x + s]
            if patch.shape[-1] != 16 or patch.shape[-2] != 16:
                patch = TF.resize(patch, [16, 16], antialias=False)
            patches.append(patch)
            coords.append([y, x, y + s, x + s])

        return torch.stack(patches, dim=0), torch.LongTensor(coords)

    def __call__(self, img):
        assert len(img.shape) == 3 and img.shape[0] == 3
        img_size = img.shape[1:]
        y_x_s = self._get_coordinates(img_size, img)
        patches, coords = self._build_patches_coords(y_x_s, img)
        return patches, coords, None

class GridSampler(SimplePatchSampler):
    def __init__(self, seed=1000000009, patch_size=(16, 16), patch_modifications=None, merge_patches=False, hard_dropout=True):
        assert patch_size[0] == patch_size[1]
        super().__init__(seed=seed, patch_size=patch_size)
        self.merge_patches = merge_patches
        self.hard_dropout = hard_dropout
        self.patch_size = patch_size
        if patch_modifications is None:
            self.patch_modifier = PatchModifier(modify_patches=False)
        else:
            self.patch_modifier = PatchModifier(modify_patches=True, patch_size=patch_size, **patch_modifications)
        
    def _get_coordinates(self, img_size, img):
        return [(i, j, self.patch_size[0]) for i in range(0, img_size[0], self.patch_size[0]) for j in
                range(0, img_size[1], self.patch_size[0])]

    def __call__(self, img):
        assert len(img.shape) == 3 and img.shape[0] == 3
        img_size = img.shape[1:]

        y_x_s = self._get_coordinates(img_size, img)
        y_x_s, mask = self.patch_modifier(img_size, y_x_s)
        patches, coords = self._build_patches_coords(y_x_s, img)
        mask = torch.Tensor(mask) if mask is not None else None
        #keeps = self.patch_modifier.get_mask(patches.shape[0])
        #keeps = torch.tensor(keeps) if keeps is not None else None

        if not self.hard_dropout and mask is not None:
            patches = mask.reshape(-1, *([1]*(len(patches.shape)-1))) * patches

        if not self.merge_patches:
            if self.hard_dropout and mask is not None:
                mask = np.arange(mask.shape[0])[mask == 1]
                return patches[mask], coords[mask], None
            else:
                return patches, coords, None

        patches = patches.reshape(14, 14, *patches.shape[1:])
        patches = patches.permute(2,0,3,1,4)
        patches = patches.flatten(3,4)
        patches = patches.flatten(1,2)

        # if DEBUG:
        #     import matplotlib.pyplot as plt
        #     plt.imsave('debug.png', patches.detach().permute(1,2,0).numpy())
        #     assert False
        
        return patches, coords, torch.arange(mask.shape[0])[mask == 1] if self.hard_dropout and mask is not None else None


class CentralMultiscaleSampler(SimplePatchSampler):
    def __init__(self, seed=1000000009, patch_sizes=[32,32,32,32,32,32,32]):
        super().__init__(seed=seed, patch_size=None)
        self.patch_sizes = patch_sizes
        
    def _get_coordinates(self, img_size, img):
        def _get_frame(filled_space, ps):
            upper = [(filled_space, i, ps) for i in range(filled_space, img_size[1]-filled_space, ps)]
            if filled_space != img_size[0]-filled_space-ps:
                lower = [(img_size[0]-filled_space-ps, i, ps) for i in range(filled_space, img_size[1]-filled_space, ps)]
            else:
                lower=[]
            left = [(i, filled_space, ps) for i in range(filled_space+ps, img_size[0]-filled_space-ps, ps)]
            if filled_space != img_size[1]-filled_space-ps:
                right = [(i, img_size[1]-filled_space-ps, ps) for i in range(filled_space+ps, img_size[0]-filled_space-ps, ps)]
            else:
                right = []
            return upper + lower + left + right
        ret = []
        filled_space = 0
        for ps in self.patch_sizes:
            ret += _get_frame(filled_space, ps)
            filled_space += ps
        return ret

class RandomMultiscaleSampler(SimplePatchSampler):
    def __init__(self, crop_sizes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.crop_sizes = np.asarray(crop_sizes)

    def _get_coordinates(self, img_size, img):
        sizes = self.crop_sizes
        ys = self.rng.integers(0, img_size[0] - sizes)
        xs = self.rng.integers(0, img_size[1] - sizes)
        return zip(ys, xs, sizes)


class RandomUniformSampler(SimplePatchSampler):
    def __init__(self, random_patches, min_patch_size=8, max_patch_size=48, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_patches = random_patches
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size

    def _get_coordinates(self, img_size, img):
        sizes = (self.rng.integers(self.min_patch_size,
                                   self.max_patch_size,
                                   size=self.random_patches)).astype(np.int64)
        ys = self.rng.integers(0, img_size[0] - sizes)
        xs = self.rng.integers(0, img_size[1] - sizes)
        return zip(ys, xs, sizes)


class SmartSampler(SimplePatchSampler):
    def __init__(self, smart_patches, min_patch_size=8, max_patch_size=48, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.smart_patches = smart_patches
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size

    def _get_coordinates(self, img_size, img):
        img = TF.rgb_to_grayscale(img).squeeze(0).numpy()
        edges = feature.canny(img, sigma=12)

        def density(i, j, p):
            return -edges[i:i + p, j:j + p].sum()

        crops = [(density(i, j, self.max_patch_size), i, j, self.max_patch_size) for i in
                 range(0, img_size[0], self.max_patch_size)
                 for j in
                 range(0, img_size[1], self.max_patch_size)]
        out_crops = []
        heapify(crops)

        while len(out_crops) + len(crops) < self.smart_patches:
            _, y, x, p = heappop(crops)
            q = p // 2
            assert q * 2 == p
            out = [
                (density(i, j, q), i, j, q)
                for i, j in [(y, x), (y + q, x), (y, x + q), (y + q, x + q)]
                if i + q < img_size[0] - 7 and j + q < img_size[1] - 7
            ]
            if q <= self.min_patch_size:
                out_crops += out
            else:
                for o in out:
                    heappush(crops, o)

        out_crops += crops

        out_crops = out_crops[:self.smart_patches]

        return [(y, x, p) for _, y, x, p in out_crops]


class RandomDelegatedSampler(PatchSampler):

    def __init__(self, samplers: List[Tuple[PatchSampler, float]], patch_size=(16, 16), seed=1000000009):
        super().__init__(patch_size, seed)

        self.samplers, self.proba = zip(*samplers)
        for sampler in self.samplers:
            sampler.rng = self.rng
        self.proba = np.asarray(self.proba)
        self.proba = self.proba / self.proba.sum()

    def __call__(self, img):
        sampler = self.rng.choice(self.samplers, p=self.proba)
        return sampler(img)


def multi_crop(image, crop_grids, interpolation_mode='bilinear', align_corners=True):
    """Crop patches defined in grid from an image.

    :param image: input image tensor, but will accept any type of multi-channel tensor of shape [c, h, w]
    :param crop_grids: list of grids, best produced by GridCropListGenerator or RandomCropListGenerator
    :param interpolation_mode: modes for torch.nn.functional.grid_sample. Can be nearest, bilinear and bicubic. Bicubic may produce artifacts, out of range values.
    :param align_corners: (tricky) value from torch.nn.functional.grid_sample intreface - see its documentation
    :return: patches as a tensor of shape [b, p, c, h, w]
    """

    if len(image.shape) == 3:
        image = image.unsqueeze(0)  # grid sampler needs batch dimension
    assert len(image.shape) == 4 and image.dtype == torch.float32
    # copy to device in separate loop because it will accumulate and optimize in cuda upload stream
    new_grid_list = []
    for crop_grid in crop_grids:
        new_grid_list.append(crop_grid.to(image.device))
    crop_grids = new_grid_list

    patch_list = []
    list_of_coords = []
    for crop_grid in crop_grids:
        x = (crop_grid[0, 0, 0] + 1.0) / 2.0 * image.shape[3]
        y = (crop_grid[0, 0, 1] + 1.0) / 2.0 * image.shape[2]
        x_max = (crop_grid[crop_grid.shape[0] - 1, crop_grid.shape[1] - 1, 0] + 1.0) / 2.0 * image.shape[3]
        y_max = (crop_grid[crop_grid.shape[0] - 1, crop_grid.shape[1] - 1, 1] + 1.0) / 2.0 * image.shape[2]
        list_of_coords.append([y, x, y_max, x_max])
        patch_list.append(torch.nn.functional.grid_sample(image, crop_grid.unsqueeze(dim=0),
                                                          mode=interpolation_mode, padding_mode='zeros',
                                                          align_corners=align_corners).squeeze(0))

    return torch.stack(patch_list), torch.Tensor(list_of_coords)


def get_prototype_grid(patch_size=(32, 32)):
    """generate protoype grid for sampling with defined resolution.

    Generates simple grid of (x,y) coordinates for torch.nn.functional.grid_sample
    function in range (-1:1) for xy (grid overlapping whole image) using
    resolution set in self.patch_size

    :param patch_size: resolution of a patch
    :return: prototype grid tensor
    """

    y_coords = torch.linspace(-1.0, 1.0, patch_size[0]).unsqueeze(1)
    y_coords_2D = y_coords.repeat(1, patch_size[1])
    x_coords = torch.linspace(-1.0, 1.0, patch_size[1]).unsqueeze(0)
    x_coords_2D = x_coords.repeat(patch_size[0], 1)
    return torch.stack([x_coords_2D, y_coords_2D], dim=2)


class GridCropListGenerator:
    def __init__(self, patches_num_x=8, patches_num_y=16, patch_size=(16, 16)):

        self.patch_size = patch_size
        self.patches_num_x = patches_num_x
        self.patches_num_y = patches_num_y
        # Generate prototype grid
        self.prototypeGrid = get_prototype_grid(self.patch_size)

        self.x_stride = 2.0 / patches_num_x
        self.y_stride = 2.0 / patches_num_y
        # Rescale and move prototype grid to left top corner + align corners
        # (ad. aligning corners => minimum x and y values should be "-1.0" (grid salmper samples from -1 to 1)
        self.rescaled_prototype_grid_y = self.prototypeGrid[:, :, 1] * self.y_stride / 2.0 - 1.0 + self.y_stride / 2.0
        self.rescaled_prototype_grid_x = self.prototypeGrid[:, :, 0] * self.x_stride / 2.0 - 1.0 + self.x_stride / 2.0

    def __call__(self):
        list_of_grids_for_sampler = []
        # Generate set of patches in checkerboard pattern by shifting x and y coord channels with stride
        for y in range(self.patches_num_y):
            for x in range(self.patches_num_x):
                tmp_rescaled_prototype_grid_y = self.rescaled_prototype_grid_y + y * self.y_stride
                tmp_rescaled_prototype_grid_x = self.rescaled_prototype_grid_x + x * self.x_stride
                list_of_grids_for_sampler.append(
                    torch.stack([tmp_rescaled_prototype_grid_x, tmp_rescaled_prototype_grid_y], dim=2))
        return list_of_grids_for_sampler


class RandomCropListGenerator:
    def __init__(self, rng, patches_num=8, patch_size=(16, 16), min_scale=0.01, max_scale=0.25):
        self.rng = rng
        self.patch_size = patch_size
        self.patches_num = patches_num
        # Generate prototype grid
        self.prototypeGrid = get_prototype_grid(self.patch_size)
        # scales is multiplied by 2 because we want sizes in <-1, +1> range
        self.normlized_min_scale = min_scale * 2
        self.normlized_max_scale = max_scale * 2

    def __call__(self):
        # Single value random - no rectangles, only squares
        crop_sizes = self.rng.uniform(self.normlized_min_scale, self.normlized_max_scale, size=self.patches_num)
        crop_positions = []
        for crop_size in crop_sizes:
            # we should find (x, y) in range <(-1.0+crop_size/2.0) : (1.0-crop_size/2.0)> so we do not sample padded borders
            a = -1.0 + (crop_size / 2.0)
            b = 1.0 - (crop_size / 2.0)
            y_center = self.rng.uniform(a, b)
            x_center = self.rng.uniform(a, b)
            crop_positions.append((x_center, y_center))

        list_of_grids_for_sampler = []

        for crop_size, coord in zip(crop_sizes, crop_positions):
            tmp_rescaled_prototype_grid_y = self.prototypeGrid[:, :, 1] * crop_size / 2.0 + coord[1]
            tmp_rescaled_prototype_grid_x = self.prototypeGrid[:, :, 0] * crop_size / 2.0 + coord[0]
            list_of_grids_for_sampler.append(
                torch.stack([tmp_rescaled_prototype_grid_x, tmp_rescaled_prototype_grid_y], dim=2))

        return list_of_grids_for_sampler


class BetterPatchSampler(PatchSampler, ABC):
    def __init__(self, *args, interpolation_mode="bilinear", align_corners=True, **kwargs):
        super().__init__(*args, **kwargs)

        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners
        self.generator = NotImplemented

    def __call__(self, img):
        grids_list = self.generator()
        return multi_crop(img, grids_list, interpolation_mode=self.interpolation_mode,
                          align_corners=self.align_corners)


class GridSamplerV2(BetterPatchSampler):
    def __init__(self, patches_num_yx=(14, 14), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = GridCropListGenerator(*patches_num_yx, patch_size=self.patch_size)


class RandomCropV2(BetterPatchSampler):
    def __init__(self, *args, patches_num=32, min_scale=0.01, max_scale=0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = RandomCropListGenerator(rng=self.rng, patches_num=patches_num, patch_size=self.patch_size,
                                                 min_scale=min_scale, max_scale=max_scale)
