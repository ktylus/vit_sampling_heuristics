import torch


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------


def get_2d_sincos_pos_embed_grid(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing='xy')  # here w goes first
    grid = torch.stack(grid, dim=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = torch.concatenate([torch.zeros([1, embed_dim]), pos_embed], dim=0)
    return pos_embed


def get_2dplus_sincos_pos_embed_grid(embed_dim, grid_size, patch_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(grid_w, grid_h, indexing='xy')  # here w goes first
    grid = torch.stack([grid_h, grid_w, grid_h + patch_size, grid_w + patch_size], dim=0)

    grid = grid.reshape([4, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = torch.concatenate([torch.zeros([1, embed_dim]), pos_embed], dim=0)
    return pos_embed


def get_2d_sincos_pos_embed_coords(embed_dim, patch_coords, cls_token=False):
    """
    patch_coords: [[x1,y1], [x2,y2], ...]
    """
    if patch_coords.shape[2] == 4:
        patch_coords = patch_coords[..., 0:2]
    b, q, t = patch_coords.shape
    assert t == 2

    patch_coords = patch_coords.reshape((b * q, 2)).float() / 16

    grid = patch_coords.permute(1, 0)
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    pos_embed = pos_embed.reshape((b, q, embed_dim))

    if cls_token:
        pos_embed = torch.cat(
            [torch.zeros([b, 1, embed_dim], dtype=patch_coords.dtype, device=patch_coords.device), pos_embed], dim=1)
    return pos_embed.detach()


def get_2dplus_sincos_pos_embed_coords(embed_dim, patch_coords, cls_token=False):
    """
    patch_coords: [[x1,y1], [x2,y2], ...]
    """
    b, q, t = patch_coords.shape
    assert t == 4

    patch_coords = patch_coords.reshape((b * q, 4)).float() / 16

    grid = patch_coords.permute(1, 0)
    pos_embed = get_2dplus_sincos_pos_embed_from_grid(embed_dim, grid)

    pos_embed = pos_embed.reshape((b, q, embed_dim))

    if cls_token:
        pos_embed = torch.cat(
            [torch.zeros([b, 1, embed_dim], dtype=patch_coords.dtype, device=patch_coords.device), pos_embed], dim=1)
    return pos_embed.detach()


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb


def get_2dplus_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 4 == 0

    # use half of dimensions to encode grid_h
    emb_h1 = get_1d_sincos_pos_embed_from_grid(embed_dim // 4, grid[0])
    emb_w1 = get_1d_sincos_pos_embed_from_grid(embed_dim // 4, grid[1])
    emb_h2 = get_1d_sincos_pos_embed_from_grid(embed_dim // 4, grid[2])
    emb_w2 = get_1d_sincos_pos_embed_from_grid(embed_dim // 4, grid[3])

    emb = torch.cat([emb_h1, emb_w1, emb_h2, emb_w2], dim=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(end=embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb
