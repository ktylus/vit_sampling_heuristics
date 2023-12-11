# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import Mlp, _cfg
from math import sqrt

from pos_utils import get_2dplus_sincos_pos_embed_coords, get_2d_sincos_pos_embed_coords


class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Attention, Mlp_block=Mlp
                 , init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Layer_scale_init_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Attention, Mlp_block=Mlp
                 , init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class Layer_scale_init_Block_paralx2(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Attention, Mlp_block=Mlp
                 , init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.attn1 = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm21 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp1 = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_1_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x))) + self.drop_path(
            self.gamma_1_1 * self.attn1(self.norm11(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x))) + self.drop_path(
            self.gamma_2_1 * self.mlp1(self.norm21(x)))
        return x


class Block_paralx2(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Attention, Mlp_block=Mlp
                 , init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.attn1 = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm21 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp1 = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.attn1(self.norm11(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.mlp1(self.norm21(x)))
        return x


class hMLP_stem(nn.Module):
    """ hMLP_stem: https://arxiv.org/pdf/2203.09795.pdf
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=nn.SyncBatchNorm):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = torch.nn.Sequential(*[nn.Conv2d(in_chans, embed_dim // 4, kernel_size=4, stride=4),
                                          norm_layer(embed_dim // 4),
                                          nn.GELU(),
                                          nn.Conv2d(embed_dim // 4, embed_dim // 4, kernel_size=2, stride=2),
                                          norm_layer(embed_dim // 4),
                                          nn.GELU(),
                                          nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=2, stride=2),
                                          norm_layer(embed_dim),
                                          ])

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbedHybrid(nn.Module):
    """ 2D Image to Patch Embedding modified for patch datset
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        if len(x.shape) == 4:
            B, C, H, W = x.shape
            P = None
            assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
            assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        elif len(x.shape) == 5:
            B, P, C, H, W = x.shape
            assert H == self.patch_size[0]
            assert W == self.patch_size[1]
            x = x.reshape(B * P, C, H, W)
        else:
            assert False

        x = self.proj(x)

        if P is None:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        else:
            x = x.reshape(B, P, self.embed_dim)
        x = self.norm(x)
        return x


class vit_models(nn.Module):
    """ Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers=Block,
                 Patch_layer=PatchEmbedHybrid, act_layer=nn.GELU,
                 Attention_block=Attention, Mlp_block=Mlp,
                 dpr_constant=True, init_scale=1e-4,
                 mlp_ratio_clstk=4.0, pretrained_cfg_overlay=None, use_learned_pos_embed=False, quantize_pos_embed=False, scale_aware_embed=True):
        super().__init__()

        self.dropout_rate = drop_rate

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size

        self.scale_aware_embed = scale_aware_embed

        self.patch_embed = Patch_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.use_learned_pos_embed = use_learned_pos_embed
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.quantize_pos_embed = quantize_pos_embed

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer, Attention_block=Attention_block, Mlp_block=Mlp_block, init_values=init_scale)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        # pos_embed = get_2dplus_sincos_pos_embed_grid(self.pos_embed.shape[-1], self.patch_embed.grid_size[0],
        #                                              self.patch_embed.patch_size[0])
        # self.pos_embed.data.copy_(pos_embed.unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def get_num_layers(self):
        return len(self.blocks)

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def quantize_coords(self, coords):
        grid_size = self.img_size // self.patch_size
        return ((coords[...,0]/self.patch_size).round()*grid_size + (coords[..., 1]/self.patch_size).round()).to(dtype=int).clip(0, grid_size**2-1)
    
    def interpolate_pos_encoding(
        self, x: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
        npatch = x.shape[1]
        nembed = pos_embed.shape[1]
        if npatch == nembed:
            return pos_embed
        assert int(sqrt(npatch))**2 == npatch

        dim = x.shape[-1]
        height, width = int(sqrt(nembed)), int(sqrt(nembed))
        scale_factor = sqrt(npatch / nembed)

        # Interpolate the position embeddings
        pos_embed = pos_embed.reshape(1, height, width, dim).permute(0, 3, 1, 2)
        pos_embed = nn.functional.interpolate(
            pos_embed,
            scale_factor=scale_factor,
            mode="bicubic",
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        # Add the class embedding back and return this
        return pos_embed

    def forward_features(self, x, coords, keeps=None):
        x = self.patch_embed(x)
        if self.use_learned_pos_embed:
            if self.quantize_pos_embed:
                quantized =  self.quantize_coords(coords)
                x = x + self.pos_embed[[0] ,quantized, :]
            else:
                interpolated = self.interpolate_pos_encoding(x, self.pos_embed)
                x = x + interpolated
        else:
            if self.scale_aware_embed:
                pos_embedding = get_2dplus_sincos_pos_embed_coords(self.embed_dim, coords, cls_token=False)
            else:
                pos_embedding = get_2d_sincos_pos_embed_coords(self.embed_dim, coords, cls_token=False)
            x = x + pos_embedding
        
        if keeps is not None:
            x = torch.gather(x, 1, keeps.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x, coords, keeps=None):

        x = self.forward_features(x, coords, keeps)

        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)
        x = self.head(x)

        return x


# DeiT III: Revenge of the ViT (https://arxiv.org/abs/2204.07118)

@register_model
def deit_tiny_patch16_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)

    return model


@register_model
def deit_small_patch16_LS(pretrained=False, pretrained_cfg=None, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_small_' + str(img_size) + '_'
        if pretrained_21k:
            name += '21k.pth'
        else:
            name += '1k.pth'

        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])

    return model


@register_model
def deit_medium_patch16_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_medium_' + str(img_size) + '_'
        if pretrained_21k:
            name += '21k.pth'
        else:
            name += '1k.pth'

        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_LS(pretrained=False, pretrained_cfg=None, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_base_' + str(img_size) + '_'
        if pretrained_21k:
            name += '21k.pth'
        else:
            name += '1k.pth'

        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_large_patch16_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_large_' + str(img_size) + '_'
        if pretrained_21k:
            name += '21k.pth'
        else:
            name += '1k.pth'

        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_huge_patch14_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_huge_' + str(img_size) + '_'
        if pretrained_21k:
            name += '21k_v1.pth'
        else:
            name += '1k_v1.pth'

        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_huge_patch14_52_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=14, embed_dim=1280, depth=52, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)

    return model


@register_model
def deit_huge_patch14_26x2_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=14, embed_dim=1280, depth=26, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block_paralx2, **kwargs)

    return model


@register_model
def deit_Giant_48_patch14_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=14, embed_dim=1664, depth=48, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    return model


@register_model
def deit_giant_40_patch14_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    # model.default_cfg = _cfg()

    return model


# Models from Three things everyone should know about Vision Transformers (https://arxiv.org/pdf/2203.09795.pdf)

@register_model
def deit_small_patch16_36_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=384, depth=36, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)

    return model


@register_model
def deit_small_patch16_36(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=384, depth=36, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    return model


@register_model
def deit_small_patch16_18x2_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=384, depth=18, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block_paralx2, **kwargs)

    return model


@register_model
def deit_small_patch16_18x2(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=384, depth=18, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Block_paralx2, **kwargs)

    return model


@register_model
def deit_base_patch16_18x2_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=768, depth=18, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block_paralx2, **kwargs)

    return model


@register_model
def deit_base_patch16_18x2(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=768, depth=18, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Block_paralx2, **kwargs)

    return model


@register_model
def deit_base_patch16_36x1_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=768, depth=36, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)

    return model


@register_model
def deit_base_patch16_36x1(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=768, depth=36, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    return model
