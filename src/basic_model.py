import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights



class BasicViT(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.vit = self.with_freezed_params(vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1))
        self.vit.heads = nn.Linear(in_features=768, out_features=kwargs.get('out_heads', 102), bias=True)

    def _process_input(self, x:torch.Tensor):
        n, c, h, w = x.shape
        p = self.vit.patch_size
        torch._assert(h == self.vit.image_size, f"Wrong image height! Expected {self.vit.image_size} but got {h}!")
        torch._assert(w == self.vit.image_size, f"Wrong image width! Expected {self.vit.image_size} but got {w}!")

        # divide an image into fragments
        # x_center is resized to bigger dim        
        x_center_small = x[:, :, h // 2 - 32 : h // 2 + 32, w // 2 - 32 : w // 2 + 32]
        
        # here we merge patches by resizing to 2 times smaller dim
        x_nw_big = x[:, :, :64, :64]
        x_ne_big = x[:, :, :64, w - 64:]
        x_sw_big = x[:, :, h - 64:, :64]
        x_se_big = x[:, :, h - 64:, w - 64:]
        
        # here are normal patches
        x_n_normal = x[:, :, :64, 64 : w - 64]
        x_w_normal = x[:, :, 64 : h - 64, :80]
        x_e_normal = x[:, :, 64 : h - 64, w - 80:]
        x_s_normal = x[:, :, h - 64:, 64 : w - 64]
        x_nc_normal = x[:, :, 64 : 64 + 16, 64 + 16: w - 64 - 16]
        x_sc_normal = x[:, :, h - 64 - 16 : h - 64, 64 + 16: w - 64 - 16]

        # resizing
        x_center_small = F.interpolate(x_center_small, scale_factor=2, mode='nearest')
        x_nw_big = F.interpolate(x_nw_big, scale_factor=0.5, mode='nearest')
        x_ne_big = F.interpolate(x_ne_big, scale_factor=0.5, mode='nearest')
        x_sw_big = F.interpolate(x_sw_big, scale_factor=0.5, mode='nearest')
        x_se_big = F.interpolate(x_se_big, scale_factor=0.5, mode='nearest')
        
        x = [x_nw_big, x_n_normal, x_ne_big, x_w_normal, x_nc_normal, x_center_small, \
             x_sc_normal, x_e_normal, x_sw_big, x_s_normal, x_se_big]

        # convolution projection
        x = [self.vit.conv_proj(x_j) for x_j in x]

        # concatenation
        x = [x_j.reshape(n, self.vit.hidden_dim, x_j.shape[2] * x_j.shape[3]) for x_j in x]
        x = torch.cat(x, dim=2)
        
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.vit.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.vit.heads(x)

        return x

    def with_freezed_params(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return model
