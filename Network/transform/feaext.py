import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from einops import rearrange
from ..layers import MLP, build_position_index
from ..layers import conv, deconv
from ..utilscom.ckbd import *


class GlobalSpatial(nn.Module):
    def __init__(
            self,
            dim=32,
            out_dim=64,
            num_heads=2) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.keys = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
        )
        self.queries = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
        )
        self.values = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
        )
        self.reprojection = nn.Conv2d(dim, out_dim * 3 // 2, kernel_size=1, stride=1, padding=0)
        self.mlp = nn.Sequential(
            nn.Conv2d(out_dim * 3 // 2, out_dim * 2, kernel_size=1, stride=1),
            nn.GELU(),
            # nn.Conv2d(out_dim * 2, out_dim * 2, kernel_size=1, stride=1),
            # nn.GELU(),
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=1, stride=1)
        )
        self.skip = nn.Conv2d(out_dim * 3 // 2, out_dim, kernel_size=1, stride=1, padding=0)
        self.out = nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x1):
        B, C, H, W = x1.shape
        queries = self.queries(x1).reshape(B, self.dim, H * W)
        keys = self.keys(x1).reshape(B, self.dim, H * W)
        values = self.values(x1).reshape(B, self.dim, H * W)
        head_dim = self.dim // self.num_heads

        attended_values = []
        for i in range(self.num_heads):
            key = F.softmax(keys[:, i * head_dim: (i + 1) * head_dim, :], dim=2)
            query = F.softmax(queries[:, i * head_dim: (i + 1) * head_dim, :], dim=1)
            value = values[:, i * head_dim: (i + 1) * head_dim, :]
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(B, head_dim, H, W)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)

        return self.out(self.skip(attention) + self.mlp(attention)) + x1