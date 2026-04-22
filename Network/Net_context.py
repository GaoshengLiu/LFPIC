"""
A simple test algorithm to rewrite the network to test how to use the code
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
    MaskedConv2d,
    GDN,
)
from compressai.ops import ste_round
from compressai.models.utils import conv, deconv, update_registered_buffers
from compressai.ans import BufferedRansEncoder, RansDecoder

from ptflops import get_model_complexity_info
import math
from compressai.models.priors import CompressionModel#, GaussianConditional
from compressai.entropy_models import GaussianConditional
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

#from utilscom.func import update_registered_buffers, get_scale_table
from .utilscom.ckbd import *
from .transform import *

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


# classes
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class postNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.norm(self.fn(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """

    N: channels of input feature maps
    heads: multi heads of layers
    dim_head: each dim of heads
    """

    def __init__(self, N=128, heads=2, dim_head=128, dropout=0.):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == N)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(N, inner_dim, bias=False)
        self.q_reshape = Rearrange('b n (h d) -> b h n d', h=self.heads)

        self.to_kv = nn.Linear(N, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, N),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()


    def forward(self, downsample_x, x):
        """

        Args:
            downsample_x: # N H/2*W/2 C
            x: # N H*W C

        Returns:

        """

        q = self.to_q(downsample_x)
        q = self.q_reshape(q)

        kv = self.to_kv(x).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Convtransformer(nn.Module):
    """
    H: height of input feature maps
    W: width of input feature maps
    N: channels of input feature maps
    heads: multi heads of layers
    dim_head: each dim of heads
    """

    def __init__(self, H, W, N, fn, depth=2, heads=2, dim_head=128, dropout=0., downsample=True, ):
        super(Convtransformer, self).__init__()
        self.depth = depth
        self.downsample = downsample

        self.conv = fn
        self.feature_map_to_tokens_conv = Rearrange('b c h w -> b (h w)  c')

        self.feature_map_to_tokens = Rearrange('b c h w -> b (h w)  c')

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                Attention(N=N, heads=heads, dim_head=dim_head, dropout=dropout),
            )

        if self.downsample:
            self.reverse = Rearrange('b (h w)  c -> b c h w', h=H // 2, w=W // 2)
        else:
            self.reverse = Rearrange('b (h w)  c -> b c h w', h=H*2, w=W*2)

    def forward(self, x):
        x_conv = self.conv(x)
        downsample_x = self.feature_map_to_tokens_conv(x_conv)
        indentity = downsample_x

        x_tockens = self.feature_map_to_tokens(x)

        for attn in self.layers:
            downsample_x = attn(downsample_x, x_tockens) + downsample_x

        if self.depth:
            return self.reverse(downsample_x+indentity)
        else:
            return self.reverse(indentity)
class DownConv(nn.Module):
    def __init__(self, ch_in, ch_out, stride, A=8):
        super(DownConv, self).__init__()
        self.view_num = A
        self.down_conv = nn.Conv2d(ch_in, ch_out,  kernel_size=3, stride=stride, padding=1)
        self.out_conv  = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.gdn = GDN(ch_out)
        if stride != 1 or ch_in != ch_out:
            self.skip = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride, padding=0)
        else:
            self.skip = None

    def forward(self, x):
        B, C, H, W = x.shape
        x = rearrange(x, 'B C (u h) (v w) -> (B u v) C h w', u=self.view_num, v=self.view_num)
        identity = x
        out = self.gdn(self.out_conv(self.lrelu(self.down_conv(x))))        
        if self.skip is not None:
            identity = self.skip(identity)    
        out = out + identity
        out = rearrange(out, '(B u v) C h w -> B C (u h) (v w)', B=B, u=self.view_num, v=self.view_num)
        return out
class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out, upscale, A=8):
        super(UpConv, self).__init__()
        self.view_num = A
        self.subpel_conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out * upscale**2, kernel_size=3, padding=1), 
                nn.PixelShuffle(upscale))
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.conv = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.igdn = GDN(ch_out, inverse=True)
        self.upsample = nn.Sequential(
                nn.Conv2d(ch_in, ch_out * upscale**2, kernel_size=1, padding=0), 
                nn.PixelShuffle(upscale))

    def forward(self, x):
        B, C, H, W = x.shape
        x = rearrange(x, 'B C (u h) (v w) -> (B u v) C h w', u=self.view_num, v=self.view_num)
        identity = x
        out = self.igdn(self.conv(self.lrelu(self.subpel_conv(x))))
        identity = self.upsample(identity)
        out += identity
        out = rearrange(out, '(B u v) C h w -> B C (u h) (v w)', B=B, u=self.view_num, v=self.view_num)
        return out 
        

class Net_context(CompressionModel):

    def __init__(self, N=128, M=192, image_size=(384, 384), **kwargs):
        super().__init__(entropy_bottleneck_channels=M, **kwargs)
        H, W = pair(image_size)
        """
             N: channel number of main network
             M: channnel number of latent space
             scale_factor: first stage of transform
             imagesize: input image size
             patchsize: split the image into patch for transform
             lenslet: the number of views in a patch
             dim:
             fdepth: first depth of the transformer
             sdepth: second depth of the transformer
             dim: the dim of attention 
             heads: the heads Multi-head attention
             dim_head: 
        """
        self.g_a = nn.Sequential(
            DownConv(3, N, stride=2),
            DownConv(N, M, stride=2),
            GlobalSpatial(dim=M, out_dim=M),
            DownConv(M, M, stride=2),
            DownConv(M, M, stride=2),
            GlobalSpatial(dim=M, out_dim=M))

        self.g_s = nn.Sequential(
            GlobalSpatial(dim=M, out_dim=M),
            UpConv(M, M, upscale=2),
            UpConv(M, M, upscale=2),
            GlobalSpatial(dim=M, out_dim=M),
            UpConv(M, N, upscale=2),
            UpConv(N, 3, upscale=2))

        self.h_a = nn.Sequential(
            conv(M, M, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            ResidualBlockWithStride(M, M, stride=2),
            conv(M, M, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            ResidualBlockWithStride(M, M, stride=2))

        self.h_s = nn.Sequential(
            ResidualBlockUpsample(M, M, 2),
            nn.LeakyReLU(inplace=True),
            conv(M, M, stride=1, kernel_size=3),
            ResidualBlockUpsample(M, M * 3 //2, 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 //2, M * 2, stride=1, kernel_size=3))
        
        slice_num = 6
        slice_ch = M // slice_num
        assert slice_ch * slice_num == M

        self.context_window = 3 ##equals to FMLI size
        self.slice_num = slice_num
        self.slice_ch = slice_ch

        self.local_context = nn.ModuleList(
            SpatialContext(dim=slice_ch, window_size=self.context_window)
            for _ in range(slice_num)
        )

        self.channel_context = nn.ModuleList(
            ChannelContext(in_dim=slice_ch * i, out_dim=slice_ch) if i else None
            for i in range(slice_num)
        )
        self.entropy_parameters_anchor = nn.ModuleList(
            EntropyParameters(in_dim=M * 2 + slice_ch * 4, out_dim=slice_ch * 2)
            if i else EntropyParameters(in_dim=M * 2, out_dim=slice_ch * 2)
            for i in range(slice_num)
        )
        self.entropy_parameters_nonanchor = nn.ModuleList(
            EntropyParameters(in_dim=M * 2 + slice_ch * 6, out_dim=slice_ch * 2)
            if i else EntropyParameters(in_dim=M * 2 + slice_ch * 2, out_dim=slice_ch * 2)
            for i in range(slice_num)
        )

        # Latent Residual Prediction
        self.lrp_anchor = nn.ModuleList(
            LatentResidualPrediction(in_dim=M + (i + 1) * slice_ch, out_dim=slice_ch)
            for i in range(slice_num)
        )
        self.lrp_nonanchor = nn.ModuleList(
            LatentResidualPrediction(in_dim=M + (i + 1) * slice_ch, out_dim=slice_ch)
            for i in range(slice_num)
        )

        self.gaussian_conditional = GaussianConditional(None)
        # self.N = int(N)
        # self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        #print(x.shape)
        y = self.g_a(x)
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = ste_round(z - z_offset) + z_offset

        # Hyper-parameters
        hyper_params = self.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)

        y_slices = y.chunk(self.slice_num, dim=1)
        y_hat_slices = []
        y_likelihoods = []
        for idx, y_slice in enumerate(y_slices):
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round anchor
                slice_anchor = ste_round(slice_anchor - means_anchor) + means_anchor
                # predict residuals cause by round
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor
                # local_ctx: [B, H, W, 2 * C]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # merge means and scales of anchor and nonanchor
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                # round slice_nonanchor
                slice_nonanchor = ste_round(slice_nonanchor - means_nonanchor) + means_nonanchor
                y_hat_slice = slice_anchor + slice_nonanchor
                # predict residuals cause by round
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [y_hat_slice]), dim=1))
                y_hat_slice = y_hat_slice + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(y_hat_slice)
                y_likelihoods.append(y_slice_likelihoods)

            else:
                # global_inter_ctx = self.global_inter_context[idx](torch.cat(y_hat_slices, dim=1))
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                # Anchor(Use channel context and hyper params)
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([channel_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round anchor
                slice_anchor = ste_round(slice_anchor - means_anchor) + means_anchor
                # predict residuals cause by round
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor(Use spatial context, channel context and hyper params)
                #global_intra_ctx = self.global_intra_context[idx](y_hat_slices[-1], slice_anchor)
                # ctx_params: [B, H, W, 2 * C]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, channel_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # merge means and scales of anchor and nonanchor
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                # round slice_nonanchor
                slice_nonanchor = ste_round(slice_nonanchor - means_nonanchor) + means_nonanchor
                y_hat_slice = slice_anchor + slice_nonanchor
                # predict residuals cause by round
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [y_hat_slice]), dim=1))
                y_hat_slice = y_hat_slice + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(y_hat_slice)
                y_likelihoods.append(y_slice_likelihoods)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihoods, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y_likelihoods": y_likelihoods, "z_likelihoods": z_likelihoods}
        }


    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        torch.cuda.synchronize()
        import imageio
        import numpy as np
        #start_time = time.time()
        # self.update_resolutions(x.size(2) // 16, x.size(3) // 16)
        y = self.g_a(x)
        # print(torch.mean(y))
        # img = torch.mean(y,dim=1,keepdim=False)
        # # img = y[:,63,:,:]
        # print(torch.mean(img))
        # print(torch.max(img), torch.min(img))
        # img = np.absolute(img[0].cpu().numpy())
        # print(np.mean(img))
        # print(np.max(img), np.min(img))
        # imageio.imwrite('/root/data1/Current_work/GACN/test_img/0001.png', (np.clip(img, 0, 1)*255).astype(np.uint8))
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        hyper_params = self.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)
        y_slices = y.chunk(self.slice_num, dim=1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for idx, y_slice in enumerate(y_slices):
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round and compress anchor
                slice_anchor = compress_anchor(self.gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)
                # predict residuals caused by round
                # print(len(indexes_list))
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # round and compress nonanchor
                slice_nonanchor = compress_nonanchor(self.gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
                # print(len(indexes_list))
                # predict residuals caused by round
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
                slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(slice_nonanchor + slice_anchor)

            else:
                # Anchor
                # global_inter_ctx = self.global_inter_context[idx](torch.cat(y_hat_slices, dim=1))
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([channel_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round and compress anchor
                slice_anchor = compress_anchor(self.gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)
                # print(len(indexes_list))
                # predict residuals caused by round
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor
                #global_intra_ctx = self.global_intra_context[idx](y_hat_slices[-1], slice_anchor)
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, channel_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # round and compress nonanchor
                slice_nonanchor = compress_nonanchor(self.gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
                # predict residuals caused by round
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
                slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(slice_nonanchor + slice_anchor)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)
        torch.cuda.synchronize()
        #end_time = time.time()

        #cost_time = end_time - start_time
        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:]
        }

    def decompress(self, strings, shape):
        torch.cuda.synchronize()
        #start_time = time.time()
        y_strings = strings[0][0]
        z_strings = strings[1]
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        # self.update_resolutions(z_hat.size(2) * 4, z_hat.size(3) * 4)
        hyper_params = self.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_strings)

        for idx in range(self.slice_num):
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # decompress anchor
                slice_anchor = decompress_anchor(self.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
                # predict residuals caused by round
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # decompress non-anchor
                slice_nonanchor = decompress_nonanchor(self.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
                # predict residuals caused by round
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
                slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(slice_nonanchor + slice_anchor)

            else:
                # Anchor
                # global_inter_ctx = self.global_inter_context[idx](torch.cat(y_hat_slices, dim=1))
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([channel_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # decompress anchor
                slice_anchor = decompress_anchor(self.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
                # predict residuals caused by round
                lrp_anchor = self.lrp_anchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1))
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)
                # Non-anchor
                # Non-anchor
                #global_intra_ctx = self.global_intra_context[idx](y_hat_slices[-1], slice_anchor)
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, channel_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # decompress non-anchor
                slice_nonanchor = decompress_nonanchor(self.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
                # predict residuals caused by round
                lrp_nonanchor = self.lrp_nonanchor[idx](torch.cat(([hyper_means] + y_hat_slices + [slice_nonanchor + slice_anchor]), dim=1))
                slice_nonanchor = slice_nonanchor + ckbd_nonanchor(lrp_nonanchor)
                y_hat_slices.append(slice_nonanchor + slice_anchor)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        torch.cuda.synchronize()
        #end_time = time.time()

        #cost_time = end_time - start_time

        return {
            "x_hat": x_hat
        }



if __name__ == "__main__":
    # model = convTransformer(H=384, W=384, lenslet_num=8, viewsize=6, C=128, depth=4, heads=4, dim_head=96, mlp_dim=96, dropout=0.1, emb_dropout=0.)
    # model = convTransformer(H=192, W=192, channels=3, patchsize=2, dim=64, depth=2, heads=4,
    #                         dim_head=64, mlp_dim=64, dropout=0.1,
    #                         emb_dropout=0.)

    model = TestModel(N=128, M=128,  image_size=(384, 384), depth=[0, 2, 2, 2], heads=4, dim_head=192, dropout=0.1)
    # model = JointAutoregressiveHierarchicalPriors(192, 192)
    # model = Cheng2020Attention(128)
    input = torch.Tensor(1, 3, 384, 384)
    # from torchvision import models
    # model = models.resnet18()
    # print(model)
    out = model(input)
    flops, params = get_model_complexity_info(model, (3, 384, 384), as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops, 'params: ', params)