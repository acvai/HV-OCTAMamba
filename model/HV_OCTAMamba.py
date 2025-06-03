import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import math
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
import time
from functools import partial
from typing import Optional, Callable
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import Tensor
from model.MDR import MultiScaleConvModule as MSDAC
from model.DAM import DualAttentionModule as FRF
from model.wtconv2d import *
###########
import torch.fft
from .H_vmunet import H_SS2D
##########


try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass


class AvgPoolingChannel(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=2, stride=2)


class MaxPoolingChannel(nn.Module):
    def forward(self, x):
        return F.max_pool2d(x, kernel_size=2, stride=2)


class SEAttention(nn.Module):
    def __init__(self, channel=3, reduction=3):
        super().__init__()
        # Pooling layer, change the width and height of each channel to 1 (average value)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            # First reduce the dimension
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            # Upgrading
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # y is the weight
        return x * y.expand_as(x)


# Multi-Stream Efficient Embedding
class MSEE(nn.Module):
    def __init__(self, out_c):
        super().__init__()

        self.out_c = out_c
        self.se = SEAttention(channel=out_c)
        self.maxpool = MaxPoolingChannel()
        self.avgpool = AvgPoolingChannel()
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
        )

        self.wtconv1 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
        )

        self.wtconv2 = nn.Sequential(
            WTConv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
        )

        self.wtconv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU()
        )
        # self.out_conv=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=self.out_c, kernel_size=1),
            nn.BatchNorm2d(num_features=self.out_c),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.init_conv(x)  # BCHW

        x1, x2, x3, x4 = x.chunk(4, dim=1)
        # branch1_maxpool
        chaneel_1_max_pool = self.maxpool(x1)
        desired_size = (x1.size(2), x1.size(3))
        channel_1_max_pool_out = F.interpolate(chaneel_1_max_pool, size=desired_size, mode='bilinear',
                                               align_corners=False)

        # branch2_avgpool
        channel_2_avg_pool = self.avgpool(x2)
        desired_size = (x2.size(2), x2.size(3))
        channel_2_avg_pool_out = F.interpolate(channel_2_avg_pool, size=desired_size, mode='bilinear',
                                               align_corners=False)

        # branch3_Wtconv
        channel_3_1 = self.wtconv1(x3)
        channel_3_2 = self.wtconv2(channel_3_1)
        channel_3_3_out = self.wtconv3(channel_3_2)

        # branch4_residual
        channel_4 = x4

        output = torch.cat((channel_1_max_pool_out, channel_2_avg_pool_out, channel_3_3_out, channel_4), dim=1)
        output = self.out_conv(output)
        output = self.se(output)
        return output


# AttentionGate
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


# DAVSSM
class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = H_SS2D(dim=hidden_dim, order=3, d_state=d_state)   #this H_SS2D comes from H_vmunet model that takes it from vmamba.py: 
        self.drop_path = DropPath(drop_path)
        self.hidden_dim = hidden_dim
        self.frf = FRF(in_channels=hidden_dim, reduction=4, kernel_size=7)

    def forward(self, input: torch.Tensor):
        x = self.ln_1(input) # B, H, W, C
        x = x.permute(0, 3, 1, 2)  # B C H W
        # x2 = self.frf(x)
        x = self.self_attention(x) #* x2
        x = x.permute(0, 2, 3, 1)  # B C H W
        x_mamba = input + self.drop_path(x)
        return x_mamba


# OCTA-Mamba Block
class HV_OCTAMambaBlock(nn.Module):
    def __init__(self, in_c, out_c, ):
        super().__init__()
        self.in_c = in_c
        self.conv = MSDAC(in_channels=in_c, out_channels=out_c)
        self.ln = nn.LayerNorm(out_c)
        self.act = nn.GELU()
        self.block = VSSBlock(hidden_dim=out_c)
        self.scale = nn.Parameter(torch.ones(1))
        self.residual_conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding="same")

    def forward(self, x):
        skip = x
        skip = self.residual_conv(skip)
        x = self.conv(x)

        x = x.permute(0, 2, 3, 1)  # B H W C
        x = self.block(x)
        x = x.permute(0, 3, 1, 2)  # B C H W

        x = x.permute(0, 2, 3, 1)  # B H W C
        x = self.act(self.ln(x))
        x = x.permute(0, 3, 1, 2)  # B C H W
        return x + skip * self.scale


# Encoder Block
class EncoderBlock(nn.Module):
    """Encoding then downsampling"""

    def __init__(self, in_c, out_c):
        super().__init__()

        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()
        self.hv_octamamba = HV_OCTAMambaBlock(in_c, out_c)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.hv_octamamba(x)
        skip = self.act(self.bn(x))
        x = self.down(skip)
        return x, skip


### Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.attGate = Attention_block(F_g=in_c, F_l=skip_c, F_int=skip_c // 2)

        self.bn2 = nn.BatchNorm2d(in_c + skip_c, out_c)
        self.hv_octamamba = HV_OCTAMambaBlock(in_c + skip_c, out_c)
        self.act = nn.ReLU()

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.attGate(x, skip)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.bn2(x))
        x = self.hv_octamamba(x)
        return x


### **Model**
class HV_OCTAMamba(nn.Module):
    def __init__(self):
        super().__init__()
        self.msee = MSEE(out_c=16)

        """Encoder"""
        self.e1 = EncoderBlock(16, 32)
        self.e2 = EncoderBlock(32, 64)
        self.e3 = EncoderBlock(64, 128)

        """Decoder"""
        self.d3 = DecoderBlock(128, 128, 64)
        self.d2 = DecoderBlock(64, 64, 32)
        self.d1 = DecoderBlock(32, 32, 16)

        """Final"""
        self.conv_out = nn.Conv2d(16, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Pre-Component"""
        x = self.msee(x)

        """Encoder"""
        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)

        """Decoder"""
        x = self.d3(x, skip3)
        x = self.d2(x, skip2)
        x = self.d1(x, skip1)

        """Final"""
        x = self.conv_out(x)
        x = self.sigmoid(x)
        return x


# Parameter calculation
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    import torch
    import time
    from fvcore.nn import FlopCountAnalysis

    img = torch.randn(1, 1, 224, 224).to('cuda')
    model = HV_OCTAMamba().to('cuda')
    # out = our_model(img)
    # print(out.shape)

    # Test Flops and parameter size
    from ptflops import get_model_complexity_info

    model = HV_OCTAMamba().to('cuda')
    macs, params = get_model_complexity_info(model, (1, 224, 224), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print(f"Total FLOPs: {macs}")
    print(f"Total params: {params}")
    print(count_parameters(model) // 1e3)

    from thop import profile
    from thop import clever_format

    input = torch.randn(1, 1, 224, 224).to('cuda')
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
