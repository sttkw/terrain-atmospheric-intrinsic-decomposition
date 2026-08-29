import torch
import torch.nn.functional as F
from torch import nn


def gaussian_blur2d(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    if kernel_size <= 1:
        return x

    if sigma <= 0.0:
        sigma = kernel_size / 6.0

    coords = torch.arange(kernel_size, device=x.device, dtype=torch.float32)
    coords = coords - (kernel_size - 1) / 2.0
    kernel = torch.exp(-0.5 * (coords / sigma).pow(2))
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    kernel = kernel.to(dtype=x.dtype)

    channels = x.shape[1]
    pad = kernel_size // 2
    kernel_x = kernel.view(1, 1, 1, kernel_size).expand(channels, 1, 1, kernel_size)
    kernel_y = kernel.view(1, 1, kernel_size, 1).expand(channels, 1, kernel_size, 1)

    x = F.pad(x, (pad, pad, 0, 0), mode="replicate")
    x = F.conv2d(x, kernel_x, groups=channels)
    x = F.pad(x, (0, 0, pad, pad), mode="replicate")
    return F.conv2d(x, kernel_y, groups=channels)


def apply_delta_spatial_filter(
    x: torch.Tensor,
    mode: str,
    kernel_size: int,
    sigma: float,
) -> torch.Tensor:
    if mode == "none":
        return x
    if mode == "global":
        return x.mean(dim=(-2, -1), keepdim=True).expand_as(x)
    if mode == "blur":
        return gaussian_blur2d(x, kernel_size, sigma)
    raise ValueError(f"Unsupported delta spatial filter mode: {mode!r}")


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, downsample: bool = False):
        super().__init__()
        stride = 2 if downsample else 1
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up_conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.block = ConvBlock(out_ch + skip_ch, out_ch, downsample=False)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up_conv(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class DifferenceEditorUNet(nn.Module):
    """U-Net for predicting log-space component deltas.

    Input:  [Sd_i, Ss_i, V_i, broadcast(delta_log_p)] -> 12ch
    Output: [delta_log_Sd, delta_log_Ss, delta_log_V] -> 9ch
    """

    def __init__(self, in_ch: int = 12, out_ch: int = 9, base_ch: int = 32):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        self.enc1 = ConvBlock(in_ch, c1, downsample=False)
        self.enc2 = ConvBlock(c1, c2, downsample=True)
        self.enc3 = ConvBlock(c2, c3, downsample=True)
        self.bottleneck = ConvBlock(c3, c4, downsample=True)

        self.dec3 = UpBlock(c4, c3, c3)
        self.dec2 = UpBlock(c3, c2, c2)
        self.dec1 = UpBlock(c2, c1, c1)

        self.out_conv = nn.Conv2d(c1, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)

        d3 = self.dec3(b, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.out_conv(d1)
