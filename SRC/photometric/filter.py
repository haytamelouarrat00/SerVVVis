"""Image filters mirroring ViSP `vpImageFilter`.

ViSP uses a 7-tap symmetric central-difference kernel for spatial gradients
(see `vpImageFilter::derivativeFilterX/Y`):

    Ix(r, c) = ( 2047*(I[r,c+1]-I[r,c-1])
               +  913*(I[r,c+2]-I[r,c-2])
               +  112*(I[r,c+3]-I[r,c-3]) ) / 8418

The three outer pixel rows / columns are zero-padded (no derivative there).

We also expose a Gaussian blur (cv2 reference behaviour, separable conv2d) and
an RGB->Y luminance conversion that matches `cv2.cvtColor(..., RGB2GRAY)`
weights (0.299, 0.587, 0.114) so behaviour with the existing NumPy controller
stays comparable.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


_VISP_KERNEL = torch.tensor(
    [-112.0, -913.0, -2047.0, 0.0, 2047.0, 913.0, 112.0],
    dtype=torch.float32,
) / 8418.0
_VISP_PAD = 3  # 7-tap kernel -> 3 zero-rows / columns on each side


def _as_tensor(image, device=None, dtype=torch.float32):
    if isinstance(image, torch.Tensor):
        t = image
    else:
        t = torch.as_tensor(image)
    t = t.to(dtype=dtype)
    if device is not None:
        t = t.to(device=device)
    return t


def rgb_to_gray(image):
    """RGB(A) float in [0,1] -> grayscale float32 (H, W)."""
    t = _as_tensor(image)
    if t.dim() == 2:
        return t
    if t.dim() != 3 or t.shape[-1] not in (3, 4):
        raise ValueError(f"rgb_to_gray expected (H,W,3|4), got {tuple(t.shape)}")
    rgb = t[..., :3]
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=t.dtype, device=t.device)
    return (rgb * weights).sum(dim=-1)


def gaussian_blur(image, sigma):
    """Isotropic Gaussian blur via separable conv2d. Passthrough for sigma <= 0."""
    t = _as_tensor(image)
    sigma = float(sigma)
    if sigma <= 0.0:
        return t
    radius = max(1, int(math.ceil(3.0 * sigma)))
    xs = torch.arange(-radius, radius + 1, dtype=t.dtype, device=t.device)
    kernel = torch.exp(-(xs ** 2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return _separable_conv2d(t, kernel, kernel)


def derivative_filter_x(image):
    """Horizontal derivative using ViSP's 7-tap central-difference kernel.

    Output has the same shape as input; the 3 outermost columns are zero.
    """
    t = _as_tensor(image)
    kernel = _VISP_KERNEL.to(dtype=t.dtype, device=t.device)
    out = _conv1d_along(t, kernel, axis=1)
    out = _zero_border(out, axis=1, pad=_VISP_PAD)
    return out


def derivative_filter_y(image):
    """Vertical derivative using ViSP's 7-tap central-difference kernel.

    Output has the same shape as input; the 3 outermost rows are zero.
    """
    t = _as_tensor(image)
    kernel = _VISP_KERNEL.to(dtype=t.dtype, device=t.device)
    out = _conv1d_along(t, kernel, axis=0)
    out = _zero_border(out, axis=0, pad=_VISP_PAD)
    return out


def border_mask(height, width, bord, device=None, dtype=torch.bool):
    """Boolean (H, W) mask: True where pixel is at least `bord` away from edge."""
    mask = torch.zeros((int(height), int(width)), dtype=dtype, device=device)
    if bord > 0 and 2 * bord < min(height, width):
        mask[bord:height - bord, bord:width - bord] = True
    elif bord == 0:
        mask[:] = True
    return mask


# ---- internals -------------------------------------------------------------


def _separable_conv2d(image, kx, ky):
    out = _conv1d_along(image, kx, axis=1)
    out = _conv1d_along(out, ky, axis=0)
    return out


def _conv1d_along(image, kernel, axis):
    """1-D convolution along `axis` (0 = vertical, 1 = horizontal) using reflect pad."""
    if image.dim() != 2:
        raise ValueError(f"_conv1d_along expects 2-D input, got {tuple(image.shape)}")
    k = kernel.flatten()
    pad = (k.numel() - 1) // 2
    inp = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    if axis == 1:
        weight = k.view(1, 1, 1, -1)
        inp = F.pad(inp, (pad, pad, 0, 0), mode="reflect")
        out = F.conv2d(inp, weight)
    elif axis == 0:
        weight = k.view(1, 1, -1, 1)
        inp = F.pad(inp, (0, 0, pad, pad), mode="reflect")
        out = F.conv2d(inp, weight)
    else:
        raise ValueError(f"axis must be 0 or 1, got {axis}")
    return out.squeeze(0).squeeze(0)


def _zero_border(image, axis, pad):
    out = image.clone()
    if axis == 1:
        out[:, :pad] = 0.0
        out[:, -pad:] = 0.0
    else:
        out[:pad, :] = 0.0
        out[-pad:, :] = 0.0
    return out
