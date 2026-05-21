"""Photometric (luminance) visual servoing ported from ViSP.

Reference: lagadic/visp
    modules/visual_features/.../vpFeatureLuminance.{h,cpp}
    example/direct-visual-servoing/photometricVisualServoing.cpp

Algorithm (Collewet & Marchand, RR-6631, 2011):
    s   = stacked pixel intensities of non-border pixels of I
    e   = s - s*
    L_I = -[Ix Iy] @ L_x(x, y, Z)               (per-pixel row of L)
    GN  : v = -lambda * pinv(L) e
    LM  : v = -lambda * (H + mu diag(H))^-1 L^T e   (ViSP example default)

Implementation is PyTorch end-to-end so it can run on CPU or GPU.
"""

from .controller import PhotometricControllerTorch
from .feature import FeatureLuminance
from .filter import (
    derivative_filter_x,
    derivative_filter_y,
    gaussian_blur,
    rgb_to_gray,
)
from .interaction import luminance_interaction, point_interaction_matrix
from .servo import PhotometricServo

__all__ = [
    "PhotometricControllerTorch",
    "FeatureLuminance",
    "PhotometricServo",
    "derivative_filter_x",
    "derivative_filter_y",
    "gaussian_blur",
    "rgb_to_gray",
    "luminance_interaction",
    "point_interaction_matrix",
]
