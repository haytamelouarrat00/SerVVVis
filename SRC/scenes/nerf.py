"""Torch-ngp NeRF scene wrapper.

Mirrors the MeshScene / GSScene interface so the same servo loop can drive
a NeRF-rendered target. Loads:
    <scene_dir>/transforms.json   - scene-space scale, offset, bound and
                                    optional model hparams
    <scene_dir>/nerf.pth          - torch-ngp checkpoint (state_dict, or
                                    {'model': state_dict, ...} as saved by
                                    nerf-navigation/main_nerf.py)

Camera poses are passed through the project's OpenCV convention; this class
internally converts them to torch-ngp's NGP-space cam2world.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Make third_party/nerf-navigation importable
_THIRD_PARTY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "third_party", "nerf-navigation")
)
if _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)

from nerf.network import NeRFNetwork  # noqa: E402
from nerf.utils import get_rays        # noqa: E402


# OpenCV (x-right, y-down, z-forward) -> Blender / OpenGL (x-right, y-up, z-back)
_CV_TO_BLENDER = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def _nerf_matrix_to_ngp(pose, scale, offset):
    """Same row permutation + column flip as nerf-navigation/provider.py.

    Input `pose` must already be in Blender (y-up, z-back) cam2world form.
    """
    p = np.asarray(pose, dtype=np.float32)
    offset = np.asarray(offset, dtype=np.float32).reshape(3)
    out = np.eye(4, dtype=np.float32)
    out[0, 0] = p[1, 0]
    out[0, 1] = -p[1, 1]
    out[0, 2] = -p[1, 2]
    out[0, 3] = p[1, 3] * scale + offset[0]
    out[1, 0] = p[2, 0]
    out[1, 1] = -p[2, 1]
    out[1, 2] = -p[2, 2]
    out[1, 3] = p[2, 3] * scale + offset[1]
    out[2, 0] = p[0, 0]
    out[2, 1] = -p[0, 1]
    out[2, 2] = -p[0, 2]
    out[2, 3] = p[0, 3] * scale + offset[2]
    return out


def _resolve_state_dict(state):
    if isinstance(state, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


class NeRFScene:
    def __init__(self, scene_dir, device=None, chunk=4096):
        scene_dir = Path(scene_dir)
        self.scene_dir = scene_dir
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.chunk = int(chunk)

        transforms_path = scene_dir / "transforms.json"
        ckpt_path = scene_dir / "nerf.pth"
        if not transforms_path.exists():
            raise FileNotFoundError(f"Missing {transforms_path}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing {ckpt_path}")

        with open(transforms_path) as f:
            self.transforms = json.load(f)

        self.scale = float(self.transforms.get("scale", 0.33))
        offset = self.transforms.get("offset", [0.0, 0.0, 0.0])
        self.offset = np.asarray(offset, dtype=np.float32).reshape(3)
        self.bound = float(
            self.transforms.get(
                "bound",
                self.transforms.get("aabb_scale", 1.0),
            )
        )

        model_cfg = dict(self.transforms.get("model", {}))
        self.model = NeRFNetwork(
            encoding=model_cfg.get("encoding", "hashgrid"),
            encoding_dir=model_cfg.get("encoding_dir", "sphere_harmonics"),
            encoding_bg=model_cfg.get("encoding_bg", "hashgrid"),
            num_layers=int(model_cfg.get("num_layers", 2)),
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            geo_feat_dim=int(model_cfg.get("geo_feat_dim", 15)),
            num_layers_color=int(model_cfg.get("num_layers_color", 3)),
            hidden_dim_color=int(model_cfg.get("hidden_dim_color", 64)),
            num_layers_bg=int(model_cfg.get("num_layers_bg", 2)),
            hidden_dim_bg=int(model_cfg.get("hidden_dim_bg", 64)),
            bound=self.bound,
            cuda_ray=bool(model_cfg.get("cuda_ray", False)),
            density_scale=float(model_cfg.get("density_scale", 1.0)),
            min_near=float(model_cfg.get("min_near", 0.05)),
            density_thresh=float(model_cfg.get("density_thresh", 10.0)),
            bg_radius=float(model_cfg.get("bg_radius", -1.0)),
        ).to(self.device)

        state = torch.load(ckpt_path, map_location=self.device)
        state_dict = _resolve_state_dict(state)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[NeRFScene] missing keys ({len(missing)}): {missing[:3]}")
        if unexpected:
            print(f"[NeRFScene] unexpected keys ({len(unexpected)}): {unexpected[:3]}")
        self.model.eval()

        self._render_kwargs = dict(model_cfg.get("render", {}))
        self._render_kwargs.setdefault("perturb", False)
        self._render_kwargs.setdefault("bg_color", None)
        self._last_camera = None

    def _pose_to_ngp(self, camera):
        pose_blender = (camera.T_world_cam @ _CV_TO_BLENDER).astype(np.float32)
        return _nerf_matrix_to_ngp(pose_blender, self.scale, self.offset)

    @torch.no_grad()
    def _render_rays(self, camera):
        H, W = int(camera.H), int(camera.W)
        pose_ngp = self._pose_to_ngp(camera)
        poses = torch.from_numpy(pose_ngp).float().unsqueeze(0).to(self.device)
        intrinsics = torch.tensor(
            [camera.fx, camera.fy, camera.cx, camera.cy],
            dtype=torch.float32,
            device=self.device,
        )
        rays = get_rays(poses, intrinsics, H, W, N=-1)
        rays_o = rays["rays_o"]
        rays_d = rays["rays_d"]
        results = self.model.render(
            rays_o,
            rays_d,
            staged=True,
            max_ray_batch=self.chunk,
            **self._render_kwargs,
        )
        return results, H, W

    def render(self, camera):
        self._last_camera = camera
        results, H, W = self._render_rays(camera)
        image = results["image"].reshape(H, W, 3).clamp(0.0, 1.0)
        return image.detach().cpu().numpy().astype(np.float32)

    def render_depth(self, camera=None):
        if camera is None:
            camera = self._last_camera
        if camera is None:
            raise NotImplementedError("NeRFScene.render_depth requires a camera")
        results, H, W = self._render_rays(camera)
        depth = results["depth"].reshape(H, W).detach().cpu().numpy().astype(np.float32)
        # NGP-space depth -> metric depth (undo the scene scale used during training).
        if self.scale > 0.0:
            depth = depth / self.scale
        return depth
