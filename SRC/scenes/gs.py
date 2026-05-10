import sys
import os
import math
import numpy as np
import torch
import plyfile

# Ensure diff-gaussian-rasterization is importable
third_party_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'third_party'))
sys.path.append(third_party_dir)

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def getProjectionMatrix(znear, zfar, fx, fy, cx, cy, W, H):
    """OpenCV-convention perspective projection with asymmetric frustum.

    Matches the INRIA gaussian-splatting CUDA rasterizer (z_sign = +1, depth in
    [0, 1] over [znear, zfar]) but allows an off-center principal point.
    Pixel mapping in CUDA is (NDC + 1) * size / 2, so:
        u = 0  -> NDC.x = -1, x/z = -cx / fx
        u = W  -> NDC.x = +1, x/z = (W - cx) / fx
        v = 0  -> NDC.y = -1, y/z = -cy / fy   (OpenCV y-down)
        v = H  -> NDC.y = +1, y/z = (H - cy) / fy
    """
    left = -cx / fx * znear
    right = (W - cx) / fx * znear
    top = -cy / fy * znear           # OpenCV y-down: top of image -> negative y_cam
    bottom = (H - cy) / fy * znear

    P = torch.zeros(4, 4)
    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (bottom - top)
    P[0, 2] = -(right + left) / (right - left)
    P[1, 2] = -(bottom + top) / (bottom - top)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class GSScene:
    def __init__(self, ply_path):
        plydata = plyfile.PlyData.read(ply_path)
        v = plydata['vertex']

        prop_names = {p.name for p in v.properties}
        f_rest_count = sum(1 for n in prop_names if n.startswith('f_rest_'))
        if f_rest_count % 3 != 0:
            raise ValueError(
                f"Expected f_rest_* count divisible by 3, got {f_rest_count}"
            )
        n_per_channel = f_rest_count // 3
        # SH coefs per channel = (deg+1)^2; minus 1 DC term => n_per_channel.
        sh_degree = int(round(math.sqrt(n_per_channel + 1) - 1))
        if (sh_degree + 1) ** 2 - 1 != n_per_channel:
            raise ValueError(
                f"f_rest count {f_rest_count} does not match any SH degree"
            )
        self.sh_degree = sh_degree

        self.xyz = torch.tensor(np.stack([v['x'], v['y'], v['z']], axis=-1)).float().cuda()
        self.opacities = torch.sigmoid(torch.tensor(v['opacity']).float().unsqueeze(-1)).cuda()
        self.scales = torch.exp(torch.tensor(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=-1)).float()).cuda()

        rot = torch.tensor(np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=-1)).float().cuda()
        self.rotations = rot / rot.norm(dim=-1, keepdim=True)

        self.sh_dc = torch.tensor(np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=-1)).float().unsqueeze(1).cuda()

        if n_per_channel > 0:
            f_rest_names = [f'f_rest_{i}' for i in range(f_rest_count)]
            f_rest = np.stack([v[name] for name in f_rest_names], axis=-1)
            # INRIA PLY storage: (N, 3, n_per_channel) flattened row-major to (N, 3*n_per_channel).
            # Reverse: reshape -> (N, 3, n_per_channel), transpose -> (N, n_per_channel, 3).
            self.sh_rest = (
                torch.tensor(f_rest)
                .float()
                .reshape(-1, 3, n_per_channel)
                .transpose(1, 2)
                .cuda()
            )
        else:
            N = self.xyz.shape[0]
            self.sh_rest = torch.zeros((N, 0, 3), device='cuda')
        self._last_camera = None

    def render(self, camera):
        self._last_camera = camera

        # tan(fov/2) values are used by the rasterizer to recover focal lengths
        # for the screen-space covariance Jacobian: focal = size / (2 * tanfov).
        # Off-center cx, cy is handled via the asymmetric projection matrix below.
        tanfovx = camera.W / (2.0 * camera.fx)
        tanfovy = camera.H / (2.0 * camera.fy)

        viewmatrix = torch.tensor(camera.T_cam_world).float().transpose(0, 1).cuda()
        projmatrix = getProjectionMatrix(
            znear=0.01, zfar=100.0,
            fx=camera.fx, fy=camera.fy,
            cx=camera.cx, cy=camera.cy,
            W=camera.W, H=camera.H,
        ).float().transpose(0, 1).cuda()

        # Pytorch's bmm expects (B, N, M), we unsqueeze and squeeze
        full_projmatrix = (viewmatrix.unsqueeze(0).bmm(projmatrix.unsqueeze(0))).squeeze(0)

        campos = torch.tensor(camera.T_world_cam[:3, 3]).float().cuda()

        raster_settings = GaussianRasterizationSettings(
            image_height=camera.H,
            image_width=camera.W,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.zeros(3).cuda(),
            scale_modifier=1.0,
            viewmatrix=viewmatrix,
            projmatrix=full_projmatrix,
            sh_degree=self.sh_degree,
            campos=campos,
            prefiltered=False,
            debug=False,
            antialiasing=False
        )

        rasterizer = GaussianRasterizer(raster_settings)

        shs = torch.cat([self.sh_dc, self.sh_rest], dim=1)

        outputs = rasterizer(
            means3D=self.xyz,
            means2D=torch.zeros_like(self.xyz),
            shs=shs,
            opacities=self.opacities,
            scales=self.scales,
            rotations=self.rotations,
            colors_precomp=None
        )
        rendered_image = outputs[0]

        rendered_image = rendered_image.permute(1, 2, 0).clamp(0, 1)
        return rendered_image.cpu().numpy()

    def render_depth(self, camera=None):
        if camera is None:
            camera = self._last_camera
        if camera is None:
            raise NotImplementedError("GSScene.render_depth requires a camera")

        tanfovx = camera.W / (2.0 * camera.fx)
        tanfovy = camera.H / (2.0 * camera.fy)

        T_cam_world = torch.tensor(camera.T_cam_world).float().cuda()
        viewmatrix = T_cam_world.transpose(0, 1)
        projmatrix = getProjectionMatrix(
            znear=0.01, zfar=100.0,
            fx=camera.fx, fy=camera.fy,
            cx=camera.cx, cy=camera.cy,
            W=camera.W, H=camera.H,
        ).float().transpose(0, 1).cuda()

        full_projmatrix = (viewmatrix.unsqueeze(0).bmm(projmatrix.unsqueeze(0))).squeeze(0)

        campos = torch.tensor(camera.T_world_cam[:3, 3]).float().cuda()

        raster_settings = GaussianRasterizationSettings(
            image_height=camera.H,
            image_width=camera.W,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.zeros(3).cuda(),
            scale_modifier=1.0,
            viewmatrix=viewmatrix,
            projmatrix=full_projmatrix,
            sh_degree=self.sh_degree,
            campos=campos,
            prefiltered=False,
            debug=False,
            antialiasing=False
        )

        rasterizer = GaussianRasterizer(raster_settings)

        ones = torch.ones((self.xyz.shape[0], 1), dtype=self.xyz.dtype, device=self.xyz.device)
        xyz_h = torch.cat([self.xyz, ones], dim=1)
        z_cam = (xyz_h @ T_cam_world.T)[:, 2:3]
        depth_colors = torch.cat([z_cam, z_cam, ones], dim=1)

        outputs = rasterizer(
            means3D=self.xyz,
            means2D=torch.zeros_like(self.xyz),
            shs=None,
            opacities=self.opacities,
            scales=self.scales,
            rotations=self.rotations,
            colors_precomp=depth_colors
        )
        depth_numerator = outputs[0][0]
        accum_alpha = outputs[0][2]
        rendered_depth = torch.zeros_like(depth_numerator)
        valid = accum_alpha > 1e-6
        rendered_depth[valid] = depth_numerator[valid] / accum_alpha[valid]

        return rendered_depth.detach().cpu().numpy().astype(np.float32, copy=False)
