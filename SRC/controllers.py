"""Image-based visual servo controller (Chaumette & Hutchinson, 2006, Part I).

Implements the basic IBVS scheme:
    e   = s - s*                                     (paper eq. 1)
    s   = stacked normalized image points (x, y)     (paper eq. 6)
    L_x = per-point interaction matrix               (paper eq. 11)
    v_c = -lambda * pinv(L_e) * e                    (paper eq. 5)
"""

import numpy as np

from depth import get_depth
from features import FeatureMatcher, filter_matches


def normalize_points(kpts, camera):
    """Pixels -> normalized image coordinates (x, y) = ((u-cu)/f, (v-cv)/f)."""
    kpts = np.asarray(kpts, dtype=np.float32).reshape(-1, 2)
    center = np.array([camera.cx, camera.cy], dtype=np.float32)
    inv_focal = np.array([1.0 / camera.fx, 1.0 / camera.fy], dtype=np.float32)
    return (kpts - center) * inv_focal


def sample_depth_nearest(depth, kpts, min_depth=1e-4):
    depth = np.asarray(depth, dtype=np.float32)
    kpts = np.asarray(kpts, dtype=np.float32).reshape(-1, 2)

    xs = np.rint(kpts[:, 0]).astype(np.int32)
    ys = np.rint(kpts[:, 1]).astype(np.int32)
    valid = (
        (xs >= 0)
        & (xs < depth.shape[1])
        & (ys >= 0)
        & (ys < depth.shape[0])
    )

    values = np.zeros(len(kpts), dtype=np.float32)
    values[valid] = depth[ys[valid], xs[valid]]
    valid &= np.isfinite(values) & (values > min_depth)
    return values, valid


def backproject_pixels(kpts, depths, camera):
    kpts = np.asarray(kpts, dtype=np.float32).reshape(-1, 2)
    depths = np.asarray(depths, dtype=np.float32).reshape(-1)
    if len(kpts) != len(depths):
        raise ValueError("kpts and depths must have the same length")

    norm = normalize_points(kpts, camera)
    points_cam = np.empty((len(kpts), 3), dtype=np.float32)
    points_cam[:, :2] = norm * depths[:, None]
    points_cam[:, 2] = depths
    return points_cam


def camera_points_to_world(points_cam, camera):
    points_cam = np.asarray(points_cam, dtype=np.float32).reshape(-1, 3)
    R = camera.T_world_cam[:3, :3]
    t = camera.T_world_cam[:3, 3]
    return (points_cam @ R.T + t).astype(np.float32, copy=False)


def project_world_points(points_world, camera, min_depth=1e-4):
    points_world = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    R = camera.T_cam_world[:3, :3]
    t = camera.T_cam_world[:3, 3]
    points_cam = points_world @ R.T + t
    depths = points_cam[:, 2]

    valid = np.isfinite(points_cam).all(axis=1) & (depths > float(min_depth))
    pixels = np.zeros((len(points_world), 2), dtype=np.float32)
    safe_depth = np.where(valid, depths, 1.0)
    pixels[:, 0] = camera.fx * (points_cam[:, 0] / safe_depth) + camera.cx
    pixels[:, 1] = camera.fy * (points_cam[:, 1] / safe_depth) + camera.cy
    pixels[~valid] = 0.0
    valid &= (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < camera.W)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < camera.H)
    )
    return pixels, depths.astype(np.float32, copy=False), valid


def point_interaction_matrix(points, depths):
    """Stack per-point L_x (paper eq. 11). Input points are normalized (x, y)."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    depths = np.asarray(depths, dtype=np.float32).reshape(-1)
    if len(points) != len(depths):
        raise ValueError("points and depths must have the same length")

    L = np.zeros((2 * len(points), 6), dtype=np.float32)
    x = points[:, 0]
    y = points[:, 1]
    inv_z = 1.0 / depths

    L[0::2, 0] = -inv_z
    L[0::2, 2] = x * inv_z
    L[0::2, 3] = x * y
    L[0::2, 4] = -(1.0 + x * x)
    L[0::2, 5] = y

    L[1::2, 1] = -inv_z
    L[1::2, 2] = y * inv_z
    L[1::2, 3] = 1.0 + y * y
    L[1::2, 4] = -x * y
    L[1::2, 5] = -x

    return L


class IBVSController:
    """Paper IBVS with optional feature-cache refresh ratio.

    `ratio` controls when features are re-matched:
        ratio = 1  -> re-match every iteration (paper default)
        ratio = N  -> re-match every Nth iteration; reproject cached 3D points between
        ratio = 0  -> match once on first call, reproject forever
    """

    def __init__(
        self,
        matcher=None,
        feature_method="xfeat",
        gain=0.5,
        min_features=3,
        min_depth=1e-4,
        depth_provider=None,
        scene=None,
        use_intrinsic_depth=False,
        ratio=1,
        damping=0.01,
        adaptive_gain=True,
        velocity_alpha=0.8,
    ):
        if int(ratio) < 0:
            raise ValueError("ratio must be >= 0")
        self.matcher = matcher if matcher is not None else FeatureMatcher(method=feature_method)
        self.gain = float(gain)
        self.min_features = int(min_features)
        self.min_depth = float(min_depth)
        self.depth_provider = depth_provider
        self.scene = scene
        self.use_intrinsic_depth = bool(use_intrinsic_depth)
        self.ratio = int(ratio)
        self.damping = float(damping)
        self.adaptive_gain = bool(adaptive_gain)
        self.velocity_alpha = float(velocity_alpha)

        self.cached_points_world = None
        self.cached_target_kpts = None
        self.cache_refresh_iteration = None
        self.last_info = {}
        self.last_visualization = {}
        self.prev_velocity = None

    def _depth(self, rendered):
        if self.depth_provider is not None:
            return np.asarray(self.depth_provider(rendered), dtype=np.float32)
        return get_depth(
            rendered,
            scene=self.scene,
            use_intrinsic=self.use_intrinsic_depth,
        )

    def _should_refresh(self, iteration):
        if self.cached_points_world is None:
            return True
        if self.ratio == 0:
            return False
        return int(iteration) % self.ratio == 0

    def _refresh(self, rendered, target, camera, iteration):
        kpts_current, kpts_target = self.matcher.match(rendered, target)
        num_raw = len(kpts_current)
        kpts_current, kpts_target, _, _, _, _ = filter_matches(
            kpts_current,
            kpts_target,
            camera,
        )

        depth_map = self._depth(rendered)
        depths, valid = sample_depth_nearest(
            depth_map,
            kpts_current,
            min_depth=self.min_depth,
        )
        kpts_current = kpts_current[valid]
        kpts_target = kpts_target[valid]
        depths = depths[valid]

        if len(kpts_current) < self.min_features:
            raise RuntimeError(
                f"IBVS needs at least {self.min_features} matches with valid "
                f"depth, got {len(kpts_current)}"
            )

        points_cam = backproject_pixels(kpts_current, depths, camera)
        self.cached_points_world = camera_points_to_world(points_cam, camera)
        self.cached_target_kpts = kpts_target.copy()
        self.cache_refresh_iteration = int(iteration)

        return {
            "mode": "refresh",
            "num_raw_matches": int(num_raw),
            "kpts_current": kpts_current,
            "kpts_target": kpts_target,
            "depths": depths,
            "num_dropped_features": 0,
        }

    def _reproject(self, rendered, camera):
        cached_before = len(self.cached_points_world)
        kpts_current, depths_geom, valid = project_world_points(
            self.cached_points_world,
            camera,
            min_depth=self.min_depth,
        )
        kpts_current = kpts_current[valid]
        depths = depths_geom[valid]
        kpts_target = self.cached_target_kpts[valid].copy()

        # With intrinsic (exact) depth from the scene, sample current depth at
        # the reprojected pixels instead of using the geometric Zc. Closer to the
        # paper's L_x (uses current observed Z) and catches occlusion / surface
        # changes. With learned depth we trust the cached 3D point's geometric
        # Zc more than a noisy depth map.
        if self.use_intrinsic_depth and len(kpts_current) > 0:
            depth_map = self._depth(rendered)
            measured, valid_z = sample_depth_nearest(
                depth_map,
                kpts_current,
                min_depth=self.min_depth,
            )
            kpts_current = kpts_current[valid_z]
            kpts_target = kpts_target[valid_z]
            depths = measured[valid_z]

        if len(kpts_current) < self.min_features:
            raise RuntimeError(
                f"IBVS needs at least {self.min_features} reprojected features, "
                f"got {len(kpts_current)}"
            )

        return {
            "mode": "reproject",
            "num_raw_matches": int(cached_before),
            "kpts_current": kpts_current,
            "kpts_target": kpts_target,
            "depths": depths,
            "num_dropped_features": int(cached_before - len(kpts_current)),
        }

    def __call__(self, rendered, target, camera, iteration):
        if self._should_refresh(iteration):
            state = self._refresh(rendered, target, camera, iteration)
        else:
            state = self._reproject(rendered, camera)

        kpts_current = state["kpts_current"]
        kpts_target = state["kpts_target"]
        depths = state["depths"]

        # Normalize image coordinates (paper eq. 6).
        x = normalize_points(kpts_current, camera)
        x_star = normalize_points(kpts_target, camera)
        error = (x - x_star).reshape(-1)

        # Build L_e from current points + current depths (paper eq. 11).
        L = point_interaction_matrix(x, depths)

        # Control law: v_c = -lambda * pinv(L_e) * e (paper eq. 5).
        velocity = (-self.gain * (np.linalg.pinv(L) @ error)).astype(np.float32)

        self.last_info = {
            "iteration": int(iteration),
            "feature_mode": state["mode"],
            "ratio": self.ratio,
            "cache_refresh_iteration": self.cache_refresh_iteration,
            "num_raw_matches": state["num_raw_matches"],
            "num_inlier_matches": int(len(kpts_current)),
            "num_cached_features": int(len(self.cached_points_world)),
            "num_dropped_features": state["num_dropped_features"],
            "residual_norm": float(np.linalg.norm(error)),
            "velocity_norm": float(np.linalg.norm(velocity)),
            "mean_depth_m": float(np.mean(depths)),
            "min_depth_m": float(np.min(depths)),
            "max_depth_m": float(np.max(depths)),
        }
        self.last_visualization = {
            "iteration": int(iteration),
            "feature_mode": state["mode"],
            "num_raw_matches": state["num_raw_matches"],
            "kpts_current": kpts_current.copy(),
            "kpts_target": kpts_target.copy(),
        }
        return velocity
