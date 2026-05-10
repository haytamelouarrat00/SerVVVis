import numpy as np

from depth import get_depth
from features import FeatureMatcher, filter_matches


def normalize_points(kpts, camera):
    kpts = np.asarray(kpts, dtype=np.float32).reshape(-1, 2)
    points = np.empty_like(kpts)
    points[:, 0] = (kpts[:, 0] - camera.cx) / camera.fx
    points[:, 1] = (kpts[:, 1] - camera.cy) / camera.fy
    return points


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

    x = (kpts[:, 0] - camera.cx) / camera.fx
    y = (kpts[:, 1] - camera.cy) / camera.fy

    points_cam = np.empty((len(kpts), 3), dtype=np.float32)
    points_cam[:, 0] = x * depths
    points_cam[:, 1] = y * depths
    points_cam[:, 2] = depths
    return points_cam


def camera_points_to_world(points_cam, camera):
    points_cam = np.asarray(points_cam, dtype=np.float32).reshape(-1, 3)
    ones = np.ones((len(points_cam), 1), dtype=np.float32)
    points_cam_h = np.concatenate([points_cam, ones], axis=1)
    points_world_h = points_cam_h @ camera.T_world_cam.T
    return points_world_h[:, :3].astype(np.float32, copy=False)


def project_world_points(points_world, camera, min_depth=1e-4):
    points_world = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    ones = np.ones((len(points_world), 1), dtype=np.float32)
    points_world_h = np.concatenate([points_world, ones], axis=1)
    points_cam_h = points_world_h @ camera.T_cam_world.T
    points_cam = points_cam_h[:, :3]
    depths = points_cam[:, 2]

    valid = np.isfinite(points_cam).all(axis=1) & (depths > float(min_depth))
    pixels = np.zeros((len(points_world), 2), dtype=np.float32)
    pixels[valid, 0] = camera.fx * (points_cam[valid, 0] / depths[valid]) + camera.cx
    pixels[valid, 1] = camera.fy * (points_cam[valid, 1] / depths[valid]) + camera.cy
    valid &= (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < camera.W)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < camera.H)
    )

    return pixels, depths.astype(np.float32, copy=False), valid


def point_interaction_matrix(points, depths):
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


def damped_least_squares_velocity(L, error, gain, damping):
    L = np.asarray(L, dtype=np.float32)
    error = np.asarray(error, dtype=np.float32).reshape(-1)

    H = L.T @ L
    diagonal = np.diag(H).copy()
    diagonal = np.maximum(diagonal, 1e-8)
    A = H + float(damping) * np.diag(diagonal)
    b = L.T @ error

    try:
        velocity = -float(gain) * np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        velocity = -float(gain) * np.linalg.lstsq(A, b, rcond=None)[0]

    return velocity.astype(np.float32)


def clip_velocity(velocity, max_translation=None, max_rotation=None):
    velocity = np.asarray(velocity, dtype=np.float32).copy()

    if max_translation is not None:
        norm = float(np.linalg.norm(velocity[:3]))
        if norm > max_translation:
            velocity[:3] *= float(max_translation) / norm

    if max_rotation is not None:
        norm = float(np.linalg.norm(velocity[3:]))
        if norm > max_rotation:
            velocity[3:] *= float(max_rotation) / norm

    return velocity


class GeometricFeatureController:
    def __init__(
        self,
        matcher=None,
        feature_method="xfeat",
        gain=0.5,
        damping=1e-3,
        max_features=200,
        min_features=4,
        min_depth=1e-4,
        max_translation_velocity=0.05,
        max_rotation_velocity=0.05,
        depth_provider=None,
        scene=None,
        use_intrinsic_depth=False,
        ratio=1,
    ):
        if int(ratio) < 0:
            raise ValueError("ratio must be >= 0")
        self.matcher = matcher if matcher is not None else FeatureMatcher(method=feature_method)
        self.gain = float(gain)
        self.damping = float(damping)
        self.max_features = None if max_features is None else int(max_features)
        self.min_features = int(min_features)
        self.min_depth = float(min_depth)
        self.max_translation_velocity = max_translation_velocity
        self.max_rotation_velocity = max_rotation_velocity
        self.depth_provider = depth_provider
        self.scene = scene
        self.use_intrinsic_depth = bool(use_intrinsic_depth)
        self.ratio = int(ratio)
        self.cached_points_world = None
        self.cached_target_kpts = None
        self.cache_refresh_iteration = None
        self.last_info = {}
        self.last_visualization = {}

    def _depth(self, rendered):
        if self.depth_provider is not None:
            return np.asarray(self.depth_provider(rendered), dtype=np.float32)
        return get_depth(
            rendered,
            scene=self.scene,
            use_intrinsic=self.use_intrinsic_depth,
        )

    def _should_refresh(self, iteration):
        if self.cached_points_world is None or self.cached_target_kpts is None:
            return True
        if self.ratio == 0:
            return False
        return int(iteration) % self.ratio == 0

    def _refresh_features(self, rendered, target, camera, iteration):
        kpts_current, kpts_target = self.matcher.match(rendered, target)
        num_raw_matches = len(kpts_current)
        kpts_current, kpts_target, H_matrix, _, _ = filter_matches(
            kpts_current,
            kpts_target,
            camera,
        )

        if len(kpts_current) < self.min_features:
            raise RuntimeError(
                f"Geometric controller needs at least {self.min_features} inlier "
                f"matches, got {len(kpts_current)}"
            )

        if self.max_features is not None and len(kpts_current) > self.max_features:
            kpts_current = kpts_current[:self.max_features]
            kpts_target = kpts_target[:self.max_features]

        depth = self._depth(rendered)
        depths, valid_depth = sample_depth_nearest(
            depth,
            kpts_current,
            min_depth=self.min_depth,
        )
        kpts_current = kpts_current[valid_depth]
        kpts_target = kpts_target[valid_depth]
        depths = depths[valid_depth]

        if len(kpts_current) < self.min_features:
            raise RuntimeError(
                f"Geometric controller needs at least {self.min_features} valid "
                f"depth matches, got {len(kpts_current)}"
            )

        points_cam = backproject_pixels(kpts_current, depths, camera)
        self.cached_points_world = camera_points_to_world(points_cam, camera)
        self.cached_target_kpts = kpts_target.copy()
        self.cache_refresh_iteration = int(iteration)

        return {
            "mode": "refresh",
            "kpts_current": kpts_current,
            "kpts_target": kpts_target,
            "depths": depths,
            "homography_ok": H_matrix is not None,
            "num_raw_matches": int(num_raw_matches),
            "num_cached_before": int(len(kpts_current)),
            "num_dropped_features": 0,
            "num_reprojected_valid": int(len(kpts_current)),
        }

    def _reproject_features(self, camera):
        cached_before = len(self.cached_points_world)
        kpts_current, depths, valid = project_world_points(
            self.cached_points_world,
            camera,
            min_depth=self.min_depth,
        )

        self.cached_points_world = self.cached_points_world[valid]
        self.cached_target_kpts = self.cached_target_kpts[valid]
        kpts_current = kpts_current[valid]
        depths = depths[valid]
        kpts_target = self.cached_target_kpts.copy()

        if len(kpts_current) < self.min_features:
            raise RuntimeError(
                f"Geometric controller needs at least {self.min_features} "
                f"reprojected features, got {len(kpts_current)}"
            )

        return {
            "mode": "reproject",
            "kpts_current": kpts_current,
            "kpts_target": kpts_target,
            "depths": depths,
            "homography_ok": None,
            "num_raw_matches": int(cached_before),
            "num_cached_before": int(cached_before),
            "num_dropped_features": int(cached_before - len(kpts_current)),
            "num_reprojected_valid": int(len(kpts_current)),
        }

    def __call__(self, rendered, target, camera, iteration):
        if self._should_refresh(iteration):
            feature_state = self._refresh_features(rendered, target, camera, iteration)
        else:
            feature_state = self._reproject_features(camera)

        kpts_current = feature_state["kpts_current"]
        kpts_target = feature_state["kpts_target"]
        depths = feature_state["depths"]

        current_norm = normalize_points(kpts_current, camera)
        target_norm = normalize_points(kpts_target, camera)
        error = (current_norm - target_norm).reshape(-1)

        L = point_interaction_matrix(current_norm, depths)
        velocity = damped_least_squares_velocity(
            L,
            error,
            gain=self.gain,
            damping=self.damping,
        )
        velocity = clip_velocity(
            velocity,
            max_translation=self.max_translation_velocity,
            max_rotation=self.max_rotation_velocity,
        )

        self.last_info = {
            "iteration": int(iteration),
            "feature_mode": feature_state["mode"],
            "ratio": self.ratio,
            "cache_refresh_iteration": self.cache_refresh_iteration,
            "num_raw_matches": feature_state["num_raw_matches"],
            "num_cached_features": int(len(self.cached_points_world)),
            "num_cached_before": feature_state["num_cached_before"],
            "num_reprojected_valid": feature_state["num_reprojected_valid"],
            "num_dropped_features": feature_state["num_dropped_features"],
            "num_inlier_matches": int(len(kpts_current)),
            "homography_ok": feature_state["homography_ok"],
            "residual_norm": float(np.linalg.norm(error)),
            "mean_depth_m": float(np.mean(depths)),
            "min_depth_m": float(np.min(depths)),
            "max_depth_m": float(np.max(depths)),
            "velocity_norm": float(np.linalg.norm(velocity)),
        }
        self.last_visualization = {
            "iteration": int(iteration),
            "feature_mode": feature_state["mode"],
            "num_raw_matches": feature_state["num_raw_matches"],
            "kpts_current": kpts_current.copy(),
            "kpts_target": kpts_target.copy(),
        }

        return velocity
