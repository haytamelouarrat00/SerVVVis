from pathlib import Path

import numpy as np

from camera import Camera
from features import FeatureMatcher, filter_matches
from viz import save_match_visualization


def skew(vector):
    x, y, z = vector
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ], dtype=np.float32)


def se3_exp(twist):
    twist = np.asarray(twist, dtype=np.float32)
    if twist.shape != (6,):
        raise ValueError(f"twist must have shape (6,), got {twist.shape}")

    v = twist[:3]
    w = twist[3:]
    theta = float(np.linalg.norm(w))

    T = np.eye(4, dtype=np.float32)
    W = skew(w)

    if theta < 1e-8:
        R = np.eye(3, dtype=np.float32) + W
        V = np.eye(3, dtype=np.float32) + 0.5 * W
    else:
        W2 = W @ W
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        R = (
            np.eye(3, dtype=np.float32)
            + (sin_t / theta) * W
            + ((1.0 - cos_t) / (theta ** 2)) * W2
        )
        V = (
            np.eye(3, dtype=np.float32)
            + ((1.0 - cos_t) / (theta ** 2)) * W
            + ((theta - sin_t) / (theta ** 3)) * W2
        )

    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


def copy_camera_with_pose(camera, T_world_cam):
    return Camera(
        T_world_cam,
        camera.fx,
        camera.fy,
        camera.cx,
        camera.cy,
        camera.H,
        camera.W,
    )


def apply_camera_velocity(camera, velocity, dt=1.0):
    velocity = np.asarray(velocity, dtype=np.float32)
    if velocity.shape != (6,):
        raise ValueError(f"velocity must have shape (6,), got {velocity.shape}")

    # Body/camera-frame velocity: update T_world_cam by right-multiplying.
    delta_T = se3_exp(velocity * float(dt))
    T_world_cam = camera.T_world_cam @ delta_T
    return copy_camera_with_pose(camera, T_world_cam.astype(np.float32))


class FixedVelocityController:
    def __init__(self, velocity):
        self.velocity = np.asarray(velocity, dtype=np.float32)
        if self.velocity.shape != (6,):
            raise ValueError(f"velocity must have shape (6,), got {self.velocity.shape}")

    def __call__(self, rendered, target, camera, iteration):
        return self.velocity.copy()


def split_removed_matches(kpts1, kpts2, kpts1_kept, kpts2_kept):
    kept_mask = np.zeros(len(kpts1), dtype=bool)
    kept_idx = 0

    for i in range(len(kpts1)):
        if kept_idx >= len(kpts1_kept):
            break
        if (
            np.allclose(kpts1[i], kpts1_kept[kept_idx])
            and np.allclose(kpts2[i], kpts2_kept[kept_idx])
        ):
            kept_mask[i] = True
            kept_idx += 1

    if kept_idx != len(kpts1_kept):
        raise RuntimeError("Filtered keypoints are not an ordered subset of matches")

    return kpts1[~kept_mask], kpts2[~kept_mask]


def save_iteration_matches(rendered, target, camera, matcher, output_path):
    kpts1, kpts2 = matcher.match(rendered, target)
    kpts1_kept, kpts2_kept, _, _, _ = filter_matches(kpts1, kpts2, camera)
    matches_kept = [(i, i) for i in range(len(kpts1_kept))]
    empty = np.zeros((0, 2), dtype=np.float32)

    save_match_visualization(
        rendered,
        target,
        empty,
        empty.copy(),
        [],
        kpts1_kept,
        kpts2_kept,
        matches_kept,
        output_path,
        draw_removed=False,
    )

    return {
        "num_matches": int(len(kpts1)),
        "num_inliers": int(len(kpts1_kept)),
        "visualization_path": str(output_path),
    }


def save_controller_matches(rendered, target, visualization, output_path):
    kpts_current = np.asarray(
        visualization.get("kpts_current", np.zeros((0, 2), dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1, 2)
    kpts_target = np.asarray(
        visualization.get("kpts_target", np.zeros((0, 2), dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1, 2)
    if kpts_current.shape != kpts_target.shape:
        raise RuntimeError(
            f"Controller visualization keypoints are not paired: "
            f"{kpts_current.shape} vs {kpts_target.shape}"
        )

    matches_kept = [(i, i) for i in range(len(kpts_current))]
    empty = np.zeros((0, 2), dtype=np.float32)
    save_match_visualization(
        rendered,
        target,
        empty,
        empty.copy(),
        [],
        kpts_current,
        kpts_target,
        matches_kept,
        output_path,
        draw_removed=False,
    )

    return {
        "num_matches": int(visualization.get("num_raw_matches", len(kpts_current))),
        "num_inliers": int(len(kpts_current)),
        "feature_mode": visualization.get("feature_mode"),
        "visualization_path": str(output_path),
    }


def run_servo_loop(
    scene,
    initial_camera,
    target_image,
    controller,
    iterations,
    dt=1.0,
    visualization_dir=None,
    matcher=None,
    feature_method="xfeat",
    iteration_callback=None,
    viz_iter=1,
):
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    if viz_iter is not None and int(viz_iter) < 0:
        raise ValueError("viz_iter must be >= 0 or None")

    if visualization_dir is not None:
        visualization_dir = Path(visualization_dir)
        visualization_dir.mkdir(parents=True, exist_ok=True)
        if matcher is None:
            matcher = FeatureMatcher(method=feature_method)

    camera = copy_camera_with_pose(initial_camera, initial_camera.T_world_cam.copy())
    history = []

    for iteration in range(iterations):
        rendered = scene.render(camera)
        velocity = np.asarray(
            controller(rendered, target_image, camera, iteration),
            dtype=np.float32,
        )
        controller_info = getattr(controller, "last_info", {})
        next_camera = apply_camera_velocity(camera, velocity, dt=dt)

        match_info = {}
        should_save_viz = (
            visualization_dir is not None
            and viz_iter is not None
            and int(viz_iter) > 0
            and iteration % int(viz_iter) == 0
        )
        if should_save_viz:
            output_path = visualization_dir / f"iter_{iteration:04d}_matches.png"
            controller_visualization = getattr(controller, "last_visualization", None)
            if (
                controller_visualization is not None
                and controller_visualization.get("iteration") == iteration
            ):
                match_info = save_controller_matches(
                    rendered,
                    target_image,
                    controller_visualization,
                    output_path,
                )
            else:
                match_info = save_iteration_matches(
                    rendered,
                    target_image,
                    camera,
                    matcher,
                    output_path,
                )

        history_item = {
            "iteration": iteration,
            "T_world_cam": camera.T_world_cam.copy(),
            "velocity": velocity.copy(),
            "next_T_world_cam": next_camera.T_world_cam.copy(),
            "controller_info": dict(controller_info),
            **match_info,
        }
        if iteration_callback is not None:
            iteration_callback(history_item)
        history.append(history_item)

        camera = next_camera

    return {
        "camera": camera,
        "rendered": scene.render(camera),
        "history": history,
    }
