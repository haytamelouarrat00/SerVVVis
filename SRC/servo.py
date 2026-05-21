from pathlib import Path

import cv2
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


class SimpleStopper:
    """Stop when ||error|| < error_threshold OR ||v[i] - v[i-1]|| < velocity_grad_eps."""

    def __init__(self, error_threshold, velocity_grad_eps):
        self.error_threshold = float(error_threshold)
        self.velocity_grad_eps = float(velocity_grad_eps)
        self.prev_velocity = None
        self.stop_reason = None
        self.stop_iteration = None

    def __call__(self, item):
        if self.stop_reason is not None:
            return True

        info = item.get("controller_info", {})
        residual = info.get("residual_norm")
        iteration = int(item["iteration"])

        if residual is not None and np.isfinite(float(residual)):
            if float(residual) < self.error_threshold:
                self.stop_reason = "error_below_threshold"
                self.stop_iteration = iteration
                return True

        velocity = np.asarray(item.get("velocity", []), dtype=np.float32).reshape(-1)
        if velocity.shape == (6,):
            if self.prev_velocity is not None:
                grad_norm = float(np.linalg.norm(velocity - self.prev_velocity))
                if grad_norm < self.velocity_grad_eps:
                    self.stop_reason = "velocity_gradient_below_eps"
                    self.stop_iteration = iteration
                    return True
            self.prev_velocity = velocity.copy()
        return False

    def metadata(self):
        return {
            "stop_reason": self.stop_reason,
            "stop_iteration": self.stop_iteration,
        }


def save_iteration_matches(rendered, target, camera, matcher, output_path):
    kpts1, kpts2 = matcher.match(rendered, target)
    kpts1_kept, kpts2_kept, _, _, _, _ = filter_matches(kpts1, kpts2, camera)
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


def save_photometric_visualization(rendered, target, visualization, output_path):
    """Three-panel viz for photometric servoing: rendered | target | |diff| heatmap.

    Diff is computed on grayscale intensities in [0, 1]; mapped to JET colormap
    over [0, max(|diff|)] of the current frame, and a fixed-range overlay over
    [0, 0.5] is also drawn alongside so the absolute scale is visible.
    """
    def to_uint8_rgb(image):
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        arr = np.clip(arr, 0.0, 1.0)
        return (arr * 255.0 + 0.5).astype(np.uint8)

    def to_gray01(image):
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 2:
            return np.clip(arr, 0.0, 1.0)
        return cv2.cvtColor(np.clip(arr, 0.0, 1.0), cv2.COLOR_RGB2GRAY)

    gray_cur = to_gray01(rendered)
    gray_tgt = to_gray01(target)
    diff = np.abs(gray_cur - gray_tgt)
    diff_max = float(diff.max()) if diff.size else 0.0

    diff_norm = (diff / max(diff_max, 1e-6) * 255.0).astype(np.uint8)
    heat_auto = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    heat_auto = cv2.cvtColor(heat_auto, cv2.COLOR_BGR2RGB)

    rendered_rgb = to_uint8_rgb(rendered)
    target_rgb = to_uint8_rgb(target)

    H = rendered_rgb.shape[0]
    label_strip = 22
    panels = []
    for img, label in (
        (rendered_rgb, "rendered"),
        (target_rgb, "target"),
        (heat_auto, f"|I-I*|  max={diff_max:.3f}"),
    ):
        panel = np.zeros((H + label_strip, img.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            panel, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        panel[label_strip:, :, :] = img
        panels.append(panel)

    out = np.concatenate(panels, axis=1)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), out_bgr)

    info = visualization or {}
    return {
        "num_matches": int(info.get("num_pixels_used", 0)),
        "num_inliers": int(info.get("num_pixels_used", 0)),
        "feature_mode": info.get("feature_mode", "photometric"),
        "visualization_path": str(output_path),
        "diff_max": diff_max,
        "diff_mean": float(diff.mean()) if diff.size else 0.0,
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
    early_stopper=None,
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
    stop_reason = None
    stop_iteration = None

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
            controller_visualization = getattr(controller, "last_visualization", None)
            is_photometric = (
                controller_visualization is not None
                and controller_visualization.get("iteration") == iteration
                and controller_visualization.get("feature_mode") == "photometric"
            )
            if is_photometric:
                output_path = visualization_dir / f"iter_{iteration:04d}_photometric.png"
                match_info = save_photometric_visualization(
                    rendered,
                    target_image,
                    controller_visualization,
                    output_path,
                )
            else:
                output_path = visualization_dir / f"iter_{iteration:04d}_matches.png"
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
        callback_stop = False
        if iteration_callback is not None:
            callback_stop = bool(iteration_callback(history_item))

        stopper_stop = False
        if early_stopper is not None:
            stopper_stop = bool(early_stopper(history_item))

        if stopper_stop:
            stop_reason = early_stopper.stop_reason or "early_stop"
            stop_iteration = early_stopper.stop_iteration
            history_item["stop_reason"] = stop_reason
        elif callback_stop:
            stop_reason = "callback"
            stop_iteration = int(iteration)
            history_item["stop_reason"] = stop_reason

        history.append(history_item)

        if stop_reason is not None:
            break

        camera = next_camera

    if stop_reason is None:
        stop_reason = "max_iterations"

    return {
        "camera": camera,
        "rendered": scene.render(camera),
        "history": history,
        "stop_reason": stop_reason,
        "stop_iteration": stop_iteration,
    }
