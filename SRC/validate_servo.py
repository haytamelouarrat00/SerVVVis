from pathlib import Path
import tempfile

import numpy as np

from camera import Camera
from servo import FixedVelocityController, SimpleStopper, run_servo_loop


class DummyScene:
    def __init__(self):
        self.render_calls = 0

    def render(self, camera):
        self.render_calls += 1
        return np.zeros((camera.H, camera.W, 3), dtype=np.float32)


class DummyMatcher:
    def match(self, img1, img2):
        kpts = np.array([
            [0.0, 0.0],
            [3.0, 0.0],
            [3.0, 3.0],
            [0.0, 3.0],
        ], dtype=np.float32)
        return kpts, kpts.copy()


class ScriptedController:
    """Returns a (residual, velocity) sequence for testing the stopper."""

    def __init__(self, residuals, velocities):
        if len(residuals) != len(velocities):
            raise ValueError("residuals and velocities must have same length")
        self.residuals = list(residuals)
        self.velocities = [np.asarray(v, dtype=np.float32) for v in velocities]
        self.last_info = {}

    def __call__(self, rendered, target, camera, iteration):
        idx = min(iteration, len(self.residuals) - 1)
        residual = float(self.residuals[idx])
        velocity = self.velocities[idx].copy()
        self.last_info = {
            "residual_norm": residual,
            "velocity_norm": float(np.linalg.norm(velocity)),
            "num_inlier_matches": 12,
        }
        return velocity


def main():
    camera = Camera(
        np.eye(4, dtype=np.float32),
        fx=10.0,
        fy=10.0,
        cx=2.0,
        cy=2.0,
        H=4,
        W=4,
    )
    target = np.zeros((4, 4, 3), dtype=np.float32)
    scene = DummyScene()
    controller = FixedVelocityController([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    result = run_servo_loop(
        scene,
        camera,
        target,
        controller,
        iterations=3,
        dt=0.5,
        visualization_dir="/tmp/servis_servo_validate",
        matcher=DummyMatcher(),
    )

    final_t = result["camera"].T_world_cam[:3, 3]
    expected_t = np.array([0.0, 0.0, 1.5], dtype=np.float32)

    if len(result["history"]) != 3:
        raise AssertionError("Servo loop did not record one history entry per iteration")
    if scene.render_calls != 4:
        raise AssertionError("Servo loop should render once per iteration plus final render")
    if not np.allclose(final_t, expected_t, atol=1e-6):
        raise AssertionError(f"Expected final translation {expected_t}, got {final_t}")
    for iteration in range(3):
        path = Path(f"/tmp/servis_servo_validate/iter_{iteration:04d}_matches.png")
        if not path.exists():
            raise AssertionError(f"Missing iteration visualization: {path}")
    if result["history"][0]["num_matches"] != 4:
        raise AssertionError("Servo loop did not record feature correspondence counts")

    with tempfile.TemporaryDirectory() as tmpdir:
        run_servo_loop(
            scene,
            camera,
            target,
            controller,
            iterations=4,
            dt=0.0,
            visualization_dir=tmpdir,
            matcher=DummyMatcher(),
            viz_iter=2,
        )
        saved = sorted(path.name for path in Path(tmpdir).glob("*.png"))
        expected = ["iter_0000_matches.png", "iter_0002_matches.png"]
        if saved != expected:
            raise AssertionError(f"Expected viz_iter=2 files {expected}, got {saved}")

    flat_v = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    big_v = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v_a = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v_b = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Stop on error below threshold. velocity_grad_eps=0 disables grad check.
    error_stop = run_servo_loop(
        DummyScene(),
        camera,
        target,
        ScriptedController(
            [5.0, 4.0, 0.5, 0.5, 0.5],
            [big_v, v_b, v_a, v_b, v_a],
        ),
        iterations=5,
        early_stopper=SimpleStopper(error_threshold=1.0, velocity_grad_eps=0.0),
    )
    if error_stop["stop_reason"] != "error_below_threshold":
        raise AssertionError(
            f"Expected error_below_threshold, got {error_stop['stop_reason']!r}"
        )
    if error_stop["stop_iteration"] != 2:
        raise AssertionError(
            f"Expected stop_iteration 2, got {error_stop['stop_iteration']}"
        )

    # Stop on velocity gradient below eps. Iter 0 stores v, iter 1 same v -> grad=0 < eps.
    grad_stop = run_servo_loop(
        DummyScene(),
        camera,
        target,
        ScriptedController(
            [5.0, 5.0, 5.0],
            [flat_v, flat_v, flat_v],
        ),
        iterations=3,
        early_stopper=SimpleStopper(error_threshold=0.0, velocity_grad_eps=1e-4),
    )
    if grad_stop["stop_reason"] != "velocity_gradient_below_eps":
        raise AssertionError(
            f"Expected velocity_gradient_below_eps, got {grad_stop['stop_reason']!r}"
        )
    if grad_stop["stop_iteration"] != 1:
        raise AssertionError(
            f"Expected stop_iteration 1, got {grad_stop['stop_iteration']}"
        )

    # Neither condition met -> run to max_iterations.
    no_stop = run_servo_loop(
        DummyScene(),
        camera,
        target,
        ScriptedController(
            [5.0, 5.0, 5.0],
            [v_a, v_b, v_a],
        ),
        iterations=3,
        early_stopper=SimpleStopper(error_threshold=1.0, velocity_grad_eps=1e-4),
    )
    if no_stop["stop_reason"] != "max_iterations":
        raise AssertionError(
            f"Expected max_iterations, got {no_stop['stop_reason']!r}"
        )

    print("Servo loop validation passed")


if __name__ == "__main__":
    main()
