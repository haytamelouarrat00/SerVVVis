from pathlib import Path
import tempfile

import numpy as np

from camera import Camera
from servo import FixedVelocityController, run_servo_loop


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

    print("Servo loop validation passed")


if __name__ == "__main__":
    main()
