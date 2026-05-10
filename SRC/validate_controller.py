import numpy as np

from camera import Camera
from controllers import (
    GeometricFeatureController,
    normalize_points,
    point_interaction_matrix,
)


class DummyMatcher:
    def __init__(self, offset=(0.0, 0.0)):
        self.offset = np.asarray(offset, dtype=np.float32)
        self.call_count = 0

    def match(self, img1, img2):
        self.call_count += 1
        current = np.array([
            [10.0, 10.0],
            [50.0, 10.0],
            [50.0, 50.0],
            [10.0, 50.0],
            [30.0, 30.0],
            [42.0, 24.0],
        ], dtype=np.float32)
        target = current + self.offset
        return current, target


def make_camera():
    return Camera(
        np.eye(4, dtype=np.float32),
        fx=80.0,
        fy=80.0,
        cx=32.0,
        cy=32.0,
        H=64,
        W=64,
    )


def constant_depth(rendered):
    return np.full(rendered.shape[:2], 2.0, dtype=np.float32)


def test_interaction_matrix_shape():
    camera = make_camera()
    kpts = np.array([[32.0, 32.0], [40.0, 32.0]], dtype=np.float32)
    points = normalize_points(kpts, camera)
    L = point_interaction_matrix(points, np.array([2.0, 2.0], dtype=np.float32))
    if L.shape != (4, 6):
        raise AssertionError(f"Expected interaction matrix shape (4, 6), got {L.shape}")
    if not np.isfinite(L).all():
        raise AssertionError("Interaction matrix contains non-finite values")


def test_zero_error_returns_zero_velocity():
    camera = make_camera()
    rendered = np.zeros((64, 64, 3), dtype=np.float32)
    target = rendered.copy()
    controller = GeometricFeatureController(
        matcher=DummyMatcher(offset=(0.0, 0.0)),
        depth_provider=constant_depth,
        max_translation_velocity=None,
        max_rotation_velocity=None,
    )

    velocity = controller(rendered, target, camera, iteration=0)
    if not np.allclose(velocity, 0.0, atol=1e-6):
        raise AssertionError(f"Zero feature error should produce zero velocity, got {velocity}")


def test_shifted_features_return_finite_velocity():
    camera = make_camera()
    rendered = np.zeros((64, 64, 3), dtype=np.float32)
    target = rendered.copy()
    controller = GeometricFeatureController(
        matcher=DummyMatcher(offset=(2.0, -1.0)),
        depth_provider=constant_depth,
        max_translation_velocity=0.05,
        max_rotation_velocity=0.05,
    )

    velocity = controller(rendered, target, camera, iteration=0)
    if velocity.shape != (6,) or not np.isfinite(velocity).all():
        raise AssertionError(f"Controller returned invalid velocity: {velocity}")
    if np.linalg.norm(velocity) <= 0.0:
        raise AssertionError("Non-zero feature error should produce non-zero velocity")
    if controller.last_info["num_inlier_matches"] < 4:
        raise AssertionError("Controller did not record enough inlier matches")


def test_ratio_zero_refreshes_once_then_reprojects():
    camera = make_camera()
    rendered = np.zeros((64, 64, 3), dtype=np.float32)
    target = rendered.copy()
    matcher = DummyMatcher(offset=(0.0, 0.0))
    controller = GeometricFeatureController(
        matcher=matcher,
        depth_provider=constant_depth,
        ratio=0,
        max_translation_velocity=None,
        max_rotation_velocity=None,
    )

    controller(rendered, target, camera, iteration=0)
    if controller.last_info["feature_mode"] != "refresh":
        raise AssertionError("ratio=0 should refresh on the first iteration")
    if matcher.call_count != 1:
        raise AssertionError("Initial refresh should call the matcher once")

    controller(rendered, target, camera, iteration=1)
    if controller.last_info["feature_mode"] != "reproject":
        raise AssertionError("ratio=0 should reproject after the first iteration")
    if matcher.call_count != 1:
        raise AssertionError("ratio=0 reproject iteration should not call matcher")
    if controller.last_info["num_dropped_features"] != 0:
        raise AssertionError("Same-pose reprojection should not drop features")


def test_ratio_n_refreshes_every_n_iterations():
    camera = make_camera()
    rendered = np.zeros((64, 64, 3), dtype=np.float32)
    target = rendered.copy()
    matcher = DummyMatcher(offset=(0.0, 0.0))
    controller = GeometricFeatureController(
        matcher=matcher,
        depth_provider=constant_depth,
        ratio=2,
        max_translation_velocity=None,
        max_rotation_velocity=None,
    )

    controller(rendered, target, camera, iteration=0)
    controller(rendered, target, camera, iteration=1)
    if controller.last_info["feature_mode"] != "reproject":
        raise AssertionError("ratio=2 should reproject on iteration 1")
    controller(rendered, target, camera, iteration=2)
    if controller.last_info["feature_mode"] != "refresh":
        raise AssertionError("ratio=2 should refresh on iteration 2")
    if matcher.call_count != 2:
        raise AssertionError(f"Expected 2 matcher calls, got {matcher.call_count}")


def main():
    test_interaction_matrix_shape()
    test_zero_error_returns_zero_velocity()
    test_shifted_features_return_finite_velocity()
    test_ratio_zero_refreshes_once_then_reprojects()
    test_ratio_n_refreshes_every_n_iterations()
    print("Geometric controller validation passed")


if __name__ == "__main__":
    main()
