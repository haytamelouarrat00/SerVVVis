import random
from pathlib import Path
import numpy as np
from PIL import Image

from dataset import load_scannet, load_colmap
from scenes.mesh import MeshScene
from scenes.gs import GSScene
from features import FeatureMatcher, filter_matches
from viz import save_match_visualization

# Resolve paths relative to this file so the smoke test runs from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEED = 42
FEATURE_METHOD = 'xfeat'


def load_real_rgb(rgb_path, camera):
    real_rgb = Image.open(rgb_path).convert("RGB")
    real_rgb = real_rgb.resize((camera.W, camera.H))
    return np.array(real_rgb, dtype=np.float32) / 255.0


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


def save_render_matches(rendered, real, camera, matcher, output_path):
    kpts1, kpts2 = matcher.match(rendered, real)
    kpts1_f, kpts2_f, H_matrix, _, _ = filter_matches(kpts1, kpts2, camera)
    kpts1_removed, kpts2_removed = split_removed_matches(kpts1, kpts2, kpts1_f, kpts2_f)

    matches_removed = [(i, i) for i in range(len(kpts1_removed))]
    matches_kept = [(i, i) for i in range(len(kpts1_f))]

    save_match_visualization(
        rendered,
        real,
        kpts1_removed,
        kpts2_removed,
        matches_removed,
        kpts1_f,
        kpts2_f,
        matches_kept,
        output_path,
    )

    return len(kpts1), len(kpts1_f), H_matrix


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    scene_dir = PROJECT_ROOT / "DATA" / "kitchen"
    matcher = FeatureMatcher(method=FEATURE_METHOD)
    print(f"Using {FEATURE_METHOD} feature matcher")

    # Mesh test
    # NOTE: mesh.ply is in the ScanNet/TSDF coordinate frame, while gs.ply is
    # trained in the COLMAP frame. Until the meshes are re-aligned to COLMAP,
    # each renderer is fed cameras from its own source.
    print("Running Mesh test...")
    scannet_data = load_scannet(scene_dir)
    if not scannet_data:
        raise RuntimeError(f"No valid ScanNet frames in {scene_dir}")
    camera, rgb_path, depth_path = random.choice(scannet_data)

    mesh_scene = MeshScene(scene_dir / "mesh.ply")
    rendered_mesh = mesh_scene.render(camera)
    real_mesh = load_real_rgb(rgb_path, camera)

    n_mesh, n_mesh_f, H_mesh = save_render_matches(
        rendered_mesh, real_mesh, camera, matcher, "output_mesh.png"
    )
    print(
        f"Saved output_mesh.png ({n_mesh} matches, {n_mesh_f} RANSAC inliers, "
        f"H={'ok' if H_mesh is not None else 'failed'})"
    )

    # GS test
    print("Running GS test...")
    colmap_data = load_colmap(scene_dir)
    if not colmap_data:
        raise RuntimeError(f"No COLMAP images in {scene_dir}")
    camera_gs, rgb_path_gs = random.choice(colmap_data)

    gs_scene = GSScene(scene_dir / "gs.ply")
    rendered_gs = gs_scene.render(camera_gs)
    real_gs = load_real_rgb(rgb_path_gs, camera_gs)

    n_gs, n_gs_f, H_gs = save_render_matches(
        rendered_gs, real_gs, camera_gs, matcher, "output_gs.png"
    )
    print(
        f"Saved output_gs.png ({n_gs} matches, {n_gs_f} RANSAC inliers, "
        f"H={'ok' if H_gs is not None else 'failed'})"
    )


if __name__ == "__main__":
    main()
