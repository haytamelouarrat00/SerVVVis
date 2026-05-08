import random
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

from dataset import load_scannet, load_colmap
from scenes.mesh import MeshScene
from scenes.gs import GSScene

# Resolve paths relative to this file so the smoke test runs from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEED = 0


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    scene_dir = PROJECT_ROOT / "DATA" / "kitchen"

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

    real_rgb = Image.open(rgb_path).convert("RGB")
    real_rgb = real_rgb.resize((camera.W, camera.H))
    real_rgb_np = np.array(real_rgb, dtype=np.float32) / 255.0

    output_mesh = np.concatenate([rendered_mesh, real_rgb_np], axis=1)
    output_mesh_uint8 = (output_mesh * 255.0).astype(np.uint8)

    # Save using OpenCV, BGR format
    cv2.imwrite("output_mesh.png", cv2.cvtColor(output_mesh_uint8, cv2.COLOR_RGB2BGR))
    print("Saved output_mesh.png")

    # GS test
    print("Running GS test...")
    colmap_data = load_colmap(scene_dir)
    if not colmap_data:
        raise RuntimeError(f"No COLMAP images in {scene_dir}")
    camera_gs, rgb_path_gs = random.choice(colmap_data)

    gs_scene = GSScene(scene_dir / "gs.ply")
    rendered_gs = gs_scene.render(camera_gs)

    real_rgb_gs = Image.open(rgb_path_gs).convert("RGB")
    real_rgb_gs = real_rgb_gs.resize((camera_gs.W, camera_gs.H))
    real_rgb_np_gs = np.array(real_rgb_gs, dtype=np.float32) / 255.0

    output_gs = np.concatenate([rendered_gs, real_rgb_np_gs], axis=1)
    output_gs_uint8 = (output_gs * 255.0).astype(np.uint8)

    # Save using OpenCV, BGR format
    cv2.imwrite("output_gs.png", cv2.cvtColor(output_gs_uint8, cv2.COLOR_RGB2BGR))
    print("Saved output_gs.png")


if __name__ == "__main__":
    main()
