import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from controllers import GeometricFeatureController
from dataset import load_colmap, load_scannet
from features import FeatureMatcher
from scenes.gs import GSScene
from scenes.mesh import MeshScene
from servo import run_servo_loop
from viz import save_error_evolution


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = PROJECT_ROOT / "RUNS"
CONTROLLER_DIR = "FBVS"

# Edit these values to define the frame-to-frame servo experiment.
SCENE_DIR = PROJECT_ROOT / "DATA" / "kitchen"
RENDERER = "gs"  # "mesh" or "gs"
START_INDEX = 34
INDEX_AWAY = 50
TARGET_INDEX = None  # Set to an integer to override START_INDEX + INDEX_AWAY.
ITERATIONS = 100
DT = 1.0
DEPTH_MODE = "intrinsic"  # "learned" = MoGe2, "intrinsic" = scene.render_depth()
FEATURE_METHOD = "xfeat"
RATIO = 1  # 0 = refresh once, N = refresh every Nth iteration.
VIZ_ITER = 1  # Save visualization every VIZ_ITER iterations. 0 disables images.
GAIN = 1
DAMPING = 1e-3
MAX_FEATURES = 200
MIN_FEATURES = 4
MAX_TRANSLATION_VELOCITY = 1
MAX_ROTATION_VELOCITY = 1
RUN_NAME = None


def normalize_frame_id(value):
    text = str(value)
    match = re.search(r"(\d+)", text)
    if match is None:
        raise ValueError(f"Could not parse frame id from {value!r}")
    return f"frame-{int(match.group(1)):06d}"


def frame_id_from_path(path):
    return normalize_frame_id(Path(path).name)


def frame_number(frame_id):
    return int(normalize_frame_id(frame_id).split("-")[1])


def load_rgb(rgb_path, width, height):
    image = Image.open(rgb_path).convert("RGB")
    image = image.resize((int(width), int(height)))
    return np.asarray(image, dtype=np.float32) / 255.0


def save_rgb(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_u8 = (np.asarray(image) * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(image_u8).save(path)


def make_frame_index(records, renderer):
    index = {}
    for record in records:
        if renderer == "mesh":
            camera, rgb_path, _ = record
        else:
            camera, rgb_path = record
        index[frame_id_from_path(rgb_path)] = {
            "camera": camera,
            "rgb_path": rgb_path,
        }
    return index


def available_frame_hint(frame_index, limit=8):
    frame_ids = sorted(frame_index, key=frame_number)
    if not frame_ids:
        return "none"
    head = ", ".join(frame_ids[:limit])
    tail = ", ".join(frame_ids[-limit:])
    if len(frame_ids) <= 2 * limit:
        return ", ".join(frame_ids)
    return f"{head}, ..., {tail}"


def sorted_frame_ids(frame_index):
    return sorted(frame_index, key=frame_number)


def resolve_target_index(start_index, target_index, index_away):
    if target_index is not None:
        return int(target_index)
    return int(start_index) + int(index_away)


def resolve_frame_from_index(frame_index, renderer, logical_index):
    logical_index = int(logical_index)
    if logical_index < 1:
        raise ValueError("Frame indexes are 1-based; use START_INDEX >= 1")

    if renderer == "mesh":
        frame_id = f"frame-{logical_index:06d}"
        if frame_id not in frame_index:
            raise RuntimeError(
                f"Missing logical index {logical_index} as {frame_id} for mesh. "
                f"Available frames include: {available_frame_hint(frame_index)}"
            )
        return frame_id

    frame_ids = sorted_frame_ids(frame_index)
    position = logical_index - 1
    if position >= len(frame_ids):
        raise RuntimeError(
            f"Missing logical index {logical_index} for {renderer}; "
            f"only {len(frame_ids)} frames are loaded. "
            f"Available frames include: {available_frame_hint(frame_index)}"
        )
    return frame_ids[position]


def load_scene_and_frames(scene_dir, renderer):
    if renderer == "mesh":
        records = load_scannet(scene_dir)
        scene = MeshScene(scene_dir / "mesh.ply")
    elif renderer == "gs":
        records = load_colmap(scene_dir)
        scene = GSScene(scene_dir / "gs.ply")
    else:
        raise ValueError(f"Unknown renderer {renderer!r}")

    frame_index = make_frame_index(records, renderer)
    if not frame_index:
        raise RuntimeError(f"No frames loaded for {renderer} from {scene_dir}")
    return scene, frame_index


def rotation_error_from_pose(T_world_cam, target_T_world_cam):
    R_current = T_world_cam[:3, :3]
    R_target = target_T_world_cam[:3, :3]
    R_delta = R_target.T @ R_current
    cos_angle = (np.trace(R_delta) - 1.0) * 0.5
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def rotation_error_deg(camera, target_camera):
    return rotation_error_from_pose(camera.T_world_cam, target_camera.T_world_cam)


def translation_error_from_pose(T_world_cam, target_T_world_cam):
    delta = T_world_cam[:3, 3] - target_T_world_cam[:3, 3]
    return float(np.linalg.norm(delta))


def translation_error_m(camera, target_camera):
    return translation_error_from_pose(camera.T_world_cam, target_camera.T_world_cam)


def write_history_csv(path, history):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "iteration",
        "num_matches",
        "num_inliers",
        "feature_mode",
        "controller_inliers",
        "residual_norm",
        "translation_error_m",
        "rotation_error_deg",
        "cached_features",
        "dropped_features",
        "reprojected_valid",
        "mean_depth_m",
        "min_depth_m",
        "max_depth_m",
        "velocity_norm",
        "vx",
        "vy",
        "vz",
        "wx",
        "wy",
        "wz",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in history:
            info = item.get("controller_info", {})
            velocity = np.asarray(item["velocity"], dtype=np.float32)
            writer.writerow({
                "iteration": item["iteration"],
                "num_matches": item.get("num_matches", ""),
                "num_inliers": item.get("num_inliers", ""),
                "feature_mode": item.get("feature_mode", ""),
                "controller_inliers": info.get("num_inlier_matches", ""),
                "residual_norm": info.get("residual_norm", ""),
                "translation_error_m": item.get("translation_error_m", ""),
                "rotation_error_deg": item.get("rotation_error_deg", ""),
                "cached_features": info.get("num_cached_features", ""),
                "dropped_features": info.get("num_dropped_features", ""),
                "reprojected_valid": info.get("num_reprojected_valid", ""),
                "mean_depth_m": info.get("mean_depth_m", ""),
                "min_depth_m": info.get("min_depth_m", ""),
                "max_depth_m": info.get("max_depth_m", ""),
                "velocity_norm": info.get("velocity_norm", ""),
                "vx": float(velocity[0]),
                "vy": float(velocity[1]),
                "vz": float(velocity[2]),
                "wx": float(velocity[3]),
                "wy": float(velocity[4]),
                "wz": float(velocity[5]),
            })


def history_for_json(history):
    rows = []
    for item in history:
        rows.append({
            "iteration": int(item["iteration"]),
            "T_world_cam": np.asarray(
                item["T_world_cam"],
                dtype=np.float32,
            ).tolist(),
            "next_T_world_cam": np.asarray(
                item["next_T_world_cam"],
                dtype=np.float32,
            ).tolist(),
            "velocity": np.asarray(item["velocity"], dtype=np.float32).tolist(),
            "num_matches": item.get("num_matches"),
            "num_inliers": item.get("num_inliers"),
            "feature_mode": item.get("feature_mode"),
            "translation_error_m": item.get("translation_error_m"),
            "rotation_error_deg": item.get("rotation_error_deg"),
            "visualization_path": item.get("visualization_path"),
            "controller_info": item.get("controller_info", {}),
        })
    return rows


def next_available_dir(path):
    path = Path(path)
    if not path.exists():
        return path

    index = 1
    while True:
        candidate = path.with_name(f"{path.name}_run-{index:03d}")
        if not candidate.exists():
            return candidate
        index += 1


def make_output_dir(renderer, start_index, target_index, depth_mode):
    base_dir = RUNS_ROOT / renderer.upper() / CONTROLLER_DIR
    if RUN_NAME is not None:
        return next_available_dir(base_dir / RUN_NAME)
    name = (
        f"idx-{int(start_index):06d}_to_idx-{int(target_index):06d}_"
        f"{depth_mode}_ratio-{RATIO}"
    )
    return next_available_dir(base_dir / name)


def camera_metadata(camera):
    return {
        "fx": camera.fx,
        "fy": camera.fy,
        "cx": camera.cx,
        "cy": camera.cy,
        "height": camera.H,
        "width": camera.W,
    }


def experiment_config(start_index, target_index, start_frame, target_frame, output_dir):
    return {
        "script": str(Path(__file__).resolve()),
        "project_root": str(PROJECT_ROOT),
        "runs_root": str(RUNS_ROOT),
        "output_dir": str(output_dir),
        "controller": "GeometricFeatureController",
        "controller_dir": CONTROLLER_DIR,
        "renderer": RENDERER,
        "scene_dir": str(SCENE_DIR),
        "frame_selection": "logical_index",
        "frame_index_base": 1,
        "start_index": int(start_index),
        "target_index": int(target_index),
        "index_away": int(INDEX_AWAY),
        "start_frame": start_frame,
        "target_frame": target_frame,
        "target_index_override": TARGET_INDEX,
        "iterations": int(ITERATIONS),
        "dt": float(DT),
        "depth_mode": DEPTH_MODE,
        "feature_method": FEATURE_METHOD,
        "ratio": int(RATIO),
        "viz_iter": int(VIZ_ITER),
        "gain": float(GAIN),
        "damping": float(DAMPING),
        "max_features": None if MAX_FEATURES is None else int(MAX_FEATURES),
        "min_features": int(MIN_FEATURES),
        "max_translation_velocity": MAX_TRANSLATION_VELOCITY,
        "max_rotation_velocity": MAX_ROTATION_VELOCITY,
        "run_name": RUN_NAME,
    }


def main():
    if DEPTH_MODE not in ("learned", "intrinsic"):
        raise ValueError("DEPTH_MODE must be 'learned' or 'intrinsic'")

    scene, frame_index = load_scene_and_frames(SCENE_DIR, RENDERER)
    start_index = int(START_INDEX)
    target_index = resolve_target_index(start_index, TARGET_INDEX, INDEX_AWAY)
    start_frame = resolve_frame_from_index(frame_index, RENDERER, start_index)
    target_frame = resolve_frame_from_index(frame_index, RENDERER, target_index)

    start = frame_index[start_frame]
    target = frame_index[target_frame]
    start_camera = start["camera"]
    target_camera = target["camera"]

    output_dir = make_output_dir(RENDERER, start_index, target_index, DEPTH_MODE)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    visualizations_dir = output_dir / "visualizations"
    logs_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir.mkdir(parents=True, exist_ok=True)

    matcher = FeatureMatcher(method=FEATURE_METHOD)
    controller = GeometricFeatureController(
        matcher=matcher,
        gain=GAIN,
        damping=DAMPING,
        max_features=MAX_FEATURES,
        min_features=MIN_FEATURES,
        max_translation_velocity=MAX_TRANSLATION_VELOCITY,
        max_rotation_velocity=MAX_ROTATION_VELOCITY,
        scene=scene,
        use_intrinsic_depth=DEPTH_MODE == "intrinsic",
        ratio=RATIO,
    )

    target_image = load_rgb(target["rgb_path"], start_camera.W, start_camera.H)
    initial_render = scene.render(start_camera)
    save_rgb(output_dir / "target.png", target_image)
    save_rgb(output_dir / "initial_render.png", initial_render)

    initial_translation_error = translation_error_m(start_camera, target_camera)
    initial_rotation_error = rotation_error_deg(start_camera, target_camera)
    target_T_world_cam = target_camera.T_world_cam.copy()

    def record_iteration_metrics(item):
        translation_error = translation_error_from_pose(
            item["T_world_cam"],
            target_T_world_cam,
        )
        rotation_error = rotation_error_from_pose(
            item["T_world_cam"],
            target_T_world_cam,
        )
        item["translation_error_m"] = translation_error
        item["rotation_error_deg"] = rotation_error

        info = item.get("controller_info", {})
        residual = info.get("residual_norm", float("nan"))
        inliers = info.get("num_inlier_matches", 0)
        print(
            f"iter {item['iteration']:04d}: "
            f"mode={info.get('feature_mode')} "
            f"error_norm={residual:.6f} "
            f"pose_distance={translation_error:.4f}m "
            f"rotation_distance={rotation_error:.4f}deg "
            f"cached={info.get('num_cached_features', 0)} "
            f"dropped={info.get('num_dropped_features', 0)} "
            f"inliers={inliers}"
        )

    result = run_servo_loop(
        scene,
        start_camera,
        target_image,
        controller,
        iterations=ITERATIONS,
        dt=DT,
        visualization_dir=visualizations_dir / "matches",
        matcher=matcher,
        feature_method=FEATURE_METHOD,
        iteration_callback=record_iteration_metrics,
        viz_iter=VIZ_ITER,
    )

    final_camera = result["camera"]
    final_render = result["rendered"]
    save_rgb(output_dir / "final_render.png", final_render)
    write_history_csv(output_dir / "history.csv", result["history"])
    write_history_csv(logs_dir / "history.csv", result["history"])
    save_error_evolution(
        result["history"],
        visualizations_dir / "error_evolution.png",
    )
    save_error_evolution(
        result["history"],
        logs_dir / "error_evolution.png",
    )

    final_translation_error = translation_error_m(final_camera, target_camera)
    final_rotation_error = rotation_error_deg(final_camera, target_camera)
    summary = {
        "config": experiment_config(
            start_index,
            target_index,
            start_frame,
            target_frame,
            output_dir,
        ),
        "renderer": RENDERER,
        "controller_dir": CONTROLLER_DIR,
        "scene_dir": str(SCENE_DIR),
        "frame_selection": "logical_index",
        "frame_index_base": 1,
        "start_index": int(start_index),
        "target_index": int(target_index),
        "index_away": int(INDEX_AWAY),
        "start_frame": start_frame,
        "target_frame": target_frame,
        "start_rgb": str(start["rgb_path"]),
        "target_rgb": str(target["rgb_path"]),
        "depth": DEPTH_MODE,
        "feature_method": FEATURE_METHOD,
        "ratio": int(RATIO),
        "viz_iter": int(VIZ_ITER),
        "logs_dir": str(logs_dir),
        "visualizations_dir": str(visualizations_dir),
        "error_evolution_plot": str(logs_dir / "error_evolution.png"),
        "iterations": ITERATIONS,
        "dt": DT,
        "camera": camera_metadata(start_camera),
        "start_T_world_cam": start_camera.T_world_cam.tolist(),
        "target_T_world_cam": target_camera.T_world_cam.tolist(),
        "final_T_world_cam": final_camera.T_world_cam.tolist(),
        "initial_translation_error_m": initial_translation_error,
        "final_translation_error_m": final_translation_error,
        "initial_rotation_error_deg": initial_rotation_error,
        "final_rotation_error_deg": final_rotation_error,
        "history": history_for_json(result["history"]),
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "config.json", "w") as f:
        json.dump(summary["config"], f, indent=2)
    with open(logs_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(logs_dir / "config.json", "w") as f:
        json.dump(summary["config"], f, indent=2)

    last = result["history"][-1] if result["history"] else {}
    info = last.get("controller_info", {})
    print(
        f"Servo {RENDERER}: index {start_index} -> {target_index} "
        f"({start_frame} -> {target_frame}), "
        f"depth={DEPTH_MODE}, ratio={RATIO}, iterations={ITERATIONS}"
    )
    print(
        f"Translation error: {initial_translation_error:.4f}m -> "
        f"{final_translation_error:.4f}m"
    )
    print(
        f"Rotation error: {initial_rotation_error:.4f}deg -> "
        f"{final_rotation_error:.4f}deg"
    )
    if info:
        print(
            f"Last iteration: {info['num_inlier_matches']} controller inliers, "
            f"residual={info['residual_norm']:.6f}, "
            f"|v|={info['velocity_norm']:.6f}"
        )
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
