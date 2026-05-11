"""Compiled trajectory of mini servo tasks across one or more datasets.

For each consecutive pair (i, i + STRIDE) in a dataset:
  - servo from the previous task's final pose to the *real* RGB at frame i+STRIDE
  - the final pose becomes the start pose of the next mini task
Append the final pose to a sim trajectory, the GT pose of frame i+STRIDE to a
GT trajectory, write both as TUM files and evaluate with evo (APE/RPE + plots).
"""

import argparse
import csv
import json
import re
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from camera import Camera
from controllers import IBVSController
from dataset import load_colmap, load_scannet
from experiment_config import (
    TRAJECTORY_CONFIG_KEYS,
    apply_config,
    format_applied_config,
    load_cli_config,
)
from features import FeatureMatcher
from main_servo_frames import (
    PROJECT_ROOT,
    RUNS_ROOT,
    camera_metadata,
    load_rgb,
    rotation_error_from_pose,
    save_rgb,
    sorted_frame_ids,
    translation_error_from_pose,
)
from scenes.gs import GSScene
from scenes.mesh import MeshScene
from servo import SimpleStopper, copy_camera_with_pose, run_servo_loop
from viz import save_side_by_side


# ---- experiment configuration ----------------------------------------------
DATASETS = ["kitchen"]  # scene folders under DATA/
RENDERER = "mesh"          # "mesh", "gs", or "nerf"
NERF_POSE_SOURCE = "scannet"  # "colmap" or "scannet" (only used when RENDERER == "nerf")
NERF_RENDER_SCALE = 0.25  # lower than 1.0 avoids full-res slow NeRF renders
MESH_PATH = "mesh.ply"  # relative to each scene dir, or absolute
MESH_POSE_SOURCE = "scannet"  # "scannet" for mesh.ply, "colmap" for COLMAP meshes
STRIDE = 1               # frames between consecutive mini-servo tasks
MINI_ITERATIONS = 10     # max iterations per mini servo
DT = 1.0
DEPTH_MODE = "intrinsic"  # "learned" or "intrinsic"
FEATURE_METHOD = "xfeat"
GAIN = 0.75
MIN_FEATURES = 3
RATIO = 1
START_INDEX = 1          # 1-based logical index of the first GT pose
MAX_PAIRS = None         # None = all pairs, int = limit number of mini tasks
EARLY_STOP_ERROR_THRESHOLD = 1e-5
EARLY_STOP_VELOCITY_GRAD_EPS = 1e-8
RPE_DELTA = 1            # frames between RPE pose pairs
RUN_TAG = "BF_XFEAT_INTRINSIC"        # optional override for the run directory name
SAVE_TASK_VIZ = True    # save side-by-side (final render | target RGB) per mini task
TASK_VIZ_EVERY = 1       # save every Nth task only (SAVE_TASK_VIZ must be True)


def frame_id_from_path(path):
    match = re.search(r"(\d+)", Path(path).name)
    if match is None:
        raise ValueError(f"Could not parse frame id from {path!r}")
    return f"frame-{int(match.group(1)):06d}"


def make_frame_index(records):
    index = {}
    for record in records:
        if len(record) == 3:
            camera, rgb_path, _ = record
        elif len(record) == 2:
            camera, rgb_path = record
        else:
            raise ValueError(f"Unexpected frame record arity: {len(record)}")
        index[frame_id_from_path(rgb_path)] = {
            "camera": camera,
            "rgb_path": rgb_path,
        }
    return index


def scale_camera(camera, scale):
    scale = float(scale)
    if scale == 1.0:
        return camera

    height = max(1, int(round(camera.H * scale)))
    width = max(1, int(round(camera.W * scale)))
    return Camera(
        camera.T_world_cam,
        camera.fx * scale,
        camera.fy * scale,
        camera.cx * scale,
        camera.cy * scale,
        height,
        width,
    )


def scale_frame_records(records, scale):
    scale = float(scale)
    if scale == 1.0:
        return records

    scaled = []
    for record in records:
        if len(record) == 3:
            camera, rgb_path, depth_path = record
            scaled.append((scale_camera(camera, scale), rgb_path, depth_path))
        elif len(record) == 2:
            camera, rgb_path = record
            scaled.append((scale_camera(camera, scale), rgb_path))
        else:
            raise ValueError(f"Unexpected frame record arity: {len(record)}")
    return scaled


def resolve_scene_asset(scene_dir, path_value):
    if path_value is None:
        return scene_dir / "mesh.ply"

    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path

    scene_path = scene_dir / path
    if scene_path.exists():
        return scene_path

    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return project_path

    return scene_path


def load_trajectory_scene_and_frames(scene_dir):
    if RENDERER == "mesh":
        pose_source = str(MESH_POSE_SOURCE).lower()
        if pose_source == "scannet":
            records = load_scannet(scene_dir)
        elif pose_source == "colmap":
            records = load_colmap(scene_dir)
        else:
            raise ValueError(
                f"MESH_POSE_SOURCE must be 'scannet' or 'colmap', got {pose_source!r}"
            )
        mesh_path = resolve_scene_asset(scene_dir, MESH_PATH)
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
        print(f"Using mesh {mesh_path} with {pose_source} poses")
        scene = MeshScene(mesh_path)
    elif RENDERER == "gs":
        records = load_colmap(scene_dir)
        scene = GSScene(scene_dir / "gs.ply")
    elif RENDERER == "nerf":
        pose_source = str(NERF_POSE_SOURCE).lower()
        if pose_source == "scannet":
            records = load_scannet(scene_dir)
        elif pose_source == "colmap":
            records = load_colmap(scene_dir)
        else:
            raise ValueError(
                f"NERF_POSE_SOURCE must be 'scannet' or 'colmap', got {pose_source!r}"
            )
        records = scale_frame_records(records, NERF_RENDER_SCALE)
        if float(NERF_RENDER_SCALE) != 1.0:
            print(f"Using NeRF render scale {float(NERF_RENDER_SCALE):g}")
        from scenes.nerf import NeRFScene
        scene = NeRFScene(scene_dir)
    else:
        raise ValueError(f"Unknown renderer {RENDERER!r}")

    frame_index = make_frame_index(records)
    if not frame_index:
        raise RuntimeError(f"No frames loaded for {RENDERER} from {scene_dir}")
    return scene, frame_index


def pose_to_tum_row(timestamp, T_world_cam):
    t = T_world_cam[:3, 3]
    R = T_world_cam[:3, :3]
    q = Rotation.from_matrix(R).as_quat()  # (x, y, z, w)
    return (
        f"{timestamp:.6f} "
        f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
        f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n"
    )


def write_tum(path, timestamps, poses):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ts, T in zip(timestamps, poses):
            f.write(pose_to_tum_row(float(ts), np.asarray(T, dtype=np.float64)))


def poses_to_evo_trajectory(timestamps, poses):
    from evo.core.trajectory import PoseTrajectory3D

    positions = np.array([np.asarray(T)[:3, 3] for T in poses], dtype=np.float64)
    quats_xyzw = np.array(
        [Rotation.from_matrix(np.asarray(T)[:3, :3]).as_quat() for T in poses],
        dtype=np.float64,
    )
    # evo expects quaternions in (w, x, y, z) order.
    quats_wxyz = quats_xyzw[:, [3, 0, 1, 2]]
    ts = np.asarray(timestamps, dtype=np.float64)
    return PoseTrajectory3D(positions, quats_wxyz, ts)


def stat_dict(metric):
    return {k: float(v) for k, v in metric.get_all_statistics().items()}


def evaluate_and_plot(timestamps, sim_poses, gt_poses, out_dir, scene_name):
    from evo.core import metrics as evo_metrics
    from evo.core import sync
    from evo.core.metrics import PoseRelation, Unit
    from evo.tools import plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traj_sim = poses_to_evo_trajectory(timestamps, sim_poses)
    traj_gt = poses_to_evo_trajectory(timestamps, gt_poses)

    traj_gt_synced, traj_sim_synced = sync.associate_trajectories(
        traj_gt, traj_sim, max_diff=1e-3,
    )
    pair = (traj_gt_synced, traj_sim_synced)

    ape_t = evo_metrics.APE(PoseRelation.translation_part)
    ape_t.process_data(pair)
    ape_r = evo_metrics.APE(PoseRelation.rotation_angle_deg)
    ape_r.process_data(pair)

    rpe_t = evo_metrics.RPE(
        PoseRelation.translation_part,
        delta=float(RPE_DELTA),
        delta_unit=Unit.frames,
        rel_delta_tol=0.0,
        all_pairs=False,
    )
    rpe_t.process_data(pair)
    rpe_r = evo_metrics.RPE(
        PoseRelation.rotation_angle_deg,
        delta=float(RPE_DELTA),
        delta_unit=Unit.frames,
        rel_delta_tol=0.0,
        all_pairs=False,
    )
    rpe_r.process_data(pair)

    metrics_out = {
        "ape_translation_m": stat_dict(ape_t),
        "ape_rotation_deg": stat_dict(ape_r),
        "rpe_translation_m": stat_dict(rpe_t),
        "rpe_rotation_deg": stat_dict(rpe_r),
        "num_poses": int(traj_gt_synced.num_poses),
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_xyz = plt.figure(figsize=(10, 8))
    ax_xyz = plot.prepare_axis(fig_xyz, plot.PlotMode.xyz)
    plot.traj(ax_xyz, plot.PlotMode.xyz, traj_gt_synced,
              style="-", color="green", label="GT")
    plot.traj(ax_xyz, plot.PlotMode.xyz, traj_sim_synced,
              style="--", color="red", label="sim")
    ax_xyz.legend()
    ax_xyz.set_title(f"{scene_name}: sim vs GT (3D)")
    fig_xyz.savefig(out_dir / "trajectory_xyz.png", dpi=120, bbox_inches="tight")
    plt.close(fig_xyz)

    fig_xy = plt.figure(figsize=(8, 8))
    ax_xy = plot.prepare_axis(fig_xy, plot.PlotMode.xy)
    plot.traj(ax_xy, plot.PlotMode.xy, traj_gt_synced,
              style="-", color="green", label="GT")
    plot.traj(ax_xy, plot.PlotMode.xy, traj_sim_synced,
              style="--", color="red", label="sim")
    ax_xy.legend()
    ax_xy.set_title(f"{scene_name}: sim vs GT (top-down xy)")
    fig_xy.savefig(out_dir / "trajectory_xy.png", dpi=120, bbox_inches="tight")
    plt.close(fig_xy)

    fig_err = plt.figure(figsize=(10, 4))
    ax_err = fig_err.add_subplot(111)
    ax_err.plot(traj_sim_synced.timestamps, ape_t.error,
                color="red", label="APE trans (m)")
    ax_err.set_xlabel("task index")
    ax_err.set_ylabel("APE trans (m)")
    ax_err.set_title(f"{scene_name}: APE translation per task")
    ax_err.grid(True, alpha=0.3)
    ax_err.legend()
    fig_err.savefig(out_dir / "ape_translation.png",
                    dpi=120, bbox_inches="tight")
    plt.close(fig_err)

    fig_rot = plt.figure(figsize=(10, 4))
    ax_rot = fig_rot.add_subplot(111)
    ax_rot.plot(traj_sim_synced.timestamps, ape_r.error,
                color="orange", label="APE rot (deg)")
    ax_rot.set_xlabel("task index")
    ax_rot.set_ylabel("APE rot (deg)")
    ax_rot.set_title(f"{scene_name}: APE rotation per task")
    ax_rot.grid(True, alpha=0.3)
    ax_rot.legend()
    fig_rot.savefig(out_dir / "ape_rotation.png",
                    dpi=120, bbox_inches="tight")
    plt.close(fig_rot)

    print(f"\n=== {scene_name} evo metrics ({metrics_out['num_poses']} poses) ===")
    for name in ("ape_translation_m", "ape_rotation_deg",
                 "rpe_translation_m", "rpe_rotation_deg"):
        stats = metrics_out[name]
        print(
            f"  {name}: "
            f"rmse={stats.get('rmse', float('nan')):.6f} "
            f"mean={stats.get('mean', float('nan')):.6f} "
            f"median={stats.get('median', float('nan')):.6f} "
            f"std={stats.get('std', float('nan')):.6f} "
            f"min={stats.get('min', float('nan')):.6f} "
            f"max={stats.get('max', float('nan')):.6f}"
        )

    return metrics_out


PER_TASK_FIELDS = [
    "task_index",
    "src_frame",
    "target_frame",
    "src_rgb",
    "target_rgb",
    "iterations_run",
    "stop_reason",
    "translation_error_m",
    "rotation_error_deg",
    "viz_path",
]


def write_per_task_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_TASK_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in PER_TASK_FIELDS})


def append_task_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_TASK_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in PER_TASK_FIELDS})


def read_per_task_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def read_tum(path):
    path = Path(path)
    timestamps = []
    poses = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0].startswith("#"):
                continue
            if len(parts) != 8:
                raise ValueError(f"Bad TUM row in {path}: {line!r}")
            ts, tx, ty, tz, qx, qy, qz, qw = (float(p) for p in parts)
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float32)
            T[:3, 3] = [tx, ty, tz]
            timestamps.append(ts)
            poses.append(T)
    return timestamps, poses


def find_latest_resumable_run(renderer, tag):
    base = RUNS_ROOT / "trajectory" / renderer
    if not base.exists():
        return None
    candidates = []
    for entry in base.iterdir():
        if not entry.is_dir():
            continue
        if not entry.name.endswith(f"_{tag}"):
            continue
        has_progress = any(
            (entry / s / "per_task_errors.csv").exists() for s in DATASETS
        )
        if has_progress:
            candidates.append(entry)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_scene(scene_name, run_root, resume=False):
    scene_dir = PROJECT_ROOT / "DATA" / scene_name
    if not scene_dir.exists():
        raise RuntimeError(f"Scene directory not found: {scene_dir}")

    scene, frame_index = load_trajectory_scene_and_frames(scene_dir)
    frame_ids = sorted_frame_ids(frame_index)
    if len(frame_ids) < 1 + STRIDE:
        raise RuntimeError(
            f"{scene_name}: only {len(frame_ids)} frames, need >= {1 + STRIDE}"
        )

    start_pos = max(0, int(START_INDEX) - 1)
    if start_pos >= len(frame_ids) - STRIDE:
        raise RuntimeError(
            f"{scene_name}: START_INDEX={START_INDEX} leaves no pairs at stride {STRIDE}"
        )

    pair_starts = list(range(start_pos, len(frame_ids) - STRIDE, STRIDE))
    if MAX_PAIRS is not None:
        pair_starts = pair_starts[: int(MAX_PAIRS)]

    initial_frame = frame_index[frame_ids[start_pos]]
    initial_camera = initial_frame["camera"]

    matcher = FeatureMatcher(method=FEATURE_METHOD)
    controller = IBVSController(
        matcher=matcher,
        gain=GAIN,
        min_features=MIN_FEATURES,
        scene=scene,
        use_intrinsic_depth=(DEPTH_MODE == "intrinsic"),
        ratio=RATIO,
    )

    scene_out = Path(run_root) / scene_name
    scene_out.mkdir(parents=True, exist_ok=True)
    viz_dir = scene_out / "visualizations" if SAVE_TASK_VIZ else None
    if viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)

    sim_tum = scene_out / "sim_traj.tum"
    gt_tum = scene_out / "gt_traj.tum"
    csv_path = scene_out / "per_task_errors.csv"

    resume_offset = 0
    sim_poses = [initial_camera.T_world_cam.copy()]
    gt_poses = [initial_camera.T_world_cam.copy()]
    timestamps = [0.0]
    per_task_rows = []

    if resume and sim_tum.exists() and gt_tum.exists() and csv_path.exists():
        prev_rows = read_per_task_csv(csv_path)
        sim_ts, sim_loaded = read_tum(sim_tum)
        gt_ts, gt_loaded = read_tum(gt_tum)
        expected = len(prev_rows) + 1
        if len(sim_loaded) != expected or len(gt_loaded) != expected:
            raise RuntimeError(
                f"{scene_name}: resume mismatch — csv rows={len(prev_rows)}, "
                f"sim_tum poses={len(sim_loaded)}, gt_tum poses={len(gt_loaded)}"
            )
        sim_poses = list(sim_loaded)
        gt_poses = list(gt_loaded)
        timestamps = list(sim_ts)
        per_task_rows = prev_rows
        resume_offset = len(prev_rows)
        print(
            f"[{scene_name}] resume: {resume_offset} tasks already done, "
            f"continuing from task {resume_offset}"
        )

    if resume_offset >= len(pair_starts):
        print(f"[{scene_name}] all {len(pair_starts)} tasks complete; running eval only")
        current_camera = copy_camera_with_pose(initial_camera, sim_poses[-1])
    else:
        current_camera = copy_camera_with_pose(initial_camera, sim_poses[-1])
        if resume_offset == 0:
            # ensure files exist so subsequent resume can find them
            write_tum(sim_tum, timestamps, sim_poses)
            write_tum(gt_tum, timestamps, gt_poses)

    for task_idx, src_pos in enumerate(pair_starts):
        if task_idx < resume_offset:
            continue
        tgt_pos = src_pos + STRIDE
        src_frame_id = frame_ids[src_pos]
        tgt_frame_id = frame_ids[tgt_pos]
        tgt_frame = frame_index[tgt_frame_id]
        target_camera = tgt_frame["camera"]

        target_image = load_rgb(
            tgt_frame["rgb_path"],
            current_camera.W,
            current_camera.H,
        )
        stopper = SimpleStopper(
            error_threshold=EARLY_STOP_ERROR_THRESHOLD,
            velocity_grad_eps=EARLY_STOP_VELOCITY_GRAD_EPS,
        )

        try:
            result = run_servo_loop(
                scene,
                current_camera,
                target_image,
                controller,
                iterations=MINI_ITERATIONS,
                dt=DT,
                visualization_dir=None,
                matcher=matcher,
                feature_method=FEATURE_METHOD,
                iteration_callback=None,
                early_stopper=stopper,
                viz_iter=0,
            )
        except Exception as e:
            traceback.print_exc()
            print(
                f"[{scene_name}] task {task_idx:04d} FAILED ({type(e).__name__}: {e}); "
                f"aborting scene. Last successful task: "
                f"{per_task_rows[-1]['task_index'] if per_task_rows else 'none'}. "
                f"Proceeding to evaluation on completed tasks."
            )
            failure_info = {
                "task_index": int(task_idx),
                "src_frame": src_frame_id,
                "target_frame": tgt_frame_id,
                "error_type": type(e).__name__,
                "error_message": str(e),
            }
            try:
                with open(scene_out / "failure.json", "w") as f:
                    json.dump(failure_info, f, indent=2)
            except OSError:
                pass
            break

        final_camera = result["camera"]
        sim_T = final_camera.T_world_cam.copy()
        gt_T = target_camera.T_world_cam.copy()

        translation_err = translation_error_from_pose(sim_T, gt_T)
        rotation_err = rotation_error_from_pose(sim_T, gt_T)

        sim_poses.append(sim_T)
        gt_poses.append(gt_T)
        timestamps.append(float(task_idx + 1))

        task_viz_path = None
        if (
            viz_dir is not None
            and TASK_VIZ_EVERY > 0
            and task_idx % int(TASK_VIZ_EVERY) == 0
        ):
            final_render = result.get("rendered")
            if final_render is None:
                final_render = scene.render(final_camera)
            task_viz_path = viz_dir / f"task_{task_idx:04d}_final_vs_target.png"
            save_side_by_side(final_render, target_image, task_viz_path)

        task_row = {
            "task_index": task_idx,
            "src_frame": src_frame_id,
            "target_frame": tgt_frame_id,
            "src_rgb": str(frame_index[src_frame_id]["rgb_path"]),
            "target_rgb": str(tgt_frame["rgb_path"]),
            "iterations_run": int(len(result["history"])),
            "stop_reason": result["stop_reason"],
            "translation_error_m": float(translation_err),
            "rotation_error_deg": float(rotation_err),
            "viz_path": str(task_viz_path) if task_viz_path is not None else "",
        }
        per_task_rows.append(task_row)

        # Incremental save so a crash mid-run is recoverable via --resume.
        write_tum(sim_tum, timestamps, sim_poses)
        write_tum(gt_tum, timestamps, gt_poses)
        append_task_row(csv_path, task_row)

        print(
            f"[{scene_name}] task {task_idx:04d}: "
            f"{src_frame_id} -> {tgt_frame_id} "
            f"iters={len(result['history'])}/{MINI_ITERATIONS} "
            f"trans_err={translation_err * 1000.0:.4f}mm "
            f"rot_err={rotation_err:.8e}deg"
        )

        # Chain: next mini task starts from the previous final pose
        # (intrinsics from target are identical across the scene).
        current_camera = copy_camera_with_pose(target_camera, sim_T)

    metrics = {}
    if len(per_task_rows) >= 1:
        try:
            metrics = evaluate_and_plot(
                timestamps, sim_poses, gt_poses, scene_out, scene_name,
            )
        except Exception as e:
            traceback.print_exc()
            print(f"[{scene_name}] evaluate_and_plot failed: {type(e).__name__}: {e}")
            metrics = {"error": f"{type(e).__name__}: {e}"}
    else:
        print(f"[{scene_name}] no tasks completed; skipping evo evaluation")
        metrics = {"error": "no_completed_tasks"}

    summary = {
        "scene": scene_name,
        "scene_dir": str(scene_dir),
        "renderer": RENDERER,
        "nerf_pose_source": NERF_POSE_SOURCE,
        "nerf_render_scale": float(NERF_RENDER_SCALE),
        "mesh_path": (
            str(resolve_scene_asset(scene_dir, MESH_PATH))
            if RENDERER == "mesh"
            else None
        ),
        "mesh_pose_source": MESH_POSE_SOURCE if RENDERER == "mesh" else None,
        "stride": int(STRIDE),
        "mini_iterations": int(MINI_ITERATIONS),
        "depth_mode": DEPTH_MODE,
        "feature_method": FEATURE_METHOD,
        "gain": float(GAIN),
        "ratio": int(RATIO),
        "start_index": int(START_INDEX),
        "max_pairs": MAX_PAIRS,
        "early_stop_error_threshold": float(EARLY_STOP_ERROR_THRESHOLD),
        "early_stop_velocity_grad_eps": float(EARLY_STOP_VELOCITY_GRAD_EPS),
        "save_task_viz": bool(SAVE_TASK_VIZ),
        "task_viz_every": int(TASK_VIZ_EVERY),
        "num_tasks": len(per_task_rows),
        "camera": camera_metadata(initial_camera),
        "sim_traj": str(sim_tum),
        "gt_traj": str(gt_tum),
        "per_task_csv": str(scene_out / "per_task_errors.csv"),
        "metrics": metrics,
    }
    with open(scene_out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        help="JSON experiment config, resolved relative to CONFIGS/ if needed.",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override a config value. May be repeated, e.g. "
            "--set renderer=mesh --set datasets=kitchen."
        ),
    )
    p.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Resume an interrupted run. With no value, auto-picks the most "
            "recent run dir matching the current tag. Pass a path to target a "
            "specific run dir."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    applied_config = load_cli_config(
        args.config,
        args.set,
        TRAJECTORY_CONFIG_KEYS,
        "trajectory",
    )
    apply_config(applied_config, globals(), TRAJECTORY_CONFIG_KEYS)
    if applied_config:
        print(f"Applied trajectory config: {format_applied_config(applied_config)}")

    tag = RUN_TAG or f"{FEATURE_METHOD}_stride{STRIDE}_iters{MINI_ITERATIONS}"

    resume = args.resume is not None
    if resume:
        if args.resume == "auto":
            run_root = find_latest_resumable_run(RENDERER, tag)
            if run_root is None:
                print(
                    f"--resume: no resumable run found under "
                    f"RUNS/trajectory/{RENDERER}/ matching tag '{tag}'. "
                    f"Starting a fresh run."
                )
                resume = False
            else:
                print(f"Resuming run: {run_root}")
        else:
            run_root = Path(args.resume).resolve()
            if not run_root.exists():
                raise FileNotFoundError(f"--resume path not found: {run_root}")
            print(f"Resuming run: {run_root}")

    if not resume:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_root = RUNS_ROOT / "trajectory" / RENDERER / f"{timestamp}_{tag}"

    run_root.mkdir(parents=True, exist_ok=True)

    overall = {}
    for scene_name in DATASETS:
        try:
            overall[scene_name] = run_scene(scene_name, run_root, resume=resume)
        except Exception as e:
            traceback.print_exc()
            overall[scene_name] = {"error": str(e)}

    with open(run_root / "trajectory_summary.json", "w") as f:
        json.dump(overall, f, indent=2)
    print(f"\nWrote {run_root}")


if __name__ == "__main__":
    main()
