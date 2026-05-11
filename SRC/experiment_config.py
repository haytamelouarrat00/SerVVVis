"""JSON config helpers for SERVIS experiment entrypoints."""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = PROJECT_ROOT / "CONFIGS"


TRAJECTORY_CONFIG_KEYS = {
    "datasets": "DATASETS",
    "renderer": "RENDERER",
    "nerf_pose_source": "NERF_POSE_SOURCE",
    "nerf_render_scale": "NERF_RENDER_SCALE",
    "mesh_path": "MESH_PATH",
    "mesh_pose_source": "MESH_POSE_SOURCE",
    "stride": "STRIDE",
    "mini_iterations": "MINI_ITERATIONS",
    "dt": "DT",
    "depth_mode": "DEPTH_MODE",
    "feature_method": "FEATURE_METHOD",
    "gain": "GAIN",
    "min_features": "MIN_FEATURES",
    "ratio": "RATIO",
    "start_index": "START_INDEX",
    "max_pairs": "MAX_PAIRS",
    "early_stop_error_threshold": "EARLY_STOP_ERROR_THRESHOLD",
    "early_stop_velocity_grad_eps": "EARLY_STOP_VELOCITY_GRAD_EPS",
    "rpe_delta": "RPE_DELTA",
    "run_tag": "RUN_TAG",
    "save_task_viz": "SAVE_TASK_VIZ",
    "task_viz_every": "TASK_VIZ_EVERY",
}

SERVO_FRAMES_CONFIG_KEYS = {
    "scene_dir": "SCENE_DIR",
    "renderer": "RENDERER",
    "nerf_pose_source": "NERF_POSE_SOURCE",
    "start_index": "START_INDEX",
    "index_away": "INDEX_AWAY",
    "target_index": "TARGET_INDEX",
    "iterations": "ITERATIONS",
    "dt": "DT",
    "depth_mode": "DEPTH_MODE",
    "feature_method": "FEATURE_METHOD",
    "viz_iter": "VIZ_ITER",
    "gain": "GAIN",
    "min_features": "MIN_FEATURES",
    "ratio": "RATIO",
    "run_name": "RUN_NAME",
    "early_stop_error_threshold": "EARLY_STOP_ERROR_THRESHOLD",
    "early_stop_velocity_grad_eps": "EARLY_STOP_VELOCITY_GRAD_EPS",
}

COMMON_ALIASES = {
    "feature": "feature_method",
    "matcher": "feature_method",
    "depth": "depth_mode",
    "viz_every": "viz_iter",
    "visualize_every": "viz_iter",
    "min_matches": "min_features",
}

KIND_ALIASES = {
    "trajectory": {
        "dataset": "datasets",
        "scene": "datasets",
        "scenes": "datasets",
        "iters": "mini_iterations",
        "iterations": "mini_iterations",
        "max_tasks": "max_pairs",
        "pairs": "max_pairs",
        "save_viz": "save_task_viz",
        "task_viz": "save_task_viz",
        "task_viz_stride": "task_viz_every",
        "tag": "run_tag",
        "nerf_scale": "nerf_render_scale",
        "render_scale": "nerf_render_scale",
        "mesh_file": "mesh_path",
        "mesh_ply": "mesh_path",
        "mesh_source": "mesh_pose_source",
    },
    "servo_frames": {
        "dataset": "scene_dir",
        "scene": "scene_dir",
        "target": "target_index",
        "start": "start_index",
        "away": "index_away",
        "iters": "iterations",
        "tag": "run_name",
    },
}

RENDERERS = {"mesh", "gs", "nerf"}
DEPTH_MODES = {"learned", "intrinsic"}
NERF_POSE_SOURCES = {"colmap", "scannet"}
MESH_POSE_SOURCES = {"colmap", "scannet"}

INT_KEYS = {
    "stride",
    "mini_iterations",
    "iterations",
    "min_features",
    "ratio",
    "start_index",
    "index_away",
    "viz_iter",
    "rpe_delta",
    "task_viz_every",
}
OPTIONAL_INT_KEYS = {"max_pairs", "target_index"}
FLOAT_KEYS = {
    "dt",
    "gain",
    "nerf_render_scale",
    "early_stop_error_threshold",
    "early_stop_velocity_grad_eps",
}
BOOL_KEYS = {"save_task_viz"}
OPTIONAL_STR_KEYS = {"run_tag", "run_name", "mesh_path"}


def resolve_config_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path

    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return project_path

    return CONFIG_ROOT / path


def load_config_file(path, expected_kind=None):
    config_path = resolve_config_path(path)
    with open(config_path) as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")

    kind = config.get("kind")
    if expected_kind is not None and kind not in (None, expected_kind):
        raise ValueError(
            f"Config kind must be {expected_kind!r}, got {kind!r} in {config_path}"
        )

    return config


def parse_literal(text):
    text = str(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def canonical_config_key(raw_key, kind):
    key = str(raw_key).strip().lower().replace("-", "_")
    key = COMMON_ALIASES.get(key, key)
    key = KIND_ALIASES.get(kind, {}).get(key, key)
    return key


def parse_cli_overrides(overrides, key_map, kind):
    config = {}
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(
                f"Expected --set KEY=VALUE, got {override!r}"
            )
        key, value = override.split("=", 1)
        config[key] = parse_literal(value)
    return normalize_config(config, key_map, kind)


def load_cli_config(config_path, overrides, key_map, kind):
    config = {}
    if config_path is not None:
        config.update(normalize_config(
            load_config_file(config_path, expected_kind=kind),
            key_map,
            kind,
        ))
    config.update(parse_cli_overrides(overrides, key_map, kind))
    return config


def apply_config(config, module_globals, key_map):
    for key, value in config.items():
        module_globals[key_map[key]] = value


def normalize_config(config, key_map, kind):
    normalized = {}
    for raw_key, raw_value in config.items():
        key = canonical_config_key(raw_key, kind)
        if key == "kind":
            continue
        if key not in key_map:
            allowed = ", ".join(sorted(key_map))
            raise KeyError(
                f"Unknown {kind} config key {raw_key!r}. Allowed keys: {allowed}"
            )
        normalized[key] = coerce_config_value(key, raw_value)
    return normalized


def coerce_config_value(key, value):
    if key == "datasets":
        return coerce_str_list(value)
    if key == "scene_dir":
        return coerce_scene_dir(value)
    if key in OPTIONAL_INT_KEYS:
        if is_null_value(value):
            return None
        value = int(value)
        if value < 1:
            raise ValueError(f"{key} must be >= 1 or null")
        return value
    if key in INT_KEYS:
        value = int(value)
        validate_int_value(key, value)
        return value
    if key in FLOAT_KEYS:
        value = float(value)
        if key == "dt" and value <= 0.0:
            raise ValueError("dt must be > 0")
        if key == "nerf_render_scale" and value <= 0.0:
            raise ValueError("nerf_render_scale must be > 0")
        return value
    if key in BOOL_KEYS:
        return coerce_bool(value)
    if key in OPTIONAL_STR_KEYS:
        if is_null_value(value):
            return None
        return str(value)
    if key == "renderer":
        value = str(value).lower()
        if value not in RENDERERS:
            raise ValueError(f"renderer must be one of {sorted(RENDERERS)}")
        return value
    if key == "depth_mode":
        value = str(value).lower()
        if value not in DEPTH_MODES:
            raise ValueError(f"depth_mode must be one of {sorted(DEPTH_MODES)}")
        return value
    if key == "nerf_pose_source":
        value = str(value).lower()
        if value not in NERF_POSE_SOURCES:
            raise ValueError(
                f"nerf_pose_source must be one of {sorted(NERF_POSE_SOURCES)}"
            )
        return value
    if key == "mesh_pose_source":
        value = str(value).lower()
        if value not in MESH_POSE_SOURCES:
            raise ValueError(
                f"mesh_pose_source must be one of {sorted(MESH_POSE_SOURCES)}"
            )
        return value
    if key == "feature_method":
        return str(value).lower()
    return value


def coerce_str_list(value):
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    elif isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value]
    else:
        raise ValueError("datasets must be a string or list of strings")

    items = [item for item in items if item]
    if not items:
        raise ValueError("datasets must not be empty")
    return items


def coerce_scene_dir(value):
    if value is None:
        raise ValueError("scene_dir must not be null")

    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    if len(path.parts) == 1:
        return PROJECT_ROOT / "DATA" / path
    return PROJECT_ROOT / path


def coerce_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected boolean value, got {value!r}")


def is_null_value(value):
    return value is None or str(value).strip().lower() in {"", "none", "null"}


def validate_int_value(key, value):
    if key in {"start_index", "min_features", "stride", "rpe_delta"} and value < 1:
        raise ValueError(f"{key} must be >= 1")
    if key in {
        "mini_iterations",
        "iterations",
        "ratio",
        "viz_iter",
        "task_viz_every",
    } and value < 0:
        raise ValueError(f"{key} must be >= 0")


def format_applied_config(config):
    if not config:
        return "none"
    return ", ".join(f"{key}={value!r}" for key, value in sorted(config.items()))
