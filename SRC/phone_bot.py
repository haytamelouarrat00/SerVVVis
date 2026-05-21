"""Phone-facing Telegram bot for launching allowlisted SERVIS experiments.

Secrets live in the environment. Experiment parameters live in CONFIGS/*.json.
The bot never executes raw shell text from chat messages.
"""

import argparse
from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from urllib import request
from urllib.error import URLError

from experiment_config import (
    CONFIG_ROOT,
    DEPTH_MODES,
    PROJECT_ROOT,
    RENDERERS,
    SERVO_FRAMES_CONFIG_KEYS,
    TRAJECTORY_CONFIG_KEYS,
    canonical_config_key,
    parse_cli_overrides,
    resolve_config_path,
)


RUNS_ROOT = PROJECT_ROOT / "RUNS"
BOT_JOBS_ROOT = RUNS_ROOT / "bot_jobs"

HELP_TEXT = """Commands:
/trajectory [mesh|gs|nerf] [dataset] [key=value ...]
/servo [mesh|gs|nerf] [dataset] [key=value ...]
/status
/tail [lines]
/cancel
/whoami

Examples:
/trajectory mesh kitchen stride=1 iterations=30 gain=0.75 depth=learned max_pairs=20
/servo mesh kitchen start=1 target=2 iterations=100 depth=intrinsic feature=sift

Use config=name.json to pick another JSON file in CONFIGS/.
"""


class BotError(Exception):
    pass


class JobManager:
    def __init__(self, python_executable):
        self.python_executable = python_executable
        self.process = None
        self.kind = None
        self.log_path = None
        self.command = None
        self.started_at = None
        self.returncode = None

    def poll(self):
        if self.process is None:
            return None
        self.returncode = self.process.poll()
        return self.returncode

    def is_running(self):
        return self.process is not None and self.poll() is None

    def start(self, kind, command, metadata):
        if self.is_running():
            raise BotError(
                f"A {self.kind} job is already running with pid {self.process.pid}"
            )

        BOT_JOBS_ROOT.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_dir = BOT_JOBS_ROOT / f"{timestamp}_{kind}"
        job_dir.mkdir(parents=True, exist_ok=False)
        log_path = job_dir / "run.log"
        command_path = job_dir / "command.json"

        with open(command_path, "w") as f:
            json.dump({
                "kind": kind,
                "command": command,
                "cwd": str(PROJECT_ROOT),
                "metadata": metadata,
            }, f, indent=2)

        log_file = open(log_path, "ab", buffering=0)
        try:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_file.close()

        self.process = process
        self.kind = kind
        self.log_path = log_path
        self.command = command
        self.started_at = datetime.now()
        self.returncode = None
        return process.pid, log_path

    def cancel(self):
        if not self.is_running():
            return "No active job to cancel."

        pid = self.process.pid
        try:
            os.killpg(pid, signal.SIGTERM)
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(pid, signal.SIGKILL)
            self.process.wait(timeout=5)
        self.returncode = self.process.returncode
        return f"Cancelled {self.kind} job pid={pid}, returncode={self.returncode}."

    def status_text(self):
        if self.process is None:
            return "No job has been started in this bot process."

        returncode = self.poll()
        elapsed = ""
        if self.started_at is not None:
            seconds = int((datetime.now() - self.started_at).total_seconds())
            elapsed = f", elapsed={seconds}s"

        if returncode is None:
            state = f"running pid={self.process.pid}"
        else:
            state = f"finished returncode={returncode}"
        return (
            f"{self.kind}: {state}{elapsed}\n"
            f"log={self.log_path}"
        )


def load_env_file(path):
    if path is None:
        return
    path = Path(path).expanduser()
    if not path.exists():
        return
    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            os.environ.setdefault(key, value)


def read_allowed_chat_ids():
    if os.environ.get("SERVIS_BOT_ALLOW_ALL") == "1":
        return None
    text = os.environ.get("SERVIS_BOT_ALLOWED_CHAT_IDS", "").strip()
    if not text:
        raise SystemExit(
            "Set SERVIS_BOT_ALLOWED_CHAT_IDS, or temporarily set "
            "SERVIS_BOT_ALLOW_ALL=1 and use /whoami to get your chat id."
        )
    return {item.strip() for item in text.split(",") if item.strip()}


def api_request(token, method, payload, timeout=35):
    url = f"https://api.telegram.org/bot{token}/{method}"
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not result.get("ok"):
        raise BotError(f"Telegram API error: {result}")
    return result["result"]


def send_message(token, chat_id, text):
    if len(text) > 3900:
        text = text[:3900] + "\n... truncated"
    return api_request(
        token,
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": True,
        },
        timeout=20,
    )


def get_updates(token, offset):
    payload = {"timeout": 30}
    if offset is not None:
        payload["offset"] = offset
    return api_request(token, "getUpdates", payload, timeout=35)


def resolve_bot_config(value):
    path = resolve_config_path(value).resolve()
    config_root = CONFIG_ROOT.resolve()
    if config_root not in (path, *path.parents):
        raise BotError("Bot config paths must stay under CONFIGS/.")
    if not path.exists():
        raise BotError(f"Config not found: {path}")
    return path


def parse_experiment_args(kind, args_text):
    if kind == "trajectory":
        key_map = TRAJECTORY_CONFIG_KEYS
        default_config = os.environ.get(
            "SERVIS_TRAJECTORY_CONFIG",
            "trajectory_kitchen_mesh.json",
        )
        default_scene_key = "datasets"
    elif kind == "servo":
        key_map = SERVO_FRAMES_CONFIG_KEYS
        default_config = os.environ.get(
            "SERVIS_SERVO_CONFIG",
            "servo_kitchen_mesh.json",
        )
        default_scene_key = "scene_dir"
    else:
        raise BotError(f"Unknown experiment kind: {kind}")

    config_path = resolve_bot_config(default_config)
    overrides = {}

    try:
        tokens = shlex.split(args_text)
    except ValueError as exc:
        raise BotError(f"Could not parse command: {exc}") from exc

    for token in tokens:
        if "=" in token:
            raw_key, value = token.split("=", 1)
            if raw_key.strip().lower().replace("-", "_") == "config":
                config_path = resolve_bot_config(value)
                continue
            validation_kind = "servo_frames" if kind == "servo" else kind
            key = canonical_config_key(raw_key, validation_kind)
            if key not in key_map:
                raise BotError(f"Unknown {kind} parameter: {raw_key}")
            overrides[key] = value
            continue

        low = token.lower()
        if low in RENDERERS:
            overrides["renderer"] = low
        elif low in DEPTH_MODES:
            overrides["depth_mode"] = low
        elif default_scene_key not in overrides:
            overrides[default_scene_key] = token
        else:
            raise BotError(f"Could not place argument {token!r}; use key=value.")

    set_args = [f"{key}={value}" for key, value in overrides.items()]
    validation_kind = "servo_frames" if kind == "servo" else kind
    parse_cli_overrides(set_args, key_map, validation_kind)
    return config_path, set_args, overrides


def build_command(kind, args_text, python_executable):
    config_path, set_args, overrides = parse_experiment_args(kind, args_text)
    cli_script = PROJECT_ROOT / "SRC" / "cli.py"
    subcommand = "trajectory" if kind == "trajectory" else "servo-frames"

    command = [
        python_executable,
        str(cli_script),
        subcommand,
        "--config",
        str(config_path),
    ]
    for set_arg in set_args:
        command.extend(["--set", set_arg])

    return command, {
        "config": str(config_path),
        "overrides": overrides,
    }


def tail_file(path, line_count):
    if path is None or not Path(path).exists():
        return "No log file yet."
    with open(path, errors="replace") as f:
        lines = deque(f, maxlen=max(1, int(line_count)))
    text = "".join(lines).strip()
    return text or "Log is empty."


def split_command(text):
    parts = text.strip().split(None, 1)
    if not parts:
        return "", ""
    command = parts[0].split("@", 1)[0].lower()
    args_text = parts[1] if len(parts) > 1 else ""
    return command, args_text


def handle_message(token, manager, allowed_chat_ids, message):
    chat = message.get("chat", {})
    chat_id = str(chat.get("id"))
    text = message.get("text", "")
    if not text:
        return

    command, args_text = split_command(text)
    if allowed_chat_ids is not None and chat_id not in allowed_chat_ids:
        send_message(token, chat_id, f"Not authorized. chat_id={chat_id}")
        return

    try:
        if command in {"/help", "/start"}:
            reply = HELP_TEXT
        elif command == "/whoami":
            reply = f"chat_id={chat_id}"
        elif command in {"/trajectory", "/traj"}:
            cmd, metadata = build_command(
                "trajectory",
                args_text,
                manager.python_executable,
            )
            pid, log_path = manager.start("trajectory", cmd, metadata)
            reply = (
                f"Started trajectory job pid={pid}\n"
                f"config={metadata['config']}\n"
                f"overrides={metadata['overrides'] or 'none'}\n"
                f"log={log_path}"
            )
        elif command in {"/servo", "/frames"}:
            cmd, metadata = build_command(
                "servo",
                args_text,
                manager.python_executable,
            )
            pid, log_path = manager.start("servo", cmd, metadata)
            reply = (
                f"Started servo job pid={pid}\n"
                f"config={metadata['config']}\n"
                f"overrides={metadata['overrides'] or 'none'}\n"
                f"log={log_path}"
            )
        elif command == "/run":
            tokens = shlex.split(args_text)
            if not tokens:
                raise BotError("Use /run trajectory ... or /run servo ...")
            kind = tokens[0].lower()
            if kind in {"traj", "trajectory"}:
                kind = "trajectory"
            elif kind in {"frame", "frames", "servo"}:
                kind = "servo"
            else:
                raise BotError("Use /run trajectory ... or /run servo ...")
            cmd, metadata = build_command(
                kind,
                " ".join(shlex.quote(token) for token in tokens[1:]),
                manager.python_executable,
            )
            pid, log_path = manager.start(kind, cmd, metadata)
            reply = (
                f"Started {kind} job pid={pid}\n"
                f"config={metadata['config']}\n"
                f"overrides={metadata['overrides'] or 'none'}\n"
                f"log={log_path}"
            )
        elif command == "/status":
            reply = manager.status_text()
        elif command == "/tail":
            line_count = int(args_text.strip() or "30")
            reply = tail_file(manager.log_path, line_count)
        elif command == "/cancel":
            reply = manager.cancel()
        else:
            reply = HELP_TEXT
    except Exception as exc:
        reply = f"Error: {type(exc).__name__}: {exc}"

    send_message(token, chat_id, reply)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--env",
        default=str(CONFIG_ROOT / "bot.env"),
        help="Optional KEY=VALUE env file for bot secrets/settings.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    load_env_file(args.env)

    token = os.environ.get("SERVIS_BOT_TOKEN")
    if not token:
        raise SystemExit("Set SERVIS_BOT_TOKEN or put it in CONFIGS/bot.env")

    allowed_chat_ids = read_allowed_chat_ids()
    python_executable = os.environ.get("SERVIS_BOT_PYTHON", sys.executable)
    manager = JobManager(python_executable)
    offset = None

    print("SERVIS phone bot is running.")
    while True:
        try:
            updates = get_updates(token, offset)
            for update in updates:
                offset = int(update["update_id"]) + 1
                message = update.get("message") or update.get("edited_message")
                if message is not None:
                    handle_message(token, manager, allowed_chat_ids, message)
        except URLError as exc:
            print(f"Telegram network error: {exc}")
            time.sleep(5)
        except KeyboardInterrupt:
            print("Stopping bot.")
            break


if __name__ == "__main__":
    main()
