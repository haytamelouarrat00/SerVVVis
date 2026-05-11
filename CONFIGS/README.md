# SERVIS Experiment Configs

Experiment scripts now read JSON config files and accept one-off overrides.

Run these commands from the same Python environment you use for SERVIS. For
example, if that environment is `viservo`, prefix commands with
`conda run -n viservo`.

Run a trajectory from the repo root:

```bash
python SRC/main_trajectory.py --config trajectory_kitchen_mesh.json
python SRC/main_trajectory.py --config trajectory_kitchen_mesh.json --set renderer=mesh --set datasets=kitchen --set iterations=30
```

Run a single frame-to-frame servo trial:

```bash
python SRC/main_servo_frames.py --config servo_kitchen_mesh.json
python SRC/main_servo_frames.py --config servo_kitchen_mesh.json --set scene=kitchen --set start=1 --set target=2
```

Start the phone bot:

```bash
cp CONFIGS/bot.env.example CONFIGS/bot.env
# edit CONFIGS/bot.env with your Telegram token and chat id
python SRC/phone_bot.py --env CONFIGS/bot.env
```

If the bot is started outside the SERVIS environment, set
`SERVIS_BOT_PYTHON` in `CONFIGS/bot.env` to the environment's Python
executable. You can find it with:

```bash
conda run -n viservo python -c "import sys; print(sys.executable)"
```

Bot commands:

```text
/whoami
/trajectory mesh kitchen stride=1 iterations=30 gain=0.75 depth=learned max_pairs=20
/servo mesh kitchen start=1 target=2 iterations=100 depth=intrinsic feature=sift
/status
/tail 40
/cancel
```

Use environment variables only for bot secrets and process settings. Keep experiment choices in JSON configs or `--set` overrides so local runs and phone-triggered runs use the same path.
