# 01 — LimX Dynamics repo map

`github.com/limxdynamics` listed 57 public repos on 2026-11-25 (about 42 when this
note was first written, 2026-09-04) across several robot families (TRON1, TRON2, the
Oli and Luna humanoids) plus shared tools. This note covers the 12 whose name starts
with `tron1-` and how they stack.

## Variant codes

The simulate and deploy repos, and the Isaac Gym trainer, select a model through the
`ROBOT_TYPE` environment variable. `tron1-rl-isaaclab` does **not**: it selects by
task ID (`Isaac-Limx-SF-Blind-Flat-v0`, registered at `robots/__init__.py:74`).
`limx_rsl_rl_ppo_cfg.py:6` reads `ROBOT_TYPE`, but no line uses the value.

| Code | Meaning |
|---|---|
| `PF` | Point foot — no ankle, the leg ends in a point |
| `SF` | Sole foot — has a foot plate and an ankle pitch joint |
| `WF` | Wheel foot — the leg ends in a driven wheel |
| `WL` | Wheel-legged quadruped, 16 actuated joints (models `WL_P311D`, `WL_P311E`). It appears in `tron1-robot-description` (URDF + SRDF) and in `robot-visualization` (wheel-legged launch files) — READMEs read 2026-11-26, copies in `scripts/limx_external_2026-11-26/` — and is not a `ROBOT_TYPE` value of the simulate, deploy or training repos |

Model names: `PF_TRON1A`, `PF_TRON1B`, `SF_TRON1A`, `SF_TRON1B`, `WF_TRON1A`,
`WF_TRON1B`, plus `PF_P441A/B/C/C2`.

## The stack

```
┌─ Application ────────────────────────────────────────────┐
│  tron1-ss        SLAM + nav (Livox MID360, FAST-LIO-SAM) │
│  tron1-agent     voice agent (FunASR + Qwen3-1.7B/Ollama)│
└──────────────────────────────────────────────────────────┘
┌─ Deploy (ONNX policy -> motors) ─────────────────────────┐
│  tron1-rl-deploy-python   pure Python, no ROS            │
│  tron1-rl-deploy-ros      ROS Noetic, ros_control, C++   │
│  tron1-rl-deploy-ros2     ROS 2 Iron                     │
│  tron1-rl-deploy-arm      arm policy on SF/WF            │
└──────────────────────────────────────────────────────────┘
┌─ Train ──────────────────────────────────────────────────┐
│  tron1-rl-isaaclab   Isaac Lab + PPO   <- we use this    │
│  tron1-rl-isaacgym   legged_gym fork, Isaac Gym Preview  │
└──────────────────────────────────────────────────────────┘
┌─ Simulate ───────────────────────────────────────────────┐
│  tron1-mujoco-sim    MuJoCo, Python, no ROS              │
│  tron1-gazebo-ros    Gazebo + Noetic                     │
│  tron1-gazebo-ros2   Gazebo + Iron                       │
└──────────────────────────────────────────────────────────┘
┌─ Foundation ─────────────────────────────────────────────┐
│  limxsdk-lowlevel        SDK both controller and sim use │
│  tron1-robot-description URDF / xacro / MJCF / meshes    │
└──────────────────────────────────────────────────────────┘

shared tools: robot-visualization, robot-joystick,
              ros1-bridger, ros2-bridger, limx-cli, gradmotion-cli
```

## Which train repo

| | isaacgym | isaaclab |
|---|---|---|
| Engine | Isaac Gym Preview 4 (deprecated by NVIDIA) | Isaac Sim 4.5.0 / PhysX |
| Python | 3.8 | 3.10 |
| Base | fork of `leggedrobotics/legged_gym` | Isaac Lab manager-based API |
| Variants | PF, SF, WF (its README, 2026-11-25) | PF, SF, WF (six task IDs in `robots/__init__.py`) |
| Status | legacy | current — use this |

## Notes for me

- `tron1-rl-isaaclab` vendors its own `rsl_rl` (v2.0.2) with an added MLP encoder
  — it is **not** the same as `rsl-rl-lib` on PyPI.
- The USD assets are committed inside the repo, so `tron1-robot-description` is
  not needed for Plan A.
- Only `tron1-rl-isaaclab` is needed to train. The SDK only enters the picture at
  deploy time — see [02-sdk-vs-sim.md](02-sdk-vs-sim.md).
