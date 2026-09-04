# 01 — LimX Dynamics repo map

`github.com/limxdynamics` has ~42 repos across four robot families (TRON1, TRON2,
Oli humanoid, VLA) plus shared tools. This note covers the 12 TRON1 ones and how
they stack.

## Variant codes

Every repo selects a model through the `ROBOT_TYPE` environment variable.

| Code | Meaning |
|---|---|
| `PF` | Point foot — no ankle, the leg ends in a point |
| `SF` | Sole foot — has a foot plate and an ankle pitch joint |
| `WF` | Wheel foot — the leg ends in a driven wheel |
| `WL` | Wheel-legged (older `P311D` / `P311E` platform) |

Model names: `PF_TRON1A`, `PF_TRON1B`, `SF_TRON1A`, `SF_TRON1B`, `WF_TRON1A`,
`WF_TRON1B`, plus the older `PF_P441A/B/C/C2`.

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
│  limxsdk-lowlevel        network client library          │
│  tron1-robot-description URDF / xacro / MJCF / meshes    │
└──────────────────────────────────────────────────────────┘

shared tools: robot-visualization, robot-joystick,
              ros1-bridger, ros2-bridger, limx-cli, gradmotion-cli
```

## Which train repo

| | isaacgym | isaaclab |
|---|---|---|
| Engine | Isaac Gym Preview 3 (discontinued) | Isaac Sim / PhysX |
| Python | 3.8 | 3.10 |
| Base | fork of `leggedrobotics/legged_gym` | Isaac Lab manager-based API |
| Variants | PF + wheel-legged | PF, SF, WF |
| Status | legacy | current — use this |

## Notes for me

- `tron1-rl-isaaclab` vendors its own `rsl_rl` (v2.0.2) with an added MLP encoder
  — it is **not** the same as `rsl-rl-lib` on PyPI.
- The USD assets are committed inside the repo, so `tron1-robot-description` is
  not needed for Plan A.
- Only `tron1-rl-isaaclab` is needed to train. The SDK only enters the picture at
  deploy time — see [02-sdk-vs-sim.md](02-sdk-vs-sim.md).
