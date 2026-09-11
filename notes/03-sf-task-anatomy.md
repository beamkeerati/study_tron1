# 03 — Anatomy of `Isaac-Limx-SF-Blind-Flat-v0`

Everything below comes from the tables `train.py` prints at startup, cross-checked
against the source.

## Timing

| | Value |
|---|---|
| `sim.dt` (physics) | 0.005 s → 200 Hz |
| `decimation` | 4 |
| Policy step | 0.02 s → **50 Hz** |
| Render step | 0.04 s |
| `episode_length_s` | 20 s → 1000 policy steps max |
| `num_steps_per_env` | 24 (rollout length per PPO update) |

One PPO iteration collects `num_envs × 24` transitions. With 2048 envs that is
49 152 samples per update.

## Action — 8 dims

A single `joint_pos` term. The policy outputs joint position *offsets*; the
implicit PD actuator turns them into torques inside PhysX. The policy never emits
torque directly, which is what makes the ONNX portable to hardware where the PD
runs on the motor driver.

SF has 8 joints: 2 legs × (hip roll, hip pitch, knee, ankle pitch).

## Observations — three different vectors

This is the part worth internalising.

### `policy` — 36 dims (what ships to the robot)

| Term | Dims |
|---|---|
| `base_ang_vel` | 3 |
| `proj_gravity` | 3 |
| `joint_pos` | 8 |
| `joint_vel` | 8 |
| `last_action` | 8 |
| `gait_phase` | 2 |
| `gait_command` | 4 |

Note what is **absent**: base linear velocity, terrain height, contact forces.
An IMU plus joint encoders can produce every one of these on real hardware.
"Blind" in the task name means exactly this.

### `critic` — 227 dims (privileged, simulation only)

Adds `base_lin_vel`, joint torque/acceleration, foot linear velocity and contact
force, and — the interesting part — the randomised system parameters themselves:
`robot_mass` (10), `robot_inertia` (90), `robot_joint_stiffness` (8),
`robot_joint_damping` (8), `robot_material_properties` (27).

This is **asymmetric actor-critic**: the value function is allowed to see the
ground truth of a randomised world so its advantage estimates have lower
variance, while the actor is forced to cope with only what a real robot senses.
The critic is discarded at deployment.

### `obsHistory` — 10 × 36

The last 10 policy observations, flattened to 360 and fed to an encoder MLP
`360 → 256 → 128 → 3`. Its 3-dim latent is concatenated to the 36-dim
observation, so the actor's real input is 42:

```
actor : 42  -> 512 -> 256 -> 128 -> 8     (ELU)
critic: 230 -> 512 -> 256 -> 128 -> 1     (227 + 3 commands)
```

The encoder is LimX's addition on top of stock rsl_rl (`modules/mlp_encoder.py`),
and `EncoderCfg.output_detach=True` stops the policy gradient from flowing back
into it. A 3-dim latent from a history of proprioception is almost certainly a
**base linear velocity estimator** — the quantity the policy is not allowed to
observe directly.

## Rewards — 20 terms

| Group | Terms |
|---|---|
| Task | `keep_balance` 1.0, `rew_lin_vel_xy` 2.5, `rew_ang_vel_z` 1.0, `rew_keep_ankle_pitch_zero_in_air` 0.5, `test_gait_reward` 1.0 |
| Posture | `pen_base_height` −50, `pen_flat_orientation` −5.0, `pen_feet_distance` −100 |
| Regularity | `pen_lin_vel_z` −10, `pen_ang_vel_xy` −0.05, `pen_action_rate` −0.5, `pen_action_smoothness` −0.15 |
| Effort | `pen_joint_torque` −8e−5, `pen_joint_accel` −1e−6, `pen_joint_power_l1` −2e−5, `pen_joint_vel_l2` −5e−5 |
| Safety | `pen_joint_pos_limits` −2.0, `pen_undesired_contacts` −0.5, `pen_feet_regulation` −0.2, `pen_foot_landing_vel` −0.15 |

The effort group is the natural hook for energy-efficiency work: `pen_joint_power_l1`
is a mechanical-power L1 penalty, and its weight is three orders of magnitude
below the tracking rewards.

## Commands — 2 terms

`base_velocity` (`UniformVelocityCommand`, resampled during the episode) and
`gait_command`, a LimX custom term in `mdp/commands/gait_command.py` that drives
the 2-dim `gait_phase` clock and the 4-dim `gait_command` in the observation.
The gait is commanded, not emergent.

## Domain randomisation — `EventsCfg`

| Mode | Terms |
|---|---|
| `startup` | base mass, link mass, rigid-body mass & inertia, physics material, joint stiffness & damping, centre of mass |
| `reset` | base pose, joint states, actuator gains |
| `interval` | `push_robot` — **interval `(0.0, 0.0)`, so it is effectively off** |

Whatever `startup` randomises is exactly what shows up in the critic's privileged
observation. Note the disabled push: enabling it is a cheap first experiment.

## Terminations

`time_out` (truncation, bootstraps the value) and `base_contact` (real failure).

## PPO hyper-parameters

`lr 1e-3` with `adaptive` schedule targeting `desired_kl 0.01`, `gamma 0.99`,
`lam 0.95`, `clip 0.2`, `entropy_coef 0.01`, 5 epochs × 4 minibatches,
`max_grad_norm 1.0`, `max_iterations 15000`, `save_interval 500`, `seed 42`.

The adaptive schedule is why the printed learning rate reads `0.0000` on the
first iterations — it drops the LR until the measured KL approaches the target.

## Tasks that exist in code but are not registered

`limx_solefoot_env_cfg.py` also defines `SFBlindRoughEnvCfg`, `SFBlindStairEnvCfg`,
`SFFlatEnvCfg`, `SFRoughEnvCfg`, `SFStairEnvCfg` (+ `_PLAY` variants). Only the
Blind-Flat pair is passed to `gym.register` in `robots/__init__.py`.
Registering rough/stair terrain is a two-line change — a good first modification.
