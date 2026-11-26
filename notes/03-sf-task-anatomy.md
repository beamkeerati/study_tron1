# 03 — Anatomy of `Isaac-Limx-SF-Blind-Flat-v0`

Everything below comes from the tables the Isaac Lab managers print when `train.py` builds the environment, cross-checked
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
torque directly, which is what lets the same ONNX drive any receiver that evaluates
PD from `q`, `Kp`, `Kd`. The MuJoCo simulator is one (`simulator.py:79-83` in
`scripts/deploy_snapshot_2026-11-26/`); where the real robot evaluates it is not
readable from these repos.

SF has 8 joints: 2 legs × (hip roll, hip pitch, knee, ankle pitch).

## Observations — three different vectors

This is the part worth internalising.

### `policy` — 36 dims (what the exported policy reads)

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
An IMU plus joint encoders can produce the 22 measured values on real hardware; the
other 14 (`last_action`, `gait_phase`, `gait_command`) are the program's own.
"Blind" in the task name refers to the missing terrain height only
(`limx_solefoot_env_cfg.py:76-78` removes the height scanner from the scene and sets `heights` to `None` in both groups; only the critic group defines that term, `limx_base_env_cfg.py:201`); the
missing velocity and contact terms are the actor/critic split.

### `critic` — 227 dims (privileged: not given to the actor, not exported)

Adds `base_lin_vel`, joint torque/acceleration, foot linear velocity and contact
force, and five terms named after the robot's physical parameters: `robot_mass`
(10), `robot_inertia` (90), `robot_joint_pos` (8), `robot_joint_stiffness` (8),
`robot_joint_damping` (8), plus `robot_material_properties` (27).

**Those names are misleading.** The first five read Isaac Lab's `default_*`
buffers (`observations.py:43`, `:50`, `:58`, `:65`, `:72`), which are set once at
load: mass and inertia from the USD (`articulation.py:1239-1240`), gains from the
actuator config (`:1379-1380`, i.e. `solefoot_cfg.py:50-51`, `:61-62`) and joint
positions from `init_state` (`:1286-1289`). The
randomisation events read those buffers and write the sampled values into PhysX
(`envs/mdp/events.py:323`/`:333` for mass, `:512`/`:516` for stiffness); they
never write the buffers back. So the critic gets the *nominal* mass, inertia and
gains — the same 124 numbers in every env at every step — not the randomised
ones. Only `robot_material_properties` (27) reads PhysX directly
(`observations.py:95`) and really differs per env. Measured on the running task,
16 envs × 2000 steps, 2026-11-26 (`scripts/workbook_env_probe.py`): the spread of
those 124 columns is exactly 0 while the simulated total mass ranges 18.1–23.6 kg.

This is still **asymmetric actor-critic** — the value function sees
`base_lin_vel`, contact forces and the other true state the actor cannot — but
of the randomised world it sees only friction and restitution, not mass or
gains. The critic is discarded at deployment.

### `obsHistory` — 10 × 36

The last 10 policy observations, flattened to 360 and fed to an encoder MLP
`360 → 256 → 128 → 3`. Its 3-dim latent is concatenated with the 36-dim
observation and the 3 values of the `commands` group (`velocity_commands`,
`limx_base_env_cfg.py:226-232`), so the actor's real input is 3 + 36 + 3 = 42
(`ppo.py:137-139`):

```
actor : 42  -> 512 -> 256 -> 128 -> 8     (ELU)
critic: 230 -> 512 -> 256 -> 128 -> 1     (227 + 3 commands)
```

The encoder is LimX's addition on top of stock rsl_rl (`modules/mlp_encoder.py`),
and `EncoderCfg.output_detach=True` cuts the graph between the policy loss and the
encoder (its parameters are not in the main optimizer either, `ppo.py:85`). The
latent is a **base linear velocity estimator**: the encoder's only loss is the MSE
against the first three critic values, `base_lin_vel` (`ppo.py:313-315`) — the
quantity the policy is not allowed to observe directly.

## Rewards — 20 terms

| Group | Terms |
|---|---|
| Task | `keep_balance` 1.0, `rew_lin_vel_xy` 2.5, `rew_ang_vel_z` 1.0, `rew_keep_ankle_pitch_zero_in_air` 0.5 |
| Gait (positive weight, **penalty in effect**) | `test_gait_reward` 1.0 |
| Posture | `pen_base_height` −50, `pen_flat_orientation` −5.0, `pen_feet_distance` −100 |
| Regularity | `pen_lin_vel_z` −10, `pen_ang_vel_xy` −0.05, `pen_action_rate` −0.5, `pen_action_smoothness` −0.15 |
| Effort | `pen_joint_torque` −8e−5, `pen_joint_accel` −1e−6, `pen_joint_power_l1` −2e−5, `pen_joint_vel_l2` −5e−5 |
| Safety | `pen_joint_pos_limits` −2.0, `pen_undesired_contacts` −0.5, `pen_feet_regulation` −0.2, `pen_foot_landing_vel` −0.15 |

`test_gait_reward` does not belong with the task rewards even though its weight
is +1.0: the config sets both of its internal scales to −2.0
(`limx_base_env_cfg.py:434-435`), and with a negative scale each branch returns
a non-negative quantity times that scale (`rewards.py:466`, `:478`). The term is
therefore never above zero — it penalises feet that are off the commanded gait,
and the best it can give is 0. In the 64-env × 218-iteration run of 2026-09-04
its logged episode value is negative (−0.013 in the logger's unit, episode sum
÷ 20 s, mean of the last 20 iterations).

The effort group is the natural hook for energy-efficiency work: `pen_joint_power_l1`
is a mechanical-power L1 penalty, and its weight is about five orders of magnitude
below the tracking weights (2e-5 against 1.0 and 2.5).

## Commands — 2 terms

`base_velocity` (`UniformVelocityCommand`, resampled during the episode) and
`gait_command`, a LimX custom term in `mdp/commands/gait_command.py` that drives
the 2-dim `gait_phase` clock and the 4-dim `gait_command` in the observation.
The gait is commanded, not emergent.

## Domain randomisation — `EventsCfg`

| Mode | Terms |
|---|---|
| `startup` | base mass (−1…+2 kg — `limx_solefoot_env_cfg.py:42` overrides the ±5 of the base file), link mass ×0.8–1.2, rigid-body mass & inertia ×0.8–1.2, physics material, joint stiffness & damping, centre of mass ±0.075 m |
| `reset` | base pose, joint states, actuator gains ×0.5–2.0 (log-uniform) |
| `interval` | `push_robot` — `interval_range_s=(0.0, 0.0)`, **which means "every step", not "off"** |

Three things here are not what they look like.

- **The push is on.** With a zero interval the event manager's countdown is
  below zero on every step, so it calls the term on every step for every env
  (`event_manager.py:207-209`, `:222-229`). The real rate is the term's own
  `probability=0.002` (`events.py:45-47`): one push per env every 500 steps
  (10 s) on average, lasting one policy step, up to ±500 N in x and y. Measured:
  2000 calls in 2000 steps, 75 pushes over 16 envs (64 expected). The `_PLAY`
  variant removes it (`limx_solefoot_env_cfg.py:61`). An earlier version of this
  note called the push "effectively off"; that was wrong.
- **The startup gains never reach a rollout.** `randomize_actuator_gains` always
  starts from the default gains (`envs/mdp/events.py:512`, `:520`), and the
  `reset` term runs for every env before its first step, so the `startup` draw
  (kp 32–48, kd 2–3) is overwritten. What the policy trains on is
  kp 45 × [0.5, 2] = 22.5–90 and kd 1.5 × [0.5, 2] = 0.75–3.0 (legs),
  0.8 × [0.5, 2] = 0.4–1.6 (ankles). Measured after the first reset:
  kp 22.98–89.30.
- **The critic does not see most of it** — see the `critic` section above: only
  the material properties are observed; mass, inertia and gains are nominal.

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
Registering rough/stair terrain takes one more `gym.register` block per task (9 lines each, `robots/__init__.py:73-81`) — a good first modification, not tried here.
