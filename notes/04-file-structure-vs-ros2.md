# 04 — Where everything lives, translated from ROS 2

Written for someone whose mental model is `colcon` workspaces.

## The three-layer picture

```
Isaac Sim   = the physics + rendering engine        (pip package, in the conda env)
Isaac Lab   = the RL framework on top of it         (git clone, editable install)
tron1-rl-isaaclab = one project that uses Isaac Lab (git clone, editable install)
```

Rough ROS 2 analogy — imperfect but useful:

| ROS 2 | here |
|---|---|
| Gazebo / the binary you `apt install` | Isaac Sim |
| `ros2_control` + `nav2` — the framework you build against | Isaac Lab |
| your own `my_robot_bringup` package | `tron1-rl-isaaclab` |
| `colcon build --symlink-install` | `pip install -e` |
| `source install/setup.bash` | `conda activate tron1` |
| `package.xml` | `setup.py` / `pyproject.toml` |
| `ros2 run pkg node` finding a node via the ament index | `gym.make("Isaac-Limx-SF-Blind-Flat-v0")` finding an env via the gymnasium registry |

The biggest difference: **there is no build step and no workspace**. `pip install -e`
writes a path pointer into the env's `site-packages`; after that, editing the
source takes effect on the next `python` run. No rebuild, no re-source.

`conda activate` replaces `source setup.bash`, and one env can hold many projects
at once — no `install/` or `devel/` tree, no overlay/underlay ordering.

## "Do I have to clone Isaac Lab for every project?"

**No — clone it once.** It is a library, not a template.

```
~/rc_lab/study_tron1/
├── IsaacLab/               <- clone once, tag v2.1.0
├── tron1-rl-isaaclab/      <- project 1
├── my_next_robot/          <- project 2, imports the same isaaclab
└── ...
```

Because it was installed with `pip install -e`, `import isaaclab` resolves to that
one clone from anywhere in the `tron1` env. Every future Isaac Lab project — your
own or someone else's — just needs `pip install -e` on its own extension folder.

Two exceptions worth knowing:

1. **A project pinned to a different Isaac Lab version.** Then make a second conda
   env with its own Isaac Sim + Isaac Lab clone. Envs are the isolation unit here,
   the way separate workspaces are in ROS.
2. **You want to modify Isaac Lab itself** (add a sensor, patch a manager). Since
   the install is editable, edit the clone in place — but that change is now global
   to every project in the env, so branch it and record it.

## Isaac Lab's own layout

```
IsaacLab/
├── isaaclab.sh                 launcher: sets up Kit, then runs python
├── apps/*.kit                  which Kit extensions to load (headless vs GUI)
├── source/
│   ├── isaaclab/               core: managers, assets, sensors, terrains, sim
│   ├── isaaclab_tasks/         NVIDIA's bundled example environments
│   ├── isaaclab_assets/        robot configs NVIDIA ships (ANYmal, G1, ...)
│   ├── isaaclab_rl/            thin wrappers to rsl_rl / rl_games / skrl / sb3
│   └── isaaclab_mimic/         imitation-learning utilities
└── scripts/                    tutorials, demos, NVIDIA's own train/play
```

You will spend most reading time in `source/isaaclab/isaaclab/managers/` and
`source/isaaclab/isaaclab/envs/mdp/`.

## tron1-rl-isaaclab's layout

```
tron1-rl-isaaclab/
├── scripts/rsl_rl/
│   ├── train.py                 entry point — CLI, AppLauncher, build env, run PPO
│   ├── play.py                  load checkpoint, roll out, export .pt + .onnx
│   └── cli_args.py
├── rsl_rl/                      VENDORED PPO — LimX's fork, not PyPI's
│   └── rsl_rl/
│       ├── algorithm/ppo.py             the PPO update itself
│       ├── runner/on_policy_runner.py   collect -> update -> log loop
│       ├── modules/actor_critic.py      the networks
│       ├── modules/mlp_encoder.py       LimX's added history encoder
│       └── storage/rollout_storage.py   the rollout buffer
└── exts/bipedal_locomotion/bipedal_locomotion/
    ├── assets/
    │   ├── usd/{PF,SF,WF}_TRON1A/       robot USD files (committed)
    │   └── config/solefoot_cfg.py       ArticulationCfg: joints, PD gains, limits
    ├── tasks/locomotion/
    │   ├── robots/__init__.py           gym.register(...) — the task IDs
    │   ├── robots/limx_solefoot_env_cfg.py   per-terrain env variants
    │   ├── cfg/SF/limx_base_env_cfg.py       the big one: scene/obs/action/
    │   │                                     reward/event/termination config
    │   ├── cfg/SF/terrains_cfg.py            terrain generator settings
    │   ├── agents/limx_rsl_rl_ppo_cfg.py     PPO hyper-parameters
    │   └── mdp/
    │       ├── rewards.py                    reward function bodies
    │       ├── observations.py               observation function bodies
    │       ├── events.py                     domain randomisation bodies
    │       ├── curriculums.py
    │       └── commands/gait_command.py      the custom gait clock
    └── utils/wrappers/rsl_rl/rl_mlp_cfg.py   EncoderCfg + ONNX/JIT export
```

## Config vs implementation — the one pattern to learn

Isaac Lab splits every MDP piece in two:

- **the function** lives in `mdp/*.py` and takes `(env, ...)` returning a tensor
- **the config** lives in `cfg/SF/limx_base_env_cfg.py` as a `RewTerm` / `ObsTerm`
  / `EventTerm` naming that function plus its weight and parameters

```python
# mdp/rewards.py  — implementation
def pen_joint_power_l1(env, asset_cfg): ...

# cfg/SF/limx_base_env_cfg.py — wiring
pen_joint_power_l1 = RewTerm(func=mdp.pen_joint_power_l1, weight=-2e-5)
```

The managers (`RewardManager`, `ObservationManager`, ...) walk the config at
startup, resolve each `func`, and vectorise the calls. That is why `train.py`
can print those neat tables — they are the manager's view of the config.

Closest ROS 2 equivalent: a plugin implementation plus the YAML that loads and
parameterises it. Change a weight → config. Change what is computed → `mdp/`.

## Where to change what

| Goal | File |
|---|---|
| Reward weight | `cfg/SF/limx_base_env_cfg.py` → `RewardsCfg` |
| New reward function | write in `mdp/rewards.py`, wire up in `RewardsCfg` |
| Add/remove an observation | `ObservarionsCfg` (+ `mdp/observations.py` if new) |
| Domain randomisation, enable `push_robot` | `EventsCfg` |
| Episode length, `sim.dt`, decimation | `SFEnvCfg.__post_init__` |
| Terrain | `cfg/SF/terrains_cfg.py` + the env variant class |
| Robot PD gains, joint limits, initial pose | `assets/config/solefoot_cfg.py` |
| PPO hyper-parameters, network sizes, encoder | `agents/limx_rsl_rl_ppo_cfg.py` |
| **The PPO algorithm itself** | `rsl_rl/rsl_rl/algorithm/ppo.py` |
| The training loop / logging | `rsl_rl/rsl_rl/runner/on_policy_runner.py` |
| Network architecture beyond hidden sizes | `rsl_rl/rsl_rl/modules/actor_critic.py` |
| Register a new task ID | `tasks/locomotion/robots/__init__.py` |
| CLI flags | `scripts/rsl_rl/train.py` |

## Modifying the learning algorithm

`rsl_rl` is vendored **and** editable-installed, so it is genuinely yours to edit:

```
rsl_rl/rsl_rl/
├── algorithm/ppo.py            333 lines
│     .act()          sample action, store the transition
│     .process_env_step()  reward/done bookkeeping
│     .compute_returns()   GAE
│     .update()        the surrogate loss, KL-adaptive LR, backward pass
├── runner/on_policy_runner.py  392 lines — the outer loop and tensorboard logging
├── modules/actor_critic.py     192 lines — networks, distribution, entropy
├── modules/mlp_encoder.py      127 lines — LimX's history encoder
└── storage/rollout_storage.py  the buffer and minibatch generator
```

Concrete places to start:

- a new loss term (multi-objective, energy) → `PPO.update()` in `algorithm/ppo.py`
- change how the advantage is formed → `compute_returns()`
- a different policy head (recurrent, mixture) → `modules/actor_critic.py`
- extra scalars in tensorboard → `on_policy_runner.py`

Two rules for myself:

1. Never `pip install rsl-rl-lib` into this env — the PyPI package uses the same
   `rsl_rl` module name and would shadow this fork. Verify with
   `pip show rsl_rl` → Version 2.0.2, editable path inside the repo.
2. `tron1-rl-isaaclab` is a clone of someone else's repo. Fork it, work on a
   branch, and keep `study_tron1` for notes and results — see the README.

## Runtime gotchas that come from this layout

- `import isaaclab` in a bare `python` fails with `No module named 'omni.kit'`.
  Isaac Lab is only importable **after** the Kit app boots. Use
  `./isaaclab.sh -p script.py`, or a script whose first act is `AppLauncher(...)`.
  `train.py` / `play.py` do this, so plain `python` works for them.
- Never `source /opt/ros/jazzy/setup.bash` in a shell running Isaac — it puts
  Python 3.12 paths back on `PYTHONPATH`. The conda activate hook only cleans at
  activation time.
- Logs are written relative to the current directory, so run `train.py` from the
  `tron1-rl-isaaclab` root to keep `logs/` in one place.
