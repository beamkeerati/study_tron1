# study_tron1

Setup notes and learning log for the **LimX Dynamics TRON1** bipedal robot, using
Isaac Lab for reinforcement learning training (Plan A: train-only, no robot SDK).

Verified working on this machine: **2026-09-04**

---

## 1. Target machine

| | Value |
|---|---|
| OS | Ubuntu 24.04.4 LTS (glibc 2.39) |
| GPU | NVIDIA RTX 4070 SUPER — 12 GB VRAM |
| Driver | 580.173.02 (CUDA 13.0) |
| CPU | Intel Core Ultra 7 265KF (20 threads) |
| RAM | 30 GB |
| Disk | ~30 GB needed for the full install |
| Other | conda (anaconda3), ROS 2 Jazzy in a separate `ros_env` conda env |

> Isaac Sim officially supports Ubuntu 22.04 only. 24.04 works, but is unofficial —
> the extra system packages in step 3 are what make it work.

## 2. Version pins (do not "upgrade to latest")

`tron1-rl-isaaclab` declares `Isaac Sim :: 4.5.0` and `python_requires >= 3.10`
in `exts/bipedal_locomotion/setup.py`. Installing Isaac Sim 5.x / Python 3.11
puts you on a different manager-term API.

| Component | Pinned version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.5.1 + cu121 |
| NumPy | 1.26.4 (Isaac Sim 4.5 requires `numpy < 2`) |
| Isaac Sim | `isaacsim[all,extscache]==4.5.0` |
| Isaac Lab | git tag `v2.1.0` |
| setuptools | `< 81` (see step 8) |

---

## 3. System packages

```bash
sudo apt update && sudo apt install -y \
  vulkan-tools cmake build-essential \
  libglu1-mesa libxt6 libxrender1 libxcb-cursor0 libxkbcommon-x11-0
```

Verify Vulkan reaches the NVIDIA GPU — this must pass before anything else:

```bash
vulkaninfo --summary | grep -iE "driverName|deviceName"
```

Expected (a second `llvmpipe` device is the normal software fallback):

```
deviceName = NVIDIA GeForce RTX 4070 SUPER
driverName = NVIDIA
```

## 4. Fix conda TLS (only needed on networks with TLS inspection)

Symptom: `CondaSSLError ... CERTIFICATE_VERIFY_FAILED: Missing Authority Key Identifier`

```bash
conda config --set ssl_verify truststore
```

pip is unaffected.

## 5. Create the conda environment

`--override-channels` is required: this machine has `robostack-jazzy` as a global
channel, which would otherwise pull ROS packages into the new env.

```bash
conda deactivate                       # leave ros_env
conda create -n tron1 python=3.10 libstdcxx-ng --override-channels -c conda-forge -y
conda activate tron1
pip install --upgrade pip
```

## 6. Isolate the env from ROS  ⚠️ do this before installing anything

`~/.bashrc` sources `otto_ws` / `uros_ws` / `ros_env`, which leaves Python 3.12
paths in `PYTHONPATH` and ros_env's Ogre/rviz libs in `LD_LIBRARY_PATH` in
**every** shell. Without this fix, pip silently skips packages it thinks are
already installed, and `import torch` dies with
`No module named 'numpy._core._multiarray_umath'`.

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/zz_isolate.sh" <<'EOF'
export _OLD_PYTHONPATH="${PYTHONPATH:-}"
export _OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
unset PYTHONPATH
unset LD_LIBRARY_PATH
EOF

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/zz_isolate.sh" <<'EOF'
[ -n "${_OLD_PYTHONPATH:-}" ] && export PYTHONPATH="$_OLD_PYTHONPATH"
[ -n "${_OLD_LD_LIBRARY_PATH:-}" ] && export LD_LIBRARY_PATH="$_OLD_LD_LIBRARY_PATH"
unset _OLD_PYTHONPATH _OLD_LD_LIBRARY_PATH
true
EOF

conda deactivate && conda activate tron1
```

Check — all three must be clean:

```bash
echo "PYTHONPATH=[$PYTHONPATH]"          # []
echo "LD_LIBRARY_PATH=[$LD_LIBRARY_PATH]" # []
python -c "import sys; print([p for p in sys.path if 'ros_env' in p or 'otto_ws' in p] or 'CLEAN')"
```

The hook only cleans at activation time. Sourcing a ROS `setup.bash` afterwards
re-pollutes the shell — open a fresh terminal instead.

## 7. PyTorch

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy==1.26.4"

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 2.5.1+cu121 True NVIDIA GeForce RTX 4070 SUPER
```

`cu121` is correct even on driver 580 — the CUDA runtime ships inside the wheel
and new drivers run older runtimes. Isaac Sim 4.5.0 is built against cu121; do
not "upgrade" to cu124/cu128.

## 8. Isaac Sim 4.5.0

~16 GB, 20-40 minutes.

```bash
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
pip install requests          # else isaacsim.asset.browser fails to start
```

First launch compiles the shader cache (5-15 min, do not Ctrl+C):

```bash
export OMNI_KIT_ACCEPT_EULA=YES

python -c "
from isaacsim import SimulationApp
app = SimulationApp({'headless': True})
from isaacsim.core.api import World
w = World(stage_units_in_meters=1.0)
w.scene.add_default_ground_plane()
w.reset()
for _ in range(10): w.step(render=False)
print('>>> ISAACSIM OK')
app.close()
"
```

Isaac Sim lives in the conda env (`$CONDA_PREFIX/lib/python3.10/site-packages/isaacsim`),
not in this repo. Caches go to `~/.cache/ov`, `~/.nv`, `~/.cache/warp`.

## 9. Isaac Lab v2.1.0

`flatdict==4.0.1` (an Isaac Lab dependency) imports `pkg_resources`, which
setuptools 84 removed — install it first with an older setuptools:

```bash
pip install "setuptools<81" wheel
pip install --no-build-isolation "flatdict==4.0.1"
```

Then:

```bash
cd ~/rc_lab/study_tron1
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.1.0
./isaaclab.sh --install none
```

**`none`, not plain `--install`.** The default would install `rsl-rl-lib` from
PyPI, which provides the same `rsl_rl` module name as the patched copy vendored
inside `tron1-rl-isaaclab` (it adds an MLP encoder). Look for
`[INFO] No rl-framework will be installed.` in the output. The warning
`does not provide the extra 'none'` is harmless.

Verify:

```bash
pip list | grep -E "^isaaclab|^isaacsim +"
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
```

`[INFO]: Setup complete...` is the pass signal — that tutorial loops forever by
design, so exit with Ctrl+C.

> `import isaaclab` in a bare `python` raises `No module named 'omni.kit'`.
> That is expected: Isaac Lab is only importable after the Kit app has started.
> Use `./isaaclab.sh -p <script>`, or a script that calls `AppLauncher` first
> (TRON1's `train.py` / `play.py` do, so plain `python` works for them).

## 10. TRON1

```bash
cd ~/rc_lab/study_tron1
git clone https://github.com/limxdynamics/tron1-rl-isaaclab.git
cd tron1-rl-isaaclab

pip install toml
pip install -e exts/bipedal_locomotion --no-build-isolation   # its setup.py imports toml
pip install -e rsl_rl                                          # LimX's patched copy — install last
```

`exts/bipedal_locomotion` has no `pyproject.toml`, so pip's isolated build env
lacks `toml` — hence `--no-build-isolation`.

Confirm the right `rsl_rl` won:

```bash
pip show rsl_rl | grep -E "Version|Editable"
# Version: 2.0.2
# Editable project location: .../study_tron1/tron1-rl-isaaclab/rsl_rl
```

Version 2.3.x means the PyPI copy leaked in — uninstall it and redo.

## 11. Smoke test

```bash
python scripts/rsl_rl/train.py --task Isaac-Limx-SF-Blind-Flat-v0 \
  --num_envs 16 --max_iterations 2 --headless
```

Success looks like `Learning iteration 0/2` with a reward table.

---

## 12. Task IDs — the upstream README is wrong

The README advertises `Isaac-Limx-SF-TRON1A-Blind-Flat-v0`, which does not
exist. The real registrations, from
`exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/__init__.py`:

| Variant | Train | Play / eval |
|---|---|---|
| PF (point foot) | `Isaac-Limx-PF-Blind-Flat-v0` | `Isaac-Limx-PF-Blind-Flat-Play-v0` |
| WF (wheel foot) | `Isaac-Limx-WF-Blind-Flat-v0` | `Isaac-Limx-WF-Blind-Flat-Play-v0` |
| SF (sole foot) | `Isaac-Limx-SF-Blind-Flat-v0` | `Isaac-Limx-SF-Blind-Flat-Play-v0` |

## 13. Daily commands

Always `conda activate tron1` first, and never `source` a ROS setup in the same shell.

```bash
cd ~/rc_lab/study_tron1/tron1-rl-isaaclab

# train (headless, full speed)
python scripts/rsl_rl/train.py --task Isaac-Limx-SF-Blind-Flat-v0 --num_envs 2048 --headless

# watch live — cut num_envs hard, GUI costs ~2 GB VRAM and 5-10x speed
python scripts/rsl_rl/train.py --task Isaac-Limx-SF-Blind-Flat-v0 --num_envs 64

# headless + periodic video clips -> logs/rsl_rl/<exp>/<run>/videos/
python scripts/rsl_rl/train.py --task Isaac-Limx-SF-Blind-Flat-v0 --num_envs 2048 --headless \
  --video --video_length 400 --video_interval 2000

# evaluate a trained policy; also exports .pt (jit) + .onnx to <run>/exported/
python scripts/rsl_rl/play.py --task Isaac-Limx-SF-Blind-Flat-Play-v0 --num_envs 16

# curves
tensorboard --logdir logs/rsl_rl

# GPU watch (nvtop segfaults on driver 580 with Ubuntu's nvtop 3.0)
watch -n1 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv'
```

**VRAM budget:** 12 GB is under Isaac Lab's 16 GB recommendation, and the desktop
already takes ~1 GB. Start at `--num_envs 2048`; on OOM step down 1536 → 1024.

## 14. Log noise that is safe to ignore

- `omni.isaac.* has been deprecated in favor of isaacsim.*` — 4.5 renamed the namespace
- `OmniHub is inaccessible` — no Nucleus server, not needed
- `No crash reporter present`
- `TLAS limit ... SceneDbContext` — normal RTX memory sizing
- `Joystick with unknown remapping` — it is misreading the keyboard
- `[Warning] IOMMU is enabled` — not fatal; if training feels slow, try the `iommu=pt` kernel parameter
- `ActorCritic.__init__ got unexpected arguments ['class_name','noise_std_type']`
- actuator `effort_limit` / `velocity_limit` deprecation warnings
- `Unable to resolve info:sourceAsset ... UsdPreviewSurface.mdl`
- torch `To copy construct from a tensor` UserWarning

---

## 15. Repository layout

```
study_tron1/
├── README.md                  <- this file: how to rebuild the machine
├── .gitignore
├── requirements-lock.txt      <- pip freeze of the working env
├── notes/                     <- what I understand (see notes/README.md)
│   ├── 01-repo-map.md
│   ├── 02-sdk-vs-sim.md
│   ├── 03-sf-task-anatomy.md
│   └── 04-file-structure-vs-ros2.md
├── experiments/               <- one file per experiment
│   ├── TEMPLATE.md
│   └── exp_001_baseline_sf.md
├── patches/                   <- diffs against the upstream clones
├── scripts/                   <- my own helpers
│   ├── compare_runs.py
│   ├── toy_answer_key.py      <- every number quoted in notes 00/05/06/07
│   └── notes_to_pdf.py        <- notes/*.md -> notes/pdf/*.pdf (Noto Sans Thai + MathJax)
├── IsaacLab/                  <- clone, tag v2.1.0        (ignored)
└── tron1-rl-isaaclab/         <- clone, LimX upstream     (ignored)
```

The clones and Isaac Sim are ignored on purpose: they are large, they are someone
else's code, and this README reproduces them exactly. What is tracked is the part
that cannot be re-downloaded — the setup knowledge, the understanding, and the
experiment record.

Regenerate the lock file after any environment change:

```bash
conda activate tron1 && pip freeze > requirements-lock.txt
```

## 15b. Working loop

1. Read the relevant note in `notes/`, or write one if it is missing.
2. Copy `experiments/TEMPLATE.md` to `experiments/exp_NNN_<name>.md` and fill in
   the question and hypothesis **before** running anything.
3. Make the change on a branch of the `tron1-rl-isaaclab` fork; if it is a
   throwaway edit, `git diff > ../patches/exp_NNN.diff` instead.
4. Train, then fill in the result honestly — including the runs that failed.
5. `python scripts/compare_runs.py <run dirs> --out runs/exp_NNN.png` to put the
   new run next to the baseline.

Model weights do not belong in git history. Attach the few policies worth keeping
to a GitHub **Release** instead.

## 16. Next: Plan B (sim-to-sim validation)

Not installed yet. Requires a **separate** conda env — LimX's SDK wheel is a
CPython extension linked against libpython3.8 and does not belong next to
Isaac Lab's 3.10.

```bash
git clone --recurse-submodules https://github.com/limxdynamics/tron1-mujoco-sim
# brings limxsdk-lowlevel + robot-description + robot-joystick as submodules
```

`simulator.py` opens the SDK in server mode, impersonating the real robot on the
network, and `tron1-rl-deploy-python` connects to `127.0.0.1` as the client —
the same code then runs against real hardware by changing the IP. The `.onnx`
exported by `play.py` is the input to that pipeline.
