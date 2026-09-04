# exp_001 — Baseline SF blind flat

**Date:** 2026-09-04
**Status:** planned
**Base commit:** `tron1-rl-isaaclab @ ` (fill in with `git rev-parse --short HEAD`)

## Question

What does an unmodified `Isaac-Limx-SF-Blind-Flat-v0` run actually converge to on
this machine, and how long does it take? Everything later is measured against
this, so it needs to exist before any change is made.

## Hypothesis

Default config (15 000 iterations, 24 steps/env). At 2048 envs that is roughly
740 M environment steps. Expect mean episode length to saturate near the 1000-step
cap and `error_vel_xy` to fall well below 0.1 once the gait stabilises.

## Change

None. This is the reference run.

## Command

```bash
cd ~/rc_lab/study_tron1/tron1-rl-isaaclab
python scripts/rsl_rl/train.py --task Isaac-Limx-SF-Blind-Flat-v0 \
  --num_envs 2048 --headless --seed 42
```

Run dir: `logs/rsl_rl/sf_tron_1a_flat/<timestamp>/`

## To record

- [ ] steps/s reported at iteration 100 (throughput on a 4070 SUPER)
- [ ] peak VRAM from `nvidia-smi` — confirms whether 2048 envs fits in 12 GB
- [ ] wall-clock for 15 000 iterations
- [ ] mean reward and mean episode length at 1k / 5k / 15k
- [ ] `Metrics/base_velocity/error_vel_xy` at the same checkpoints
- [ ] `Episode_Termination/base_contact` trend
- [ ] a video from `play.py` at the end

## Result

_(fill in)_

## Conclusion

_(fill in)_
