#!/usr/bin/env python3
"""Probe the REAL SF task (Isaac-Limx-SF-Blind-Flat-v0) and save what chapter 29 of the workbook quotes.

`scripts/workbook_answer_key.py` runs on the system python, which has neither torch nor Isaac Sim. The facts
below cannot be read off the config with `ast` — they depend on the robot's USD file and on what Isaac Lab's
managers do at run time — so this script starts the simulator once (16 envs, headless), measures them, and
writes a dated JSON snapshot that `ch29()` reads.

What it measures
  * the width of every observation term (the table train.py prints at start-up), body and joint names
  * which critic terms differ between envs or over time, and which are the same number everywhere
  * the PD gains right after the `startup` events and again after the first `reset` event
  * how often `push_robot` (mode="interval", interval_range_s=(0.0, 0.0)) is really called, and how many pushes land
  * the layout of the observation history at reset and one step later

Run from the repo root (≈1 minute on an RTX 4070 S); it changes nothing under tron1-rl-isaaclab/:

    cd tron1-rl-isaaclab && env -u PYTHONPATH ~/anaconda3/envs/tron1/bin/python -s ../scripts/workbook_env_probe.py --headless
"""
import argparse
import datetime
import json
import pathlib

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", default=None, help="JSON path (default: scripts/env_probe_<today>.json)")
parser.add_argument("--steps", type=int, default=2000)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from bipedal_locomotion.utils.wrappers.rsl_rl import RslRlPpoAlgorithmMlpCfg  # noqa: E402,F401  (import registers the tasks)

TASK, N_ENVS = "Isaac-Limx-SF-Blind-Flat-v0", 16
cfg = parse_env_cfg(task_name=TASK, device="cuda:0", num_envs=N_ENVS)
env = gym.make(TASK, cfg=cfg)
u = env.unwrapped
robot = u.scene["robot"]
om, em = u.observation_manager, u.event_manager


def lst(t, nd=4):
    return [round(float(x), nd) for x in t.flatten().tolist()]


def span(t):
    return [round(float(t.min()), 4), round(float(t.max()), 4)]


out = {"date": datetime.date.today().isoformat(), "task": TASK, "num_envs": N_ENVS, "steps": args.steps,
       "step_dt": u.step_dt, "physics_dt": u.physics_dt, "max_episode_length": int(u.max_episode_length),
       "body_names": list(robot.body_names), "joint_names": list(robot.joint_names),
       "ankle_bodies": robot.find_bodies("ankle_.*")[1], "LR_link_bodies": robot.find_bodies(".*_[LR]_Link")[1],
       "default_joint_pos": lst(robot.data.default_joint_pos[0])}

out["events"] = {mode: list(names) for mode, names in em.active_terms.items()}
out["curriculum_terms"] = list(u.curriculum_manager.active_terms)
out["termination_terms"] = list(u.termination_manager.active_terms)
out["command_terms"] = list(u.command_manager.active_terms)
out["reward_terms"] = len(u.reward_manager.active_terms)

# -- PD gains: after the startup events only, then after the first reset event
legs, ankles = robot.actuators["legs"].joint_indices, robot.actuators["ankles"].joint_indices
gains = {"default_stiffness": lst(robot.data.default_joint_stiffness[0]), "default_damping": lst(robot.data.default_joint_damping[0]),
         "after_startup": {"stiffness": span(robot.data.joint_stiffness), "damping_legs": span(robot.data.joint_damping[:, legs]),
                           "damping_ankles": span(robot.data.joint_damping[:, ankles])}}

obs, _ = env.reset()
gains["after_first_reset"] = {"stiffness": span(robot.data.joint_stiffness), "damping_legs": span(robot.data.joint_damping[:, legs]),
                              "damping_ankles": span(robot.data.joint_damping[:, ankles])}
out["obs_shapes"] = {k: list(v.shape[1:]) for k, v in obs.items()}
out["obs_terms"] = {g: [[n, list(d)] for n, d in zip(om.active_terms[g], om.group_obs_term_dim[g])] for g in om.active_terms}

# -- observation history: every row equal at reset? which row is the newest one step later?
h0 = obs["obsHistory"]
out["history"] = {"rows_identical_at_reset": bool((h0 == h0[:, :1]).all())}

# -- mass / inertia / material: what the simulator holds vs. what the "default" buffers hold
pm = robot.root_physx_view.get_masses()
mat = robot.root_physx_view.get_material_properties()
out["mass"] = {"default_per_body": lst(robot.data.default_mass[0]), "default_total": round(float(robot.data.default_mass[0].sum()), 3),
               "default_identical_across_envs": bool((robot.data.default_mass == robot.data.default_mass[0]).all()),
               "physx_total_range": span(pm.sum(dim=1)), "physx_base_range": span(pm[:, 0])}
out["material"] = {"shape": list(mat.shape), "static_friction_range": span(mat[..., 0]), "dynamic_friction_range": span(mat[..., 1]),
                   "restitution_range": span(mat[..., 2])}

# -- count the real calls of push_robot
tc = em.get_term_cfg("push_robot")
orig, push = tc.func, {"calls": 0, "env_ids_per_call": set(), "pushes": 0}


def counted(env_, env_ids, **kw):
    push["calls"] += 1
    push["env_ids_per_call"].add(-1 if env_ids is None else len(env_ids))
    orig(env_, env_ids, **kw)
    push["pushes"] += int((robot._external_force_b.abs().sum(dim=(1, 2)) > 0).sum())


tc.func = counted
em.set_term_cfg("push_robot", tc)

names, dims = om.active_terms["critic"], om.group_obs_term_dim["critic"]
widths = [int(torch.tensor(d).prod()) for d in dims]
lo = hi = obs["critic"].clone()
stiff, damp_l, damp_a = [robot.data.joint_stiffness.clone()], [robot.data.joint_damping[:, legs].clone()], [robot.data.joint_damping[:, ankles].clone()]
resets = 0
torch.manual_seed(0)
for step in range(args.steps):
    # small random actions, so that `last_action` is not mistaken for a constant term
    o, _, term, trunc, _ = env.step(0.1 * torch.randn(N_ENVS, 8, device=u.device))
    if step == 0:        # gait_phase carries no noise: the row of the history that equals it is the newest one
        g = slice(30, 32)
        out["history"]["newest_row_is_last"] = bool(torch.allclose(o["obsHistory"][:, -1, g], o["policy"][:, g]))
        out["history"]["oldest_row_is_first"] = not bool(torch.allclose(o["obsHistory"][:, 0, g], o["policy"][:, g]))
    c = o["critic"]
    lo, hi = torch.minimum(lo, c), torch.maximum(hi, c)
    resets += int((term | trunc).sum())
    stiff.append(robot.data.joint_stiffness.clone()); damp_l.append(robot.data.joint_damping[:, legs].clone()); damp_a.append(robot.data.joint_damping[:, ankles].clone())

# spread = (max - min) over every env and every step; 0.0 means "the same number everywhere, all the time"
crit, i = [], 0
for n, w in zip(names, widths):
    sp = float((hi[:, i:i + w].max(dim=0).values - lo[:, i:i + w].min(dim=0).values).max())
    crit.append({"name": n, "width": w, "start": i, "spread": round(sp, 6), "first_values": lst(obs["critic"][0, i:i + min(w, 4)])})
    i += w
out["critic_terms"] = crit
out["critic_total"] = i
out["critic_constant_width"] = sum(c["width"] for c in crit if c["spread"] == 0.0)
gains["over_run"] = {"resets": resets, "stiffness": span(torch.stack(stiff)), "damping_legs": span(torch.stack(damp_l)),
                     "damping_ankles": span(torch.stack(damp_a)),
                     "default_stiffness_after_run": lst(robot.data.default_joint_stiffness[0])}
out["gains"] = gains
out["push"] = {"calls": push["calls"], "env_ids_per_call": sorted(push["env_ids_per_call"]), "pushes": push["pushes"],
               "expected_pushes": args.steps * N_ENVS * tc.params["probability"], "probability": tc.params["probability"],
               "interval_range_s": list(tc.interval_range_s)}

path = pathlib.Path(args.out) if args.out else pathlib.Path(__file__).with_name(f"env_probe_{out['date']}.json")
path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"PROBE wrote {path}", flush=True)
env.close()
app.close()
