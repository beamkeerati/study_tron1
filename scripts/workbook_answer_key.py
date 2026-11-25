#!/usr/bin/env python3
"""Answer key for the exercises in notes/workbook/ (the textbook-style rewrite of notes 00-07).

Every number comes from the SAME running example as scripts/toy_answer_key.py: this file
imports that module (its data `env`, the derived `cells`, and the constants) and only adds
the "what if" variants the exercises ask about. It defines no new example data.

Chapters 1-2 have no toy numbers: ch01()/ch02() read their facts from the real code under tron1-rl-isaaclab/
and IsaacLab/, plus one dated snapshot of the GitHub repo list (limxdynamics_repos_2026-11-25.json).
Chapters 3-6 come before the table is introduced (chapter 7): ch03/ch05/ch06 read the code the same way, and
chapter 4's bag of eight cards is the r row of the toy table (box K = env index), not new data.

Run from the repo root:  python3 scripts/workbook_answer_key.py [chapter ...]
"""
import contextlib
import io
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
with contextlib.redirect_stdout(io.StringIO()):      # the toy script prints its table on import
    import toy_answer_key as toy

GAMMA, LAM, GL = toy.GAMMA, toy.LAM, toy.GL
env, cells, meanA, stdA = toy.env, toy.cells, toy.meanA, toy.stdA


def gae(e, lam=LAM, use_mask=True, r_override=None):
    """Backward pass of rollout_storage.py:187-201 for one env; returns per-step dicts t=0..3."""
    d = env[e]
    out, A_next = {}, 0.0
    for t in (3, 2, 1, 0):
        r = d["r"][t] if not r_override or t not in r_override else r_override[t]
        mask = (1.0 - d["done"][t]) if use_mask else 1.0
        V_next = d["V_last"] if t == 3 else d["V"][t + 1]
        delta = r + mask * GAMMA * V_next - d["V"][t]
        A = delta + mask * GAMMA * lam * A_next
        out[t] = dict(delta=delta, A=A, R=A + d["V"][t])
        A_next = A
    return out


def ch22():
    print("=== chapter 22 — GAE ===")
    print("-- given (same as toy_answer_key): delta, A, R, Ahat")
    for (e, t), c in cells.items():
        print(f"e{e}t{t}  delta={c['delta']:+.4f}  A={c['A']:+.4f}  R={c['R']:+.4f}  Ahat={c['Ahat']:+.4f}")
    As = [c["A"] for c in cells.values()]
    ss = sum((x - meanA) ** 2 for x in As)
    print(f"sum(A)={sum(As):+.4f}  mean={meanA:+.4f}  sum sq dev={ss:.4f}  /7={ss/7:.4f}  std_n-1={stdA:.4f}")
    std_n = math.sqrt(ss / 8)
    print(f"std_n={std_n:.4f}  Ahat(e0t0) with std_n = {(cells[(0,0)]['A'] - meanA) / std_n:+.4f}")
    hats = [c["Ahat"] for c in cells.values()]
    print(f"check: mean(Ahat)={sum(hats)/8:+.4f}  std_n-1(Ahat)={math.sqrt(sum(h*h for h in hats)/7):.4f}")

    print("-- env 1, first episode (t0..t2): Monte Carlo advantage G_t - V_t")
    d = env[1]
    for t in (0, 1, 2):
        G = sum(GAMMA ** k * d["r"][t + k] for k in range(3 - t))
        print(f"t{t}: G={G:+.4f}  G-V={G - d['V'][t]:+.4f}")
    print(f"sum gamma^k delta (t0) = {sum(GAMMA**k * cells[(1,k)]['delta'] for k in range(3)):+.4f}")
    print(f"blame passed back from delta_2=-3: x0.72 = {GL*-3:+.4f}   x0.72^2 = {GL*GL*-3:+.4f}")
    for e in (0, 1):
        terms = [GL ** k * cells[(e, k)]["delta"] for k in range(4)]
        print(f"closed form e{e} t0 terms (no mask applied): " + "  ".join(f"{x:+.4f}" for x in terms))

    print("-- two-step estimate A^(2) = delta_t + gamma*delta_{t+1}")
    print(f"e0 t0: {cells[(0,0)]['delta'] + GAMMA*cells[(0,1)]['delta']:+.4f}   "
          f"e1 t0: {cells[(1,0)]['delta'] + GAMMA*cells[(1,1)]['delta']:+.4f}")

    print("-- lambda = 0 and lambda = 1 (sum cut at the table edge, closed with V_last)")
    for e in (0, 1):
        g0, g1 = gae(e, lam=0.0), gae(e, lam=1.0)
        print(f"e{e}: lam=0 A=" + " ".join(f"{g0[t]['A']:+.4f}" for t in range(4))
              + "   lam=1 A=" + " ".join(f"{g1[t]['A']:+.4f}" for t in range(4)))

    print("-- env 1 WITHOUT the mask (the bug the mask prevents)")
    g = gae(1, use_mask=False)
    for t in (3, 2, 1, 0):
        print(f"t{t}: delta={g[t]['delta']:+.4f}  A={g[t]['A']:+.4f}")

    print("-- env 1 if t2 were a TIME-OUT instead of a fall (ppo.py:163-168: r += gamma * V_t)")
    r2 = env[1]["r"][2] + GAMMA * env[1]["V"][2]
    g = gae(1, r_override={2: r2})
    print(f"stored r_2 = {r2:+.4f}")
    for t in (2, 1, 0):
        print(f"t{t}: delta={g[t]['delta']:+.4f}  A={g[t]['A']:+.4f}  R={g[t]['R']:+.4f}")


# ---------------------------------------------------------------- chapters 1-2: facts read from the real code
# These two chapters have no toy numbers. Their "answer key" is the code itself, so every count, name and
# line number is read here with ast/regex from tron1-rl-isaaclab/ and IsaacLab/ (never typed in by hand).
import ast
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
T = ROOT / "tron1-rl-isaaclab"
EXT = T / "exts/bipedal_locomotion/bipedal_locomotion"
LOCO = EXT / "tasks/locomotion"
ISAAC = ROOT / "IsaacLab/source"


def _num(node):
    """Value of a numeric expression node such as `0.8+0.166` or `-2e-5`."""
    return eval(compile(ast.Expression(node), "<cfg>", "eval"), {"__builtins__": {}}, {"math": math})


def _calls(tree, name):
    """Every call node whose callee is `name` or `x.name`."""
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            if (isinstance(f, ast.Name) and f.id == name) or (isinstance(f, ast.Attribute) and f.attr == name):
                yield n


def _kw(call, key):
    return next((k.value for k in call.keywords if k.arg == key), None)


def show(path, a, b=None):
    """Print lines a..b of a file exactly as they are, prefixed with basename:NN."""
    lines = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
    for i in range(a, (b or a) + 1):
        print(f"    {pathlib.Path(path).name}:{i}: {lines[i - 1].rstrip()}")


# Public repos of github.com/limxdynamics as the GitHub API listed them on 2026-11-25 (name + one-line description).
# A dated snapshot, not live data: the account changes. The layer is this book's reading of the repo NAME.
SNAPSHOT = pathlib.Path(__file__).with_name("limxdynamics_repos_2026-11-25.json")


def layer_of(name):
    if "rl-isaac" in name:
        return "train"
    if "rl-deploy" in name:
        return "deploy"
    if "sim" in name or "gazebo" in name:
        return "simulate"
    return "outside the three layers"


def tron1_repos():
    import json
    repos = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    return len(repos), {r["name"]: (layer_of(r["name"]), r["description"]) for r in repos if r["name"].startswith("tron1-")}


def variants():
    """PF / SF / WF as the three asset configs define them."""
    out = {}
    for code, stem in (("PF", "pointfoot"), ("SF", "solefoot"), ("WF", "wheelfoot")):
        f = EXT / f"assets/config/{stem}_cfg.py"
        tree = ast.parse(f.read_text())
        groups = []
        for c in _calls(tree, "ImplicitActuatorCfg"):
            names = [e.value for e in _kw(c, "joint_names_expr").elts]
            gains = {}
            for key in ("stiffness", "damping", "effort_limit", "velocity_limit"):
                v = _kw(c, key)
                if isinstance(v, ast.Dict):
                    gains[key] = sorted({_num(x) for x in v.values})
                else:
                    gains[key] = [_num(v)]
            groups.append(dict(line=(c.lineno, c.end_lineno), names=names, **gains))
        pos = next(_kw(c, "pos") for c in _calls(tree, "InitialStateCfg"))
        out[code] = dict(file=f, groups=groups, z=_num(pos.elts[2]), z_line=pos.lineno)
    return out


def task_ids():
    f = LOCO / "robots/__init__.py"
    tree = ast.parse(f.read_text())
    rows = []
    for c in _calls(tree, "register"):
        kwargs = _kw(c, "kwargs")
        cfg = next(v for k, v in zip(kwargs.keys, kwargs.values) if k.value == "env_cfg_entry_point")
        rows.append((_kw(c, "id").value, _kw(c, "id").lineno, ast.unparse(cfg), ast.unparse(_kw(c, "entry_point")), (c.lineno, c.end_lineno)))
    return f, rows


def ch01():
    print("=== chapter 1 — Tron1 and its repos ===")
    total, repos = tron1_repos()
    print(f"-- {SNAPSHOT.name}: {total} public repos, {len(repos)} named tron1-*; by layer (rule: layer_of)")
    for layer in ("train", "deploy", "simulate", "outside the three layers"):
        for n, (l, desc) in repos.items():
            if l == layer:
                print(f"  {layer:<25} {n:<26} {desc[:88]}")
    print("-- the three variants (assets/config/*_cfg.py)")
    v = variants()
    for code, d in v.items():
        total = sum(len(g["names"]) for g in d["groups"])
        print(f"  {code}  {d['file'].name}  actuated joints = {total}  ({total // 2} per leg)  start height z = {d['z']:.3f} m (line {d['z_line']})")
        for g in d["groups"]:
            print(f"      lines {g['line'][0]}-{g['line'][1]}: {', '.join(g['names'])}")
            print(f"        stiffness {g['stiffness']}  damping {g['damping']}  effort_limit {g['effort_limit']}  velocity_limit {g['velocity_limit']}")
    print("  model folders:", ", ".join(sorted(p.name for p in (EXT / "assets/usd").iterdir() if p.is_dir())))
    print("  highest start:", max(v, key=lambda c: v[c]["z"]), " lowest:", min(v, key=lambda c: v[c]["z"]),
          f" WF - SF = {v['WF']['z'] - v['SF']['z']:.3f} m")

    print("-- task IDs by variant (robots/__init__.py)")
    f, rows = task_ids()
    for code in ("PF", "SF", "WF"):
        print(f"  {code}: " + ", ".join(f"{i} (:{ln})" for i, ln, *_ in rows if f"-{code}-" in i))

    print("-- who reads ROBOT_TYPE in the training repo")
    hits = []
    for p in list(EXT.rglob("*.py")) + list((T / "scripts").rglob("*.py")) + list((T / "rsl_rl/rsl_rl").rglob("*.py")):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if "robot_type" in line or "ROBOT_TYPE" in line:
                hits.append(f"{p.name}:{i}: {line.strip()}")
    print("  " + "\n  ".join(hits) + f"\n  ({len(hits)} line in total: the variable is read and never used)")
    sdk = [p for p in list(EXT.rglob("*.py")) + list((T / "scripts").rglob("*.py")) + list((T / "rsl_rl/rsl_rl").rglob("*.py"))
           if "limxsdk" in p.read_text()]
    print(f"  files in the training repo that mention limxsdk: {len(sdk)}")

    print("-- lines cited in the chapter")
    show(T / "scripts/rsl_rl/train.py", 39, 40)
    show(T / "scripts/rsl_rl/train.py", 142, 143)
    print("  files in the training repo that mention params.yaml:",
          sum("params.yaml" in q.read_text() for q in list(T.rglob("*.py")) if ".git" not in q.parts))
    show(T / "scripts/rsl_rl/play.py", 111, 122)
    show(EXT / "utils/wrappers/rsl_rl/rl_mlp_cfg.py", 37)
    show(EXT / "assets/config/pointfoot_cfg.py", 38, 39)
    show(LOCO / "agents/limx_rsl_rl_ppo_cfg.py", 6)
    for ln in (42, 80, 119):
        show(LOCO / "agents/limx_rsl_rl_ppo_cfg.py", ln)
    show(T / "scripts/rsl_rl/play.py", 107, 109)
    print("  variant-specific env cfg files:", ", ".join(sorted(q.name for q in (LOCO / "robots").glob("limx_*_env_cfg.py"))))
    show(LOCO / "robots/__init__.py", 74)
    show(LOCO / "cfg/WF/limx_base_env_cfg.py", 111, 115)
    show(EXT / "assets/config/wheelfoot_cfg.py", 50, 51)
    show(EXT / "assets/config/solefoot_cfg.py", 50, 51)
    show(T / "rsl_rl/rsl_rl/runner/on_policy_runner.py", 239)
    show(LOCO / "cfg/SF/limx_base_env_cfg.py", 484)
    show(T / "exts/bipedal_locomotion/setup.py", 63, 67)


def defined_in(name):
    """Files, in star-import order, that define `name`; the LAST one is what `mdp.name` resolves to."""
    order = ([ISAAC / "isaaclab/isaaclab/envs/mdp"]
             + [ISAAC / f"isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/{m}.py" for m in ("curriculums", "rewards", "terminations")]
             + [LOCO / "mdp/commands"] + [LOCO / f"mdp/{m}.py" for m in ("curriculums", "events", "observations", "rewards")])
    found = []
    for src in order:
        for p in (sorted(src.rglob("*.py")) if src.is_dir() else [src]):
            for i, line in enumerate(p.read_text().splitlines(), 1):
                m = re.match(rf"(def|class) {re.escape(name)}\b", line)
                if m:
                    found.append((p, i, m.group(1)))
    return found


def ch02():
    print("=== chapter 2 — where the files are ===")
    site = pathlib.Path.home() / "anaconda3/envs/tron1/lib/python3.10/site-packages"
    print("-- editable installs in the conda env `tron1` (machine-specific; skipped if the env is elsewhere)")
    for p in sorted(site.glob("__editable___*_finder.py")):
        m = re.search(r"MAPPING[^=]*= (\{.*\})", p.read_text())
        for pkg, path in ast.literal_eval(m.group(1)).items():
            print(f"  {pkg:<20} -> {pathlib.Path(path).relative_to(ROOT)}")
    sims = sorted(d.name for d in site.glob("isaacsim-*.dist-info"))
    print("  pip (not editable):", ", ".join(sims) or "(not found)")
    print(f"  IsaacLab VERSION = {(ROOT / 'IsaacLab/VERSION').read_text().strip()}   package.xml files in both repos:",
          len(list(T.rglob("package.xml"))) + len(list((ROOT / "IsaacLab").rglob("package.xml"))))

    print("-- basenames cited as file.py:NN anywhere in notes/ -> full path, line count")
    cited = set()
    for md in list((ROOT / "notes").glob("0*.md")) + list((ROOT / "notes/workbook").glob("[0-9]*.md")):
        cited |= set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*\.py):\d+", md.read_text(encoding="utf-8")))
    for name in sorted(cited):
        here = sorted(p for p in T.rglob(name) if ".git" not in p.parts)
        there = sorted(ISAAC.rglob(name))
        where = [str(p.relative_to(T)) for p in here] or [f"IsaacLab/source/{p.relative_to(ISAAC)}" for p in there] or ["(outside both repos)"]
        n = len(here[0].read_text().splitlines()) if len(here) == 1 else (len(there[0].read_text().splitlines()) if not here and len(there) == 1 else "")
        extra = f"   ALSO in IsaacLab x{len(there)}" if here and there else ""
        print(f"  {name:<28} {n!s:>4}  " + " | ".join(where) + extra)

    print("-- gym.register calls")
    f, rows = task_ids()
    for tid, ln, cfg, entry, (a, b) in rows:
        print(f"  __init__.py:{a}-{b}  {tid:<34} {cfg}   entry_point={entry}")
    print(f"  {len(rows)} task IDs, {len({e for *_, e, _ in rows})} distinct entry_point")
    readme = re.findall(r"--task (\S+)", (T / "README.md").read_text())
    print("  README.md advertises:", readme, " registered?", [r in {t for t, *_ in rows} for r in readme])

    print("-- config classes in cfg/SF/limx_base_env_cfg.py, and the chain behind the task this book uses")
    base = LOCO / "cfg/SF/limx_base_env_cfg.py"
    tree = ast.parse(base.read_text())
    for n in tree.body:
        if isinstance(n, ast.ClassDef):
            print(f"  :{n.lineno}  class {n.name}({', '.join(ast.unparse(b) for b in n.bases)})")
    sf = LOCO / "robots/limx_solefoot_env_cfg.py"
    for n in ast.parse(sf.read_text()).body:
        if isinstance(n, ast.ClassDef) and n.name in ("SFBaseEnvCfg", "SFBaseEnvCfg_PLAY", "SFBlindFlatEnvCfg", "SFBlindFlatEnvCfg_PLAY"):
            print(f"  {sf.name}:{n.lineno}  class {n.name}({', '.join(ast.unparse(b) for b in n.bases)})")

    print("-- RewardsCfg: config entry -> function -> where the function is defined")
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RewardsCfg")
    limx = isaac = same_name = classes = 0
    for a in cls.body:
        if isinstance(a, ast.Assign) and isinstance(a.value, ast.Call):
            entry, func = a.targets[0].id, _kw(a.value, "func").attr
            places = defined_in(func)
            p, ln, kind = places[-1]
            classes += kind == "class"
            owner = "project" if T in p.parents else "IsaacLab"
            limx += owner == "project"
            isaac += owner == "IsaacLab"
            same_name += entry == func
            rel = p.relative_to(T) if owner == "project" else pathlib.Path("IsaacLab/source") / p.relative_to(ISAAC)
            note = f"   (also defined, and overridden, in {len(places) - 1} earlier file)" if len(places) > 1 else ""
            print(f"  :{a.lineno:<4} {entry:<34} func=mdp.{func:<30} weight={_num(_kw(a.value, 'weight'))!s:<8} {owner:<9} {kind:<5} {rel}:{ln}{note}")
    print(f"  entries {limx + isaac}: function defined in the project {limx}, in IsaacLab {isaac}; entry name == function name in {same_name}; func is a class (not a def) in {classes}")

    print("-- timing read from the config")
    src = base.read_text()
    dec = int(re.search(r"self\.decimation = (\d+)", src).group(1))
    dt = float(re.search(r"self\.sim\.dt = ([\d.]+)", src).group(1))
    print(f"  decimation {dec} x sim.dt {dt} s = {dec * dt:.3f} s per control period = {1 / (dec * dt):.0f} Hz;  physics {1 / dt:.0f} Hz")

    print("-- the file map: lines per file (as an editor counts them)")
    for rel in ("scripts/rsl_rl/train.py", "scripts/rsl_rl/play.py", "scripts/rsl_rl/cli_args.py",
                "rsl_rl/rsl_rl/runner/on_policy_runner.py", "rsl_rl/rsl_rl/algorithm/ppo.py", "rsl_rl/rsl_rl/storage/rollout_storage.py",
                "rsl_rl/rsl_rl/modules/actor_critic.py", "rsl_rl/rsl_rl/modules/mlp_encoder.py"):
        print(f"  {len((T / rel).read_text().splitlines()):>4}  {rel}")
    for rel in ("assets/config/solefoot_cfg.py", "tasks/locomotion/robots/__init__.py", "tasks/locomotion/robots/limx_solefoot_env_cfg.py",
                "tasks/locomotion/cfg/SF/limx_base_env_cfg.py", "tasks/locomotion/agents/limx_rsl_rl_ppo_cfg.py",
                "tasks/locomotion/mdp/rewards.py", "tasks/locomotion/mdp/observations.py", "tasks/locomotion/mdp/events.py", "tasks/locomotion/mdp/commands/gait_command.py",
                "utils/wrappers/rsl_rl/rl_mlp_cfg.py"):
        print(f"  {len((EXT / rel).read_text().splitlines()):>4}  {rel}")
    print("  basenames that exist both in the project and in IsaacLab:",
          ", ".join(sorted({p.name for p in (LOCO / "mdp").glob("*.py")} & {p.name for p in ISAAC.rglob("*.py")} - {"__init__.py"})))
    print("  basenames that exist in more than one of cfg/PF, cfg/SF, cfg/WF:",
          ", ".join(sorted({p.name for p in (LOCO / "cfg/SF").glob("*.py")} & {p.name for p in (LOCO / "cfg/PF").glob("*.py")} - {"__init__.py"})))
    for name in ("app_launcher.py", "manager_based_rl_env.py", "reward_manager.py", "parse_cfg.py"):
        print(f"  {name}: {[str(p.relative_to(ISAAC)) for p in ISAAC.rglob(name)]}")

    print("-- lines cited in the chapter")
    tr = T / "scripts/rsl_rl/train.py"
    for a, b in ((14, 23), (39, 40), (42, 42), (49, 50), (65, 65), (77, 80), (88, 89), (98, 98), (123, 123), (148, 148)):
        show(tr, a, b)
    show(T / "scripts/rsl_rl/play.py", 50, 51)
    show(T / "scripts/rsl_rl/cli_args.py", 49)
    show(EXT / "__init__.py", 6)
    show(EXT / "tasks/__init__.py", 16)
    show(LOCO / "__init__.py", 3)
    show(LOCO / "mdp/__init__.py", 1, 7)
    show(LOCO / "robots/__init__.py", 73, 81)
    show(base, 22)
    show(base, 426)
    show(base, 484)
    show(base, 497, 501)
    show(LOCO / "mdp/rewards.py", 49, 54)
    show(sf, 31)
    show(sf, 56)
    show(sf, 76)
    show(base, 36)
    show(base, 393)
    show(EXT / "assets/config/solefoot_cfg.py", 48, 51)
    show(EXT / "assets/config/solefoot_cfg.py", 62)
    show(ISAAC / "isaaclab/isaaclab/envs/manager_based_rl_env.py", 125)
    show(ISAAC / "isaaclab/isaaclab/envs/manager_based_rl_env.py", 207)
    show(T / "rsl_rl/setup.py", 7, 8)
    show(ISAAC / "isaaclab/isaaclab/managers/reward_manager.py", 148)
    show(ISAAC / "isaaclab/isaaclab/managers/manager_base.py", 317, 330)
    show(ISAAC / "isaaclab/isaaclab/managers/reward_manager.py", 215)
    show(LOCO / "mdp/rewards.py", 345)
    show(LOCO / "mdp/rewards.py", 481)
    show(LOCO / "mdp/rewards.py", 496, 497)
    show(ISAAC / "isaaclab/isaaclab/envs/manager_based_rl_env.py", 124)
    show(ISAAC / "isaaclab/isaaclab/envs/manager_based_rl_env.py", 182)
    show(ISAAC / "isaaclab/isaaclab/app/app_launcher.py", 780)
    show(ISAAC / "isaaclab_tasks/isaaclab_tasks/utils/parse_cfg.py", 55)
    show(ISAAC / "isaaclab_tasks/isaaclab_tasks/utils/parse_cfg.py", 120)
    show(ISAAC / "isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/__init__.py", 8)
    show(ISAAC / "isaaclab/isaaclab/actuators/actuator_cfg.py", 172)


# ---------------------------------------------------------------- chapters 3-9 (part B: the problem)
RSL = T / "rsl_rl/rsl_rl"
BASE = LOCO / "cfg/SF/limx_base_env_cfg.py"
PPO_CFG = LOCO / "agents/limx_rsl_rl_ppo_cfg.py"
RL_ENV = ISAAC / "isaaclab/isaaclab/envs/manager_based_rl_env.py"
WRAP = ISAAC / "isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py"


def cfg_class(*names):
    """The (possibly nested) config class `names[0].names[1]...` of cfg/SF/limx_base_env_cfg.py."""
    body = ast.parse(BASE.read_text()).body
    node = None
    for nm in names:
        node = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == nm)
        body = node.body
    return node


def cfg_terms(*names):
    """[(entry name, line, call node)] for every `x = Something(...)` in that config class."""
    return [(a.targets[0].id, a.lineno, a.value) for a in cfg_class(*names).body
            if isinstance(a, ast.Assign) and isinstance(a.value, ast.Call)]


def timing():
    src_ = BASE.read_text()
    dec = int(re.search(r"self\.decimation = (\d+)", src_).group(1))
    dt = float(re.search(r"self\.sim\.dt = ([\d.]+)", src_).group(1))
    ep = float(re.search(r"self\.episode_length_s = ([\d.]+)", src_).group(1))
    return dec, dt, dec * dt, ep


def sf_ppo():
    """Keyword values of the block that actually runs: SF_TRON1AFlatPPORunnerCfg."""
    cls = next(n for n in ast.parse(PPO_CFG.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "SF_TRON1AFlatPPORunnerCfg")
    out = {}
    for a in cls.body:
        if isinstance(a, ast.Assign):
            if isinstance(a.value, ast.Call):
                for k in a.value.keywords:
                    with contextlib.suppress(Exception):
                        out[k.arg] = (ast.literal_eval(k.value), k.value.lineno)
            else:
                with contextlib.suppress(Exception):
                    out[a.targets[0].id] = (ast.literal_eval(a.value), a.lineno)
    return out


def n_joints():
    return sum(len(g["names"]) for g in variants()["SF"]["groups"])


def obs_dims(*group):
    """Width of every term of an observation group. Joint-sized terms use the SF joint count; the two gait terms are
    read from the project code; 3 is the size of a base-frame vector (angular velocity, projected gravity)."""
    nj = n_joints()
    gait_cmd = int(re.search(r"self\.gait_command = torch\.zeros\(self\.num_envs, (\d+)", (LOCO / "mdp/commands/gait_command.py").read_text()).group(1))
    obs_src = (LOCO / "mdp/observations.py").read_text()
    gait_phase = int(re.search(r"def get_gait_phase.*?Shape: \(num_envs, (\d+)\)", obs_src, re.S).group(1))
    width = {"base_ang_vel": 3, "projected_gravity": 3, "base_lin_vel": 3, "joint_pos_rel": nj, "joint_vel_rel": nj,
             "last_action": nj, "get_gait_phase": gait_phase, "get_gait_command": gait_cmd}
    return [(name, ln, _kw(call, "func").attr, width.get(_kw(call, "func").attr)) for name, ln, call in cfg_terms(*group)]


def reward_terms():
    return [(name, ln, _kw(call, "func").attr, _num(_kw(call, "weight"))) for name, ln, call in cfg_terms("RewardsCfg")]


def find_line(path, pattern, start=1):
    for i, line in enumerate(pathlib.Path(path).read_text().splitlines(), 1):
        if i >= start and re.search(pattern, line):
            return i
    raise LookupError(pattern)


def ch03():
    print("=== chapter 3 — what RL is solving ===")
    dec, dt, step_dt, _ = timing()
    print(f"decimation {dec} x sim.dt {dt} = {step_dt:.3f} s per decision = {1/step_dt:.0f} decisions per second")
    nj, nrew = n_joints(), len(reward_terms())
    scene = next(n for n in ast.walk(cfg_class("SFEnvCfg")) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "SFSceneCfg")
    nenv = _num(_kw(scene, "num_envs"))
    print(f"actuated joints (SF) = {nj}   reward entries = {nrew}   default num_envs = {nenv} (line {scene.lineno})")
    print(f"numbers the controller must produce per second: {nj} x {1/step_dt:.0f} = {nj/step_dt:.0f}")
    print(f"decisions per simulated second across the default {nenv} robots = {nenv/step_dt:,.0f}")
    print(f"exercise: with 2048 robots = {2048/step_dt:,.0f} decisions per second, {2048/step_dt*60:,.0f} per minute")
    print(f"PD loop (every physics step) = {1/dt:.0f} Hz")
    print("-- the loop in the runner")
    run = RSL / "runner/on_policy_runner.py"
    show(run, 170); show(run, 174, 177); show(run, 192)
    print("-- inside one env.step (IsaacLab)")
    for a, b in ((173, 173), (182, 189), (207, 207), (239, 239), (242, 242)):
        show(RL_ENV, a, b)
    print("-- who computes the reward: the environment's reward manager")
    show(ISAAC / "isaaclab/isaaclab/managers/reward_manager.py", 148, 150)
    show(RSL / "algorithm/ppo.py", 133); show(RSL / "algorithm/ppo.py", 159, 160)
    show(BASE, 116, 121); show(BASE, 368); show(BASE, 497); show(BASE, 501); show(BASE, 484)
    show(T / "scripts/rsl_rl/train.py", 98)


def bag():
    """Chapter 4's bag of cards: the eight r values of the toy table, box K = env index."""
    cards = [(k, x) for k in (0, 1) for x in env[k]["r"]]
    return cards


def dist(values):
    out = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    n = len(values)
    return {v: c / n for v, c in sorted(out.items(), reverse=True)}, n


def ch04():
    import random
    from fractions import Fraction as F
    print("=== chapter 4 — random variables, expectation, the bar ===")
    cards = bag()
    xs = [x for _, x in cards]
    print("box 0:", env[0]["r"], "  box 1:", env[1]["r"])
    p, n = dist(xs)
    print("-- distribution of X (whole bag, 8 cards)")
    for v, q in p.items():
        print(f"  P(X={v:+.1f}) = {F(q).limit_denominator(8)} = {q:.4f}   x*p = {v*q:+.4f}")
    EX = sum(v * q for v, q in p.items())
    print(f"  sum p = {sum(p.values()):.4f}   E[X] = {EX:+.4f}   (= {sum(xs)}/8)")
    cond = {}
    for k in (0, 1):
        pk, nk = dist(env[k]["r"])
        cond[k] = sum(v * q for v, q in pk.items())
        print(f"-- given K={k}: " + "  ".join(f"P(X={v:+.1f}|K={k})={F(q).limit_denominator(4)}" for v, q in pk.items()) + f"   E[X|K={k}] = {cond[k]:+.4f}")
    print(f"  two-level average: 0.5*{cond[0]:+.4f} + 0.5*{cond[1]:+.4f} = {0.5*cond[0] + 0.5*cond[1]:+.4f}")
    p0, _ = dist(env[0]["r"]); p1, _ = dist(env[1]["r"])
    for v in (1.0, 0.5):
        print(f"  P(X={v}|K=0) + P(X={v}|K=1) = {p0.get(v,0):.2f} + {p1.get(v,0):.2f} = {p0.get(v,0)+p1.get(v,0):.2f}  (not 1: summing over the RIGHT side)")
    # the bar turned round: P(K | X)
    for v in (1.0, 0.5, -2.0):
        ks = [k for k, x in cards if x == v]
        print(f"  P(K=0|X={v:+.1f}) = {ks.count(0)}/{len(ks)}   P(K=1|X={v:+.1f}) = {ks.count(1)}/{len(ks)}")
    print(f"  P(X>=0.5) = {sum(x >= 0.5 for x in xs)}/8   P(X>=0.5|K=1) = {sum(x >= 0.5 for x in env[1]['r'])}/4   P(X<0) = {sum(x < 0 for x in xs)}/8")
    print("-- linearity")
    print(f"  E[2X+1] = {sum((2*v+1)*q for v, q in p.items()):+.4f}  = 2*E[X]+1 = {2*EX+1:+.4f}")
    print(f"  E[0.9X] = {sum(0.9*v*q for v, q in p.items()):+.5f}   E[X^2] = {sum(v*v*q for v, q in p.items()):.4f}   (E[X])^2 = {EX*EX:.4f}")
    print(f"  E[X|K=0] + 0.9*E[X|K=1] = {cond[0] + 0.9*cond[1]:+.4f}")
    print("-- sample mean of n draws with replacement (random.Random(7)); E[X] = %+.4f" % EX)
    rng = random.Random(7)
    draws = [rng.choice(xs) for _ in range(8000)]
    print("  first 8 draws:", draws[:8])
    print(f"  the value 1.0 appears {draws[:8].count(1.0)} times in the first 8")
    for m in (1, 8, 80, 800, 8000):
        print(f"  n={m:<5} mean = {sum(draws[:m])/m:+.4f}   |mean - E[X]| = {abs(sum(draws[:m])/m - EX):.4f}")
    print("-- two draws with replacement: E[X1 + 0.9*X2]")
    pairs = [a + 0.9 * b for a in xs for b in xs]
    print(f"  by linearity {EX + 0.9*EX:.5f};  by enumerating {len(pairs)} pairs {sum(pairs)/len(pairs):.5f}")
    print(f"-- biased coin P(K=0)=3/4: 0.75*{cond[0]:+.4f} + 0.25*{cond[1]:+.4f} = {0.75*cond[0] + 0.25*cond[1]:+.5f}")
    print("-- one card removed: bag without the -2.0 card")
    ys = [x for x in xs if x != -2.0]
    print(f"  E = {sum(ys)}/{len(ys)} = {sum(ys)/len(ys):+.4f}")


def ch05():
    print("=== chapter 5 — the MDP form, and Tron1's MDP ===")
    dec, dt, step_dt, _ = timing()
    print("-- action")
    show(BASE, 116, 121)
    print(f"  joint_names ['.*'] -> every actuated joint of SF = {n_joints()}")
    print("-- policy observation group (ObservarionsCfg.PolicyCfg)")
    c = cfg_class("ObservarionsCfg", "PolicyCfg")
    print(f"  class lines {c.lineno}-{c.end_lineno}")
    rows = obs_dims("ObservarionsCfg", "PolicyCfg")
    for name, ln, func, w in rows:
        print(f"  :{ln}  {name:<14} func=mdp.{func:<20} width {w}")
    print(f"  total = {' + '.join(str(w) for *_, w in rows)} = {sum(w for *_, w in rows)}")
    print(f"  without last_action: {sum(w for n, _, _, w in rows if n != 'last_action')}")
    show(LOCO / "mdp/commands/gait_command.py", 34); show(LOCO / "mdp/commands/gait_command.py", 63, 69)
    obs_py = LOCO / "mdp/observations.py"
    a = find_line(obs_py, r"def get_gait_phase"); show(obs_py, a); show(obs_py, find_line(obs_py, r"gait_indices = torch.remainder", a)); show(obs_py, find_line(obs_py, r"sin_phase =", a), find_line(obs_py, r"cos_phase =", a))
    print("-- commands (CommandsCfg)")
    for name, ln, call in cfg_terms("CommandsCfg"):
        rng_ = _kw(call, "ranges")
        print(f"  :{ln}  {name:<14} {ast.unparse(call.func)}  resampling {ast.unparse(_kw(call, 'resampling_time_range'))}")
        for k in rng_.keywords:
            print(f"        {k.arg} = {ast.unparse(k.value)}  (line {k.value.lineno})")
    show(BASE, 226, 227)
    show(BASE, 100, 107)
    vc = ISAAC / "isaaclab/isaaclab/envs/mdp/commands/velocity_command.py"
    a = find_line(vc, r"def command\(self\)"); show(vc, a, a + 2)
    show(vc, 127, 138); show(vc, 147, 160)
    print("  -> heading_command=True and rel_heading_envs=1.0: ang_vel_z is recomputed from the heading error every step, clipped to its range")
    show(RSL / "algorithm/ppo.py", 137, 139)
    show(RSL / "runner/on_policy_runner.py", 59)
    print("-- transition P: physics")
    show(RL_ENV, 182, 189); show(BASE, 105); show(BASE, 134)
    print(f"  {dec} physics steps of {dt} s inside one decision of {step_dt:.2f} s")
    print("-- reward: 20 entries")
    rt = reward_terms()
    pos = [r for r in rt if r[3] > 0]; neg = [r for r in rt if r[3] < 0]
    for name, ln, func, w in rt:
        places = defined_in(func)
        pth, fl, kind = places[-1]
        owner = "project" if T in pth.parents else "IsaacLab"
        print(f"  :{ln:<4} {name:<34} weight {w!s:<8} mdp.{func:<28} {owner} {pth.name}:{fl}")
    print(f"  entries {len(rt)}: positive weight {len(pos)} (sum {sum(r[3] for r in pos)}), negative weight {len(neg)}")
    print(f"  largest |weight|: {max(rt, key=lambda r: abs(r[3]))[0]}   smallest |weight|: {min(rt, key=lambda r: abs(r[3]))[0]}")
    print(f"  ratio rew_lin_vel_xy / |pen_joint_power_l1| = {2.5/2e-5:,.0f}")
    print("  test_gait_reward: weight +1.0 but both inner scales are negative, so the function is <= 0:")
    show(BASE, 434, 435); show(LOCO / "mdp/rewards.py", 466); show(LOCO / "mdp/rewards.py", 478); show(LOCO / "mdp/rewards.py", 413, 418)
    show(BASE, 372, 375)
    print("-- one step of reward, by hand (func x weight x dt, dt = %.2f)" % step_dt)
    print(f"  keep_balance: 1 x 1.0 x {step_dt} = {1*1.0*step_dt:.4f} per step; over 1 s ({1/step_dt:.0f} steps) = {1.0:.2f}")
    std2 = 0.25
    for err in (0.0, 0.5, 1.0):
        f = math.exp(-err**2 / std2)
        print(f"  rew_lin_vel_xy, speed error {err} m/s: exp(-{err**2:.2f}/{std2}) = {f:.4f}; x 2.5 x {step_dt} = {f*2.5*step_dt:.4f}")
    f05 = math.exp(-0.25 / std2)
    print(f"  example r_t with only these two entries non-zero, error 0.5: {1*1.0*step_dt:.4f} + {f05*2.5*step_dt:.4f} = {step_dt + f05*2.5*step_dt:.4f}")
    print(f"  exercise: error 0 -> {step_dt:.4f} + {2.5*step_dt:.4f} = {step_dt + 2.5*step_dt:.4f};   error 1.0 -> {step_dt + math.exp(-4)*2.5*step_dt:.4f}")
    print(f"  pen_action_rate with one joint changing by 0.2 between steps: 0.2^2 x -0.5 x {step_dt} = {0.04*-0.5*step_dt:.5f}")
    iso = ISAAC / "isaaclab/isaaclab/envs/mdp/rewards.py"
    for fn in ("track_lin_vel_xy_exp", "action_rate_l2", "joint_torques_l2"):
        a = find_line(iso, rf"^def {fn}\b"); b = find_line(iso, r"^    return ", a)
        print(f"  IsaacLab rewards.py: def {fn} at :{a}, return at :{b}")
        show(iso, b - (4 if fn == "track_lin_vel_xy_exp" else 0), b)
    tr = LOCO / "mdp/rewards.py"
    show(tr, 23, 25); show(tr, find_line(tr, r"penalty = torch.sum\(torch.square\(current_action - 2"))
    show(BASE, 378, 380)
    rm = ISAAC / "isaaclab/isaaclab/managers/reward_manager.py"
    show(rm, 141); show(rm, 148, 150); show(rm, 157)
    show(RL_ENV, 207); show(RL_ENV, 242); show(RSL / "runner/on_policy_runner.py", 177)
    print("-- gamma")
    show(PPO_CFG, 102)
    print("-- where each slot lives")
    for nm in ("CommandsCfg", "ActionsCfg", "ObservarionsCfg", "RewardsCfg"):
        print(f"  class {nm} at :{cfg_class(nm).lineno}")


def ch06():
    print("=== chapter 6 — the Markov property ===")
    dec, dt, step_dt, _ = timing()
    pol = [n for n, *_ in cfg_terms("ObservarionsCfg", "PolicyCfg")]
    cri = cfg_terms("ObservarionsCfg", "CriticCfg")
    c = cfg_class("ObservarionsCfg", "CriticCfg")
    print(f"policy group terms ({len(pol)}): {', '.join(pol)}")
    print(f"critic group (lines {c.lineno}-{c.end_lineno}) terms ({len(cri)}); not in the policy group:")
    only = [(n, ln) for n, ln, _ in cri if n not in pol]
    for n, ln in only:
        print(f"  :{ln}  {n}")
    print(f"  {len(only)} terms the policy never sees;  shared: {len(cri) - len(only)}")
    sfb = LOCO / "robots/limx_solefoot_env_cfg.py"
    show(sfb, 76, 78)
    print("  (the task SFBlindFlatEnvCfg removes `heights`, so the critic group actually used has one term fewer)")
    h = cfg_class("ObservarionsCfg", "HistoryObsCfg")
    hist = int(re.search(r"self\.history_length = (\d+)", ast.get_source_segment(BASE.read_text(), h)).group(1))
    n_obs = sum(w for *_, w in obs_dims("ObservarionsCfg", "PolicyCfg"))
    print(f"history group lines {h.lineno}-{h.end_lineno}: history_length = {hist};  {hist} x {n_obs} = {hist*n_obs} numbers")
    print(f"  the buffer holds the CURRENT frame plus {hist-1} earlier ones (append, then read):")
    show(ISAAC / "isaaclab/isaaclab/managers/observation_manager.py", 320, 327)
    span = (hist - 1) * step_dt
    print(f"  oldest frame is {hist-1} x {step_dt} = {span:.2f} s before the current one")
    show(BASE, 176); show(PPO_CFG, 106); show(BASE, 217)
    show(BASE, 175); show(BASE, 183)
    show(RSL / "runner/on_policy_runner.py", 63)
    print("-- two robots with the same observation")
    for name, v in (("A", 1.0), ("B", 0.0)):
        print(f"  robot {name}: base speed {v} m/s -> moves {v*step_dt:.3f} m in one step, {v*span:.2f} m over the history window")
    print(f"-- exercises: history_length 25 -> {25*n_obs} numbers, oldest frame {24*step_dt:.2f} s back;   0.5 m/s over {span:.2f} s = {0.5*span:.2f} m")
    print(f"   critic group used by SFBlindFlatEnvCfg: {len(cri) - 1} terms")


def episodes(e):
    """Split one env's four steps into episodes: [(first t, last t, finished?)]."""
    out, start = [], 0
    for t in range(4):
        if env[e]["done"][t]:
            out.append((start, t, True)); start = t + 1
    if start <= 3:
        out.append((start, 3, False))
    return out


def ch07():
    print("=== chapter 7 — step, episode, done, one table ===")
    dec, dt, step_dt, ep_s = timing()
    cfg = sf_ppo()
    Trows, iters = cfg["num_steps_per_env"][0], cfg["max_iterations"][0]
    max_len = math.ceil(ep_s / step_dt)
    print(f"step = {dec} x {dt} = {step_dt:.2f} s ({1/step_dt:.0f} Hz); physics {1/dt:.0f} Hz")
    print(f"episode_length_s {ep_s} / {step_dt:.2f} = {max_len} steps max")
    for B in (2048, 4096, 64):
        print(f"  B={B}: T x B = {Trows} x {B} = {Trows*B:,} transitions per rollout")
    print(f"  rollout spans {Trows} x {step_dt:.2f} = {Trows*step_dt:.2f} s of simulated time per env")
    print(f"  exercise: num_steps_per_env 48 with B=2048 -> {48*2048:,} transitions, {48*step_dt:.2f} s per env")
    print(f"  a full-length episode covers {max_len}/{Trows} = {max_len/Trows:.2f} tables")
    print(f"  {iters:,} iterations x {Trows} = {iters*Trows:,} steps per env = {iters*Trows*step_dt:,.0f} s = {iters*Trows*step_dt/3600:.1f} h simulated")
    print(f"  total transitions with B=2048: {iters*Trows*2048:,}")
    print(f"  toy: T x B = 4 x 2 = 8;  toy rollout time if a step were {step_dt:.2f} s: {4*step_dt:.2f} s")
    print("-- the toy table")
    for e in (0, 1):
        d = env[e]
        print(f"  env {e}: mu={d['mu']} eps={d['eps']} r={d['r']} V={d['V']} done={d['done']} V_last={d['V_last']}")
        for a, b, fin in episodes(e):
            rs = d["r"][a:b + 1]
            print(f"     episode t{a}..t{b}: {'finished (done=1 at t%d)' % b if fin else 'still running at the table edge'};"
                  f" steps in table {len(rs)}; plain sum of r = {sum(rs):+.2f}")
    print(f"  done cells: {sum(sum(env[e]['done']) for e in (0,1))} of 8;  episodes touching the table: {sum(len(episodes(e)) for e in (0,1))}")
    print("  what the logger would record for env 1's finished episode: reward sum %+.2f, length %d" % (sum(env[1]['r'][:3]), 3))
    print("-- what-if: done at e0 t1 as well")
    print("  env 0 would hold episodes t0..t1 (finished, sum %+.2f, length 2) and t2..t3 (running, sum so far %+.2f)" % (sum(env[0]['r'][:2]), sum(env[0]['r'][2:])))
    st = RSL / "storage/rollout_storage.py"
    lines = st.read_text().splitlines()
    a = find_line(st, r"def add_transitions"); b = find_line(st, r"self\.step \+= 1", a)
    fields = [re.search(r"self\.(\w+)\[self\.step\]\.copy_", l).group(1) for l in lines[a:b] if ".copy_(" in l]
    print(f"-- add_transitions (rollout_storage.py:{a}-{b}) copies {len(fields)} slots: {', '.join(fields)}")
    print("-- lines cited")
    run = RSL / "runner/on_policy_runner.py"
    for x, y in ((170, 170), (174, 177), (192, 192), (200, 210), (221, 225), (326, 327)):
        show(run, x, y)
    show(RSL / "algorithm/ppo.py", 172); show(RSL / "algorithm/ppo.py", 331)
    show(st, 141, 142)
    for x, y in ((101, 103), (182, 189), (200, 200), (203, 207), (215, 220), (237, 239)):
        show(RL_ENV, x, y)
    term = ISAAC / "isaaclab/isaaclab/envs/mdp/terminations.py"
    a = find_line(term, r"^def time_out"); show(term, a, a + 2)
    a = find_line(term, r"^def illegal_contact"); show(term, a); show(term, find_line(term, r"return torch.any", a), find_line(term, r"return torch.any", a) + 2)
    show(BASE, 450, 454); show(BASE, 497, 501); show(BASE, 484)
    show(WRAP, 176, 178); show(WRAP, 184, 185)
    show(PPO_CFG, 81, 82)
    show(ROOT / "README.md", find_line(ROOT / "README.md", r"--num_envs 2048"))


def G_of(rs, gamma=GAMMA):
    """Return of a finished episode from each of its steps: G_t = r_t + gamma*G_{t+1}."""
    out, g = [], 0.0
    for r in reversed(rs):
        g = r + gamma * g
        out.append(g)
    return out[::-1]


def ch08():
    print("=== chapter 8 — return and the objective ===")
    ep1 = env[1]["r"][:3]
    print(f"env 1, first episode r = {ep1}  (gamma = {GAMMA})")
    for t, g in enumerate(G_of(ep1)):
        print(f"  G_{t} = {g:+.4f}")
    print(f"  expanded G_0 = {ep1[0]} + {GAMMA}*{ep1[1]} + {GAMMA**2:.2f}*{ep1[2]} = {ep1[0]:+.2f} {GAMMA*ep1[1]:+.2f} {GAMMA**2*ep1[2]:+.2f} = {G_of(ep1)[0]:+.4f}")
    for g_ in (1.0, 0.5, 0.0, 0.99):
        print(f"  gamma={g_}: G = " + "  ".join(f"{x:+.4f}" for x in G_of(ep1, g_)))
    print(f"  plain sum = {sum(ep1):+.2f}; length 3")
    rev = ep1[::-1]
    print(f"  same rewards, reversed order {rev}: G_0 = {G_of(rev)[0]:+.4f}  (plain sum still {sum(rev):+.2f})")
    worse = ep1[:2] + [-5.0]
    print(f"  if r_2 were -5.0: G = " + "  ".join(f"{x:+.4f}" for x in G_of(worse)))
    print("-- env 0: the episode is not finished inside the table")
    r0 = env[0]["r"]
    part = [GAMMA ** k * r0[k] for k in range(4)]
    print(f"  visible part of G_0 = " + " + ".join(f"{x:.4f}" for x in part) + f" = {sum(part):.4f};  missing: {GAMMA**4:.4f} x G_4")
    for t in range(4):
        print(f"  visible part of G_{t} = {sum(GAMMA**k * r0[t+k] for k in range(4-t)):.4f}   weight on the missing tail = {GAMMA**(4-t):.4f}")
    print(f"  env 1 second episode: visible part of G_3 = {env[1]['r'][3]:.1f}; missing {GAMMA} x G_4")
    print("-- discount weights gamma^k")
    for k in (1, 2, 5, 10, 22, 50, 100, 230, 1000):
        print(f"  k={k:<5} 0.9^k = {0.9**k:.4f}   0.99^k = {0.99**k:.4f}")
    for g_ in (0.9, 0.99):
        print(f"  gamma={g_}: 1/(1-gamma) = {1/(1-g_):.0f} steps = {1/(1-g_)*0.02:.1f} s at 0.02 s per step; "
              f"steps until weight < 0.5: {math.ceil(math.log(0.5)/math.log(g_))}; < 0.1: {math.ceil(math.log(0.1)/math.log(g_))}")
    print(f"  sum of gamma^k for k=0..inf: 0.9 -> {1/(1-0.9):.0f}, 0.99 -> {1/(1-0.99):.0f};  first 1000 terms at 0.99: {(1-0.99**1000)/(1-0.99):.4f}")
    print(f"  r = 1 every step forever, gamma 0.9: G = {1/(1-0.9):.1f};  r = 0.5 every step: {0.5/(1-0.9):.1f}")
    print(f"  r = 1 for 1000 steps: gamma=1 -> 1000; gamma=0.99 -> {(1-0.99**1000)/(1-0.99):.2f}")
    cfg = sf_ppo()
    print(f"-- code: gamma = {cfg['gamma'][0]} (line {cfg['gamma'][1]}), max_iterations = {cfg['max_iterations'][0]} (line {cfg['max_iterations'][1]})")
    run = RSL / "runner/on_policy_runner.py"
    show(run, 170); show(run, 200, 205); show(run, 326)
    show(PPO_CFG, 102)


def ch09():
    print("=== chapter 9 — value, Bellman, Q, advantage ===")
    print("-- Bellman check: right side r + gamma*V(next) against the critic's V (inside one episode; at a done cell the right side is r alone)")
    for e in (0, 1):
        d = env[e]
        for t in range(4):
            Vn = d["V_last"] if t == 3 else d["V"][t + 1]
            if d["done"][t]:
                rhs, how = d["r"][t], f"{d['r'][t]:+.1f} (episode ends)"
            else:
                rhs, how = d["r"][t] + GAMMA * Vn, f"{d['r'][t]:+.1f} + {GAMMA}*{Vn}"
            print(f"  e{e}t{t}: {how:<22} = {rhs:+.4f}   V = {d['V'][t]:.1f}   right - left = {rhs - d['V'][t]:+.4f}"
                  + ("   (uses V_last)" if t == 3 else ""))
    print("-- return against prediction, env 1 first episode")
    for t, g in enumerate(G_of(env[1]["r"][:3])):
        print(f"  t{t}: G = {g:+.4f}   V = {env[1]['V'][t]:.1f}   V - G = {env[1]['V'][t] - g:+.4f}")
    print("-- one-step estimate of Q and the rough advantage Q - V")
    for e, t in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1)):
        d = env[e]
        q = d["r"][t] + GAMMA * d["V"][t + 1]
        print(f"  e{e}t{t}: Q ~ {q:.4f}   A ~ {q - d['V'][t]:+.4f}")
    print("-- bootstrapping at the table edge (env 0): visible rewards + gamma^4 * V_last")
    r0 = env[0]["r"]
    for t in range(4):
        vis = sum(GAMMA ** k * r0[t + k] for k in range(4 - t))
        tail = GAMMA ** (4 - t) * env[0]["V_last"]
        print(f"  from t{t}: {vis:.4f} + {GAMMA**(4-t):.4f}*{env[0]['V_last']} = {vis + tail:.4f}   critic said V = {env[0]['V'][t]}   difference {vis + tail - env[0]['V'][t]:+.4f}")
    print(f"  env 1 from t3: {env[1]['r'][3]} + {GAMMA}*{env[1]['V_last']} = {env[1]['r'][3] + GAMMA*env[1]['V_last']:.4f}   V = {env[1]['V'][3]}")
    print("-- two-step look-ahead from e0 t0: r0 + gamma*r1 + gamma^2*V(s2)")
    print(f"  {r0[0]} + {GAMMA}*{r0[1]} + {GAMMA**2:.2f}*{env[0]['V'][2]} = {r0[0] + GAMMA*r0[1] + GAMMA**2*env[0]['V'][2]:.4f}")
    print("-- if the critic were exact on env 1's first episode it would have said V = G:", "  ".join(f"{g:+.2f}" for g in G_of(env[1]['r'][:3])))
    print("-- lines cited")
    ac = RSL / "modules/actor_critic.py"
    show(ac, 92, 94); show(ac, 98); show(ac, 111); show(ac, 170, 172)
    show(RSL / "algorithm/ppo.py", 144); show(RSL / "algorithm/ppo.py", 176, 178)
    st = RSL / "storage/rollout_storage.py"
    show(st, 143); show(st, 190, 191); show(st, 195, 199)
    hits = [f"{q.relative_to(T)}:{i}" for q in (T / "rsl_rl/rsl_rl").rglob("*.py") for i, l in enumerate(q.read_text().splitlines(), 1)
            if re.search(r"\bq_value|\bQ\(|q_function|QFunction|q_net", l, re.I)]
    print(f"  lines in rsl_rl/ that mention a Q-function: {len(hits)} {hits[:3]}")



CHAPTERS = {"01": ch01, "02": ch02, "03": ch03, "04": ch04, "05": ch05, "06": ch06, "07": ch07, "08": ch08, "09": ch09, "22": ch22}

if __name__ == "__main__":
    for key in sys.argv[1:] or CHAPTERS:
        CHAPTERS[key.zfill(2)]()
