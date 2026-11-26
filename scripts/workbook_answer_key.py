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



# ---------------------------------------------------------------- part C (chapters 10-15): the policy
SIG = toy.SIGMA
TORCH_NORMAL = pathlib.Path.home() / "anaconda3/envs/tron1/lib/python3.10/site-packages/torch/distributions/normal.py"


def pdf(a, mu, s=SIG):
    return math.exp(-(a - mu) ** 2 / (2 * s * s)) / (s * math.sqrt(2 * math.pi))


def a_of(e, t):
    return cells[(e, t)]["a"]


def ch10():
    print("=== chapter 10 — the Gaussian bell and log ===")
    print(f"-- peak height 1/(sigma*sqrt(2pi)); sqrt(2pi) = {math.sqrt(2*math.pi):.5f}")
    for s in (1.0, 0.5, 0.3, 0.25, 0.1):
        print(f"  sigma={s:<5} peak = {1/(s*math.sqrt(2*math.pi)):.4f}   log(peak) = {math.log(1/(s*math.sqrt(2*math.pi))):+.4f}")
    print("-- attenuation exp(-z^2/2)")
    for z in (0, 0.4, 1, 2, 3):
        print(f"  z={z:<4} {math.exp(-z*z/2):.4f}")
    print("-- area within k sigma (erf(k/sqrt2))")
    for k in (1, 2, 3):
        print(f"  |a-mu| < {k} sigma: {math.erf(k/math.sqrt(2)):.4f}")
    print("-- the 8 steps at e0t0 (mu=0.20, sigma=0.5, a=0.40)")
    mu, a = 0.20, 0.40
    s1 = SIG * math.sqrt(2 * math.pi); s2 = 1 / s1; d = a - mu; e = -(d * d) / (2 * SIG * SIG)
    print(f"  sigma*sqrt(2pi)={s1:.5f}  peak={s2:.5f}  a-mu={d:.2f}  (a-mu)^2={d*d:.2f}  2sigma^2={2*SIG*SIG}  exponent={e:.2f}  exp={math.exp(e):.5f}  p={s2*math.exp(e):.5f}")
    print(f"  z = {d/SIG:.2f}")
    print("-- the same bell at other points (06 §3.5.2 row)")
    for x in (-1.0, 0.0, 0.20, 0.40, 1.5):
        print(f"  p({x:+.2f}) = {pdf(x, 0.20):.4f}")
    print("-- small interval around 0.40: density x width, against the exact area")
    for w in (0.02, 0.2):
        lo, hi = a - w / 2, a + w / 2
        exact = 0.5 * (math.erf((hi - mu) / (SIG * math.sqrt(2))) - math.erf((lo - mu) / (SIG * math.sqrt(2))))
        print(f"  width {w}: {pdf(a, mu):.4f} x {w} = {pdf(a, mu)*w:.5f}   exact = {exact:.5f}")
    print(f"  P(a > mu) = 0.5;  P(a < 0.40) = {0.5*(1+math.erf((a-mu)/(SIG*math.sqrt(2)))):.4f}")
    C = -0.5 * math.log(2 * math.pi) - math.log(SIG)
    print(f"-- log form: -0.5 ln 2pi = {-0.5*math.log(2*math.pi):.4f}   -ln sigma = {-math.log(SIG):+.4f}   constant = {C:.4f}")
    print("-- table: a = mu + sigma*eps (chapter 12 derives it); log p = C - (a-mu)^2/(2 sigma^2)")
    for (e_, t), c in cells.items():
        d = c["a"] - c["mu"]
        print(f"  e{e_}t{t}: mu={c['mu']:+.2f} a={c['a']:+.2f} a-mu={d:+.2f} term={d*d/(2*SIG*SIG):.4f} log p={c['logpi_old']:+.4f}  p={math.exp(c['logpi_old']):.4f}  z={d/SIG:+.2f}  attenuation={math.exp(-d*d/(2*SIG*SIG)):.4f}")
    print(f"  sigma=0.25 at the peak: log p = {math.log(1/(0.25*math.sqrt(2*math.pi))):+.4f}")
    print(f"  sigma=0.25, a=0.40, mu=0.20: p = {pdf(0.40, 0.20, 0.25):.4f}  log = {math.log(pdf(0.40,0.20,0.25)):+.4f}")
    print(f"  sigma=1.0, a=0.40, mu=0.20: p = {pdf(0.40, 0.20, 1.0):.4f}  log = {math.log(pdf(0.40,0.20,1.0)):+.4f}")
    print("-- products vs sums: 8 densities of 0.7365 multiplied:", f"{0.73654**8:.5f}", " sum of logs:", f"{8*math.log(0.73654):.4f}")
    print("-- lines cited"); show(TORCH_NORMAL, 80, 92); show(RSL / "modules/actor_critic.py", 157); show(RSL / "modules/actor_critic.py", 164)


def ch11():
    print("=== chapter 11 — what the policy is ===")
    ac = RSL / "modules/actor_critic.py"
    dims = [42, 512, 256, 128, 8]
    net = sum(i * o + o for i, o in zip(dims, dims[1:]))
    print(f"  actor numbers {net:,} + logstd 8 = theta {net + 8:,}")
    for l in (0.0, math.log(0.5), -2.302585, 0.5, -100.0):
        print(f"  ell={l:+.4f}  sigma=exp(ell)={math.exp(l):.4g}")
    print("-- the bell at e0t0 is N(0.20, 0.5); the density at several a (type change: one function, many numbers)")
    for x in (-1.0, 0.0, 0.20, 0.40, 1.5):
        print(f"  pi(a={x:+.2f}|s) = {pdf(x, 0.20):.4f}")
    print("-- the mu row and the bell of every cell (sigma fixed 0.5)")
    for (e, t), c in cells.items():
        print(f"  e{e}t{t}: N({c['mu']:+.2f}, 0.5)  peak at a={c['mu']:+.2f}  logpi(a={c['a']:+.2f}) = {c['logpi_old']:+.4f}")
    print("-- (15) eight joints, glossary 8.6: joint 1 = e0t0")
    mu8 = [0.20, -0.30, 0.10, 0.40, -0.10, 0.60, -0.50, 0.30]
    eps8 = [0.40, -0.20, 0.60, 0.00, -0.60, 1.20, -0.80, 0.20]
    a8 = [m + SIG * e for m, e in zip(mu8, eps8)]
    lp = [toy.logpi(a, m) for a, m in zip(a8, mu8)]
    print("  a =", [round(x, 2) for x in a8])
    print("  log p_i =", [round(x, 4) for x in lp], " p_i =", [round(math.exp(x), 4) for x in lp])
    print(f"  sum = {sum(lp):.4f}   product of p_i = {math.prod(math.exp(x) for x in lp):.5f}   exp(sum) = {math.exp(sum(lp)):.5f}")
    print(f"  first four joints: sum = {sum(lp[:4]):.4f}  product = {math.exp(sum(lp[:4])):.5f};  last four: sum = {sum(lp[4:]):.4f}")
    print(f"  if all 8 joints sat at their mean (eps=0): sum = {8*toy.C:.4f}  product = {math.exp(8*toy.C):.5f}")
    print("-- deterministic (deploy) uses mu itself: e0 row", env[0]["mu"], " e1 row", env[1]["mu"])
    print("-- lines cited")
    show(ac, 117, 118); show(ac, 155, 168); show(PPO_CFG, 87)
    show(RSL / "algorithm/ppo.py", 137, 139); show(RSL / "algorithm/ppo.py", 147, 149)


def ch12():
    print("=== chapter 12 — sampling the action, explore and exploit ===")
    for (e, t), c in cells.items():
        print(f"  e{e}t{t}: mu={c['mu']:+.2f} eps={c['eps']:+.2f} sigma*eps={SIG*c['eps']:+.2f} a={c['a']:+.2f}   z=(a-mu)/sigma={(c['a']-c['mu'])/SIG:+.2f}   log p = C - eps^2/2 = {toy.C - c['eps']**2/2:+.4f}")
    print(f"  C = {toy.C:.4f}")
    print("-- how often |eps| exceeds k (two-sided)")
    for k in (0.5, 1, 2, 3):
        print(f"  P(|eps| > {k}) = {1 - math.erf(k/math.sqrt(2)):.4f}")
    print("-- deviation sigma*eps for one eps at several sigma (eps = +0.80, e0t2)")
    for s in (1.0, 0.5, 0.25, 0.1):
        print(f"  sigma={s}: a - mu = {s*0.80:+.3f}")
    print("-- eight joints (glossary 8.6)")
    mu8 = [0.20, -0.30, 0.10, 0.40, -0.10, 0.60, -0.50, 0.30]
    eps8 = [0.40, -0.20, 0.60, 0.00, -0.60, 1.20, -0.80, 0.20]
    print("  a =", [f"{m + SIG*e:+.2f}" for m, e in zip(mu8, eps8)])
    print("  same eps, sigma = 1.0 (Tron1 at the start):", [f"{m + 1.0*e:+.2f}" for m, e in zip(mu8, eps8)])
    print("-- numbers of draws per step in Tron1: 2048 env x 8 joints =", 2048 * 8)
    print("-- lines cited")
    ac = RSL / "modules/actor_critic.py"
    show(ac, 155, 161); show(TORCH_NORMAL, 70, 78)
    run = RSL / "runner/on_policy_runner.py"
    show(run, 269); show(run, 284); show(run, 324)


def ch13():
    print("=== chapter 13 — inside the neural network ===")
    cfg = sf_ppo()
    hid = cfg["actor_hidden_dims"][0]
    print(f"  actor_hidden_dims = {hid} (line {cfg['actor_hidden_dims'][1]})  critic_hidden_dims = {cfg['critic_hidden_dims'][0]}  activation = {cfg['activation'][0]}")
    nj = n_joints()
    pol = sum(w for *_, w in obs_dims("ObservarionsCfg", "PolicyCfg"))
    print(f"  joints = {nj};  policy obs = {pol};  actor input = 3 + {pol} + 3 = {3 + pol + 3}")

    def count(dims, show_layers=True):
        tot = 0
        for i, o in zip(dims, dims[1:]):
            n = i * o + o
            tot += n
            if show_layers:
                print(f"    Linear({i}, {o}): {i}x{o} = {i*o:,} weights + {o} biases = {n:,}")
        return tot
    a_dims = [3 + pol + 3] + list(hid) + [nj]
    print("-- actor", a_dims)
    A = count(a_dims)
    wts = sum(i * o for i, o in zip(a_dims, a_dims[1:])); bs = sum(a_dims[1:])
    print(f"  total {A:,} = weights {wts:,} + biases {bs:,};  theta = {A + nj:,}")
    print("-- critic [230, 512, 256, 128, 1] (230 = 227 privileged + 3 commands, chapter 20)")
    Cn = count([230] + list(hid) + [1])
    print(f"  total {Cn:,};  actor + critic + logstd = {A + Cn + nj:,}")
    print("-- if every ELU were removed: the stack collapses to one Linear(42, 8):", 42 * 8 + 8)
    print("-- class default dims [256,256,256] (actor_critic.py:49):", f"{count([42,256,256,256,8], False):,}")
    print("-- bigger [1024,512,256]:", f"{count([42,1024,512,256,8], False):,}", "   smaller [128,64]:", f"{count([42,128,64,8], False):,}")
    print("-- share of actor numbers per layer")
    for i, o in zip(a_dims, a_dims[1:]):
        print(f"    {i}->{o}: {(i*o+o)/A*100:.1f}%")
    print("-- ELU, tanh, ReLU at a few points (value / slope)")
    elu = lambda x: x if x > 0 else math.exp(x) - 1
    delu = lambda x: 1.0 if x > 0 else math.exp(x)
    for x in (2.0, 0.5, 0.0, -0.5, -1.0, -2.0, -3.0):
        print(f"  x={x:+.1f}: ELU={elu(x):+.4f} (slope {delu(x):.4f})   tanh={math.tanh(x):+.4f} (slope {1-math.tanh(x)**2:.4f})   ReLU={max(0.0,x):+.1f} (slope {1.0 if x > 0 else 0.0})")
    for y in (0.9, 0.99, 0.999):
        u = math.atanh(y)
        print(f"  tanh(u) = {y}: u = {u:.3f}, slope = {1 - y*y:.4f}")
    print("-- shapes for the toy B = 2:", " -> ".join(f"(2, {d})" for d in a_dims))
    print("-- lines cited")
    ac = RSL / "modules/actor_critic.py"
    show(ac, 69, 77); show(ac, 88, 90); show(ac, 98); show(ac, 113); show(ac, 175, 177)
    show(PPO_CFG, 88, 90); show(RSL / "algorithm/ppo.py", 137, 139)


def ch14():
    print("=== chapter 14 — loss, gradient, gradient descent ===")
    a, mu0, l0 = 0.40, 0.20, math.log(SIG)
    alpha = 0.1   # glossary §5: lr in the toy = 0.1 (SGD, to see the numbers)
    def L(mu, l):  # -log density of a under N(mu, e^l)
        return (a - mu) ** 2 / (2 * math.exp(2 * l)) + l + 0.5 * math.log(2 * math.pi)
    def dmu(mu, l): return -(a - mu) / math.exp(2 * l)
    def dl(mu, l): return 1 - ((a - mu) / math.exp(l)) ** 2
    print(f"  L(0.20, ln0.5) = {L(mu0, l0):.4f}   dL/dmu = {dmu(mu0, l0):+.4f}   dL/dl = {dl(mu0, l0):+.4f}  (z^2 = {((a-mu0)/SIG)**2:.2f})")
    print("-- 1-D: only the last-layer bias b moves (dmu/db = 1), sigma fixed 0.5")
    for al in (0.1, 0.25, 0.5, 0.6):
        mu, row = mu0, []
        for k in range(6):
            row.append(f"{mu:.5f}")
            mu = mu - al * dmu(mu, l0)
        print(f"  alpha={al}: factor 1-alpha/sigma^2 = {1 - al/SIG**2:+.2f}   mu_k = " + "  ".join(row))
    mu = mu0
    print("  alpha=0.1 table:")
    for k in range(6):
        print(f"    k={k}: mu={mu:.5f}  a-mu={a-mu:.5f}  L={L(mu, l0):.4f}  dL/db={dmu(mu, l0):+.4f}  step={-alpha*dmu(mu, l0):+.5f}")
        mu = mu - alpha * dmu(mu, l0)
    print(f"  minimum: mu = a = 0.40, L = {L(a, l0):.4f}")
    print("-- numerical slope check at mu = 0.20 (central difference, h = 0.01)")
    h = 0.01
    print(f"  (L(0.21) - L(0.19)) / 0.02 = {(L(mu0+h, l0) - L(mu0-h, l0))/(2*h):+.4f}")
    print("-- 2-D: both knobs b and ell, one SGD step alpha = 0.1")
    g = (dmu(mu0, l0), dl(mu0, l0))
    mu1, l1 = mu0 - alpha * g[0], l0 - alpha * g[1]
    print(f"  grad = ({g[0]:+.4f}, {g[1]:+.4f})   |grad| = {math.hypot(*g):.4f}")
    print(f"  new mu = {mu1:.4f}  new ell = {l1:.4f}  sigma = {math.exp(l1):.4f}   L: {L(mu0, l0):.4f} -> {L(mu1, l1):.4f}")
    print(f"  only b moved: L = {L(mu1, l0):.4f};  only ell moved: L = {L(mu0, l1):.4f}")
    print(f"  ell-only minimum: sigma = |a-mu| = 0.20, ell = {math.log(0.2):.4f}, L = {L(mu0, math.log(0.2)):.4f}")
    print(f"-- keep descending both knobs at alpha = 0.1: mu factor 1 - alpha/sigma^2 leaves (-1, 1) below sigma = {math.sqrt(alpha/2):.4f}")
    mu, l = mu0, l0
    for k in range(1, 20):
        mu, l = mu - alpha * dmu(mu, l), l - alpha * dl(mu, l)
        if k in (5, 8, 9, 16, 17, 18, 19):
            print(f"    step {k}: mu = {mu:+.4f}  sigma = {math.exp(l):.4g}")
    print("-- the same loss at e0t2 (mu=0.10, a=0.50) and e0t3 (mu=0.40, a=0.40)")
    for (e, t) in ((0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)):
        c = cells[(e, t)]
        d = c["a"] - c["mu"]; z = d / SIG
        print(f"  e{e}t{t}: L={-c['logpi_old']:.4f}  dL/dmu = {-d/SIG**2:+.4f}  dL/dell = {1 - z*z:+.4f}  -> mu after one step {c['mu'] + alpha*d/SIG**2:+.4f}, ell after {l0 - alpha*(1-z*z):+.4f} (sigma {math.exp(l0 - alpha*(1-z*z)):.4f})")
    print("-- tanh at the output would multiply the slope by 1 - tanh^2:")
    for y in (0.5, 0.9, 0.99):
        print(f"  mu = tanh(u) = {y}: dmu/du = {1-y*y:.4f}")
    print("-- knob counts: actor 187,272 + critic 282,625 + logstd 8 =", f"{187272 + 282625 + 8:,}")
    print(f"-- Tron1 lr = {sf_ppo()['learning_rate'][0]} (line {sf_ppo()['learning_rate'][1]});  toy alpha = 0.1")
    print("-- lines cited")
    ppo = RSL / "algorithm/ppo.py"
    show(ppo, 85); show(ppo, 286, 290); show(PPO_CFG, 100)


def ch15():
    print("=== chapter 15 — from mu to the motor ===")
    base = BASE.read_text().splitlines()
    print("  ActionsCfg:"); show(BASE, 116, 121)
    sole = EXT / "assets/config/solefoot_cfg.py"
    show(sole, 32, 34); show(sole, 48, 51); show(sole, 59, 62)
    jact = ISAAC / "isaaclab/isaaclab/envs/mdp/actions/joint_actions.py"
    show(jact, 134); show(jact, 155, 156); show(jact, 160)
    show(RL_ENV, 173); show(RL_ENV, 182, 189)
    dec, dt, step_dt, _ = timing()
    print(f"  decimation {dec} x {dt} s = {step_dt} s per policy step ({1/step_dt:.0f} Hz); PD every {dt} s ({1/dt:.0f} Hz)")
    scale, kp, kd_leg, kd_ank, lim = 0.25, 45.0, 1.5, 0.8, 80.0
    deg = 180 / math.pi
    print("-- (17) q_target = 0.25 a + 0 for every cell; deploy would use mu instead of a")
    for (e, t), c in cells.items():
        q = scale * c["a"]; qd = scale * c["mu"]
        print(f"  e{e}t{t}: a={c['a']:+.2f} -> q_target={q:+.4f} rad = {q*deg:+.2f} deg   (deploy: mu={c['mu']:+.2f} -> {qd:+.4f} rad = {qd*deg:+.2f} deg)")
    print("-- the scale table")
    for x in (0.40, 1.0, 2.0):
        print(f"  a={x}: no scale {x:.2f} rad = {x*deg:.1f} deg;  scale 0.25: {scale*x:.2f} rad = {scale*x*deg:.1f} deg")
    print("-- (18) PD torque at rest (qdot = 0) from the default pose q = 0")
    for (e, t) in ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)):
        q = scale * a_of(e, t)
        print(f"  e{e}t{t}: tau = 45 x {q:+.4f} = {kp*q:+.4f} N m")
    print(f"  saturation: |q_target - q| > {lim}/{kp} = {lim/kp:.4f} rad = {lim/kp*deg:.1f} deg (at rest)")
    print(f"  that error needs |a| > {lim/kp/scale:.3f} from the default pose")
    print(f"  example: q = 0.05, qdot = +1.0, q_target = 0.10: leg tau = 45x0.05 - 1.5x1.0 = {kp*0.05 - kd_leg*1.0:+.2f};  ankle = {kp*0.05 - kd_ank*1.0:+.2f}")
    print(f"  q = 0.10 reached, qdot = +0.5: leg tau = {kp*0 - kd_leg*0.5:+.2f}")
    print(f"  q = 0.12 overshoot, qdot = 0: tau = {kp*(0.10-0.12):+.2f}")
    print(f"  if scale were 0.5, a that gives 0.10 rad: {0.10/0.5:.2f}")

# ---------------------------------------------------------------- parts D-E (chapters 16-21): dead ends, then actor/critic/encoder
import collections
import pickle
import zipfile

PPO_PY = RSL / "algorithm/ppo.py"
AC_PY = RSL / "modules/actor_critic.py"
ENC_PY = RSL / "modules/mlp_encoder.py"
STORE_PY = RSL / "storage/rollout_storage.py"
RUN_PY = RSL / "runner/on_policy_runner.py"
PLAY_PY = T / "scripts/rsl_rl/play.py"
OBS_MGR = ISAAC / "isaaclab/isaaclab/managers/observation_manager.py"


def G1(rs):
    """Return of a finished episode from its first step."""
    return G_of(rs)[0]


def mlp_count(dims, label=None):
    """Parameters of Linear(d0,d1) -> ... -> Linear(d_{n-1}, d_n); prints one line per layer when label is given."""
    tot = 0
    for i, o in zip(dims, dims[1:]):
        n = i * o + o
        tot += n
        if label:
            print(f"    {label} Linear({i}, {o}): {i}x{o} = {i*o:,} + {o} = {n:,}")
    return tot


def ckpt_shapes():
    """Tensor shapes stored in the newest training checkpoint, read WITHOUT torch and without running any pickled
    code: every class the pickle names is replaced by a stub, so only the shape tuples survive. None if no file."""
    runs = sorted((T / "logs/rsl_rl/sf_tron_1a_flat").glob("*/model_*.pt"))
    if not runs:
        return None, None

    class Stub:
        def __init__(self, *a, **k):
            pass

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == "_rebuild_tensor_v2":
                return lambda storage, offset, size, *rest: tuple(size)
            if name == "_rebuild_parameter":
                return lambda data, *rest: data
            if name == "OrderedDict":
                return collections.OrderedDict
            return Stub

        def persistent_load(self, pid):
            return None

    with zipfile.ZipFile(runs[-1]) as z:
        d = U(z.open(next(n for n in z.namelist() if n.endswith("data.pkl")))).load()
    return runs[-1], {k: dict(d[k]) for k in ("model_state_dict", "encoder_state_dict")}


def chain_dp(e, sweeps=5):
    """Chapter 16's what-if: read ONE column of the toy table as a tiny world with 4 states visited in order, known
    rewards and known successors (the DP assumption), the state after the table pinned at V_last, and a done cell
    having no successor. Synchronous Bellman sweeps V <- r + mask*gamma*V(next), starting from the critic's row."""
    d = env[e]
    V = list(d["V"])
    rows = [list(V)]
    for _ in range(sweeps):
        V = [d["r"][t] + (1 - d["done"][t]) * GAMMA * (d["V_last"] if t == 3 else V[t + 1]) for t in range(4)]
        rows.append(list(V))
    return rows


def ch16():
    print("=== chapter 16 — why a table cannot hold Tron1's values ===")
    for e in (0, 1):
        print(f"-- env {e} column read as a 4-state chain (model known, V after the table pinned at V_last = {env[e]['V_last']})")
        for k, row in enumerate(chain_dp(e)):
            print(f"  sweep {k}: " + "  ".join(f"{v:+.4f}" for v in row))
    print("  env 0 check: 2.6245 + 0.9^4 x 3.0 =", f"{sum(GAMMA**k * env[0]['r'][k] for k in range(4)) + GAMMA**4 * 3.0:.4f}")
    print("  env 1 check: G_0, G_1, G_2 =", [round(G1(env[1]['r'][t:3]), 4) for t in range(3)])
    cfg = sf_ppo()
    T_, iters = cfg["num_steps_per_env"][0], cfg["max_iterations"][0]
    B_ = 2048
    per_iter = T_ * B_
    total = per_iter * iters
    print(f"-- transitions: {T_} x {B_} = {per_iter:,} per iteration (lines {cfg['num_steps_per_env'][1]}, {cfg['max_iterations'][1]});  x {iters:,} iterations = {total:,} = {total:.3e}")
    nj = n_joints()
    pol = sum(w for *_, w in obs_dims("ObservarionsCfg", "PolicyCfg"))
    n_in = 3 + pol + 3
    for lv in (2, 10):
        rows = lv ** n_in
        print(f"  {lv} levels per number, {n_in} numbers: {lv}^{n_in} = {rows:.4e} rows;  rows / transitions of the whole run = {rows/total:.3e}")
        print(f"     with the action too ({nj} more numbers): {lv}^{n_in + nj} = {lv ** (n_in + nj):.4e} cells for a Q table")
    print(f"  2^42 exactly = {2**42:,}")
    print(f"  one table row per transition ever seen = {total:,} rows  vs  policy network {mlp_count([n_in, 512, 256, 128, nj]):,} numbers, critic {mlp_count([230, 512, 256, 128, 1]):,}")
    print(f"  rows of the 2-level table per number of the policy network: {2**42 / mlp_count([n_in, 512, 256, 128, nj]):.4e}")
    print(f"  share of a 2-level table the whole run could touch at most: {total / 2**42 * 100:.4f} %")
    print("-- toy: the table of the toy has 8 cells = transitions, not states")
    print("-- exercises")
    print(f"  3 levels: 3^{n_in} = {3**n_in:,} = {3**n_in:.4e};  / transitions = {3**n_in/total:.3e}")
    t4096 = T_ * 4096 * iters
    print(f"  4096 envs (cfg default): {T_} x 4096 x {iters:,} = {t4096:,};  share of 2^42 = {t4096/2**42*100:.4f} %")
    print(f"  a table with as many rows as the critic has numbers: log2(282,625) = {math.log2(282625):.2f}  (2^18 = {2**18:,}, 2^19 = {2**19:,})")
    print("  env 0 in ONE backward pass (last state first): ", end="")
    V = [0.0] * 4
    for t in (3, 2, 1, 0):
        V[t] = env[0]["r"][t] + GAMMA * (env[0]["V_last"] if t == 3 else V[t + 1])
    print("  ".join(f"s{t} = {V[t]:.4f}" for t in (3, 2, 1, 0)))
    print("  env 0 with gamma = 0.5, one backward pass: ", end="")
    for t in (3, 2, 1, 0):
        V[t] = env[0]["r"][t] + 0.5 * (env[0]["V_last"] if t == 3 else V[t + 1])
    print("  ".join(f"s{t} = {V[t]:.4f}" for t in (3, 2, 1, 0)))
    print("-- lines cited")
    show(PPO_PY, 331); show(STORE_PY, 184, 185); show(STORE_PY, 131, 133); show(STORE_PY, 143); show(AC_PY, 90); show(AC_PY, 111)
    show(PPO_CFG, 81, 82); show(RUN_PY, 177); show(RUN_PY, 355, 365); show(BASE, 484)


def nstep_value(e, t, k):
    """Value estimate of s_t from k real rewards, then the critic's number (stops early at an episode end)."""
    d = env[e]
    tot = 0.0
    for j in range(k):
        tot += GAMMA ** j * d["r"][t + j]
        if d["done"][t + j]:
            return tot
    nxt = t + k
    return tot + GAMMA ** k * (d["V_last"] if nxt == 4 else d["V"][nxt])


def ch17():
    print("=== chapter 17 — Monte Carlo, TD, and the TD error ===")
    print("-- three numbers for the first state of env 1")
    d = env[1]
    print(f"  critic V(s0) = {d['V'][0]}   one step: 0.5 + 0.9 x 2.0 = {nstep_value(1, 0, 1):.4f}   whole episode G_0 = {G1(d['r'][0:3]):+.4f}")
    print("-- k real steps, then the critic (value estimates)")
    for e, kmax in ((1, 3), (0, 4)):
        for t in range(3 if e == 1 else 1):
            print(f"  e{e} s{t}: " + "  ".join(f"k={k}: {nstep_value(e, t, k):+.4f}" for k in range(0, kmax - t + 1)))
    print("-- env 1, first episode: critic / one-step / Monte Carlo, and the two 'errors'")
    for t in range(3):
        G = G1(d["r"][t:3])
        print(f"  s{t}: V = {d['V'][t]:.1f}   one-step = {nstep_value(1, t, 1):+.4f}   G = {G:+.4f}   delta = {cells[(1,t)]['delta']:+.4f}   G - V = {G - d['V'][t]:+.4f}")
    print("-- delta, all 8 cells, with the pieces")
    for (e, t), c in cells.items():
        dd = env[e]
        Vn = dd["V_last"] if t == 3 else dd["V"][t + 1]
        m = 1 - c["done"]
        print(f"  e{e}t{t}: r={c['r']:+.1f}  mask={m}  gamma*V_next = 0.9 x {Vn} = {GAMMA*Vn:.2f}  V={c['V']:.1f}  ->  delta = {c['r']:+.1f} + {m} x {GAMMA*Vn:.2f} - {c['V']:.1f} = {c['delta']:+.4f}")
    print(f"  without the mask at e1t2: {-2.0 + GAMMA*4.0 - 1.0:+.4f}")
    print("-- what-ifs for the exercises")
    print(f"  e0 t1 if the robot fell there (done = 1): delta = 0.5 + 0 - 4.5 = {0.5 - 4.5:+.4f}")
    print(f"  e0 t0 with gamma = 0.99: {1.0 + 0.99*4.5 - 5.0:+.4f}")
    print(f"  e1 t3 if V_last were 0: {0.0 + 0.9*0.0 - 4.0:+.4f};  if V_last were 4.0/0.9 = {4.0/0.9:.4f}: delta = 0")
    print(f"  e0 t0 if the critic had said V(s0) = 5.05: delta = {1.0 + 0.9*4.5 - 5.05:+.4f}")
    print(f"  sum of delta over env 0 = {sum(cells[(0,t)]['delta'] for t in range(4)):+.4f};  env 1 = {sum(cells[(1,t)]['delta'] for t in range(4)):+.4f}")
    print(f"  cells with delta > 0: {[f'e{e}t{t}' for (e,t),c in cells.items() if c['delta'] > 0]}")
    dec, dt, step_dt, ep = timing()
    nj = n_joints()
    print(f"-- how many random draws sit inside each estimate: episode {ep:.0f} s / {step_dt} s = {ep/step_dt:.0f} steps x {nj} joints = {ep/step_dt*nj:.0f} draws (MC);  one step = {nj} draws (TD)")
    print(f"   table rows {sf_ppo()['num_steps_per_env'][0]} vs episode {ep/step_dt:.0f} steps")
    print("-- lines cited")
    show(STORE_PY, 187, 199); show(RUN_PY, 200, 205); show(RUN_PY, 225); show(PPO_PY, 176, 178)
    show(BASE, find_line(BASE, r"class TerminationsCfg"), find_line(BASE, r"class TerminationsCfg") + 12)


def rho_of(c, shift):
    return math.exp(toy.logpi(c["a"], c["mu"] + shift) - c["logpi_old"])


def integrate(f, lo=-8.0, hi=8.0, n=160001):
    """Plain trapezoid rule; the Gaussians here are negligible beyond +-8."""
    h = (hi - lo) / (n - 1)
    s = 0.5 * (f(lo) + f(hi)) + sum(f(lo + i * h) for i in range(1, n - 1))
    return s * h


def ch18():
    print("=== chapter 18 — on-policy, off-policy, importance sampling ===")
    cards = bag()
    print("-- the bag of chapter 4: samples drawn with a fair coin (q), average wanted under the weighted coin (p)")
    q = {0: 0.5, 1: 0.5}
    p = {0: 0.75, 1: 0.25}
    w = {k: p[k] / q[k] for k in (0, 1)}
    direct = sum(p[k] * sum(env[k]["r"]) / 4 for k in (0, 1))
    plain = sum(x for _, x in cards) / 8
    weighted = sum(w[k] * x for k, x in cards) / 8
    print(f"  weights p/q: box 0 = {w[0]}, box 1 = {w[1]}")
    print(f"  plain average of the 8 cards (fair coin) = {plain:.5f}")
    print(f"  weighted: (1.5 x {sum(env[0]['r'])} + 0.5 x {sum(env[1]['r'])}) / 8 = {weighted:.5f};   direct E_p[X] = 0.75 x 0.75 + 0.25 x (-0.125) = {direct:.5f}")
    print(f"  average weight = {sum(w[k] for k, _ in cards)/8}")
    p2 = {0: 0.25, 1: 0.75}
    print(f"  exercise: p = (1/4, 3/4): weights {p2[0]/0.5}, {p2[1]/0.5};  weighted = {sum(p2[k]/0.5 * x for k, x in cards)/8:.5f};  direct = {sum(p2[k]*sum(env[k]['r'])/4 for k in (0,1)):.5f}")
    print("-- rho with mu_new = mu_old + 0.05 in every cell (the demo shift of glossary 8.1)")
    for (e, t), c in cells.items():
        lp_new = toy.logpi(c["a"], c["mu"] + 0.05)
        print(f"  e{e}t{t}: a={c['a']:+.2f} mu_old={c['mu']:+.2f} mu_new={c['mu']+0.05:+.2f}  (a-mu_new)^2/0.5 = {(c['a']-c['mu']-0.05)**2/0.5:.4f}  "
              f"logpi_new={lp_new:+.4f}  logpi_old={c['logpi_old']:+.4f}  diff={lp_new - c['logpi_old']:+.4f}  rho={math.exp(lp_new - c['logpi_old']):.4f}")
    c = cells[(0, 0)]
    print(f"  e0t0 densities: old {pdf(0.40, 0.20):.4f}  new {pdf(0.40, 0.25):.4f}  ratio {pdf(0.40, 0.25)/pdf(0.40, 0.20):.4f}")
    print(f"  mean rho over the 8 cells = {sum(rho_of(c, 0.05) for c in cells.values())/8:.4f}")
    print("-- how far rho goes as the bell moves (e0 t2: a = 0.50, mu_old = 0.10)")
    c = cells[(0, 2)]
    for s in (0.0, 0.05, 0.15, 0.40, 0.80, -0.15, -0.40):
        print(f"  shift {s:+.2f}: mu_new = {c['mu']+s:+.2f}  rho = {rho_of(c, s):.4f}")
    print(f"  1-D ceiling with sigma fixed (mu_new = a): exp(z^2/2), z = 0.8 -> {math.exp(0.8**2/2):.4f}")
    print(f"  if all 8 joints had that ceiling: {math.exp(0.8**2/2):.4f}^8 = {math.exp(0.8**2/2)**8:.2f}")
    c = cells[(0, 3)]
    print("-- e0 t3 (a = mu_old): " + "  ".join(f"shift {s:+.2f} -> {rho_of(c, s):.4f}" for s in (0.05, -0.05, 0.15, 0.5)))
    print("-- rho is a product over 8 joints: " + "  ".join(f"{x}^8 = {x**8:.4f}" for x in (1.2, 1.1, 1.0356, 0.9, 0.8)))
    print(f"   8-D widening of glossary 8.6: exp(-3.2863 - (-3.3263)) = {math.exp(-3.2863 + 3.3263):.4f}")
    print("-- (20) checked by numerical integration on e0 t0's two bells (old mu 0.20, new mu 0.25, sigma 0.5)")
    po = lambda a: pdf(a, 0.20)
    pn = lambda a: pdf(a, 0.25)
    for name, f in (("f(a) = 1", lambda a: 1.0), ("f(a) = a", lambda a: a), ("f(a) = a^2", lambda a: a * a)):
        left = integrate(lambda a: pn(a) * f(a))
        right = integrate(lambda a: po(a) * (pn(a) / po(a)) * f(a))
        naive = integrate(lambda a: po(a) * f(a))
        print(f"  {name}: under new = {left:.6f}   under old with weights = {right:.6f}   under old, no weights = {naive:.6f}")
    cfg = sf_ppo()
    print(f"-- gradient steps on one table: {cfg['num_learning_epochs'][0]} x {cfg['num_mini_batches'][0]} = {cfg['num_learning_epochs'][0]*cfg['num_mini_batches'][0]} (lines {cfg['num_learning_epochs'][1]}, {cfg['num_mini_batches'][1]})")
    print("-- lines cited")
    show(RUN_PY, 46); show(PPO_PY, 331); show(PPO_PY, 147, 149); show(PPO_PY, 203, 214); show(PPO_PY, 252, 254)
    show(STORE_PY, 144); show(STORE_PY, 140)


def ch19():
    print("=== chapter 19 — the policy gradient ===")
    cards = bag()
    print("-- the coin of chapter 4 as a one-knob policy: P(K=1) = p, P(K=0) = 1 - p")
    m0, m1 = sum(env[0]["r"]) / 4, sum(env[1]["r"]) / 4
    print(f"  E[X | K=0] = {m0}   E[X | K=1] = {m1}   J(p) = {m0}(1-p) + ({m1})p   dJ/dp = {m1 - m0}")
    for pp in (0.5, 0.25):
        J = m0 * (1 - pp) + m1 * pp
        s0, s1 = -1 / (1 - pp), 1 / pp
        print(f"  p = {pp}: J = {J:.5f}   score(K=0) = d log(1-p)/dp = {s0:+.4f}   score(K=1) = d log p/dp = {s1:+.4f}")
        print(f"     E[score] = {(1-pp)*s0 + pp*s1:+.4f}   E[score x X] = {(1-pp)*s0*m0 + pp*s1*m1:+.5f}   E[score x (X - J)] = {(1-pp)*s0*(m0-J) + pp*s1*(m1-J):+.5f}")
    sc = {0: -2.0, 1: 2.0}
    J = 0.3125
    print(f"  from the 8 equally likely cards (p = 1/2): mean(score x card) = ({sc[0]} x {sum(env[0]['r'])} + {sc[1]} x {sum(env[1]['r'])})/8 = {sum(sc[k]*x for k, x in cards)/8:+.5f}")
    print(f"     per card score x card: " + " ".join(f"{sc[k]*x:+.2f}" for k, x in cards))
    print(f"     per card score x (card - 0.3125): " + " ".join(f"{sc[k]*(x-J):+.3f}" for k, x in cards) + f"   mean = {sum(sc[k]*(x-J) for k, x in cards)/8:+.5f}")
    print(f"     spread (max - min): raw {max(sc[k]*x for k,x in cards) - min(sc[k]*x for k,x in cards):.3f}   with baseline {max(sc[k]*(x-J) for k,x in cards) - min(sc[k]*(x-J) for k,x in cards):.3f}")
    print("  the same cards with +10 written on every one (which box is better does not change):")
    print("     per card score x (card + 10): " + " ".join(f"{sc[k]*(x+10):+.1f}" for k, x in cards) + f"   mean = {sum(sc[k]*(x+10) for k, x in cards)/8:+.5f}")
    print(f"     per card score x (card + 10 - {J+10}): " + " ".join(f"{sc[k]*(x-J):+.3f}" for k, x in cards))
    print(f"  one ascent step alpha = 0.1 from p = 0.5: p <- 0.5 + 0.1 x ({m1-m0}) = {0.5 + 0.1*(m1-m0):.4f};  J: {J:.5f} -> {m0*(1-(0.5+0.1*(m1-m0))) + m1*(0.5+0.1*(m1-m0)):.5f}")
    print("-- (22) on e0 t0: d pi/d mu = pi x (a - mu)/sigma^2")
    a, mu = 0.40, 0.20
    print(f"  pi = {pdf(a, mu):.4f}   score = {(a-mu)/SIG**2:.4f}   product = {pdf(a, mu)*(a-mu)/SIG**2:.4f}   central difference = {(pdf(a, mu+0.01) - pdf(a, mu-0.01))/0.02:.4f}")
    print(f"  d log pi / d mu by central difference = {(toy.logpi(a, mu+0.01) - toy.logpi(a, mu-0.01))/0.02:.4f}")
    print("-- the mean of the score under its own bell is 0 (why a baseline costs nothing)")
    print(f"  integral of pi(a) (a - mu)/sigma^2 da = {integrate(lambda x: pdf(x, 0.20) * (x - 0.20) / SIG**2):+.6f}")
    print(f"  d/dmu of E[a] must be 1: integral of pi(a) (a-mu)/sigma^2 a da = {integrate(lambda x: pdf(x, 0.20) * (x - 0.20) / SIG**2 * x):.6f}")
    print("-- (24) in every cell, and the rough advantage delta_t of chapter 17 (the real number is chapter 22's)")
    alpha = 0.1
    for (e, t), c in cells.items():
        g = (c["a"] - c["mu"]) / SIG ** 2
        z = (c["a"] - c["mu"]) / SIG
        print(f"  e{e}t{t}: a={c['a']:+.2f} mu={c['mu']:+.2f}  (a-mu)/s^2 = {g:+.2f}  delta = {c['delta']:+.2f}  product = {c['delta']*g:+.4f}  "
              f"mu after one step alpha 0.1 = {c['mu'] + alpha*c['delta']*g:+.4f}   z^2-1 = {z*z-1:+.2f}  delta(z^2-1) = {c['delta']*(z*z-1):+.4f}   "
              f"[sign of chapter 22's number: {'+' if c['Ahat'] > 0 else '-'}]")
    print("-- the loss whose slope is that product: L = -delta x log pi;  e0 t0 and e1 t1")
    for (e, t) in ((0, 0), (1, 1)):
        c = cells[(e, t)]
        g = (c["a"] - c["mu"]) / SIG ** 2
        print(f"  e{e}t{t}: L = {-c['delta']*c['logpi_old']:+.4f}   dL/dmu = {-c['delta']*g:+.4f}   mu: {c['mu']:+.2f} -> {c['mu'] + alpha*c['delta']*g:+.4f}")
    print("-- chapter 14's loss -log pi is the same thing with the advantage replaced by 1: dL/dmu at e0 t0 =", f"{-(0.40-0.20)/SIG**2:+.2f}")
    print("-- slope of rho itself (score-function identity applied to the ratio): d rho/d mu_new = rho (a - mu_new)/sigma^2")
    c = cells[(0, 0)]
    for s in (0.0, 0.05):
        mu_n = c["mu"] + s
        r_ = rho_of(c, s)
        fd = (rho_of(c, s + 0.001) - rho_of(c, s - 0.001)) / 0.002
        print(f"  e0t0 mu_new = {mu_n:.2f}: rho = {r_:.4f}   rho (a - mu_new)/s^2 = {r_*(c['a']-mu_n)/SIG**2:.4f}   central difference = {fd:.4f}")
    c = cells[(0, 2)]
    for sft in (0.0, 0.05):
        mu_n = c["mu"] + sft
        print(f"  e0t2 mu_new = {mu_n:.2f}: rho = {rho_of(c, sft):.4f}   (a - mu_new)/s^2 = {(c['a']-mu_n)/SIG**2:.2f}   product = {rho_of(c, sft)*(c['a']-mu_n)/SIG**2:.4f}")
    print(f"  e0t0 with sigma = 0.25: (a - mu)/sigma^2 = 0.20/{0.25**2} = {0.20/0.25**2:.2f}   (sigma 0.5: {0.20/0.5**2:.2f},  sigma 1.0: {0.20/1.0:.2f})")
    print(f"  coin p = 1/4: scores {-1/0.75:+.4f}, {1/0.25:+.4f};  E[score x X] = 0.75 x ({-1/0.75:+.4f}) x 0.75 + 0.25 x 4 x (-0.125) = {0.75*(-1/0.75)*0.75 + 0.25*4*(-0.125):+.4f}")
    print(f"-- theta = actor {mlp_count([42,512,256,128,8]):,} + logstd 8 = {mlp_count([42,512,256,128,8]) + 8:,}")
    print("-- lines cited")
    show(TORCH_NORMAL, 70, 73); show(AC_PY, 159, 164); show(PPO_PY, 137, 139); show(PPO_PY, 252, 256); show(PPO_PY, 288)


def ch20():
    print("=== chapter 20 — actor-critic, and critic vs reward function ===")
    a_n = mlp_count([42, 512, 256, 128, 8], "actor ")
    c_n = mlp_count([230, 512, 256, 128, 1], "critic")
    print(f"  actor {a_n:,}   critic {c_n:,}   logstd 8   theta = {a_n + 8:,}   one optimizer holds {a_n + c_n + 8:,}")
    print(f"  critic - actor = {c_n - a_n:,};  first layers {230*512+512:,} vs {42*512+512:,};  last layers {128*1+1} vs {128*8+8:,}")
    print(f"  a critic that saw only the actor's 42 numbers: {mlp_count([42, 512, 256, 128, 1]):,}  (difference {c_n - mlp_count([42, 512, 256, 128, 1]):,} = 188 x 512 = {188*512:,})")
    path, sd = ckpt_shapes()
    if sd:
        m = sd["model_state_dict"]
        print(f"-- checkpoint {path.relative_to(T)}")
        for k in ("actor.0.weight", "actor.6.weight", "critic.0.weight", "critic.6.weight", "logstd"):
            print(f"    {k}: {m[k]}")
        tot = sum(math.prod(s) for s in m.values())
        print(f"    model_state_dict total = {tot:,};  keys starting 'actor' = {sum(math.prod(s) for k, s in m.items() if k.startswith('actor')):,}, 'critic' = {sum(math.prod(s) for k, s in m.items() if k.startswith('critic')):,}")
    else:
        print("-- no checkpoint under tron1-rl-isaaclab/logs (shape check skipped)")
    print("-- the critic observation group")
    terms = obs_dims("ObservarionsCfg", "CriticCfg")
    pol = {n for n, *_ in obs_dims("ObservarionsCfg", "PolicyCfg")}
    nj = n_joints()
    extra = {"robot_joint_torque": nj, "robot_joint_acc": nj, "robot_joint_pos": nj, "robot_joint_stiffness": nj,
             "robot_joint_damping": nj, "robot_pos": 3, "robot_vel": 6, "feet_lin_vel": 2 * 3, "robot_contact_force": 2 * 3}
    known = 0
    unknown = []
    for name, ln, func, w in terms:
        if name == "heights":
            print(f"    :{ln} {name:<28} removed by the task (limx_solefoot_env_cfg.py:78)")
            continue
        w = w if w is not None else extra.get(func)
        tag = "also in the policy group" if name in pol else "critic only"
        if w is None:
            unknown.append(name)
            print(f"    :{ln} {name:<28} {func:<28} width set by the robot asset   [{tag}]")
        else:
            known += w
            print(f"    :{ln} {name:<28} {func:<28} {w:>3}   [{tag}]")
    used = [t for t in terms if t[0] != "heights"]
    shared_w = sum(w for n, _, _, w in terms if n in pol)
    print(f"  terms used: {len(used)};  shared with the policy group: {len([t for t in used if t[0] in pol])} terms = {shared_w} numbers;  critic only: {len([t for t in used if t[0] not in pol])} terms")
    print(f"  widths readable from the code add to {known};  230 - 3 = 227;  227 - {known} = {227 - known} left for {unknown}")
    print("  (note 03 records 10 + 90 + 27 = 127 for those three, from the table train.py prints at start-up)")
    print("  ankle bodies matched by 'ankle_.*': 2 (ankle_L, ankle_R);  root_vel_w is linear + angular = 6")
    show(BASE, 149); show(BASE, 222); show(OBS_MGR, 409, 410); show(OBS_MGR, 333)
    cfg = sf_ppo()
    T_ = cfg["num_steps_per_env"][0]
    print(f"-- how often the critic is asked in one iteration: {T_} x 2048 = {T_*2048:,} while collecting, + 2048 for V_last = {T_*2048 + 2048:,}")
    print(f"-- why not 'try every action and keep the best': 10 levels on each of {nj} joints = 10^{nj} = {10**nj:,} candidates per decision;  x 2048 robots = {10**nj*2048:.3e} per step")
    print(f"   exercises: 5 levels -> 5^{nj} = {5**nj:,};  with 4096 envs: {T_} x 4096 + 4096 = {T_*4096 + 4096:,}")
    grp = {"same names as the policy group": shared_w, "base_lin_vel": 3,
           "state the robot's sensors do not give (torque, acc, feet_lin_vel, robot_pos, robot_vel, contact force)": 8 + 8 + 6 + 3 + 6 + 6,
           "settings of that robot readable from code (default joint pos, stiffness, damping)": 8 + 8 + 8,
           "mass, inertia, material (width set by the asset)": 227 - known}
    for k, v in grp.items():
        print(f"     {v:>4}  {k}")
    print(f"     {sum(grp.values()):>4}  total")
    d = env[1]
    print(f"-- toy e1 t0: reward (fact) = {d['r'][0]}   critic (prediction) = {d['V'][0]}   what actually followed G_0 = {G1(d['r'][0:3]):+.2f}")
    print("-- who fills which row of the toy table: mu <- actor, V <- critic, eps <- the dice, r and done <- the environment")
    print("-- lines cited")
    show(AC_PY, 39); show(AC_PY, 69, 71); show(AC_PY, 90); show(AC_PY, 92, 94); show(AC_PY, 98); show(AC_PY, 111); show(AC_PY, 118)
    show(AC_PY, 155, 157); show(AC_PY, 166, 172)
    show(PPO_PY, 62); show(PPO_PY, 85); show(PPO_PY, 133, 144); show(PPO_PY, 177); show(PPO_PY, 216)
    show(RUN_PY, 61); show(RUN_PY, 69, 77); show(PLAY_PY, 111, 122)
    show(BASE, find_line(BASE, r"class RewardsCfg")); show(BASE, 179, 183)


def ch21():
    print("=== chapter 21 — the encoder and v-hat ===")
    cfg = sf_ppo()
    hid, out = cfg["hidden_dims"][0], cfg["num_output_dim"][0]
    hist = cfg["obs_history_len"][0]
    pol = sum(w for *_, w in obs_dims("ObservarionsCfg", "PolicyCfg"))
    print(f"  obs_history_len = {hist} (line {cfg['obs_history_len'][1]}), policy obs = {pol}, encoder input = {hist} x {pol} = {hist*pol};  hidden {hid}, output {out} (lines {cfg['hidden_dims'][1]}, {cfg['num_output_dim'][1]}), output_detach = {cfg['output_detach'][0]} (line {cfg['output_detach'][1]})")
    e_n = mlp_count([hist * pol] + list(hid) + [out], "encoder")
    a_n, c_n = mlp_count([42, 512, 256, 128, 8]), mlp_count([230, 512, 256, 128, 1])
    print(f"  encoder psi = {e_n:,};  actor {a_n:,} + critic {c_n:,} + logstd 8 = {a_n + c_n + 8:,};  everything = {a_n + c_n + 8 + e_n:,}")
    print(f"  share of the first layer: {(hist*pol*hid[0] + hid[0])/e_n*100:.1f} %")
    print(f"  actor input = {out} + {pol} + 3 = {out + pol + 3}")
    path, sd = ckpt_shapes()
    if sd:
        en = sd["encoder_state_dict"]
        print(f"-- checkpoint {path.relative_to(T)}: " + ", ".join(f"{k} {v}" for k, v in en.items()))
        print(f"    encoder_state_dict total = {sum(math.prod(s) for s in en.values()):,};  both dicts = {sum(math.prod(s) for dct in sd.values() for s in dct.values()):,}")
    print("-- what-ifs for the exercises")
    print(f"  history of 5 steps: input {5*pol}, psi = {mlp_count([5*pol] + list(hid) + [out]):,}")
    print(f"  history of 25 steps: input {25*pol}, psi = {mlp_count([25*pol] + list(hid) + [out]):,}")
    print(f"  encoder output 6 instead of 3: psi = {mlp_count([hist*pol] + list(hid) + [6]):,};  actor input {6 + pol + 3}, actor = {mlp_count([6 + pol + 3, 512, 256, 128, 8]):,}  (+{mlp_count([6 + pol + 3, 512, 256, 128, 8]) - a_n:,})")
    print(f"  feeding all 360 numbers straight into the actor: input {hist*pol + pol + 3}, actor = {mlp_count([hist*pol + pol + 3, 512, 256, 128, 8]):,};  encoder + actor today = {e_n + a_n:,}")
    print("-- MSE on pairs the reader already has (prediction vs what happened)")
    d = env[1]
    errs = [d["V"][t] - G1(d["r"][t:3]) for t in range(3)]
    print(f"  env 1 first episode, V - G: {[round(x, 2) for x in errs]}  squares {[round(x*x, 4) for x in errs]}  sum {sum(x*x for x in errs):.4f}  MSE = {sum(x*x for x in errs)/3:.4f}  (root {math.sqrt(sum(x*x for x in errs)/3):.4f})")
    print(f"     mean of the plain errors = {sum(errs)/3:.4f};  mean of |errors| = {sum(abs(x) for x in errs)/3:.4f}")
    print(f"     slope of the MSE w.r.t. each prediction = 2 x error / 3: {[round(2*x/3, 4) for x in errs]}")
    for e in (0, 1):
        ds = [cells[(e, t)]["delta"] for t in range(4)]
        print(f"  env {e}, left vs right of (4) = delta: squares {[round(x*x, 4) for x in ds]}  sum {sum(x*x for x in ds):.4f}  MSE = {sum(x*x for x in ds)/4:.5f}")
    print(f"  errors +1 and -1: mean error 0, MSE {(1+1)/2};  errors 0.1 x 3 and 3.0 once (4 numbers): MSE = {(3*0.01 + 9)/4:.4f}, the big one carries {9/(3*0.01+9)*100:.1f} %")
    ep, mb = cfg["num_learning_epochs"][0], cfg["num_mini_batches"][0]
    print(f"-- encoder gradient steps per iteration: {ep} x {mb} = {ep*mb}, each on {24*2048//mb:,} cells x 3 numbers = {24*2048//mb*3:,} squared differences")
    print("-- lines cited")
    show(ENC_PY, 39, 41); show(ENC_PY, 67, 69); show(ENC_PY, 75); show(ENC_PY, 85); show(ENC_PY, 95, 103)
    show(PPO_PY, 61); show(PPO_PY, 85, 89); show(PPO_PY, 136, 139); show(PPO_PY, 154); show(PPO_PY, 203); show(PPO_PY, 289); show(PPO_PY, 299, 321)
    show(STORE_PY, 289, 291); show(RUN_PY, 63); show(RUN_PY, 65, 67); show(RUN_PY, 173); show(RUN_PY, 180)
    show(PPO_CFG, 106, 111); show(BASE, 175, 176); show(BASE, 183); show(PLAY_PY, 117, 122)
    show(ISAAC / "isaaclab/isaaclab/envs/mdp/observations.py", 40, 44)


# ---------------------------------------------------------------- chapters 23-24: the PPO loss
EPS = toy.EPS_CLIP
SET1, SET2 = toy.mb0, toy.mb1          # the two groups of four cells one gradient step uses (glossary 8.3)


def clipf(x, lo, hi):
    return min(max(x, lo), hi)


def surr_cell(c, shift, eps=EPS, sigma_new=SIG):
    """One cell of ppo.py:252-260 with the new bell at mu_old + shift (and optionally a new sigma)."""
    mu_n = c["mu"] + shift
    lp_new = -0.5 * math.log(2 * math.pi) - math.log(sigma_new) - (c["a"] - mu_n) ** 2 / (2 * sigma_new ** 2)
    rho = math.exp(lp_new - c["logpi_old"])
    rc = clipf(rho, 1 - eps, 1 + eps)
    raw, cl = -c["Ahat"] * rho, -c["Ahat"] * rc
    picks_clip = cl > raw
    g = 0.0 if picks_clip else -c["Ahat"] * rho * (c["a"] - mu_n) / sigma_new ** 2
    return dict(mu_n=mu_n, rho=rho, rc=rc, raw=raw, cl=cl, L=max(raw, cl), clipped=picks_clip, g=g)


def knob_walk(sched, use_clip, n, alpha=0.1):
    """One shared knob b (the last-layer bias of chapter 14: every mu moves by b), SGD on the mean of the set.
    Returns, for k = 0..n, the state BEFORE gradient step k+1: (b, force, rho of all 8 cells)."""
    b, out = 0.0, []
    for k in range(n + 1):
        cs = [surr_cell(cells[key], b) for key in sched(k)]
        gs = [(x["g"] if use_clip else -cells[key]["Ahat"] * x["rho"] * (cells[key]["a"] - x["mu_n"]) / SIG ** 2)
              for key, x in zip(sched(k), cs)]
        g = sum(gs) / len(gs)
        out.append((b, g, [surr_cell(c, b)["rho"] for c in cells.values()]))
        b -= alpha * g
    return out


def ch23():
    print("=== chapter 23 — the clipped surrogate ===")
    name = lambda k: f"e{k[0]}t{k[1]}"
    print(f"  the two sets: set 1 = {[name(k) for k in SET1]}   set 2 = {[name(k) for k in SET2]}   epsilon = {EPS}  band [{1-EPS}, {1+EPS}]")
    print("-- candidate 1 (-Ahat log pi) against the surrogate (-Ahat rho): slope w.r.t. mu, e0 t0")
    c = cells[(0, 0)]
    for s in (0.0, 0.05):
        x = surr_cell(c, s)
        print(f"  mu_new = {x['mu_n']:.2f}: -Ahat (a-mu)/s^2 = {-c['Ahat']*(c['a']-x['mu_n'])/SIG**2:+.4f}    -Ahat rho (a-mu)/s^2 = {-c['Ahat']*x['rho']*(c['a']-x['mu_n'])/SIG**2:+.4f}   (rho {x['rho']:.4f})")
    print("-- the unclipped surrogate has no bottom: one cell, L_i = -Ahat rho")
    c = cells[(1, 1)]
    for s in (0.0, -0.15, -0.40, -1.00):
        x = surr_cell(c, s)
        print(f"  e1t1 (Ahat {c['Ahat']:+.4f}) mu_new = {x['mu_n']:+.2f}: rho = {x['rho']:.4f}   -Ahat rho = {x['raw']:+.4f}")
    c = cells[(0, 2)]
    for s, sg in ((0.0, 0.5), (0.15, 0.5), (0.40, 0.5), (0.40, 0.25), (0.40, 0.10)):
        x = surr_cell(c, s, sigma_new=sg)
        print(f"  e0t2 (Ahat {c['Ahat']:+.4f}) mu_new = {x['mu_n']:+.2f} sigma_new = {sg}: rho = {x['rho']:.4f}   -Ahat rho = {x['raw']:+.4f}")
    print("-- clip(x, 0.8, 1.2): " + "  ".join(f"{v} -> {clipf(v, 0.8, 1.2)}" for v in (1.0356, 1.2153, 0.7520, 1.2, 0.8, 2.1815)))
    print("-- (29) with mu_new = mu_old + 0.05 in every cell")
    for label, keys in (("set 1", SET1), ("set 2", SET2), ("env 0", [(0, t) for t in range(4)]), ("env 1", [(1, t) for t in range(4)])):
        xs = [surr_cell(cells[k], 0.05) for k in keys]
        for k, x in zip(keys, xs):
            print(f"  {name(k)}: rho {x['rho']:.4f} clip {x['rc']:.4f}  Ahat {cells[k]['Ahat']:+.4f}  -rho Ahat {x['raw']:+.4f}  -clip Ahat {x['cl']:+.4f}  L_i {x['L']:+.4f}  "
                  f"{'CLIPPED' if x['clipped'] else 'raw'}  (a - mu_new) {cells[k]['a']-x['mu_n']:+.2f}  slope (G) {x['g']:+.4f}  alone, alpha 0.1: mu {x['mu_n']:+.2f} -> {x['mu_n'] - 0.1*x['g']:+.4f}")
        print(f"     {label}: sum {sum(x['L'] for x in xs):+.4f}  mean L^CLIP = {sum(x['L'] for x in xs)/4:+.4f}   mean slope on a shared knob = {sum(x['g'] for x in xs)/4:+.4f}")
    print(f"  all 8 cells: mean L^CLIP = {sum(surr_cell(c, 0.05)['L'] for c in cells.values())/8:+.4f}")
    print("-- the paper's form: min(rho Ahat, clip Ahat) = -L_i  (to be maximised)")
    for k in SET1:
        x = surr_cell(cells[k], 0.05)
        A = cells[k]["Ahat"]
        print(f"  {name(k)}: min({x['rho']*A:+.4f}, {x['rc']*A:+.4f}) = {min(x['rho']*A, x['rc']*A):+.4f}   -L_i = {-x['L']:+.4f}")
    print("-- the first gradient step (rho = 1 in every cell it uses): L_i = -Ahat, slope = -Ahat (a - mu)/sigma^2")
    for label, keys in (("set 1", SET1), ("set 2", SET2)):
        xs = [surr_cell(cells[k], 0.0) for k in keys]
        print(f"  {label}: L_i " + " ".join(f"{x['L']:+.4f}" for x in xs) + f"  mean = {sum(x['L'] for x in xs)/4:+.4f};   slopes " + " ".join(f"{x['g']:+.4f}" for x in xs)
              + f"  mean = {sum(x['g'] for x in xs)/4:+.4f}  -> shared knob b = {-0.1*sum(x['g'] for x in xs)/4:+.4f}")
    print(f"  all 8: mean of -Ahat = {-sum(c['Ahat'] for c in cells.values())/8:+.6f} (the normalisation of (28) makes it zero)")
    print("-- when the clip acts: the twins e0 t2 / e1 t1, moved 0.15 the right way and the wrong way")
    for k in ((0, 2), (1, 1)):
        c = cells[k]
        right = 0.15 if c["Ahat"] > 0 else -0.15
        for lab, s in (("right way", right), ("wrong way", -right)):
            x = surr_cell(c, s)
            print(f"  {name(k)} Ahat {c['Ahat']:+.4f} {lab} {s:+.2f}: mu_new {x['mu_n']:+.2f}  rho {x['rho']:.4f}  -rho Ahat {x['raw']:+.4f}  -clip Ahat {x['cl']:+.4f}  "
                  f"L_i {x['L']:+.4f}  {'CLIPPED slope 0' if x['clipped'] else 'raw'}  slope {x['g']:+.4f}")
    print("-- where the band edge is, sigma fixed at 0.5: rho = exp(((a-mu_old)^2 - (a-mu_new)^2)/(2 sigma^2))")
    for z in (0.4, 0.6, 0.8):
        d = z * SIG
        top = math.exp(z * z / 2)
        up = d - math.sqrt(d * d - 0.5 * math.log(1 + EPS)) if top >= 1 + EPS else None
        dn = math.sqrt(d * d - 0.5 * math.log(1 - EPS)) - d
        print(f"  |a - mu_old| = {d:.2f} (z {z}): ceiling exp(z^2/2) = {top:.4f};  rho = 1.2 at a shift of {('%.4f' % up) if up else 'never'} toward a;  rho = 0.8 at {dn:.4f} away")
    print(f"  smallest |z| whose ceiling reaches 1.2: sqrt(2 ln 1.2) = {math.sqrt(2*math.log(1.2)):.4f}")
    c = cells[(0, 2)]
    print(f"  e0t2 moved +0.15 with sigma = 0.25 on both bells: rho = exp((0.40^2 - 0.25^2)/(2 x 0.0625)) = {math.exp((0.4**2 - 0.25**2)/(2*0.25**2)):.4f}")
    print(f"-- eight joints share one band: 1.2^(1/8) = {1.2**(1/8):.4f}   0.8^(1/8) = {0.8**(1/8):.4f}")
    print("-- exercise: e1 t0 moved 0.15 each way")
    c = cells[(1, 0)]
    for s in (+0.15, -0.15):
        x = surr_cell(c, s)
        print(f"  e1t0 Ahat {c['Ahat']:+.4f} a {c['a']:+.2f} shift {s:+.2f}: mu_new {x['mu_n']:+.2f} rho {x['rho']:.4f}  -rho Ahat {x['raw']:+.4f}  -clip Ahat {x['cl']:+.4f}  L_i {x['L']:+.4f}  {'CLIPPED' if x['clipped'] else 'raw'}  slope {x['g']:+.4f}")
    print("-- exercise: a narrower band, set 1 at +0.05")
    for eps in (0.1, 0.05):
        xs = [surr_cell(cells[k], 0.05, eps=eps) for k in SET1]
        print(f"  eps = {eps}: " + "  ".join(f"{name(k)} {x['L']:+.4f}{'*' if x['clipped'] else ''}" for k, x in zip(SET1, xs)) + f"   mean = {sum(x['L'] for x in xs)/4:+.4f}   (* = clipped term chosen, slope 0)")
    print("-- one shared knob b, alpha 0.1, sets taken in turn (1, 2, 1, 2, ...): with and without the clip")
    sched = lambda k: SET1 if k % 2 == 0 else SET2
    for use_clip in (False, True):
        w = knob_walk(sched, use_clip, 100)
        print(f"  {'with clip' if use_clip else 'no clip  '}: " + "   ".join(f"after {k}: b {w[k][0]:+.4f} rho {min(w[k][2]):.4f}..{max(w[k][2]):.4f}" for k in (1, 4, 10, 20, 100)))
        if use_clip:
            tail = [x[0] for x in w[20:]]
            print(f"     steps 20..100 with clip: b stays in [{min(tail):+.4f}, {max(tail):+.4f}];  largest rho seen {max(max(x[2]) for x in w):.4f}, smallest {min(min(x[2]) for x in w):.4f}")
            first = next(k for k in range(100) if any(r > 1 + EPS or r < 1 - EPS for r in w[k][2]))
            print(f"     first state with a cell outside the band: after {first} steps: " + ", ".join(f"{name(key)} {r:.4f}" for key, r in zip(cells, w[first][2]) if r > 1 + EPS or r < 1 - EPS))
            used = sched(first)
            xs = {key: surr_cell(cells[key], w[first][0]) for key in used}
            print(f"     step {first + 1} uses set {1 if used is SET1 else 2}: " + "  ".join(f"{name(key)} slope {x['g']:+.4f}{' (clipped)' if x['clipped'] else ''}" for key, x in xs.items())
                  + f"  mean {w[first][1]:+.4f}  ->  b {w[first][0]:+.4f} -> {w[first + 1][0]:+.4f}")
        else:
            print(f"     without clip, after 11: b {w[11][0]:+.4f}")
    print("-- sign of Ahat (chapter 22) against sign of delta (chapter 19 used delta): " + "  ".join(
        f"{name(k)} {'same' if (c['delta'] > 0) == (c['Ahat'] > 0) else 'DIFFERENT'}" for k, c in cells.items()))
    print("-- lines cited")
    show(PPO_PY, 50); show(PPO_PY, 96); show(PPO_PY, 197); show(PPO_PY, 252, 260); show(PPO_CFG, 96)
    show(PPO_CFG, 99); show(STORE_PY, 103, 105); show(STORE_PY, 112, 114); show(AC_PY, 164)
    show(STORE_PY, 204, 207); show(STORE_PY, 230); show(STORE_PY, 247); show(STORE_PY, 278, 279); show(PPO_PY, 288)


def vloss_cell(V_old, R, V_new, eps=EPS):
    """One cell of ppo.py:264-269."""
    Vc = V_old + clipf(V_new - V_old, -eps, eps)
    raw, cl = (V_new - R) ** 2, (Vc - R) ** 2
    picks_clip = cl > raw
    return dict(Vc=Vc, raw=raw, cl=cl, L=max(raw, cl), clipped=picks_clip, slope=0.0 if picks_clip else 2 * (V_new - R))


def H_gauss(sigma):
    return 0.5 + 0.5 * math.log(2 * math.pi) + math.log(sigma)


def ch24():
    print("=== chapter 24 — value loss, entropy, total loss ===")
    name = lambda k: f"e{k[0]}t{k[1]}"
    print("-- value loss at the first gradient step: V_new = V_old, so (V_new - R)^2 = A^2")
    for label, keys in (("set 1", SET1), ("set 2", SET2), ("all 8", list(cells))):
        sq = [(cells[k]["V"] - cells[k]["R"]) ** 2 for k in keys]
        if len(keys) == 4:
            for k, s in zip(keys, sq):
                c = cells[k]
                print(f"  {name(k)}: V {c['V']:.1f}  R {c['R']:+.4f}  V - R {c['V']-c['R']:+.4f} (= -A, A = {c['A']:+.4f})  square {s:.4f}  slope 2(V - R) = {2*(c['V']-c['R']):+.4f}")
        print(f"     {label}: sum {sum(sq):.4f}  mean = {sum(sq)/len(sq):.4f}   largest share {max(sq)/sum(sq)*100:.1f} %")
    c = cells[(0, 3)]
    print(f"  the target holds later predictions, not its own: e0t3 R = r + gamma V_last = {c['r']} + {GAMMA} x {env[0]['V_last']} = {c['r'] + GAMMA*env[0]['V_last']:.4f} (table {c['R']:.4f});  e1t2 (done) R = r = {cells[(1,2)]['R']:.4f}")
    print("-- the clipped form (32): V_clip = V_old + clip(V_new - V_old, -0.2, 0.2); the larger square is charged")
    for k, vs in (((0, 0), (4.9, 4.8, 4.75, 4.5, 5.3)), ((1, 1), (1.9, 1.5, 2.5, -0.26)), ((0, 2), (3.934, 3.7))):
        c = cells[k]
        print(f"  {name(k)}: V_old {c['V']:.1f}  R {c['R']:+.4f}  |R - V_old| = {abs(c['A']):.4f}")
        for v in vs:
            x = vloss_cell(c["V"], c["R"], v)
            print(f"     V_new {v:+.4f}: V_clip {x['Vc']:+.4f}  (V_new - R)^2 {x['raw']:.4f}  (V_clip - R)^2 {x['cl']:.4f}  charged {x['L']:.4f}  "
                  f"{'CLIPPED slope 0' if x['clipped'] else 'raw'}  slope {x['slope']:+.4f}")
    print("-- the stretch of V_new where the clipped term wins (slope 0): between V_clip and its mirror image about R")
    for k, c in cells.items():
        if abs(c["A"]) > EPS:
            Vc = c["V"] + (EPS if c["R"] > c["V"] else -EPS)
            lo, hi = sorted((Vc, 2 * c["R"] - Vc))
            print(f"  {name(k)}: |A| = {abs(c['A']):.4f} > 0.2: V_clip {Vc:+.4f}, mirror {2*c['R']-Vc:+.4f}  ->  slope 0 for V_new in ({lo:+.4f}, {hi:+.4f})")
        else:
            print(f"  {name(k)}: |A| = {abs(c['A']):.4f} <= 0.2: R is inside the band, no such stretch")
    print("-- entropy of the coin of chapter 4: H = -sum p log p")
    for p in (0.5, 0.75, 0.9, 0.99, 1.0):
        h = -sum(q * math.log(q) for q in (p, 1 - p) if q > 0)
        print(f"  P(box 0) = {p}: H = {h:.4f}")
    print("-- (31): H = 1/2 + 1/2 ln 2 pi + ln sigma, per joint")
    print(f"  1/2 ln 2 pi = {0.5*math.log(2*math.pi):.4f};  1/2 + 1/2 ln 2 pi = {0.5 + 0.5*math.log(2*math.pi):.4f} (six digits {0.5 + 0.5*math.log(2*math.pi):.6f}; x 8 = {8*(0.5 + 0.5*math.log(2*math.pi)):.4f})")
    for s in (2.0, 1.0, 0.5, 0.25, 0.1):
        print(f"  sigma = {s}: ln sigma = {math.log(s):+.4f}  H = {H_gauss(s):+.4f}   x 8 joints = {8*H_gauss(s):+.4f}   -0.01 x 8H = {-0.01*8*H_gauss(s):+.4f}")
    print(f"  H = 0 at sigma = exp(-(1/2 + 1/2 ln 2 pi)) = {math.exp(-(0.5 + 0.5*math.log(2*math.pi))):.4f};  halving sigma changes H by -ln 2 = {-math.log(2):.4f}")
    print(f"  by numerical integration, sigma 0.5: -integral p log p = {integrate(lambda a: -pdf(a, 0.2) * math.log(pdf(a, 0.2))):.4f};  E[(a - mu)^2] = {integrate(lambda a: pdf(a, 0.2) * (a - 0.2) ** 2):.4f}")
    print(f"  toy, one joint: H = {H_gauss(SIG):.4f}, -c_H H = {-0.01*H_gauss(SIG):+.6f}")
    print("-- (35) at the first gradient step (rho = 1): slope of L_i w.r.t. ell = -Ahat (z^2 - 1); slope of -c_H H = -0.01")
    f = {}
    for k, c in cells.items():
        z = (c["a"] - c["mu"]) / SIG
        f[k] = -c["Ahat"] * (z * z - 1)
        print(f"  {name(k)}: z {z:+.2f}  z^2 - 1 {z*z-1:+.2f}  Ahat {c['Ahat']:+.4f}  slope {f[k]:+.4f}  -> {'narrower' if f[k] > 0 else 'wider'}")
    zs = [((c["a"] - c["mu"]) / SIG) ** 2 - 1 for c in cells.values()]
    print(f"     mean of z^2 - 1 over the 8 cells = {sum(zs)/8:+.4f} (not 0: eight draws, all |z| < 1);  mean Ahat = {sum(c['Ahat'] for c in cells.values())/8:+.6f}")
    for label, keys in (("set 1", SET1), ("set 2", SET2), ("all 8", list(cells))):
        m = sum(f[k] for k in keys) / len(keys)
        print(f"     {label}: mean surrogate slope {m:+.4f};  with entropy {m - 0.01:+.4f}")
    c = cells[(0, 0)]
    h = 1e-5
    def Li(ell, c=c):
        s = math.exp(ell)
        lp = -0.5 * math.log(2 * math.pi) - ell - (c["a"] - c["mu"]) ** 2 / (2 * s * s)
        return -c["Ahat"] * math.exp(lp - c["logpi_old"]) - 0.01 * (0.5 + 0.5 * math.log(2 * math.pi) + ell)
    l0 = math.log(SIG)
    print(f"  e0t0 by central difference (surrogate - 0.01 H): {(Li(l0 + h) - Li(l0 - h)) / (2*h):+.4f}   formula: {f[(0,0)]:+.4f} - 0.01 = {f[(0,0)] - 0.01:+.4f}")
    g = f[(0, 0)] - 0.01
    print(f"  e0t0 alone, alpha = 1e-3 (Tron1's starting value): ell {l0:.6f} -> {l0 - 1e-3*g:.6f}   sigma -> {math.exp(l0 - 1e-3*g):.6f}")
    print("-- (33) at the first gradient step (rho = 1, V_new = V_old): L = L^CLIP + 1.0 L_V - 0.01 H")
    for label, keys in (("set 1", SET1), ("set 2", SET2)):
        Lc = sum(-cells[k]["Ahat"] for k in keys) / 4
        Lv = sum((cells[k]["V"] - cells[k]["R"]) ** 2 for k in keys) / 4
        print(f"  {label}: L^CLIP {Lc:+.4f}  L_V {Lv:.4f}  H {H_gauss(SIG):.4f}  ->  L = {Lc:+.4f} + {Lv:.4f} - {0.01*H_gauss(SIG):.4f} = {Lc + Lv - 0.01*H_gauss(SIG):+.4f}")
    a_n, c_n = mlp_count([42, 512, 256, 128, 8]), mlp_count([230, 512, 256, 128, 1])
    print(f"-- which knobs each term reaches: L^CLIP -> theta = actor {a_n:,} + logstd 8 = {a_n + 8:,};  L_V -> phi = {c_n:,};  H -> logstd 8;  optimizer holds {a_n + 8 + c_n:,}")
    cfg = sf_ppo()
    print("-- config: " + ", ".join(f"{k} = {cfg[k][0]} (line {cfg[k][1]})" for k in ("value_loss_coef", "use_clipped_value_loss", "clip_param", "entropy_coef")))
    print("-- lines cited")
    show(PPO_PY, 53, 54); show(PPO_PY, 57); show(PPO_PY, 99, 100); show(PPO_PY, 104); show(PPO_PY, 196, 198); show(PPO_PY, 216, 219); show(PPO_PY, 263, 278)
    show(PPO_PY, 293, 294); show(AC_PY, 118); show(AC_PY, 151, 157); show(TORCH_NORMAL, 104, 105)
    show(STORE_PY, 243, 244); show(STORE_PY, 269, 273); show(RUN_PY, 277); show(RUN_PY, 281); show(RUN_PY, 322, 323)
    show(PPO_CFG, 94, 97); show(PPO_CFG, 100); show(STORE_PY, 201); show(PPO_PY, 221); show(PPO_PY, 244); show(PPO_PY, 287, 290); show(PPO_PY, 326, 329)


# ---------------------------------------------------------------- chapters 25-27: the gradient step, the KL gauge, one whole iteration
import statistics
import struct

TORCH = pathlib.Path.home() / "anaconda3/envs/tron1/lib/python3.10/site-packages/torch"
ADAM_PY = TORCH / "optim/adam.py"
CLIPGRAD_PY = TORCH / "nn/utils/clip_grad.py"
OPTIM_PY = TORCH / "optim/optimizer.py"
TRAIN_PY = T / "scripts/rsl_rl/train.py"
B1, B2, ADAM_EPS = 0.9, 0.999, 1e-8      # torch defaults (adam.py:37-38); ppo.py:85 passes only lr
KNOBS = ("b", "ell", "b_V")
ELL0 = math.log(SIG)


def three_knobs(keys, x=(0.0, ELL0, 0.0), eps=EPS, cV=1.0, cH=0.01):
    """ppo.py:252-278 on the cells `keys`, when three SHARED knobs sit at x = (b, ell, b_V):
    b   = last-layer bias of the actor (chapter 14): mu_new = mu_old + b in every cell,
    ell = logstd: sigma_new = e^ell in every cell,
    b_V = last-layer bias of the critic, built the same way: V_new = V_old + b_V in every cell.
    The knobs are this book's assumption (the toy table has no network); the formulas are the code's.
    Returns the three terms, the loss, and the analytic gradient that loss.backward() would leave in .grad."""
    b, ell, bV = x
    s = math.exp(ell)
    rows = []
    for k in keys:
        c = cells[k]
        mu_n = c["mu"] + b
        z = (c["a"] - mu_n) / s
        lp = -0.5 * math.log(2 * math.pi) - ell - 0.5 * z * z
        rho = math.exp(lp - c["logpi_old"])
        raw, cl = -c["Ahat"] * rho, -c["Ahat"] * clipf(rho, 1 - eps, 1 + eps)
        pc = cl > raw
        v = vloss_cell(c["V"], c["R"], c["V"] + bV, eps)
        rows.append(dict(key=k, rho=rho, z=z, Li=max(raw, cl), pc=pc, g_b=0.0 if pc else -c["Ahat"] * rho * z / s,
                         g_l=0.0 if pc else -c["Ahat"] * rho * (z * z - 1), Lv=v["L"], vc=v["clipped"], g_V=v["slope"]))
    n = len(rows)
    Lc, Lv, H = sum(r["Li"] for r in rows) / n, sum(r["Lv"] for r in rows) / n, H_gauss(s)
    grad = [sum(r["g_b"] for r in rows) / n, sum(r["g_l"] for r in rows) / n - cH, cV * sum(r["g_V"] for r in rows) / n]
    return dict(rows=rows, Lc=Lc, Lv=Lv, H=H, L=Lc + cV * Lv - cH * H, grad=grad)


def clip_grad(g, max_norm=1.0):
    """clip_grad.py:105-109: one norm over every knob, every knob multiplied by the same coefficient."""
    n = math.sqrt(sum(x * x for x in g))
    coef = min(1.0, max_norm / (n + 1e-6))
    return [x * coef for x in g], n, coef


class AdamPy:
    """adam.py:378-379 and :414-432 written out for a list of scalars (no weight decay, no amsgrad)."""
    def __init__(self, n):
        self.m, self.v, self.k = [0.0] * n, [0.0] * n, 0

    def step(self, x, g, lr):
        self.k += 1
        out, info = [], []
        for i in range(len(x)):
            self.m[i] = B1 * self.m[i] + (1 - B1) * g[i]
            self.v[i] = B2 * self.v[i] + (1 - B2) * g[i] ** 2
            mh, vh = self.m[i] / (1 - B1 ** self.k), self.v[i] / (1 - B2 ** self.k)
            d = -lr * mh / (math.sqrt(vh) + ADAM_EPS)
            out.append(x[i] + d)
            info.append(dict(m=self.m[i], v=self.v[i], mh=mh, vh=vh, d=d, ratio=mh / (math.sqrt(vh) + ADAM_EPS)))
        return out, info


def kl_gauss(mu_o, s_o, mu_n, s_n, guard=0.0):
    """(30) for one joint; guard = 1e-5 reproduces the term inside the log at ppo.py:224."""
    return math.log(s_n / s_o + guard) + (s_o ** 2 + (mu_o - mu_n) ** 2) / (2 * s_n ** 2) - 0.5


def lr_rule(lr, kl, desired=0.01):
    """(37) = ppo.py:238-241."""
    if kl > desired * 2.0:
        return max(1e-5, lr / 1.5), "cut"
    if kl < desired / 2.0 and kl > 0.0:
        return min(1e-2, lr * 1.5), "raise"
    return lr, "hold"


def toy_update(order=(SET1, SET2), epochs=2, lr=1e-3, rule=True, adam=True, clipnorm=True, max_norm=1.0):
    """The loop of ppo.py:190-295 on the toy table: for every set, KL -> lr rule -> loss -> backward -> clip_grad_norm_ -> step.
    The split into sets is made once and reused by every epoch (rollout_storage.py:230 is outside the loop at :251)."""
    x, opt, log = [0.0, ELL0, 0.0], AdamPy(3), []
    for ep in range(epochs):
        for keys in order:
            kl = kl_gauss(0.0, SIG, x[0], math.exp(x[1]), guard=1e-5)   # one joint, shared knobs: the same in every cell
            act = "-"
            if rule:
                lr, act = lr_rule(lr, kl)
            r = three_knobs(keys, x)
            g, norm, coef = clip_grad(r["grad"], max_norm) if clipnorm else (list(r["grad"]), math.sqrt(sum(v * v for v in r["grad"])), 1.0)
            if adam:
                x_new, info = opt.step(x, g, lr)
            else:
                x_new, info = [xi - lr * gi for xi, gi in zip(x, g)], None
            log.append(dict(ep=ep + 1, keys=keys, kl=kl, act=act, lr=lr, r=r, raw=list(r["grad"]), norm=norm, coef=coef, g=g, x=list(x), x_new=x_new, info=info))
            x = x_new
    return log, x


def _varint(buf, i):
    n = shift = 0
    while True:
        byte = buf[i]
        i += 1
        n |= (byte & 0x7F) << shift
        if byte < 0x80:
            return n, i
        shift += 7


def _pb(buf):
    """Fields of one protobuf message as (field number, wire type, value) - just enough for a TensorBoard scalar."""
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            val, i = _varint(buf, i)
        elif wt == 1:
            val, i = buf[i:i + 8], i + 8
        elif wt == 2:
            ln, i = _varint(buf, i)
            val, i = buf[i:i + ln], i + ln
        elif wt == 5:
            val, i = buf[i:i + 4], i + 4
        else:
            return
        yield f, wt, val


def tfevents_scalars(path):
    """Every scalar in one TensorBoard event file, read with the standard library only (TFRecord framing:
    8-byte length, 4-byte crc, payload, 4-byte crc; Event.step = field 2, Event.summary = 5, Value.tag = 1,
    Value.simple_value = 2). Returns {tag: [(step, value), ...]}."""
    data, pos, out = pathlib.Path(path).read_bytes(), 0, {}
    while pos + 12 <= len(data):
        n = struct.unpack("<Q", data[pos:pos + 8])[0]
        rec = data[pos + 12:pos + 12 + n]
        pos += 12 + n + 4
        step, summ = 0, None
        for f, wt, val in _pb(rec):
            if f == 2 and wt == 0:
                step = val
            elif f == 5 and wt == 2:
                summ = val
        for f, wt, val in _pb(summ or b""):
            if f == 1 and wt == 2:
                tag = sv = None
                for f2, wt2, v2 in _pb(val):
                    if f2 == 1 and wt2 == 2:
                        tag = v2.decode()
                    elif f2 == 2 and wt2 == 5:
                        sv = struct.unpack("<f", v2)[0]
                if tag is not None and sv is not None:
                    out.setdefault(tag, []).append((step, sv))
    return out


def training_runs():
    """The runs under logs/ (outside git): (folder name, num_envs, max_iterations, scalars). Empty list if none."""
    out = []
    for d in sorted((T / "logs/rsl_rl/sf_tron_1a_flat").glob("*")):
        ev = sorted(d.glob("events.out.tfevents.*"))
        if not ev:
            continue
        n_env = int(re.search(r"num_envs: (\d+)", (d / "params/env.yaml").read_text()).group(1))
        max_it = int(re.search(r"max_iterations: (\d+)", (d / "params/agent.yaml").read_text()).group(1))
        out.append((d.name, n_env, max_it, tfevents_scalars(ev[0])))
    return out


def ckpt_optimizer():
    """optimizer_state_dict of the newest checkpoint, read without torch: the param_groups (plain Python) and, for
    every parameter tensor, the shape of exp_avg plus the VALUE of the 0-d `step` tensor (4 bytes from the zip)."""
    runs = sorted((T / "logs/rsl_rl/sf_tron_1a_flat").glob("*/model_*.pt"))
    if not runs:
        return None, None

    class Stub:
        def __init__(self, *a, **k):
            pass

    with zipfile.ZipFile(runs[-1]) as z:
        names = z.namelist()

        class U(pickle.Unpickler):
            def find_class(self, mod, name):
                if name == "_rebuild_tensor_v2":
                    def rebuild(storage, offset, size, *rest):
                        if tuple(size) == () and storage is not None:
                            raw = z.read(next(n for n in names if n.endswith("data/" + storage)))
                            return struct.unpack("<f", raw[offset * 4:offset * 4 + 4])[0]
                        return tuple(size)
                    return rebuild
                if name == "_rebuild_parameter":
                    return lambda data, *rest: data
                if name == "OrderedDict":
                    return collections.OrderedDict
                return Stub

            def persistent_load(self, pid):
                return pid[2]          # ('storage', type, key, location, numel) -> key

        d = U(z.open(next(n for n in names if n.endswith("data.pkl")))).load()
    return runs[-1], d["optimizer_state_dict"]


def ch25():
    print("=== chapter 25 — Adam and optimizer.step() ===")
    name = lambda k: f"e{k[0]}t{k[1]}"
    cfg = sf_ppo()
    max_norm = cfg["max_grad_norm"][0]
    print(f"  three shared knobs (this book's assumption): b (actor's last bias), ell (logstd), b_V (critic's last bias); start (0, {ELL0:.6f}, 0)")
    print(f"  config: learning_rate = {cfg['learning_rate'][0]} (line {cfg['learning_rate'][1]}), max_grad_norm = {max_norm} (line {cfg['max_grad_norm'][1]});  Adam defaults: beta1 {B1}, beta2 {B2}, eps {ADAM_EPS}")
    print("-- what loss.backward() leaves in .grad at the first gradient step: the MEAN over the set of the per-cell slopes")
    for label, keys in (("e0 t0 alone", [(0, 0)]), ("set 1", SET1), ("set 2", SET2), ("all 8", list(cells))):
        r = three_knobs(keys)
        if 1 < len(keys) <= 4:
            for row in r["rows"]:
                print(f"     {name(row['key'])}: dL_i/dmu {row['g_b']:+.4f}   dL_i/dell {row['g_l']:+.4f}   2(V - R) {row['g_V']:+.4f}")
        g2, norm, coef = clip_grad(r["grad"], max_norm)
        print(f"  {label}: L = {r['L']:+.4f};  .grad (b, ell, b_V) = " + "  ".join(f"{g:+.4f}" for g in r["grad"])
              + f";  squares " + " + ".join(f"{g*g:.4f}" for g in r["grad"]) + f" = {norm*norm:.4f}, norm {norm:.4f}, share of b_V {r['grad'][2]**2/norm**2*100:.1f} %;  coef {coef:.4f}  -> " + "  ".join(f"{g:+.4f}" for g in g2))
    h = 1e-6
    r0 = three_knobs(SET1)
    num = []
    for i in range(3):
        xp, xm = [0.0, ELL0, 0.0], [0.0, ELL0, 0.0]
        xp[i] += h; xm[i] -= h
        num.append((three_knobs(SET1, xp)["L"] - three_knobs(SET1, xm)["L"]) / (2 * h))
    print("  central difference on L of set 1: " + "  ".join(f"{g:+.4f}" for g in num) + "   (analytic " + "  ".join(f"{g:+.4f}" for g in r0["grad"]) + ")")
    print("  1/N: the slope of the averaged L w.r.t. ONE cell's mu is 1/N of the per-cell slope: e0t0 in set 1 "
          f"{r0['rows'][0]['g_b']:+.4f} / 4 = {r0['rows'][0]['g_b']/4:+.4f};  Tron1 N = 49152/4 = {49152//4:,}")
    print("-- the three clips")
    print(f"  rho: clip_param {cfg['clip_param'][0]} -> [{1-EPS}, {1+EPS}];  V: +-{cfg['clip_param'][0]};  gradient norm: max_grad_norm {max_norm}")
    print("-- one step from set 1's .grad, alpha = 1e-3: SGD (12) against Adam (36)")
    g_raw = r0["grad"]
    g, norm, coef = clip_grad(g_raw, max_norm)
    x0 = [0.0, ELL0, 0.0]
    for label, gg in (("SGD, unclipped .grad", g_raw), ("SGD, clipped .grad", g)):
        x1 = [xi - 1e-3 * gi for xi, gi in zip(x0, gg)]
        print(f"  {label}: steps " + "  ".join(f"{a-b_:+.6f}" for a, b_ in zip(x1, x0)) + f"   sigma -> {math.exp(x1[1]):.6f}")
    opt = AdamPy(3)
    x1, info = opt.step(x0, g, 1e-3)
    for nme, gi, inf in zip(KNOBS, g, info):
        print(f"  Adam step 1, {nme}: g {gi:+.6f}  m {inf['m']:+.6f}  v {inf['v']:.9f}  m^ {inf['mh']:+.6f}  v^ {inf['vh']:.6f}  sqrt {math.sqrt(inf['vh']):.6f}  ratio {inf['ratio']:+.6f}  step {inf['d']:+.6f}")
    print(f"     sigma after Adam's first step = 0.5 e^-0.001 = {math.exp(x1[1]):.6f}   (chapter 24's SGD figure, one cell: 0.499670; SGD with set 1's mean: {math.exp(ELL0 - 1e-3*g_raw[1]):.6f})")
    print(f"     bias correction at k = 1: 1 - beta1 = {1-B1:.1f}, 1 - beta2 = {1-B2:.3f};  without it: m/sqrt(v) = {(1-B1)/math.sqrt(1-B2):.4f} x sign(g)")
    print("  the first step for any gradient size: g/(|g| + eps) = " + "  ".join(f"g={gv:g}: {gv/(abs(gv)+ADAM_EPS):.4f}" for gv in (100.0, 1.0, 1e-4, 1e-8, 1e-9, 0.0)))
    print("-- exercise: the first gradient step taken on set 2 instead")
    r2 = three_knobs(SET2)
    g2c, n2, c2 = clip_grad(r2["grad"], max_norm)
    print("  (12), alpha 1e-3, clipped .grad: steps " + "  ".join(f"{-1e-3*gi:+.6f}" for gi in g2c) + ";   Adam: " + "  ".join(f"{-1e-3*(gi/(abs(gi)+ADAM_EPS)):+.6f}" for gi in g2c)
          + f";   ell -> {ELL0 + 1e-3:.6f}, sigma -> {math.exp(ELL0 + 1e-3):.6f}")
    print(f"  exercise: constant +g, third step: m3 = 0.9 x 0.19 + 0.1 = {0.9*0.19+0.1:.3f} g, 1 - 0.9^3 = {1-0.9**3:.3f}")
    print(f"  exercise: 4 x 469,905 = {4*469905:,};  2 x 125,699 = {2*125699:,};  -0.6380/12288 = {-0.6380/12288:.7f}")
    print("-- pure cases, any size g: constant sign, and alternating sign")
    for label, seq in (("constant +g", lambda k: 1.0), ("alternating +g, -g", lambda k: 1.0 if k % 2 == 0 else -1.0), ("constant, g x 100", lambda k: 100.0)):
        o, xs, ds = AdamPy(1), [0.0], []
        for k in range(200):
            xs, inf = o.step(xs, [seq(k)], 1.0)
            ds.append(inf[0]["d"])
        print(f"  {label}: steps/alpha 1..6 = " + " ".join(f"{d:+.4f}" for d in ds[:6]) + f"   net after 200 = {sum(ds):+.3f}   last two {ds[-2]:+.4f} {ds[-1]:+.4f}")
    print(f"  alternating, steady amplitude of m = (1-beta1)/(1+beta1) g = {(1-B1)/(1+B1):.4f} g")
    print(f"  memory: weights (1-beta) beta^j sum to 1; 1/(1-beta1) = {1/(1-B1):.0f} gradient steps, 1/(1-beta2) = {1/(1-B2):.0f} gradient steps = {1/(1-B2)/20:.0f} iterations of 20")
    print(f"  after max_norm clipping |g_i| <= {max_norm}: one step adds at most (1-beta1) x 1 = {(1-B1)*max_norm:.1f} to m and (1-beta2) x 1 = {(1-B2)*max_norm**2:.3f} to v")
    print("-- four gradient steps on the toy table, alpha fixed at 1e-3 (no lr rule yet), sets 1, 2, 1, 2")
    for label, kw in (("Adam + clip_grad_norm_", dict()), ("Adam, no clip_grad_norm_", dict(clipnorm=False)), ("SGD + clip_grad_norm_", dict(adam=False))):
        log, x = toy_update(rule=False, **kw)
        print(f"  {label}:")
        for i, e in enumerate(log, 1):
            line = (f"    step {i} set {1 if e['keys'] is SET1 else 2}: .grad " + " ".join(f"{g:+.4f}" for g in e["raw"]) + f"  norm {e['norm']:.4f} coef {e['coef']:.4f}  clipped " + " ".join(f"{g:+.4f}" for g in e["g"]))
            if e["info"]:
                line += "  | ratio m^/sqrt(v^) " + " ".join(f"{inf['ratio']:+.4f}" for inf in e["info"])
            line += "  | step " + " ".join(f"{a-b_:+.6f}" for a, b_ in zip(e["x_new"], e["x"]))
            print(line)
            if e["info"] and not kw:
                print("         m " + " ".join(f"{inf['m']:+.6f}" for inf in e["info"]) + "   v " + " ".join(f"{inf['v']:.8f}" for inf in e["info"])
                      + "   m^ " + " ".join(f"{inf['mh']:+.6f}" for inf in e["info"]) + "   sqrt(v^) " + " ".join(f"{math.sqrt(inf['vh']):.6f}" for inf in e["info"]))
        print(f"    after 4: b {x[0]:+.6f}  ell {x[1]:.6f} (sigma {math.exp(x[1]):.6f})  b_V {x[2]:+.6f};   total moved " + " ".join(f"{a-b_:+.6f}" for a, b_ in zip(x, [0.0, ELL0, 0.0])))
    print("-- the numbers an Adam optimizer carries: value, .grad, m, v per knob + one count")
    a_n, c_n, e_n = mlp_count([42, 512, 256, 128, 8]), mlp_count([230, 512, 256, 128, 1]), mlp_count([360, 256, 128, 3])
    print(f"  main optimizer: {a_n:,} + {c_n:,} + 8 = {a_n + c_n + 8:,} knobs in 8 + 8 + 1 = 17 tensors;  m and v: 2 x {a_n + c_n + 8:,} = {2*(a_n + c_n + 8):,} more numbers")
    print(f"  encoder optimizer: {e_n:,} knobs in 6 tensors")
    path, osd = ckpt_optimizer()
    if osd:
        st, pg = osd["state"], osd["param_groups"][0]
        print(f"  checkpoint {path.parent.name}/{path.name}: {len(st)} tensors with state; keys {sorted(next(iter(st.values())).keys())};  step = {sorted({v['step'] for v in st.values()})};  "
              f"lr = {pg['lr']}, betas = {pg['betas']}, eps = {pg['eps']}, weight_decay = {pg['weight_decay']}, amsgrad = {pg['amsgrad']}")
        print("     exp_avg shapes: " + " ".join(str(v["exp_avg"]) for v in st.values()))
    else:
        print("  (no checkpoint under logs/: skipped)")
    print("-- lines cited")
    show(PPO_PY, 56); show(PPO_PY, 85); show(PPO_PY, 88, 90); show(PPO_PY, 103); show(PPO_PY, 260); show(PPO_PY, 269); show(PPO_PY, 273); show(PPO_PY, 286, 290); show(PPO_PY, 319, 321)
    show(PPO_CFG, 100); show(PPO_CFG, 105); show(RUN_PY, 360); show(RUN_PY, 367); show(RUN_PY, 371, 372); show(TRAIN_PY, 136)
    show(ADAM_PY, 36, 38); show(ADAM_PY, 364); show(ADAM_PY, 378, 379); show(ADAM_PY, 414, 419); show(ADAM_PY, 421); show(ADAM_PY, 430); show(ADAM_PY, 432); show(ADAM_PY, 439)
    show(CLIPGRAD_PY, 69); show(CLIPGRAD_PY, 86); show(CLIPGRAD_PY, 94, 96); show(CLIPGRAD_PY, 105); show(CLIPGRAD_PY, 109); show(CLIPGRAD_PY, 114); show(OPTIM_PY, 917); show(OPTIM_PY, 951, 952)


def ch26():
    print("=== chapter 26 — KL and the learning-rate rule ===")
    name = lambda k: f"e{k[0]}t{k[1]}"
    cfg = sf_ppo()
    d = cfg["desired_kl"][0]
    print(f"  config: schedule = {cfg['schedule'][0]!r} (line {cfg['schedule'][1]}), desired_kl = {d} (line {cfg['desired_kl'][1]});  thresholds {2*d} and {d/2}")
    print("-- the simple gauges fail: the mean of rho is 1 for any new bell; the mean of -log rho is the KL")
    for dm, sn in ((0.05, 0.5), (0.15, 0.5), (0.05, 0.45)):
        f_old = lambda a: pdf(a, 0.2, SIG)
        rho = lambda a: pdf(a, 0.2 + dm, sn) / pdf(a, 0.2, SIG)
        print(f"  e0t0's bell N(0.20, 0.5) -> N({0.2+dm:.2f}, {sn}): E_old[rho] = {integrate(lambda a: f_old(a) * rho(a)):.4f}   E_old[-log rho] = {integrate(lambda a: -f_old(a) * math.log(rho(a))):.6f}   (30) = {kl_gauss(0.2, SIG, 0.2+dm, sn):.6f}")
    print("-- from eight sampled actions instead of the whole bell (mu shifted +0.05, sigma kept):")
    ls = {k: -(c["s05_logpi_new"] - c["logpi_old"]) for k, c in cells.items()}
    for k, v in ls.items():
        c = cells[k]
        print(f"  {name(k)}: z {(c['a']-c['mu'])/SIG:+.2f}  rho {c['s05_rho']:.4f}  -log rho {v:+.4f}")
    zbar = sum((c["a"] - c["mu"]) / SIG for c in cells.values()) / 8
    print(f"  mean of -log rho over the 8 cells = {sum(ls.values())/8:+.4f}   (true KL 0.0050; = 0.005 - (0.05/0.5) x mean z, mean z = {zbar:+.4f});  mean rho over the 8 cells = {sum(c['s05_rho'] for c in cells.values())/8:.4f}")
    print("-- (30) piece by piece: E_old[(a - mu_o)^2] = sigma_o^2,  E_old[(a - mu_n)^2] = sigma_o^2 + (mu_o - mu_n)^2")
    print(f"  by integration, old N(0.20, 0.5), mu_n = 0.25: {integrate(lambda a: pdf(a, 0.2) * (a - 0.2) ** 2):.4f} and {integrate(lambda a: pdf(a, 0.2) * (a - 0.25) ** 2):.4f}  (0.25 + 0.0025 = 0.2525)")
    print("-- (30) values, one joint")
    for mo, so, mn, sn in ((0.2, 0.5, 0.2, 0.5), (0.2, 0.5, 0.25, 0.5), (0.2, 0.5, 0.35, 0.5), (0.2, 0.5, 0.3, 0.5), (0.2, 0.5, 0.2, 0.45), (0.2, 0.5, 0.25, 0.45),
                           (0.2, 0.5, 0.2, 0.25), (0.2, 0.25, 0.2, 0.5), (0.2, 0.5, 0.2, 1.0), (0.2, 0.25, 0.25, 0.25), (0.2, 1.0, 0.25, 1.0), (0.2, 0.5, 0.2, 0.55)):
        terms = (math.log(sn / so), (so ** 2 + (mo - mn) ** 2) / (2 * sn ** 2), -0.5)
        print(f"  N({mo}, {so}) || N({mn}, {sn}): ln {terms[0]:+.6f}  middle {terms[1]:.6f}  -0.5  =  {sum(terms):.6f}   [(dmu/sigma)^2/2 = {((mo-mn)/sn)**2/2:.6f}]")
    print("  reversed order of the two pairs with different sigma: " + f"KL(N(.2,.5)||N(.25,.45)) = {kl_gauss(0.2,0.5,0.25,0.45):.6f},  KL(N(.25,.45)||N(.2,.5)) = {kl_gauss(0.25,0.45,0.2,0.5):.6f}")
    print("-- where the thresholds sit: equal sigma, KL = n_joints x (dmu/sigma)^2 / 2")
    for nj in (1, 8):
        lo, hi = math.sqrt(2 * (d / 2) / nj), math.sqrt(2 * (2 * d) / nj)
        print(f"  {nj} joint(s): KL = {d/2} at dmu/sigma = {lo:.4f},  KL = {2*d} at dmu/sigma = {hi:.4f};  sigma 0.5: {lo*0.5:.4f} .. {hi*0.5:.4f};  sigma 1.0: {lo:.4f} .. {hi:.4f} -> x 0.25 rad = {lo*0.25:.4f} .. {hi*0.25:.4f} rad = {math.degrees(lo*0.25):.2f} .. {math.degrees(hi*0.25):.2f} deg")
    print("  sigma only (mu kept): KL(sigma_o || sigma_n) for sigma_n/sigma_o = " + "  ".join(f"{q}: {kl_gauss(0, 1, 0, q):.5f}" for q in (0.8, 0.9, 0.95, 1.05, 1.1, 1.2)))
    print(f"  for a small change of ell, KL ~ (d ell)^2: d ell = 0.1 -> {kl_gauss(0, 1, 0, math.exp(0.1)):.5f} and {kl_gauss(0, 1, 0, math.exp(-0.1)):.5f}")
    print("-- the code's guard inside the log (ppo.py:224): the first gradient step of an iteration, bells identical")
    g1 = math.log(1 + 1e-5)
    print(f"  ln(1 + 1e-5) = {g1:.3e} per joint;  1 joint: {kl_gauss(0, SIG, 0, SIG, 1e-5):.3e};  8 joints: {8*g1:.3e}  -> 0 < KL < {d/2}: lr x 1.5 every time")
    print("-- (37) on the toy shifts (one joint, sigma 0.5, with the guard)")
    for dm in (0.0, 0.05, 0.07, 0.10, 0.15):
        kl = kl_gauss(0, SIG, dm, SIG, 1e-5)
        lr, act = lr_rule(1e-3, kl)
        print(f"  dmu = {dm:.2f}: KL = {dm*dm/(2*SIG**2):.4f} (code {kl:.6f}) -> {act}: lr 1e-3 -> {lr:.6g}")
    print("-- the ladder of lr values")
    lr, k = 1e-3, 0
    seq = [lr]
    while lr < 1e-2:
        lr = min(1e-2, lr * 1.5); k += 1; seq.append(lr)
    print(f"  up from 1e-3: " + " ".join(f"{v:.5g}" for v in seq) + f"   ({k} raises reach the ceiling; 1e-3 x 1.5^6 = {1e-3*1.5**6:.5g})")
    lr, k = 1e-3, 0
    seq = [lr]
    while lr > 1e-5:
        lr = max(1e-5, lr / 1.5); k += 1; seq.append(lr)
    print(f"  down from 1e-3: " + " ".join(f"{v:.4g}" for v in seq) + f"   ({k} cuts reach the floor; 1e-3 / 1.5^11 = {1e-3/1.5**11:.4g}, / 1.5^12 = {1e-3/1.5**12:.4g})")
    print("  from the floor: " + " ".join(f"{1e-5*1.5**j:.4g}" for j in range(0, 18)) + f"   (1e-5 x 1.5^17 = {1e-5*1.5**17:.4g}, x 1.5^18 = {1e-5*1.5**18:.4g} -> ceiling)")
    print(f"  1 raise then 19 cuts from 1e-3: {max(1e-5, 1.5e-3/1.5**19):.1e};  the floor is reached after 1 raise + {next(j for j in range(30) if 1.5e-3/1.5**j <= 1e-5)} cuts")
    print("  printed with 4 decimals (on_policy_runner.py:325): " + "  ".join(f"{v:.4g} -> '{v:.4f}'" for v in (1e-5, 3.4e-5, 5.1e-5, 1.14e-4, 3.84e-4, 1e-3)))
    print("-- the rule on the toy table: four gradient steps, Adam, alpha starts at 1e-3")
    log, x = toy_update(rule=True)
    for i, e in enumerate(log, 1):
        s_n = math.exp(e["x"][1])
        print(f"  step {i} set {1 if e['keys'] is SET1 else 2}: knobs before b {e['x'][0]:+.6f} sigma {s_n:.6f}  ->  KL {e['kl']:.3e} (without guard {kl_gauss(0, SIG, e['x'][0], s_n):.3e})  {e['act']}: lr = {e['lr']:.6g}   steps " + " ".join(f"{a-b_:+.6f}" for a, b_ in zip(e["x_new"], e["x"])))
    print(f"  after 4: b {x[0]:+.6f}  sigma {math.exp(x[1]):.6f}  b_V {x[2]:+.6f};  KL of the final bell {kl_gauss(0, SIG, x[0], math.exp(x[1])):.3e}")
    print("-- clip and KL do not measure the same thing: one joint, equal sigma, mu moved dmu toward a: rho = exp(dmu/sigma x z - (dmu/sigma)^2/2)")
    for q in (0.1, 0.2):
        zc = (math.log(1.2) + q * q / 2) / q
        print(f"  dmu/sigma = {q}: KL = {q*q/2:.4f};  rho reaches 1.2 for z >= {zc:.4f}")
    for k in ((0, 0), (0, 2)):
        c = cells[k]
        for dm in (0.10, 0.12, 0.15):
            print(f"  {name(k)} (z {(c['a']-c['mu'])/SIG:+.2f}) moved {dm:+.2f}: rho {rho_of(c, dm):.4f}   KL {dm*dm/(2*SIG**2):.4f}")
    print("-- exercises")
    for label, mo, so, mn, sn in (("e1t3 bell moved to -0.18", -0.30, 0.5, -0.18, 0.5), ("e1t0 bell moved to -0.02", -0.10, 0.5, -0.02, 0.5), ("sigma 0.5 -> 0.55", 0.2, 0.5, 0.2, 0.55),
                                  ("sigma 0.5 -> 0.25", 0.2, 0.5, 0.2, 0.25), ("reversed: N(0.25,0.45)||N(0.2,0.5)", 0.25, 0.45, 0.2, 0.5), ("e0t1 bell 0.30 -> 0.36, sigma 0.5 -> 0.48", 0.30, 0.5, 0.36, 0.48)):
        kl = kl_gauss(mo, so, mn, sn)
        print(f"  {label}: ln {math.log(sn/so):+.6f}  middle {(so**2 + (mo-mn)**2)/(2*sn**2):.6f}  KL = {kl:.6f}  -> {lr_rule(1e-3, kl)[1]}")
    print(f"  shift that gives KL = 0.02 at sigma 0.25: {0.2*0.25:.4f};  at sigma 1.0: {0.2:.4f};  8 joints, sigma 0.5, equal shifts: {math.sqrt(2*0.02/8)*0.5:.4f}")
    for label, lr0, kls in (("ladder A", 1e-3, (8e-5, 0.003, 0.025, 0.012, 0.04)), ("ladder B", 5e-3, (8e-5, 0.004, 0.012, 0.031, 0.0199)), ("ladder C", 1.5e-5, (0.03, 0.05, 8e-5, 0.021))):
        lr, seq = lr0, []
        for kl in kls:
            lr, act = lr_rule(lr, kl)
            seq.append(f"{kl:g} {act} {lr:.6g}")
        print(f"  {label} from {lr0:g}: " + " | ".join(seq))
    e0 = [ls[(0, t)] for t in range(4)]
    e1 = [ls[(1, t)] for t in range(4)]
    print(f"  mean of -log rho over env 0's four cells: ({' '.join(f'{v:+.4f}' for v in e0)})/4 = {sum(e0)/4:+.4f};  env 1: {sum(e1)/4:+.4f};  mean z env 0 {sum((cells[(0,t)]['a']-cells[(0,t)]['mu'])/SIG for t in range(4))/4:+.2f}")
    print(f"  net raises from 1e-5 to 1.139e-4: log(11.39)/log(1.5) = {math.log(1.139e-4/1e-5)/math.log(1.5):.2f}")
    runs = training_runs()
    if len(runs) >= 3:
        a, b_ = runs[1][3], runs[2][3]
        same = 0
        for (s1, v1), (s2, v2) in zip(a["Loss/learning_rate"], b_["Loss/learning_rate"]):
            if v1 != v2:
                break
            same += 1
        samek = 0
        for (s1, v1), (s2, v2) in zip(a["Policy/mean_kl"], b_["Policy/mean_kl"]):
            if v1 != v2:
                break
            samek += 1
        print(f"  the two 64-env runs (same seed): learning_rate identical for the first {same} iterations, mean_kl for the first {samek}")
        lr = [v for _, v in a["Loss/learning_rate"]]
        print(f"  218-iteration run: iterations whose end lr is the floor 1e-5: {sum(1 for v in lr if abs(v-1e-5) < 1e-9)};  ever back to 1e-3 or above: {sum(1 for v in lr if v >= 1e-3)};  median rung: {statistics.median(round(math.log(v/1e-5)/math.log(1.5)) for v in lr)}")
    print("-- real runs under logs/ (not the 2048-env configuration): what the rule did")
    for nm, n_env, max_it, sc in training_runs():
        lr = [v for _, v in sc["Loss/learning_rate"]]
        kl = [v for _, v in sc["Policy/mean_kl"]]
        rung = [round(math.log(v / 1e-5) / math.log(1.5)) for v in lr]
        off = max(abs(v - 1e-5 * 1.5 ** r_) / v for v, r_ in zip(lr, rung))
        cnt = collections.Counter(rung)
        print(f"  {nm}: {n_env} envs, max_iterations {max_it}, {len(lr)} iterations logged ({24*n_env:,} transitions per table, {24*n_env//4:,} per set)")
        print(f"     lr at the end of iteration 0: {lr[0]:.3g};  min {min(lr):.3g}  max {max(lr):.3g} = 1e-5 x 1.5^{max(rung)};  largest relative distance from the ladder {off:.1e}")
        print("     iterations on each rung k (lr = 1e-5 x 1.5^k): " + " ".join(f"{k_}:{cnt[k_]}" for k_ in sorted(cnt)) + f";  printed as 0.0000 (< 5e-5): {sum(1 for v in lr if v < 5e-5)} of {len(lr)}")
        skl = sorted(kl)
        print(f"     mean_kl: iteration 0 = {kl[0]:.4f};  median {statistics.median(kl):.4f}  mean {sum(kl)/len(kl):.4f}  min {skl[0]:.4f}  max {skl[-1]:.4f};  inside [0.005, 0.02]: {sum(1 for v in kl if 0.005 <= v <= 0.02)} of {len(kl)}")
        print("     first 10 (lr, mean_kl): " + "  ".join(f"({a:.3g}, {b_:.4f})" for a, b_ in list(zip(lr, kl))[:10]))
    print("-- lines cited")
    show(PPO_PY, 58, 59); show(PPO_PY, 63, 64); show(PPO_PY, 71, 74); show(PPO_PY, 150, 151); show(PPO_PY, 200, 201); show(PPO_PY, 217, 218); show(PPO_PY, 221, 249)
    show(PPO_PY, 280, 284); show(PPO_PY, 295); show(PPO_PY, 330); show(STORE_PY, 145, 146); show(STORE_PY, 248, 249); show(STORE_PY, 281, 285)
    show(RUN_PY, 283); show(RUN_PY, 285); show(RUN_PY, 325); show(PPO_CFG, 100, 101); show(PPO_CFG, 104)


def ch27():
    print("=== chapter 27 — one whole iteration ===")
    name = lambda k: f"e{k[0]}t{k[1]}"
    cfg = sf_ppo()
    Tn, ep, mb = cfg["num_steps_per_env"][0], cfg["num_learning_epochs"][0], cfg["num_mini_batches"][0]
    B = 2048
    print(f"  config: num_steps_per_env {Tn} (line {cfg['num_steps_per_env'][1]}), num_learning_epochs {ep} (line {cfg['num_learning_epochs'][1]}), num_mini_batches {mb} (line {cfg['num_mini_batches'][1]}), "
          f"max_iterations {cfg['max_iterations'][0]:,} (line {cfg['max_iterations'][1]}), save_interval {cfg['save_interval'][0]} (line {cfg['save_interval'][1]})")
    print(f"-- sizes: table {Tn} x {B} = {Tn*B:,};  mini-batch {Tn*B}//{mb} = {Tn*B//mb:,};  gradient steps {ep} x {mb} = {ep*mb};  each cell used {ep} times;  toy: 8 cells, 2 sets of 4, 2 x 2 = 4")
    for n_env in (16, 64, 4096):
        print(f"     {n_env} envs: table {Tn*n_env:,}, mini-batch {Tn*n_env//mb:,}")
    print(f"  flatten(0, 1) of a (T, B) table: cell (t, e) -> index t x B + e;  toy (B = 2): " + " ".join(f"{name((e, t))}->{t*2+e}" for t in range(4) for e in range(2)))
    idx = {k: k[1] * 2 + k[0] for k in cells}
    print(f"  the toy split as indices: set 1 = {[idx[k] for k in SET1]}  set 2 = {[idx[k] for k in SET2]}  -> one permutation that gives it: {[idx[k] for k in SET1 + SET2]}")
    print(f"  number of ways to split 8 cells into an ordered pair of sets of 4: C(8,4) = {math.comb(8, 4)}")
    print("-- how many times each thing happens in one iteration (toy / Tron1)")
    for what, toy_n, real_n in (("policy steps (act + env.step)", 4, Tn), ("transitions written", 8, Tn * B), ("GAE backward pass", 1, 1), ("normalisation of A", 1, 1), ("shuffle", 1, 1),
                                ("epochs", 2, ep), ("mini-batches per epoch", 2, mb), ("KL checks = gradient steps", 4, ep * mb), ("encoder gradient steps", "-", ep * mb), ("storage.clear()", 1, 1)):
        print(f"  {what}: {toy_n} / {real_n:,}" if isinstance(real_n, int) else f"  {what}: {toy_n} / {real_n}")
    it = cfg["max_iterations"][0]
    print(f"-- whole training: {it:,} iterations x {ep*mb} = {it*ep*mb:,} gradient steps (and as many for the encoder);  {it:,} x {Tn*B:,} = {it*Tn*B:,} transitions")
    dec, dt, step_dt, ep_s = timing()
    ep_steps = round(ep_s / step_dt)
    print(f"  one episode of {ep_steps} steps spans {ep_steps}/{Tn} = {ep_steps/Tn:.2f} tables = {ep_steps/Tn*ep*mb:.0f} gradient steps;  one table = {Tn} x {step_dt} s = {Tn*step_dt:.2f} s of simulated time")
    print("-- (34): the horizon of GAE, 1/(1 - gamma lambda)")
    g, l = cfg["gamma"][0], cfg["lam"][0]
    for label, gg, ll, Tt in (("toy", GAMMA, LAM, 4), ("Tron1", g, l, Tn)):
        gl = gg * ll
        print(f"  {label}: gamma {gg} x lambda {ll} = {gl:.4f};  1/(1 - {gl:.4f}) = {1/(1-gl):.2f} steps;  weight of the delta {Tt} steps ahead (gamma lambda)^{Tt} = {gl**Tt:.4f};  share of the weight inside {Tt} steps 1 - that = {1-gl**Tt:.4f};  1/(1-gamma) = {1/(1-gg):.0f}")
    gl = g * l
    for Tt in (4, 8, 16, 17, 24, 48, 100, 200):
        print(f"     Tron1, table of {Tt} rows: share of the GAE weight that falls inside the table for the top row = {1-gl**Tt:.4f}")
    print(f"  mean over the {Tn} rows of the share inside the table (row t has {Tn}-t steps left): {sum(1-gl**(Tn-t) for t in range(Tn))/Tn:.4f};  toy: {sum(1-GL**(4-t) for t in range(4))/4:.4f}")
    print("-- memory of one table: floats per transition (rollout_storage.py:71-120)")
    parts = dict(observations=36, next_observations=36, critic_obs=230, observation_history=360, commands=3, rewards=1, actions=8, actions_log_prob=1, values=1, returns=1, advantages=1, mu=8, sigma=8)
    nf = sum(parts.values())
    print("  " + " + ".join(f"{v}" for v in parts.values()) + f" = {nf} floats x 4 bytes + 1 byte (dones) = {nf*4+1} bytes")
    for Tt in (24, 100, 200):
        print(f"     {Tt} rows x {B} envs: {Tt*B*(nf*4+1)/1e6:.1f} MB ({Tt*B*(nf*4+1)/2**20:.1f} MiB)")
    print(f"  share of the history: {parts['observation_history']}/{nf} = {parts['observation_history']/nf*100:.0f} %")
    copied = nf - parts["next_observations"] - parts["rewards"]      # rollout_storage.py:236-249 index with a tensor: 11 copies
    print(f"  mini_batch_generator copies {copied} of the {nf} floats per cell (advanced indexing): {copied*4*Tn*B/1e6:.1f} MB more at {Tn} x {B} while update() runs")
    print("-- the toy iteration end to end: phases A and B are chapters 7 and 22; phase C = 4 gradient steps (Adam, lr rule, clip_grad_norm_)")
    log, x = toy_update(rule=True)
    for i, e in enumerate(log, 1):
        r = e["r"]
        print(f"  step {i} (epoch {e['ep']}, set {1 if e['keys'] is SET1 else 2}): KL {e['kl']:.3e} {e['act']} lr {e['lr']:.6g} | rho " + " ".join(f"{row['rho']:.4f}" for row in r["rows"])
              + f" | L^CLIP {r['Lc']:+.4f}  L_V {r['Lv']:.4f}  H {r['H']:.4f}  L {r['L']:+.4f} | .grad " + " ".join(f"{g:+.4f}" for g in e["raw"]) + f" norm {e['norm']:.4f} coef {e['coef']:.4f} | steps " + " ".join(f"{a-b_:+.6f}" for a, b_ in zip(e["x_new"], e["x"]))
              + f" | after: b {e['x_new'][0]:+.6f} sigma {math.exp(e['x_new'][1]):.6f} b_V {e['x_new'][2]:+.6f}")
        if any(row["pc"] or row["vc"] for row in r["rows"]):
            print("       clipped terms chosen: " + ", ".join(f"{name(row['key'])}{' rho' if row['pc'] else ''}{' V' if row['vc'] else ''}" for row in r["rows"] if row["pc"] or row["vc"]))
    print(f"  mean over the 4 steps (what the log prints): Value function loss {sum(e['r']['Lv'] for e in log)/4:.4f}   Surrogate loss {sum(e['r']['Lc'] for e in log)/4:+.4f}   mean_kl {sum(e['kl'] for e in log)/4:.3e}")
    print("  the other order (set 2 first):")
    log2, x2 = toy_update(order=(SET2, SET1), rule=True)
    for i, e in enumerate(log2, 1):
        r = e["r"]
        print(f"  step {i} (epoch {e['ep']}, set {1 if e['keys'] is SET1 else 2}): KL {e['kl']:.3e} {e['act']} lr {e['lr']:.6g} | L^CLIP {r['Lc']:+.4f}  L_V {r['Lv']:.4f}  L {r['L']:+.4f} | .grad " + " ".join(f"{g:+.4f}" for g in e["raw"])
              + f" norm {e['norm']:.4f} coef {e['coef']:.4f} | steps " + " ".join(f"{a-b_:+.6f}" for a, b_ in zip(e["x_new"], e["x"])))
    print(f"     after 4: b {x2[0]:+.6f} sigma {math.exp(x2[1]):.6f} b_V {x2[2]:+.6f}   (order 1,2: b {x[0]:+.6f} sigma {math.exp(x[1]):.6f} b_V {x[2]:+.6f})")
    print("  one gradient step on the whole table instead of four on halves (same lr rule):")
    log3, x3 = toy_update(order=(list(cells),), epochs=1, rule=True)
    e = log3[0]
    print(f"     L {e['r']['L']:+.4f}  .grad " + " ".join(f"{g:+.4f}" for g in e["raw"]) + f"  norm {e['norm']:.4f}  steps " + " ".join(f"{a-b_:+.6f}" for a, b_ in zip(e["x_new"], e["x"])))
    print("-- exercises")
    print(f"  8 epochs x 8 mini-batches: {8*8} gradient steps, mini-batch {Tn*B//8:,}, each cell used 8 times;  cell (t=3, e=10) -> index 3 x 2048 + 10 = {3*2048+10}")
    for ll in (0.9, 0.99):
        gl2 = g * ll
        print(f"  gamma {g}, lambda {ll}: gamma lambda = {gl2:.4f}, horizon {1/(1-gl2):.2f}, share inside 24 rows (top row) {1-gl2**24:.4f}")
    print("  toy weights (gamma lambda)^k: " + " ".join(f"{GL**k:.4f}" for k in range(4)) + f"  sum of 4 = {sum(GL**k for k in range(4)):.4f} of {1/(1-GL):.4f};  share by rows left: " + " ".join(f"{n} left: {1-GL**n:.4f}" for n in (4, 3, 2, 1)))
    print("  Tron1 share by rows left: " + " ".join(f"{n} left: {1-gl**n:.4f}" for n in (24, 12, 6, 1)))
    print(f"  memory, 48 rows x 4096 envs: 48 x 4096 x {nf*4+1} = {48*4096*(nf*4+1):,} bytes = {48*4096*(nf*4+1)/1e6:.1f} MB;  24 rows x 4096 envs: {24*4096*(nf*4+1)/1e6:.1f} MB")
    vals = []
    for f_ in sorted((ISAAC / "isaaclab_tasks").rglob("rsl_rl_ppo_cfg.py")):
        vals += [(int(v), "locomotion" in str(f_)) for v in re.findall(r"num_steps_per_env\s*=\s*(\d+)", f_.read_text())]
    cnt = collections.Counter(v for v, _ in vals)
    print(f"  num_steps_per_env in Isaac Lab's own rsl_rl configs ({len(vals)} classes): " + " ".join(f"{k}:{cnt[k]}" for k in sorted(cnt))
          + f";  the locomotion tasks: {sorted(collections.Counter(v for v, loco in vals if loco).items())}")
    limx = re.findall(r"num_steps_per_env = (\d+)", PPO_CFG.read_text())
    print(f"  LimX file: {limx}")
    print("-- real runs under logs/: time per phase (NOT the 2048-env configuration)")
    for nm, n_env, max_it, sc in training_runs():
        col = [v for _, v in sc["Perf/collection time"]]
        lrn = [v for _, v in sc["Perf/learning_time"]]
        fps = [v for _, v in sc["Perf/total_fps"]]
        med = statistics.median
        print(f"  {nm}: {n_env} envs, {len(col)} iterations: collection median {med(col):.3f} s, learning median {med(lrn):.3f} s ({med(lrn)/(med(col)+med(lrn))*100:.0f} % of the iteration), total_fps median {med(fps):.1f};  {24*n_env} / ({med(col):.3f} + {med(lrn):.3f}) = {24*n_env/(med(col)+med(lrn)):.0f}")
    print("-- lines cited")
    show(RUN_PY, 81); show(RUN_PY, 170, 177); show(RUN_PY, 192); show(RUN_PY, 212, 213); show(RUN_PY, 218); show(RUN_PY, 225); show(RUN_PY, 227, 234); show(RUN_PY, 237, 239); show(RUN_PY, 250)
    show(RUN_PY, 270, 274); show(RUN_PY, 314); show(RUN_PY, 320, 327)
    show(PPO_PY, 97, 98); show(PPO_PY, 176, 178); show(PPO_PY, 180, 190); show(PPO_PY, 203, 205); show(PPO_PY, 212); show(PPO_PY, 216); show(PPO_PY, 292, 303); show(PPO_PY, 326, 333)
    show(STORE_PY, 132, 133); show(STORE_PY, 148); show(STORE_PY, 184, 185); show(STORE_PY, 222, 236); show(STORE_PY, 251, 255); show(STORE_PY, 287); show(STORE_PY, 289, 294); show(STORE_PY, 304, 305)
    show(PPO_CFG, 81, 83); show(PPO_CFG, 98, 99); show(PPO_CFG, 102, 103)


# ---------------------------------------------------------------- chapters 28-30 (part 6: the whole system)
import json

PROBE = pathlib.Path(__file__).with_name("env_probe_2026-11-26.json")           # written by scripts/workbook_env_probe.py
DEPLOY = pathlib.Path(__file__).with_name("deploy_snapshot_2026-11-26")         # read-only copies, see its SOURCE.md
SOLE_CFG = EXT / "assets/config/solefoot_cfg.py"
SF_VARIANTS = LOCO / "robots/limx_solefoot_env_cfg.py"
PROJ_EVENTS, PROJ_OBS, PROJ_CURR = LOCO / "mdp/events.py", LOCO / "mdp/observations.py", LOCO / "mdp/curriculums.py"
EV_MGR = ISAAC / "isaaclab/isaaclab/managers/event_manager.py"
REW_MGR = ISAAC / "isaaclab/isaaclab/managers/reward_manager.py"
ISAAC_EVENTS = ISAAC / "isaaclab/isaaclab/envs/mdp/events.py"
ARTIC = ISAAC / "isaaclab/isaaclab/assets/articulation/articulation.py"
MLP_CFG = EXT / "utils/wrappers/rsl_rl/rl_mlp_cfg.py"


def probe():
    return json.loads(PROBE.read_text(encoding="utf-8")) if PROBE.exists() else None


def wide_cell(shift=0.05, ahat=None, sigma_new=SIG):
    """The 8-joint widening of cell e0 t0 (glossary 8.6, toy_answer_key.mu8/eps8) pushed through the whole line:
    sample -> log-density per joint -> sum -> new bell at mu + shift -> rho -> L_i -> dL_i/dmu_j."""
    ahat = cells[(0, 0)]["Ahat"] if ahat is None else ahat
    mu, eps = toy.mu8, toy.eps8
    a = [m + SIG * e for m, e in zip(mu, eps)]
    lp_old = [toy.logpi(x, m) for x, m in zip(a, mu)]
    mu_n = [m + shift for m in mu]
    lp_new = [-0.5 * math.log(2 * math.pi) - math.log(sigma_new) - (x - m) ** 2 / (2 * sigma_new ** 2) for x, m in zip(a, mu_n)]
    rho = math.exp(sum(lp_new) - sum(lp_old))
    rc = clipf(rho, 1 - EPS, 1 + EPS)
    raw, cl = -ahat * rho, -ahat * rc
    g = [0.0 if cl > raw else -ahat * rho * (x - m) / sigma_new ** 2 for x, m in zip(a, mu_n)]
    kl = sum(math.log(sigma_new / SIG) + (SIG ** 2 + (mo - mn) ** 2) / (2 * sigma_new ** 2) - 0.5 for mo, mn in zip(mu, mu_n))
    return dict(mu=mu, eps=eps, a=a, lp_old=lp_old, mu_n=mu_n, lp_new=lp_new, S_old=sum(lp_old), S_new=sum(lp_new), rho=rho,
                rho_i=[math.exp(n - o) for n, o in zip(lp_new, lp_old)], raw=raw, cl=cl, L=max(raw, cl), clipped=cl > raw, g=g, kl=kl,
                H=8 * H_gauss(sigma_new), ahat=ahat)


def ch28():
    print("=== chapter 28 — the value line E1-E15, with shapes ===")
    sf = sf_ppo()
    T_, ep, mb = sf["num_steps_per_env"][0], sf["num_learning_epochs"][0], sf["num_mini_batches"][0]
    B = 2048
    N = T_ * B // mb
    a_n, c_n, e_n = mlp_count([42, 512, 256, 128, 8]), mlp_count([230, 512, 256, 128, 1]), mlp_count([360, 256, 128, 3])
    print(f"B = {B}   table {T_} x {B} = {T_*B:,} cells   N = {N:,} per mini-batch   gradient steps {ep} x {mb} = {ep*mb}")
    print(f"knobs the main optimizer holds: actor {a_n:,} + logstd 8 + critic {c_n:,} = {a_n + 8 + c_n:,};  encoder {e_n:,} (its own optimizer)")
    print("-- the fifteen stations: the lines the chapter cites")
    for label, path, a, b in (("E1 encoder", PPO_PY, 136, None), ("   encode()", ENC_PY, 95, 100), ("E2 concat", PPO_PY, 137, 139),
                              ("E3-E5 mean, sigma, bell", AC_PY, 155, 157), ("   logstd", AC_PY, 118, None), ("E6 sample", AC_PY, 159, 161),
                              ("   sample()", TORCH_NORMAL, 70, 73), ("E7-E8 log_prob()", TORCH_NORMAL, 84, 92), ("E9 sum over joints", AC_PY, 163, 164),
                              ("E10 store", PPO_PY, 147, 151), ("   add_transitions", STORE_PY, 140, 146), ("E11 new bell", PPO_PY, 203, 210),
                              ("E12 new log-prob", PPO_PY, 212, 219), ("E13 ratio", PPO_PY, 252, 254), ("E14 loss", PPO_PY, 256, 260),
                              ("   value loss, entropy, total", PPO_PY, 264, 278), ("E15 backward and step", PPO_PY, 287, 290),
                              ("critic branch", PPO_PY, 144, None), ("   evaluate", AC_PY, 170, 172), ("   GAE", STORE_PY, 187, 207),
                              ("entropy sum", AC_PY, 151, 153), ("KL sum over joints", PPO_PY, 223, 233), ("storage shapes", STORE_PY, 92, 120),
                              ("flatten", STORE_PY, 236, 249), ("yield (12 values)", STORE_PY, 287, None), ("loop header", PPO_PY, 190, 202),
                              ("runner: history flatten", RUN_PY, 180, None), ("runner: sizes", RUN_PY, 56, 63), ("runner: inference_mode", RUN_PY, 173, 175)):
        print(f"  {label}")
        show(path, a, b)
    print("-- shapes at Tron1 scale")
    for name, shp in (("h", (B, 360)), ("v_hat", (B, 3)), ("s", (B, 42)), ("mu", (B, 8)), ("ell / sigma", (8,)), ("a", (B, 8)),
                      ("log-density per joint", (B, 8)), ("log pi", (B,)), ("stored log pi", (T_, B, 1)), ("V", (B, 1)),
                      ("mini-batch: s", (N, 42)), ("mini-batch: new log pi", (N,)), ("mini-batch: old log pi, A_hat, R, V_old", (N, 1)),
                      ("rho", (N,)), ("entropy, kl", (N,)), ("old mu, old sigma", (N, 8))):
        print(f"    {name:42s} {shp}   numbers: {math.prod(shp):,}")
    print(f"    encoder input per collection step {B} x 360 = {B*360:,};  per-joint log-densities in one gradient step {N} x 8 = {N*8:,}")
    print(f"    cells through E1-E10 per iteration {T_*B:,};  cell-visits through E11-E15 per iteration {ep*mb} x {N:,} = {ep*mb*N:,} = {ep} per cell")
    print(f"    with 4096 envs: N = {T_*4096//mb:,};  with 64 envs: N = {T_*64//mb:,}")

    w = wide_cell()
    print("-- example: e0 t0 widened to 8 joints (glossary 8.6), A_hat = %+.4f" % w["ahat"])
    print("   j   mu    eps     a    a-mu  log-density   mu_new  a-mu_new  log-density_new   rho_j    dL_i/dmu_j")
    for j in range(8):
        print(f"   {j+1}  {w['mu'][j]:+.2f}  {w['eps'][j]:+.2f}  {w['a'][j]:+.2f}  {w['a'][j]-w['mu'][j]:+.2f}   {w['lp_old'][j]:+.4f}     {w['mu_n'][j]:+.2f}   {w['a'][j]-w['mu_n'][j]:+.2f}     {w['lp_new'][j]:+.4f}       {w['rho_i'][j]:.4f}   {w['g'][j]:+.4f}")
    print(f"   sum log pi_old = {w['S_old']:.4f}  pi_old = {math.exp(w['S_old']):.4f}   sum log pi_new = {w['S_new']:.4f}   difference {w['S_new']-w['S_old']:+.4f}   rho = {w['rho']:.4f}")
    print(f"   product of rho_j = {math.prod(w['rho_i']):.4f};  raw {w['raw']:+.4f}  clipped {w['cl']:+.4f}  L_i = {w['L']:+.4f}  picks clipped: {w['clipped']}")
    print(f"   one joint alone (e0 t0 of the table): log pi {cells[(0,0)]['logpi_old']:.4f}, density {math.exp(cells[(0,0)]['logpi_old']):.4f}, rho {surr_cell(cells[(0,0)], 0.05)['rho']:.4f}")
    print(f"   peak log-density {toy.C:.4f} = log {math.exp(toy.C):.4f};  8 x peak = {8*toy.C:.4f}")
    print(f"   H = 8 x {H_gauss(SIG):.4f} = {w['H']:.4f};  KL = 8 x {0.05**2/(2*SIG**2):.4f} = {w['kl']:.4f}  -> rule: {lr_rule(1e-3, w['kl'])}")
    print(f"   sum of dL_i/dmu_j = {sum(w['g']):+.4f};  largest |.| at joint {max(range(8), key=lambda j: abs(w['g'][j])) + 1}")

    print("-- exercises")
    for label, kw in (("A_hat = e1 t1's", dict(ahat=cells[(1, 1)]["Ahat"])), ("shift -0.05", dict(shift=-0.05)), ("shift +0.10", dict(shift=0.10)),
                      ("shift +0.20", dict(shift=0.20)), ("shift +0.20, A_hat = e1 t1's", dict(shift=0.20, ahat=cells[(1, 1)]["Ahat"]))):
        v = wide_cell(**kw)
        print(f"   {label}: A_hat {v['ahat']:+.4f}  S_new {v['S_new']:.4f}  rho {v['rho']:.4f}  raw {v['raw']:+.4f}  clipped {v['cl']:+.4f}  L_i {v['L']:+.4f}  picks clipped {v['clipped']}  KL {v['kl']:.4f}")
        print(f"       dL_i/dmu_j = {[round(x, 4) for x in v['g']]}")
    half = wide_cell(shift=0.05)
    first4 = sum(half["lp_new"][:4]) - sum(half["lp_old"][:4])
    print(f"   joints 1-4 only: sum log pi_old {sum(half['lp_old'][:4]):.4f}  new {sum(half['lp_new'][:4]):.4f}  rho {math.exp(first4):.4f};  joints 5-8: rho {math.exp(sum(half['lp_new'][4:]) - sum(half['lp_old'][4:])):.4f}")
    print("   -- the squeeze trap on mini-batch 1 with mu + 0.05 (ppo.py:253, :256-257): (N,) against (N,1) broadcasts to (N,N)")
    rows = [surr_cell(cells[k], 0.05) for k in SET1]
    ah = [cells[k]["Ahat"] for k in SET1]
    rho, rc = [r["rho"] for r in rows], [r["rc"] for r in rows]
    n = len(ah)
    right = sum(max(-ah[i] * rho[i], -ah[i] * rc[i]) for i in range(n)) / n
    # both squeezes missing: surrogate[i][j] = -A_i rho_j and clipped[i][j] = -A_i clip(rho_j)
    both = sum(max(-ah[i] * rho[j], -ah[i] * rc[j]) for i in range(n) for j in range(n)) / n ** 2
    # only :256 missing: surrogate is (N,N), the clipped term stays (N,) and broadcasts along the rows as -A_j clip(rho_j)
    only256 = sum(max(-ah[i] * rho[j], -ah[j] * rc[j]) for i in range(n) for j in range(n)) / n ** 2
    print(f"      A_hat {[round(x, 4) for x in ah]}  rho {[round(x, 4) for x in rho]}")
    print(f"      right (both squeezed): {right:+.4f};  :256 and :257 both unsqueezed: {both:+.4f} = -mean(A_hat) x mean(rho) = {-sum(ah)/n * sum(rho)/n:+.4f};  only :256 unsqueezed: {only256:+.4f}")
    print("      (all three agree with torch 2.5.1: -0.2958, -0.3013, +0.1140 - checked 2026-11-26, not part of this script)")
    print(f"   element counts: a 16 x 8 batch -> log-densities {16*8}, log pi {16}, loss 1")


def _event_terms():
    """[(name, line, mode, func name, params dict, interval)] of EventsCfg; SceneEntityCfg values become their keyword dict."""
    out = []
    for name, ln, call in cfg_terms("EventsCfg"):
        params = {}
        node = _kw(call, "params")
        for k, v in zip(node.keys, node.values) if node is not None else ():
            try:
                params[k.value] = ast.literal_eval(v)
            except Exception:
                params[k.value] = {kk.arg: ast.literal_eval(kk.value) for kk in v.keywords} | {"_entity": ast.literal_eval(v.args[0])}
        iv = _kw(call, "interval_range_s")
        out.append((name, ln, ast.literal_eval(_kw(call, "mode")), _kw(call, "func").attr, params, ast.literal_eval(iv) if iv is not None else None))
    return out


def _ppo_defaults():
    """Default values in the signature of PPO.__init__ (ppo.py), with their line numbers."""
    fn = next(n for n in ast.walk(ast.parse(PPO_PY.read_text())) if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    args = fn.args.args[-len(fn.args.defaults):]
    return {a.arg: (ast.literal_eval(d), a.lineno) for a, d in zip(args, fn.args.defaults)}


def ch29():
    print("=== chapter 29 — the whole SF task: every config class, and from the toy table to full size ===")
    print("-- the config classes of limx_base_env_cfg.py and how many entries each holds")
    for cls in ("CommandsCfg", "ActionsCfg", "EventsCfg", "RewardsCfg", "TerminationsCfg", "CurriculumCfg"):
        t = cfg_terms(cls)
        print(f"   {cls} at :{cfg_class(cls).lineno}  {len(t)} entries: {[n for n, *_ in t]}")
    for grp in ("PolicyCfg", "HistoryObsCfg", "CriticCfg", "CommandsObsCfg"):
        t = cfg_terms("ObservarionsCfg", grp)
        print(f"   ObservarionsCfg.{grp} at :{cfg_class('ObservarionsCfg', grp).lineno}  {len(t)} entries")
    show(BASE, 484); show(BASE, 495, 502)

    print("-- EventsCfg: the eleven entries")
    ev = _event_terms()
    for name, ln, mode, fn, params, iv in ev:
        short = {k: v for k, v in params.items() if k != "asset_cfg"}
        ent = params.get("asset_cfg", {})
        print(f"   :{ln:<3d} {mode:8s} {name:34s} func={fn}  bodies/joints={ {k: v for k, v in ent.items() if k != '_entity'} }  {short}{'  interval=' + str(iv) if iv else ''}")
    modes = {m: [n for n, _, mm, *_ in ev if mm == m] for m in ("startup", "reset", "interval")}
    print(f"   count by mode: { {m: len(v) for m, v in modes.items()} }  total {len(ev)}")
    print("   overrides in the task variant, and what the play variant switches off")
    show(SF_VARIANTS, 41, 42); show(SF_VARIANTS, 56, 63); show(SF_VARIANTS, 76, 80)
    print("   when each mode runs")
    show(RL_ENV, 134, 135); show(RL_ENV, 235, 236); show(RL_ENV, 360)
    print("   interval mode with a zero range")
    show(EV_MGR, 207, 209); show(EV_MGR, 222, 229); show(EV_MGR, 405, 407)
    print("   the push function")
    show(PROJ_EVENTS, 37, 50); show(PROJ_EVENTS, 65)

    p = probe()
    d0 = dict(zip(p["body_names"], p["mass"]["default_per_body"])) if p else {}
    print("-- derived ranges")
    base = d0.get("base_Link", float("nan"))
    lo, hi = next(pp for n, *_, pp, _ in ev if n == "add_base_mass")["mass_distribution_params"]
    print(f"   add_base_mass in the base file: {lo, hi};  the SF variant overrides it to (-1.0, 2.0)")
    s_lo, s_hi = next(pp for n, *_, pp, _ in ev if n == "radomize_rigid_body_mass_inertia")["mass_inertia_distribution_params"]
    l_lo, l_hi = next(pp for n, *_, pp, _ in ev if n == "add_link_mass")["mass_distribution_params"]
    print(f"   base_Link: default {base} kg;  after add (-1, 2): [{base-1:.3f}, {base+2:.3f}];  after the x[{s_lo}, {s_hi}] of all bodies: [{(base-1)*s_lo:.3f}, {(base+2)*s_hi:.3f}] kg")
    print(f"   a leg link: default x [{l_lo}, {l_hi}] x [{s_lo}, {s_hi}] = x [{l_lo*s_lo:.2f}, {l_hi*s_hi:.2f}];  knee_L_Link {d0.get('knee_L_Link')} kg -> [{d0.get('knee_L_Link', 0)*l_lo*s_lo:.4f}, {d0.get('knee_L_Link', 0)*l_hi*s_hi:.4f}]")
    tot_lo = (base - 1) * s_lo + sum(v * l_lo * s_lo for k, v in d0.items() if k.endswith(("_L_Link", "_R_Link"))) + d0.get("limx_imu", 0) * s_lo
    tot_hi = (base + 2) * s_hi + sum(v * l_hi * s_hi for k, v in d0.items() if k.endswith(("_L_Link", "_R_Link"))) + d0.get("limx_imu", 0) * s_hi
    print(f"   whole robot: default {sum(d0.values()):.3f} kg;  possible range [{tot_lo:.3f}, {tot_hi:.3f}] kg")
    g_st = next(pp for n, *_, pp, _ in ev if n == "robot_joint_stiffness_and_damping")
    g_re = next(pp for n, *_, pp, _ in ev if n == "randomize_actuator_gains")
    show(SOLE_CFG, 48, 51); show(SOLE_CFG, 59, 62)
    a, b = g_re["stiffness_distribution_params"]
    print(f"   startup gains: operation {g_st['operation']}  stiffness {g_st['stiffness_distribution_params']}  damping {g_st['damping_distribution_params']}")
    print(f"   reset gains:   operation {g_re['operation']} ({g_re['distribution']})  x {a, b}:  stiffness 45 -> [{45*a}, {45*b}];  damping 1.5 -> [{1.5*a}, {1.5*b}];  0.8 -> [{0.8*a}, {0.8*b}]")
    print(f"   log-uniform on [0.5, 2]: median factor = sqrt(0.5 x 2) = {math.sqrt(a*b):.1f};  P(factor > 1) = {math.log(b/1)/math.log(b/a):.2f}")
    print("   both gain events start from the default buffers, which they never write")
    show(ISAAC_EVENTS, 510, 516); show(ISAAC_EVENTS, 323); show(ISAAC_EVENTS, 327, 331)
    show(ARTIC, 1239, 1240); show(ARTIC, 1379, 1380)

    push = next(pp for n, *_, pp, _ in ev if n == "push_robot")
    pr, F, tq = push["probability"], push["force_range"]["x"][1], push["torque_range"]["x"][1]
    dec, dt, step_dt, ep_s = timing()
    n_ep = int(ep_s / step_dt)
    m_def = sum(d0.values()) if d0 else float("nan")
    print(f"-- push_robot: probability {pr} per env per step -> one push every {1/pr:.0f} steps = {step_dt/pr:.0f} s on average")
    print(f"   expected pushes in a full {n_ep}-step episode = {n_ep*pr:.1f};  P(no push in a full episode) = {(1-pr)**n_ep:.4f};  P(no push in 24 steps) = {(1-pr)**24:.4f}")
    print(f"   with 2048 envs: {2048*pr:.3f} pushes per step, {2048*24*pr:.1f} per table;  with 16 envs x 2000 steps: {16*2000*pr:.0f} expected")
    print(f"   force up to {F:.0f} N on x and on y for one step ({step_dt} s): impulse {F*step_dt:.0f} N s per axis;  on the default {m_def:.3f} kg that is {F*step_dt/m_def:.3f} m/s per axis, {math.hypot(F, F)*step_dt/m_def:.3f} m/s diagonal;  torque up to {tq:.0f} N m")
    if p:
        print(f"-- measured on the real task ({p['date']}, {p['num_envs']} envs, {p['steps']} steps): scripts/{PROBE.name}")
        print(f"   bodies {len(p['body_names'])}: {p['body_names']}")
        print(f"   joints {p['joint_names']}")
        print(f"   bodies matching ankle_.* : {p['ankle_bodies']};  matching .*_[LR]_Link : {len(p['LR_link_bodies'])}")
        print(f"   events {p['events']};  curriculum terms {p['curriculum_terms']}")
        print(f"   push: {p['push']}")
        print(f"   gains after startup only {p['gains']['after_startup']}")
        print(f"   gains after first reset  {p['gains']['after_first_reset']}")
        print(f"   gains over the run       { {k: v for k, v in p['gains']['over_run'].items()} }")
        print(f"   mass {p['mass']}")
        print(f"   material {p['material']}")
        print(f"   history {p['history']}")
        print("   critic terms: name, width, first column, spread (max - min over every env and step)")
        for c in p["critic_terms"]:
            print(f"      {c['name']:28s} {c['width']:3d}  col {c['start']:3d}  spread {c['spread']:<12}  {'CONSTANT ' + str(c['first_values']) if c['spread'] == 0.0 else ''}")
        const = [c for c in p["critic_terms"] if c["spread"] == 0.0]
        cw = sum(c["width"] for c in const)
        print(f"   constant terms {[c['name'] for c in const]}: {' + '.join(str(c['width']) for c in const)} = {cw} of {p['critic_total']} ({100*cw/p['critic_total']:.1f}%);  varying {p['critic_total']-cw};  with the 3 commands the critic reads {p['critic_total']-cw+3} varying numbers of 230")
        w3 = {c["name"]: c["width"] for c in p["critic_terms"]}
        print(f"   the three model-dependent widths: robot_mass {w3['robot_mass']} + robot_inertia {w3['robot_inertia']} + robot_material_properties {w3['robot_material_properties']} = {w3['robot_mass']+w3['robot_inertia']+w3['robot_material_properties']};  inertia = {len(p['body_names'])} bodies x 9;  material = {p['material']['shape'][1]} shapes x 3")
    print("   where the critic's terms read from")
    for a_ in (43, 50, 58, 65, 72, 95):
        show(PROJ_OBS, a_)

    print("-- noise on the policy's observations (order: noise, clip, scale)")
    show(OBS_MGR, 313, 318)
    for name, ln, call in cfg_terms("ObservarionsCfg", "PolicyCfg"):
        nz, sc = _kw(call, "noise"), _kw(call, "scale")
        std = ast.literal_eval(_kw(nz, "std")) if nz is not None else None
        scale = ast.literal_eval(sc) if sc is not None else None
        crit = next((c for n, _, c in cfg_terms("ObservarionsCfg", "CriticCfg") if n == name), None)
        cs = ast.literal_eval(_kw(crit, "scale")) if crit is not None and _kw(crit, "scale") is not None else None
        print(f"   :{ln} {name:14s} noise std {std}  scale {scale}  -> std after scale {None if std is None else round(std*scale, 5)}   critic scale {cs}")

    print("-- curriculum")
    show(BASE, 461); show(BASE, 463, 472); show(SF_VARIANTS, 80)
    print(f"   functions in the project's curriculums.py: {[n.name for n in ast.parse(PROJ_CURR.read_text()).body if isinstance(n, ast.FunctionDef)]}")

    print("-- hyperparameter table: the SF block")
    sf, dflt = sf_ppo(), _ppo_defaults()
    for k, (v, ln) in sorted(sf.items(), key=lambda kv: kv[1][1]):
        dv = dflt.get(k)
        print(f"   cfg:{ln:<4d} {k:24s} = {v!r:18}  {'class default ' + repr(dv[0]) + ' at ppo.py:' + str(dv[1]) + ('  (overridden)' if dv[0] != v else '  (same)') if dv else ''}")
    print(f"   PPO.__init__ defaults that the SF block does not set: { {k: v for k, v in dflt.items() if k not in sf} }")
    show(AC_PY, 117, 118); show(PPO_PY, 238, 241)
    print("-- from the toy table to full size")
    T_, ep, mb, it = sf["num_steps_per_env"][0], sf["num_learning_epochs"][0], sf["num_mini_batches"][0], sf["max_iterations"][0]
    B = 2048
    a_n, c_n, e_n = mlp_count([42, 512, 256, 128, 8]), mlp_count([230, 512, 256, 128, 1]), mlp_count([360, 256, 128, 3])
    print(f"   cells 2 x 4 = 8  vs  {B} x {T_} = {B*T_:,};  mini-batch 4 vs {B*T_//mb:,};  gradient steps 2 x 2 = 4 vs {ep} x {mb} = {ep*mb}")
    print(f"   gamma*lambda 0.72 vs {sf['gamma'][0]*sf['lam'][0]:.4f};  1/(1-gl) {1/(1-0.72):.2f} vs {1/(1-sf['gamma'][0]*sf['lam'][0]):.2f};  1/(1-gamma) 10 vs {1/(1-sf['gamma'][0]):.0f}")
    print(f"   knobs: toy 3 (b, ell, b_V)  vs  actor {a_n:,} + 8 + critic {c_n:,} = {a_n+8+c_n:,} and encoder {e_n:,}; all {a_n+8+c_n+e_n:,}")
    trans = it * B * T_
    print(f"   whole run: {it:,} iterations, {trans:,} transitions, {it*ep*mb:,} gradient steps;  simulated time {trans*step_dt:,.0f} s = {trans*step_dt/3600:,.0f} robot-hours = {trans*step_dt/86400:.1f} robot-days")
    print(f"   one env: {it*T_:,} steps = {it*T_*step_dt:,.0f} s = {it*T_*step_dt/3600:.0f} h;  at least {it*T_/n_ep:.0f} full episodes;  checkpoints {it//sf['save_interval'][0]} + the first = {it//sf['save_interval'][0] + 1}")
    print(f"   table memory {2777*B*T_/1e6:.1f} MB (chapter 27)")
    for n_ in (1024, 4096, 8192):
        print(f"   {n_} envs: table {n_*T_:,} cells, mini-batch {n_*T_//mb:,}, transitions over the run {it*n_*T_:,}, table memory {2777*n_*T_/1e6:.1f} MB")

    runs = training_runs()
    if runs:
        name, n_env, _, sc = runs[1]
        wts = {n: w for n, _, _, w in reward_terms()}
        n_it = len(sc["Loss/learning_rate"])
        rows = sorted((statistics.mean(v for _, v in sc[t][-20:]), t.split("/")[-1]) for t in sc if "Episode_Reward" in t)
        print(f"-- per-term episode reward, run {name}: {n_env} envs, {n_it} iterations; mean of the last 20 iterations (unit: episode sum / {ep_s:.0f} s)")
        show(REW_MGR, 118, 119)
        for m, t in rows:
            print(f"   {t:36s} weight {wts[t]:>9}   {m:+.6f}  -> {m:+.4f}")
        print(f"   sum {sum(m for m, _ in rows):+.5f}   positive terms {sum(m for m, _ in rows if m > 0):+.5f}   negative terms {sum(m for m, _ in rows if m < 0):+.5f}")
        d = dict((t, m) for m, t in rows)
        print(f"   pen_joint_accel / pen_feet_distance = {d['pen_joint_accel']/d['pen_feet_distance']:.1f} while |weights| are 1e-6 and 100")
        print(f"   mean episode length, last 20 iterations: {statistics.mean(v for _, v in sc['Train/mean_episode_length'][-20:]):.2f} steps;  mean reward {statistics.mean(v for _, v in sc['Train/mean_reward'][-20:]):.3f}")
    print("-- exercises")
    print(f"   push with probability 0.01: every {1/0.01:.0f} steps = {step_dt/0.01:.0f} s; per full episode {n_ep*0.01:.0f}; per table at 2048 envs {2048*24*0.01:.1f}")
    print(f"   damping factor needed for 3.0 from 1.5: {3.0/1.5};  for 1.5 from 0.8: {1.5/0.8:.3f} (inside [0.5, 2]: {0.5 <= 1.5/0.8 <= 2})")
    print(f"   friction buckets 48, shapes {p['material']['shape'][1] if p else '?'} per robot")
    print(f"   2000 x 16 x 0.002 = {2000*16*0.002:.0f} expected pushes; measured {p['push']['pushes'] if p else '?'}; sd of a binomial = {math.sqrt(2000*16*0.002*0.998):.2f}")
    print(f"   max_iterations 2000 at 2048 envs: {2000*B*T_:,} transitions = {2000*B*T_*step_dt/3600:.0f} robot-hours;  ratio to 15000 = {2000/15000:.4f}")


def _yaml(path):
    """The small subset of YAML that params.yaml uses: nested maps by indentation, `- item` lists, numbers, quoted strings, # comments."""
    root, stack = {}, [(-1, None)]
    stack = [(-1, root)]
    for raw in pathlib.Path(path).read_text().splitlines():
        line = raw.split("#")[0].rstrip()
        if not line.strip():
            continue
        ind = len(line) - len(line.lstrip())
        while stack[-1][0] >= ind:
            stack.pop()
        cur = stack[-1][1]
        s = line.strip()
        if s.startswith("- "):
            cur.setdefault("_list", []).append(s[2:].strip().strip('"'))
            continue
        k, _, v = s.partition(":")
        v = v.strip()
        if not v:
            cur[k] = {}
            stack.append((ind, cur[k]))
        else:
            try:
                cur[k] = float(v) if any(c in v for c in ".e") else int(v)
            except ValueError:
                cur[k] = v.strip('"')
    return root


def torque_clamp(q_rel, dq, kp, kd, tau_max, scale):
    """SolefootController.py:234-243 — the action range inside which |kp (q_des - q) - kd dq| <= tau_max."""
    return ((q_rel + (kd * dq - tau_max) / kp) / scale, (q_rel + (kd * dq + tau_max) / kp) / scale)


def ch30():
    print("=== chapter 30 — from training to the real robot ===")
    print("-- act and act_inference; who calls which")
    show(AC_PY, 155, 168); show(RUN_PY, 376, 386); show(RUN_PY, 175); show(PLAY_PY, 101, 102); show(PLAY_PY, 131, 139)
    print("-- what a checkpoint holds, and what play.py exports")
    show(RUN_PY, 355, 374); show(PLAY_PY, 104, 122); show(MLP_CFG, 35, 62); show(TRAIN_PY, 136)
    show(SF_VARIANTS, 50, 63)
    print("-- toy table: act (a) and act_inference (mu) side by side; scale 0.25 of chapter 15")
    for (e, t), c in cells.items():
        print(f"   e{e}t{t}  mu {c['mu']:+.2f}  a {c['a']:+.2f}  a - mu = sigma*eps = {c['a']-c['mu']:+.2f}   target angle from a {0.25*c['a']:+.4f} rad, from mu {0.25*c['mu']:+.4f} rad, difference {0.25*(c['a']-c['mu']):+.4f} rad = {math.degrees(0.25*(c['a']-c['mu'])):+.2f} deg")
    sd = math.sqrt(sum((c["a"] - c["mu"]) ** 2 for c in cells.values()) / 8)
    print(f"   rms of a - mu over the 8 cells {sd:.4f};  at sigma 1.0 (start of training) one sigma moves the target by {0.25*1.0:.2f} rad = {math.degrees(0.25):.1f} deg; at sigma 0.5: {math.degrees(0.125):.1f} deg")

    shp = json.loads((DEPLOY / "onnx_shapes.json").read_text())
    print(f"-- the .onnx files LimX ships for SF_TRON1A (commit {shp['commit']}, read {shp['date']})")
    for kind in ("isaaclab", "isaacgym"):
        for net in ("policy", "encoder"):
            s = shp[kind][net]
            n = sum(a * b for a, b in s["weights"]) + sum(s["biases"])
            print(f"   {kind:9s} {net:8s} in {s['input']:3d} out {s['output']}  layers {s['weights']}  parameters {n:,}  x4 bytes = {4*n:,} of {s['bytes']:,} ({s['bytes']-4*n:,} bytes of structure)  ops {'-'.join(s['ops'])}")
    a_n, e_n, c_n = mlp_count([42, 512, 256, 128, 8]), mlp_count([360, 256, 128, 3]), mlp_count([230, 512, 256, 128, 1])
    print(f"   this book's counts: actor {a_n:,}  encoder {e_n:,}  -> on the robot {a_n + e_n:,};  left behind: critic {c_n:,} + logstd 8 = {c_n + 8:,};  all {a_n+e_n+c_n+8:,}  share on the robot {100*(a_n+e_n)/(a_n+e_n+c_n+8):.1f}%")
    print(f"   isaacgym policy input 44 = 3 + 36 + 5;  isaacgym encoder input 180 = 5 x 36;  isaaclab 42 = 3 + 36 + 3;  360 = 10 x 36")

    y = _yaml(DEPLOY / "params.yaml")["PointfootCfg"]
    ctl, size = y["control"], y["size"]
    print(f"-- params.yaml of SF_TRON1A (snapshot {DEPLOY.name})")
    print(f"   loop_frequency {y['loop_frequency']}  decimation {ctl['decimation']}  -> policy every {ctl['decimation']/y['loop_frequency']:.3f} s = {y['loop_frequency']/ctl['decimation']:.0f} Hz;  command sent every {1000/y['loop_frequency']:.0f} ms")
    print(f"   joint order {y['joint_names']['_list']}")
    print(f"   default angles {y['init_state']['default_joint_angle']}")
    print(f"   control {ctl}")
    print(f"   obs_scales {y['normalization']['obs_scales']}  clip {y['normalization']['clip_scales']}")
    print(f"   size {size}  gait {y['gait']}  stand {y['stand_mode']}  user_cmd_scales {y['user_cmd_scales']}")
    for a, b in ((1, 12), (17, 34), (38, 60)):
        show(DEPLOY / "params.yaml", a, b)
    print("-- main.py and the controller: the lines the chapter cites")
    for path, a, b in (("main.py", 9, 9), ("main.py", 16, 16), ("main.py", 25, 39), ("SolefootController.py", 24, 26), ("SolefootController.py", 35, 41),
                       ("SolefootController.py", 60, 68), ("SolefootController.py", 100, 104), ("SolefootController.py", 136, 136), ("SolefootController.py", 180, 191),
                       ("SolefootController.py", 195, 205), ("SolefootController.py", 213, 225), ("SolefootController.py", 231, 247), ("SolefootController.py", 252, 262),
                       ("SolefootController.py", 264, 272), ("SolefootController.py", 285, 285), ("SolefootController.py", 295, 299), ("SolefootController.py", 313, 323),
                       ("SolefootController.py", 333, 341), ("SolefootController.py", 344, 362), ("SolefootController.py", 379, 389), ("SolefootController.py", 401, 411),
                       ("SolefootController.py", 427, 431), ("SolefootController.py", 443, 446), ("SolefootController.py", 479, 481), ("SolefootController.py", 488, 500),
                       ("simulator.py", 55, 55), ("simulator.py", 70, 87), ("simulator.py", 108, 108), ("simulator.py", 126, 136)):
        show(DEPLOY / path, a, b)
    p = probe()
    lab = p["joint_names"] if p else ["abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint", "knee_L_Joint", "knee_R_Joint", "ankle_L_Joint", "ankle_R_Joint"]
    sdk = y["joint_names"]["_list"]
    idx = [0, 4, 1, 5, 2, 6, 3, 7]
    print(f"-- joint order: Isaac Lab {lab}")
    print(f"   SDK {sdk}")
    print(f"   swap_positions: lab[i] = sdk[idx[i]] with idx {idx}: {[sdk[i] for i in idx] == lab};  ankle joints in SDK order are indices {[i for i, n in enumerate(sdk) if n.startswith('ankle')]} ((i+1) % 4 == 0)")
    print("-- observation layout on the robot vs PolicyCfg")
    pol = obs_dims("ObservarionsCfg", "PolicyCfg")
    print(f"   PolicyCfg {[(n, w) for n, _, _, w in pol]} = {sum(w for *_, w in pol)}")
    print(f"   controller: gyro x {y['normalization']['obs_scales']['ang_vel']} (3), gravity (3), q - default x {y['normalization']['obs_scales']['dof_pos']} (8), dq x {y['normalization']['obs_scales']['dof_vel']} (8), last actions (8), clock (2), gait (4) = {3+3+8+8+8+2+4}")
    print(f"   policy input: encoder {size['encoder_output_size']} + obs {size['observations_size']} + commands 3 = {size['encoder_output_size']+size['observations_size']+3}")
    print(f"   history: buffer {shp['isaaclab']['encoder']['input']} numbers = {shp['isaaclab']['encoder']['input']//36} observations; first fill writes obs_history_length {size['obs_history_length']} x 36 = {size['obs_history_length']*36}; "
          f"after the first shift {size['obs_history_length']-1} copies + {shp['isaaclab']['encoder']['input']//36 - size['obs_history_length']} zero rows + 1 newest; real observations fill it after {shp['isaaclab']['encoder']['input']//36} policy steps = {shp['isaaclab']['encoder']['input']//36 * 0.02:.1f} s")
    if p:
        print(f"   training side (probe): {p['history']}")
    print("-- training config vs deploy file")
    dec, dt, step_dt, _ = timing()
    ev = {n: pp for n, _, _, _, pp, _ in _event_terms()}
    a, b = ev["randomize_actuator_gains"]["damping_distribution_params"]
    cmd = next(c for n, _, c in cfg_terms("CommandsCfg") if n == "base_velocity")
    rng = {k.arg: ast.literal_eval(k.value) if not isinstance(k.value, ast.Tuple) or all(isinstance(e, ast.Constant) or isinstance(e, ast.UnaryOp) and isinstance(e.operand, ast.Constant) for e in k.value.elts) else "(-pi, pi)" for k in _kw(cmd, "ranges").keywords}
    gait = next(c for n, _, c in cfg_terms("CommandsCfg") if n == "gait_command")
    grng = {k.arg: ast.literal_eval(k.value) for k in _kw(gait, "ranges").keywords}
    us = y["user_cmd_scales"]
    rows = [("policy rate", f"{1/step_dt:.0f} Hz (decimation {dec} x {dt} s)", f"{y['loop_frequency']/ctl['decimation']:.0f} Hz ({y['loop_frequency']} Hz / {ctl['decimation']})"),
            ("PD update", f"every physics step, {1/dt:.0f} Hz", f"command sent at {y['loop_frequency']} Hz"),
            ("stiffness", f"45 default, reset draws 45 x [{a}, {b}] = [{45*a}, {45*b}]", f"{ctl['stiffness']}"),
            ("damping, legs", f"1.5 default, [{1.5*a}, {1.5*b}]", f"{ctl['damping']}"),
            ("damping, ankles", f"0.8 default, [{0.8*a}, {0.8*b}]", f"{ctl['ankle_joint_damping']}"),
            ("torque limit", "80 on all 8 joints", f"{ctl['user_torque_limit']} legs, {ctl['ankle_joint_torque_limit']} ankles (as an action clamp)"),
            ("action scale", "0.25", f"{ctl['action_scale_pos']}"),
            ("default hip angle", "0.0", f"{y['init_state']['default_joint_angle']['hip_L_Joint']} / {y['init_state']['default_joint_angle']['hip_R_Joint']}"),
            ("command range", f"{rng}", f"joystick x 0.5 x scales -> vx +-{0.5*us['lin_vel_x']}, vy +-{0.5*us['lin_vel_y']}, wz +-{0.5*us['ang_vel_yaw']}"),
            ("gait", f"{grng}", f"frequency {y['gait']['frequencies']}, offset 0.5, duration 0.5, swing height {y['gait']['swing_height']}"),
            ("history length", "10", f"yaml {size['obs_history_length']}; buffer from the encoder file: {shp['isaaclab']['encoder']['input']//36}")]
    for r in rows:
        print(f"   {r[0]:18s} | {r[1]:62s} | {r[2]}")
    print(f"   deploy damping inside the training range: legs {1.5*a <= ctl['damping'] <= 1.5*b} ({ctl['damping']} vs upper edge {1.5*b});  ankles {0.8*a <= ctl['ankle_joint_damping'] <= 0.8*b} ({ctl['ankle_joint_damping']} vs upper edge {0.8*b})")
    print("-- the torque clamp as an action range (SolefootController.py:234-243)")
    kp, kd, kda, tl, ta, sc = ctl["stiffness"], ctl["damping"], ctl["ankle_joint_damping"], ctl["user_torque_limit"], ctl["ankle_joint_torque_limit"], ctl["action_scale_pos"]
    for label, args_ in (("leg, q-q0 = 0.1, dq = 2", (0.1, 2.0, kp, kd, tl, sc)), ("ankle, q-q0 = 0.1, dq = 2", (0.1, 2.0, kp, kda, ta, sc)),
                         ("leg at rest, q-q0 = 0", (0.0, 0.0, kp, kd, tl, sc)), ("ankle at rest, q-q0 = 0", (0.0, 0.0, kp, kda, ta, sc)),
                         ("leg, q-q0 = -0.2, dq = -3", (-0.2, -3.0, kp, kd, tl, sc)), ("ankle, q-q0 = -0.2, dq = -3", (-0.2, -3.0, kp, kda, ta, sc))):
        lo, hi = torque_clamp(*args_)
        print(f"   {label:28s} action in [{lo:+.4f}, {hi:+.4f}]  width {hi-lo:.4f} = 2 tau_max / (kp scale) = {2*args_[4]/(args_[2]*args_[5]):.4f}")
    print(f"   at rest: leg target may differ from q by tau/kp = {tl/kp:.4f} rad ({math.degrees(tl/kp):.1f} deg); ankle {ta/kp:.4f} rad ({math.degrees(ta/kp):.1f} deg)")
    print(f"   check: leg, a at the upper edge -> torque {kp*(sc*torque_clamp(0.1, 2.0, kp, kd, tl, sc)[1] - 0.1) - kd*2.0:.4f}")
    print("-- PD: the simulator's line against equation (18)")
    q_t, q, dq = 0.25 * 0.40, 0.0, 0.0
    print(f"   e0 t0 with mu (act_inference): target 0.25 x 0.20 = {0.25*0.20:.3f} rad; at q = 0, dq = 0: tau = 45 x {0.25*0.20:.3f} = {45*0.25*0.20:.2f} N m (chapter 15 used a = 0.40: {45*q_t:.2f} N m)")
    print(f"   with dq = 1 rad/s: training default kd 1.5 -> {45*0.05 - 1.5*1:.2f};  deploy kd 3.0 -> {45*0.05 - 3.0*1:.2f}")
    print("-- exercises")
    print(f"   loop 1000 Hz and 50 Hz policy -> decimation {1000//50};  decimation 10 at 500 Hz: commands per policy step {ctl['decimation']}, per second {y['loop_frequency']}")
    print(f"   joystick axis 0.6 on x: command {0.6*0.5*us['lin_vel_x']:.2f} m/s; on y: {0.6*0.5*us['lin_vel_y']:.2f} m/s")
    g, resets = 0.0, []
    for tick in range(1, 200):                      # SolefootController.py:296-298: add, then reset to 0.0 (not wrap) above 1.0
        g += 0.02 * y["gait"]["frequencies"]
        if g > 1.0:
            g = 0.0
            resets.append(tick)
    per = resets[1] - resets[0]
    print(f"   gait clock: +0.02 x {y['gait']['frequencies']} = {0.02*y['gait']['frequencies']:.3f} per policy step; nominal cycle {1/(0.02*y['gait']['frequencies']):.2f} steps = {1/y['gait']['frequencies']:.3f} s;"
          f" with the reset at :297-298 the clock restarts at ticks {resets[:3]}: {per} steps = {per*0.02:.2f} s = {1/(per*0.02):.2f} Hz  (38 x 0.026 = {38*0.026:.3f}, 39 x 0.026 = {39*0.026:.3f})")
    print(f"   stand mode: {y['stand_mode']['stand_duration']} s x {y['loop_frequency']} Hz = {y['stand_mode']['stand_duration']*y['loop_frequency']:.0f} loop iterations")
    print(f"   sdk order of lab vector [1..8]: {[ [1,2,3,4,5,6,7,8][idx.index(i)] for i in range(8)]}")



CHAPTERS = {"01": ch01, "02": ch02, "03": ch03, "04": ch04, "05": ch05, "06": ch06, "07": ch07, "08": ch08, "09": ch09,
            "10": ch10, "11": ch11, "12": ch12, "13": ch13, "14": ch14, "15": ch15, "16": ch16, "17": ch17, "18": ch18,
            "19": ch19, "20": ch20, "21": ch21, "22": ch22, "23": ch23, "24": ch24, "25": ch25, "26": ch26, "27": ch27, "28": ch28, "29": ch29, "30": ch30}

if __name__ == "__main__":
    for key in sys.argv[1:] or CHAPTERS:
        CHAPTERS[key.zfill(2)]()
