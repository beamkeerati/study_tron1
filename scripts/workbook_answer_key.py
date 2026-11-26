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


CHAPTERS = {"01": ch01, "02": ch02, "03": ch03, "04": ch04, "05": ch05, "06": ch06, "07": ch07, "08": ch08, "09": ch09,
            "10": ch10, "11": ch11, "12": ch12, "13": ch13, "14": ch14, "15": ch15, "16": ch16, "17": ch17, "18": ch18,
            "19": ch19, "20": ch20, "21": ch21, "22": ch22, "23": ch23, "24": ch24}

if __name__ == "__main__":
    for key in sys.argv[1:] or CHAPTERS:
        CHAPTERS[key.zfill(2)]()
