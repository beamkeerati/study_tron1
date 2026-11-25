#!/usr/bin/env python3
"""Answer key for the exercises in notes/workbook/ (the textbook-style rewrite of notes 00-07).

Every number comes from the SAME running example as scripts/toy_answer_key.py: this file
imports that module (its data `env`, the derived `cells`, and the constants) and only adds
the "what if" variants the exercises ask about. It defines no new example data.

Chapters 1-2 have no toy numbers: ch01()/ch02() read their facts from the real code under tron1-rl-isaaclab/
and IsaacLab/, plus one dated snapshot of the GitHub repo list (limxdynamics_repos_2026-11-25.json).

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
                "tasks/locomotion/mdp/rewards.py", "tasks/locomotion/mdp/observations.py", "tasks/locomotion/mdp/events.py",
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


CHAPTERS = {"01": ch01, "02": ch02, "22": ch22}

if __name__ == "__main__":
    for key in sys.argv[1:] or CHAPTERS:
        CHAPTERS[key.zfill(2)]()
