#!/usr/bin/env python3
"""Answer key for the exercises in notes/workbook/ (the textbook-style rewrite of notes 00-07).

Every number comes from the SAME running example as scripts/toy_answer_key.py: this file
imports that module (its data `env`, the derived `cells`, and the constants) and only adds
the "what if" variants the exercises ask about. It defines no new example data.

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


CHAPTERS = {"22": ch22}

if __name__ == "__main__":
    for key in sys.argv[1:] or CHAPTERS:
        CHAPTERS[key]()
