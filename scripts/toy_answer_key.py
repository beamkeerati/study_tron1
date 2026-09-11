"""Answer key for the running example used by notes 00 (glossary §8), 05, 06, 07.

Single source of truth: 2 env x 4 steps x 1-D action, sigma 0.5 fixed, gamma 0.9,
lambda 0.8, clip 0.2, mini-batches mb0 = {e0t0, e0t2, e1t1, e1t3}, mb1 = the rest.
Run from the repo root:  python3 scripts/toy_answer_key.py
"""
import math

SIGMA, GAMMA, LAM, EPS_CLIP = 0.5, 0.9, 0.8, 0.2
GL = GAMMA * LAM  # 0.72
env = {
    0: dict(mu=[0.20, 0.30, 0.10, 0.40], eps=[+0.40, -0.20, +0.80, 0.00],
            r=[1.0, 0.5, 1.0, 0.5], V=[5.0, 4.5, 4.0, 3.5], done=[0, 0, 0, 0], V_last=3.0),
    1: dict(mu=[-0.10, 0.00, 0.20, -0.30], eps=[-0.60, +0.80, +0.20, +0.40],
            r=[0.5, 1.0, -2.0, 0.0], V=[3.0, 2.0, 1.0, 4.0], done=[0, 0, 1, 0], V_last=3.5),
}
C = -0.5 * math.log(2 * math.pi) - math.log(SIGMA)  # constant part of log pi = -0.2258


def logpi(a, mu):
    return C - (a - mu) ** 2 / (2 * SIGMA ** 2)


cells = {}
for e, d in env.items():
    for t in range(4):
        a = d["mu"][t] + SIGMA * d["eps"][t]
        cells[(e, t)] = dict(mu=d["mu"][t], eps=d["eps"][t], a=a, r=d["r"][t], V=d["V"][t],
                             done=d["done"][t], logpi_old=logpi(a, d["mu"][t]))

# --- GAE backward (rollout_storage.py:187-201) ---
for e, d in env.items():
    A_next = 0.0
    for t in (3, 2, 1, 0):
        c = cells[(e, t)]
        mask = 1.0 - c["done"]
        V_next = d["V_last"] if t == 3 else d["V"][t + 1]
        c["delta"] = c["r"] + mask * GAMMA * V_next - c["V"]
        c["A"] = c["delta"] + mask * GL * A_next
        A_next = c["A"]
        c["R"] = c["A"] + c["V"]

# --- normalise (rollout_storage.py:204-207), torch.std is n-1 ---
As = [c["A"] for c in cells.values()]
meanA = sum(As) / len(As)
stdA = math.sqrt(sum((x - meanA) ** 2 for x in As) / (len(As) - 1))
for c in cells.values():
    c["Ahat"] = (c["A"] - meanA) / (stdA + 1e-8)

# --- one gradient step with mu_new = mu + 0.05 (07 §5.2) ---
def step(c, shift):
    mu_new = c["mu"] + shift
    lp_new = logpi(c["a"], mu_new)
    rho = math.exp(lp_new - c["logpi_old"])
    clipped = min(max(rho, 1 - EPS_CLIP), 1 + EPS_CLIP)
    s1, s2 = -rho * c["Ahat"], -clipped * c["Ahat"]
    L = max(s1, s2)
    picks_clipped = s2 > s1
    dLdmu = 0.0 if picks_clipped else -c["Ahat"] * rho * (c["a"] - mu_new) / SIGMA ** 2
    return dict(mu_new=mu_new, logpi_new=lp_new, rho=rho, clip=clipped, L=L,
                clipped=picks_clipped, dLdmu=dLdmu)


for c in cells.values():
    c.update({f"s05_{k}": v for k, v in step(c, +0.05).items()})

mb0 = [(0, 0), (0, 2), (1, 1), (1, 3)]
mb1 = [(0, 1), (0, 3), (1, 0), (1, 2)]

print(f"sigma={SIGMA} gamma={GAMMA} lam={LAM} gl={GL:.2f}  const C={C:.4f}")
print(f"mean(A)={meanA:.4f}  std_n-1(A)={stdA:.4f}\n")
hdr = "cell   mu     eps    a      logpi_old  delta    A        R        Ahat     | +0.05: logpi_new  rho     L_i      dL/dmu"
print(hdr)
for (e, t), c in cells.items():
    print(f"e{e}t{t}  {c['mu']:+.2f}  {c['eps']:+.2f}  {c['a']:+.2f}  {c['logpi_old']:+.4f}   "
          f"{c['delta']:+.4f}  {c['A']:+.4f}  {c['R']:+.4f}  {c['Ahat']:+.4f}  | "
          f"{c['s05_logpi_new']:+.4f}   {c['s05_rho']:.4f}  {c['s05_L']:+.4f}  {c['s05_dLdmu']:+.4f}")
print(f"\nmb0 L^CLIP mean = {sum(cells[k]['s05_L'] for k in mb0)/4:+.4f}")
print(f"mb1 L^CLIP mean = {sum(cells[k]['s05_L'] for k in mb1)/4:+.4f}")
print(f"all 8 mean      = {sum(c['s05_L'] for c in cells.values())/8:+.4f}")

print("\n--- 07 §5.3 rule: mu_new = mu_old + sign(Ahat)*0.15, twin pair e0t2 / e1t1, both directions ---")
for key in [(0, 2), (1, 1)]:
    c = cells[key]
    for shift in (+0.15, -0.15):
        s = step(c, shift)
        print(f"e{key[0]}t{key[1]} Ahat={c['Ahat']:+.4f} shift={shift:+.2f} -> rho={s['rho']:.4f} "
              f"{'CLIPPED grad=0' if s['clipped'] else 'raw   grad flows'}  L={s['L']:+.4f}  dL/dmu={s['dLdmu']:+.4f}")

print("\n--- KL (equal sigma): (dmu)^2/(2 sigma^2) ---")
for dm in (0.05, 0.15):
    kl = dm ** 2 / (2 * SIGMA ** 2)
    print(f"dmu={dm:.2f}: KL={kl:.4f}  -> {'> 0.02: lr/1.5' if kl > 0.02 else ('< 0.005: lr*1.5' if kl < 0.005 else 'in band')}")

print("\n--- per-dim constants ---")
H = 0.5 + 0.5 * math.log(2 * math.pi) + math.log(SIGMA)
print(f"H(sigma=0.5) = {H:.4f}   x8 = {8*H:.4f}")
print(f"pdf(a=0.40|mu=0.20,s=0.5) = {math.exp(logpi(0.40, 0.20)):.4f}   peak = {1/(SIGMA*math.sqrt(2*math.pi)):.4f}")
print(f"ell = ln(0.5) = {math.log(0.5):.4f}")

# --- 8-D widening (glossary §8.6, note 06 §3.5.5): joint 1 = e0t0, joints 2-8 free ---
print("\n--- 8-D widening: joint 1 = e0t0, mu_new = mu + 0.05 on every joint ---")
mu8 = [0.20, -0.30, 0.10, 0.40, -0.10, 0.60, -0.50, 0.30]
eps8 = [0.40, -0.20, 0.60, 0.00, -0.60, 1.20, -0.80, 0.20]
a8 = [m + SIGMA * e for m, e in zip(mu8, eps8)]
lp8 = [logpi(a, m) for a, m in zip(a8, mu8)]
S_old = sum(lp8)
mu8n = [m + 0.05 for m in mu8]
S_new = sum(logpi(a, m) for a, m in zip(a8, mu8n))
rho8 = math.exp(S_new - S_old)
Ahat8 = cells[(0, 0)]["Ahat"]
clip8 = min(max(rho8, 1 - EPS_CLIP), 1 + EPS_CLIP)
L8 = max(-Ahat8 * rho8, -Ahat8 * clip8)
dL8 = [-Ahat8 * rho8 * (a - m) / SIGMA ** 2 for a, m in zip(a8, mu8n)]
KL8 = 8 * 0.05 ** 2 / (2 * SIGMA ** 2)
print("a      =", [round(x, 2) for x in a8])
print("logN_i =", [round(x, 4) for x in lp8])
print(f"sum log pi_old = {S_old:.4f}   pi = {math.exp(S_old):.5f}   sum log pi_new = {S_new:.4f}   rho = {rho8:.4f}")
print(f"L^CLIP (Ahat {Ahat8:+.4f}) = {L8:+.4f}   dL/dmu_i = {[round(x, 3) for x in dL8]}")
print(f"H x8 = {8*H:.4f}   KL (8 dims) = {KL8:.3f}")
