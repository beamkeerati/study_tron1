#!/usr/bin/env python3
"""Cross-check of scripts/workbook_answer_key.py (workbook chapters 25-27) against PyTorch itself.

The answer key computes Adam, clip_grad_norm_, the KL of (30) and the lr rule of (37) in plain Python so that it runs
anywhere. This script repeats the same four-gradient-step walks with the real torch.optim.Adam, the real
nn.utils.clip_grad_norm_ and real autograd, with the body of the loop typed in line for line from ppo.py:221-290
(self.* replaced by locals), and asserts that every KL, lr, loss, .grad and knob value agrees.

System python3 has no torch. Run from the repo root with the conda env `tron1` (torch 2.5.1):

    env -u PYTHONPATH ~/anaconda3/envs/tron1/bin/python -s scripts/workbook_torch_check.py
"""
import contextlib, io, math, sys
import torch, torch.nn as nn, torch.optim as optim
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
with contextlib.redirect_stdout(io.StringIO()):
    import workbook_answer_key as wk
torch.set_default_dtype(torch.float64)
print("torch", torch.__version__)
cells, SET1, SET2 = wk.cells, wk.SET1, wk.SET2

def run(order, epochs=2, lr0=1e-3, schedule="adaptive", use_clip=True):
    b = nn.Parameter(torch.zeros(1)); logstd = nn.Parameter(torch.full((1,), math.log(0.5))); bV = nn.Parameter(torch.zeros(1))
    params = [b, logstd, bV]
    optimizer = optim.Adam([{"params": params}], lr=lr0)
    learning_rate, desired_kl, clip_param, value_loss_coef, entropy_coef, max_grad_norm = lr0, 0.01, 0.2, 1.0, 0.01, 1.0
    out = []
    for ep in range(epochs):
        for keys in order:
            T = lambda f: torch.tensor([f(cells[k]) for k in keys])
            old_mu_batch = T(lambda c: c["mu"]).unsqueeze(1); old_sigma_batch = torch.full((len(keys), 1), 0.5)
            actions_batch = T(lambda c: c["a"]).unsqueeze(1)
            old_actions_log_prob_batch = T(lambda c: c["logpi_old"]).unsqueeze(1)
            advantages_batch = T(lambda c: c["Ahat"]).unsqueeze(1)
            target_values_batch = T(lambda c: c["V"]).unsqueeze(1); returns_batch = T(lambda c: c["R"]).unsqueeze(1)
            # the three knobs stand in for the networks
            mean = old_mu_batch + b
            std = mean * 0.0 + torch.exp(logstd)                        # actor_critic.py:157
            dist = torch.distributions.Normal(mean, std)
            actions_log_prob_batch = dist.log_prob(actions_batch).sum(dim=-1)
            value_batch = target_values_batch + bV
            mu_batch, sigma_batch = mean, std
            entropy_batch = dist.entropy().sum(dim=-1)
            # ---- ppo.py:221-244
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)
            if desired_kl != None and schedule == "adaptive":
                with torch.inference_mode():
                    if kl_mean > desired_kl * 2.0:
                        learning_rate = max(1e-5, learning_rate / 1.5)
                    elif kl_mean < desired_kl / 2.0 and kl_mean > 0.0:
                        learning_rate = min(1e-2, learning_rate * 1.5)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = learning_rate
            # ---- ppo.py:252-278
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-clip_param, clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
            entropy_batch_mean = entropy_batch.mean()
            loss = surrogate_loss + value_loss_coef * value_loss - entropy_coef * entropy_batch_mean
            # ---- ppo.py:287-290
            optimizer.zero_grad()
            loss.backward()
            raw = [p.grad.item() for p in params]
            if use_clip:
                nn.utils.clip_grad_norm_(params, max_grad_norm)
            before = [p.item() for p in params]
            optimizer.step()
            out.append(dict(kl=kl_mean.item(), lr=learning_rate, L=loss.item(), raw=raw, g=[p.grad.item() for p in params], x=before, x_new=[p.item() for p in params]))
    return out

worst = 0.0
for label, order, kw_t, kw_p in (("adaptive, sets 1,2", (SET1, SET2), {}, dict(rule=True)),
                                 ("adaptive, sets 2,1", (SET2, SET1), {}, dict(rule=True)),
                                 ("fixed lr, sets 1,2", (SET1, SET2), dict(schedule="fixed"), dict(rule=False)),
                                 ("fixed lr, no clip_grad_norm_", (SET1, SET2), dict(schedule="fixed", use_clip=False), dict(rule=False, clipnorm=False)),
                                 ("adaptive, whole table once", (list(cells),), dict(epochs=1), dict(rule=True, epochs=1))):
    t = run(order, **kw_t)
    p, _ = wk.toy_update(order=order, **kw_p)
    d = 0.0
    for a, e in zip(t, p):
        d = max(d, abs(a["kl"] - e["kl"]), abs(a["lr"] - e["lr"]), abs(a["L"] - e["r"]["L"]),
                *(abs(u - v) for u, v in zip(a["raw"], e["raw"])), *(abs(u - v) for u, v in zip(a["g"], e["g"])), *(abs(u - v) for u, v in zip(a["x_new"], e["x_new"])))
    worst = max(worst, d)
    print(f"{label}: {len(t)} gradient steps, largest |torch - answer key| over KL, lr, L, .grad (raw and clipped), knobs = {d:.2e}")
    print("   torch: " + " | ".join(f"L {a['L']:+.4f} grad " + " ".join(f"{g:+.4f}" for g in a["raw"]) + f" lr {a['lr']:.6g}" for a in t))
    print("   knobs after: " + " ".join(f"{v:+.6f}" for v in t[-1]["x_new"]) + f"  sigma {math.exp(t[-1]['x_new'][1]):.6f}")
assert worst < 1e-9, worst
# pure Adam cases
for label, seq in (("constant", lambda k: 1.0), ("alternating", lambda k: 1.0 if k % 2 == 0 else -1.0)):
    x = nn.Parameter(torch.zeros(1)); o = optim.Adam([x], lr=1.0); ds = []
    for k in range(6):
        o.zero_grad(); (seq(k) * x).sum().backward(); prev = x.item(); o.step(); ds.append(x.item() - prev)
    print(label, "steps/alpha:", " ".join(f"{d:+.4f}" for d in ds))
# which implementation runs: foreach default on cuda / cpu
from torch.optim.optimizer import _default_to_fused_or_foreach
print("cpu params -> foreach:", _default_to_fused_or_foreach([torch.zeros(1)], False, use_fused=False))
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda params -> foreach:", _default_to_fused_or_foreach([torch.zeros(1, device='cuda')], False, use_fused=False))
print("ALL MATCH, worst", f"{worst:.2e}")
# backward() ADDS into .grad; zero_grad() empties the box (set_to_none=True)
x = nn.Parameter(torch.ones(1)); o = optim.Adam([x], lr=1e-3)
(3.0 * x).sum().backward(); g1 = x.grad.item(); (3.0 * x).sum().backward(); g2 = x.grad.item(); o.zero_grad()
print("backward twice without zero_grad:", g1, "->", g2, "; after zero_grad(): .grad is", x.grad)
