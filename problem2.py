"""HW1 Problem 2: single-precision quadrature of I = int_0^1 exp(-t) dt.

Midpoint, trapezoid and Simpson rules, all accumulated in float32 with a naive
running sum so that the roundoff floor is visible at large N.
"""
import numpy as np
import matplotlib.pyplot as plt

from plotstyle import use_science

use_science()

f32 = np.float32
EPS32 = float(np.finfo(np.float32).eps)
EXACT = 1.0 - np.exp(-1.0)          # 0.6321205588285577

fexp = lambda t: np.exp(-f32(t), dtype=np.float32)


def _sum32(values):
    """Naive left-to-right accumulation in float32 (no pairwise tricks)."""
    total = f32(0.0)
    for v in values:
        total += f32(v)
    return total


def midpoint(f, a, b, n):
    h = f32((b - a) / n)
    t = f32(a) + (np.arange(n, dtype=np.float32) + f32(0.5)) * h
    return h * _sum32(f(t))


def trapezoid(f, a, b, n):
    h = f32((b - a) / n)
    t = f32(a) + np.arange(n + 1, dtype=np.float32) * h
    w = np.full(n + 1, 2.0, dtype=np.float32)
    w[0] = w[-1] = f32(1.0)
    return h / f32(2.0) * _sum32(w * f(t))


def simpson(f, a, b, n):
    """Composite Simpson; n must be even."""
    if n % 2:
        raise ValueError("Simpson's rule needs an even number of bins")
    h = f32((b - a) / n)
    t = f32(a) + np.arange(n + 1, dtype=np.float32) * h
    w = np.full(n + 1, 2.0, dtype=np.float32)
    w[1::2] = f32(4.0)
    w[0] = w[-1] = f32(1.0)
    return h / f32(3.0) * _sum32(w * f(t))


METHODS = [("midpoint", midpoint, 2), ("trapezoid", trapezoid, 2),
           ("Simpson", simpson, 4)]
COLORS = {"midpoint": "C0", "trapezoid": "C2", "Simpson": "C1"}

N = 2 ** np.arange(1, 23)           # 2 ... 4194304 bins

fig, ax = plt.subplots(figsize=(6, 4.2))
best = []
for name, rule, order in METHODS:
    eps = np.array([abs(np.float64(rule(fexp, 0.0, 1.0, int(n))) - EXACT) / EXACT
                    for n in N])
    eps[eps == 0] = np.nan
    ax.loglog(N, eps, "o-", ms=3, lw=1.2, color=COLORS[name], label=f"{name} (O(h$^{order}$))")
    k = np.nanargmin(eps)
    best.append((name, order, N[k], eps[k]))

n_ref = np.array([2.0, 3e2])
n_ref2 = np.array([2.0, 4e1])
ax.loglog(n_ref, 3e-2 * (n_ref / 2) ** -2.0, "k--", lw=0.8)
ax.loglog(n_ref2, 3e-4 * (n_ref2 / 2) ** -4.0, "k:", lw=0.8)
ax.text(60, 4e-5, "$N^{-2}$", fontsize=7)
ax.text(46, 5e-7, "$N^{-4}$", fontsize=7)
ax.set_ylim(2e-9, 3e0)
ax.loglog(N, EPS32 * np.sqrt(N.astype(float)), color="0.5", ls="-.", lw=0.8,
          label=r"$\epsilon_m\sqrt{N}$ random-walk roundoff")
ax.axhline(EPS32, color="0.4", ls=":", lw=1)
ax.set_xlabel("number of bins $N$")
ax.set_ylabel(r"relative error $\varepsilon$")
ax.set_title(r"Single-precision quadrature of $\int_0^1 e^{-t}\,dt$")
ax.legend(fontsize=7, loc="upper center", framealpha=0.9)
fig.savefig("figs/problem2_error.png")

print(f"exact I = {EXACT:.10f}")
print(f"{'method':>10} {'order':>5} {'N_best':>9} {'eps_min':>10} {'digits':>7}")
for name, order, n, e in best:
    print(f"{name:>10} {order:>5} {n:>9} {e:10.2e} {-np.log10(e):7.1f}")
print(f"\nsingle-precision eps_m = {EPS32:.2e}")
