"""HW1 Problem 1: single-precision numerical differentiation.

Forward, central and extrapolated (Richardson) difference formulas applied to
cos(x) and exp(x) at x = 0.1 and x = 10.  Every operation inside a difference
formula is carried out in float32, so roundoff enters at the single-precision
machine epsilon (about 1.2e-7) rather than at the double-precision one.
"""
import numpy as np
import matplotlib.pyplot as plt

from plotstyle import use_science

use_science()

f32 = np.float32
EPS32 = float(np.finfo(np.float32).eps)


def forward(f, x, h):
    """Forward difference, truncation error O(h)."""
    x, h = f32(x), f32(h)
    return (f(x + h) - f(x)) / h


def central(f, x, h):
    """Central difference, truncation error O(h^2)."""
    x, h = f32(x), f32(h)
    return (f(x + h) - f(x - h)) / (f32(2.0) * h)


def extrapolated(f, x, h):
    """Richardson extrapolation of the central difference, error O(h^4)."""
    d_h = central(f, x, h)
    d_h2 = central(f, x, f32(h) / f32(2.0))
    return (f32(4.0) * d_h2 - d_h) / f32(3.0)


cos32 = lambda x: np.cos(f32(x), dtype=np.float32)
exp32 = lambda x: np.exp(f32(x), dtype=np.float32)

FUNCS = [("cos", r"\cos", cos32, lambda x: -np.sin(np.float64(x))),
         ("exp", r"\exp", exp32, lambda x: np.exp(np.float64(x)))]
POINTS = [0.1, 10.0]
METHODS = [("forward", forward, 1), ("central", central, 2),
           ("extrapolated", extrapolated, 4)]
COLORS = {"forward": "C0", "central": "C2", "extrapolated": "C1"}

# Powers of two are exactly representable, so h itself carries no rounding.
h = 2.0 ** np.arange(-27.0, 1.0, 0.25)


def binned_median(x, y, nbins=28):
    """Median of y in equal log-x bins: the trend under the roundoff noise."""
    edges = np.logspace(np.log10(x.min()), np.log10(x.max()) * 1.0001, nbins + 1)
    idx = np.digitize(x, edges) - 1
    xb, yb = [], []
    for b in range(nbins):
        m = idx == b
        if m.sum():
            xb.append(np.exp(np.mean(np.log(x[m]))))
            yb.append(np.nanmedian(y[m]))
    return np.array(xb), np.array(yb)


fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True, sharey=True)
summary = []
for i, (fname, ftex, f, dfexact) in enumerate(FUNCS):
    for j, x in enumerate(POINTS):
        ax = axes[i][j]
        exact = dfexact(x)
        for mname, method, order in METHODS:
            approx = np.array([np.float64(method(f, x, hh)) for hh in h])
            eps = np.abs((approx - exact) / exact)
            eps[eps == 0] = np.nan
            ax.loglog(h, eps, color=COLORS[mname], lw=0.6, alpha=0.35)
            xb, yb = binned_median(h, eps)
            ax.loglog(xb, yb, color=COLORS[mname], lw=1.6, label=mname)
            k = np.nanargmin(eps)
            summary.append((fname, x, mname, order, h[k], eps[k]))
        ax.axhline(EPS32, color="0.4", ls=":", lw=1)
        # guides for the pure-truncation scalings h^1, h^2, h^4
        hg = np.array([3e-3, 3e-1])
        for order, anchor in [(1, 3e-2), (2, 3e-3), (4, 3e-5)]:
            ax.loglog(hg, anchor * hg ** order, color="0.55", ls="--", lw=0.6)
        if i == 0 and j == 0:
            bb = dict(fc="white", ec="none", pad=0.8)
            ax.text(0.36, 1.6e-2, "$h$", fontsize=6, color="0.45", bbox=bb)
            ax.text(0.36, 4.9e-4, "$h^2$", fontsize=6, color="0.45", bbox=bb)
            ax.text(0.36, 4.4e-7, "$h^4$", fontsize=6, color="0.45", bbox=bb)
        ax.set_title(rf"$\dfrac{{\mathrm{{d}}}}{{\mathrm{{d}}x}}{ftex}(x)$ at $x = {x:g}$")
        ax.set_ylim(1e-8, 1e1)
        if i == 1:
            ax.set_xlabel("step size $h$")
        if j == 0:
            ax.set_ylabel(r"relative error $\varepsilon$")
axes[0][0].legend(loc="lower left", fontsize=7)
axes[0][1].text(0.04, 0.16, r"single-precision $\epsilon_m$", fontsize=6, color="0.4",
                transform=axes[0][1].transAxes,
                bbox=dict(fc="white", ec="none", pad=0.8))
fig.suptitle(r"Single-precision differentiation: relative error $\varepsilon$ vs step size $h$")
fig.savefig("figs/problem1_error.png")

print(f"{'f':>4} {'x':>6} {'method':>13} {'ord':>4} {'h_best':>10} {'eps_min':>10} {'digits':>7}")
for fname, x, mname, order, hb, em in summary:
    print(f"{fname:>4} {x:>6} {mname:>13} {order:>4} {hb:10.2e} {em:10.2e} {-np.log10(em):7.1f}")

print("\nsimple estimates (relative error, machine epsilon eps_m = %.2e):" % EPS32)
for order, name in [(1, "forward"), (2, "central"), (4, "extrapolated")]:
    p = order + 1
    print(f"  {name:>13}: h_opt ~ eps_m^(1/{p}) = {EPS32 ** (1.0 / p):.2e}, "
          f"eps_min ~ eps_m^({order}/{p}) = {EPS32 ** (order / p):.2e}, "
          f"~{-np.log10(EPS32 ** (order / p)):.1f} significant digits")
