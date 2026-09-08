"""HW1 Problem 3: the BAO peak in the matter correlation function.

xi(r) = 1/(2 pi^2) int dk k^2 P(k) sin(kr)/(kr)

P(k) is read from the tabulated lcdm_z0.matter_pk file (columns: k, P(k), ...),
interpolated with a cubic spline in log-log space, and the integral is done with
composite Simpson on a k grid fine enough to resolve the sin(kr) oscillations.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks

from plotstyle import use_science

use_science()

DATA = "lcdm_z0.matter_pk"
K_SPLIT = 1e-2          # below this the integrand is smooth: log-spaced grid
DK_FINE = 2.0e-4        # linear spacing above K_SPLIT (h/Mpc)

kt, pt = np.loadtxt(DATA, usecols=(0, 1), unpack=True)
logP = CubicSpline(np.log(kt), np.log(pt))


def P(k):
    """Spline-interpolated power spectrum; zero outside the tabulated range."""
    k = np.asarray(k, dtype=float)
    out = np.zeros_like(k)
    m = (k >= kt[0]) & (k <= kt[-1])
    out[m] = np.exp(logP(np.log(k[m])))
    return out


def _grids(kmax, dk=DK_FINE):
    k_lo = np.logspace(np.log10(kt[0]), np.log10(K_SPLIT), 601)
    n = int(np.ceil((kmax - K_SPLIT) / dk))
    n += n % 2                                   # Simpson wants an even count
    k_hi = np.linspace(K_SPLIT, kmax, n + 1)
    return k_lo, k_hi


def xi(r, kmax=10.0, dk=DK_FINE, k_damp=None):
    """xi(r) for an array of r.  k_damp applies a Gaussian cutoff exp(-(k/k_damp)^2)."""
    r = np.atleast_1d(np.asarray(r, dtype=float))
    total = np.zeros_like(r)
    for k in _grids(kmax, dk):
        w = k ** 2 * P(k)
        if k_damp is not None:
            w = w * np.exp(-(k / k_damp) ** 2)
        kr = np.outer(r, k)                       # (n_r, n_k)
        integrand = w[None, :] * np.sinc(kr / np.pi)   # np.sinc(x) = sin(pi x)/(pi x)
        total += simpson(integrand, x=k, axis=1)
    return total / (2.0 * np.pi ** 2)


def peak_of(r, y):
    """Position of the BAO bump in r^2 xi(r).

    r^2 xi(r) is still falling steeply at r = 50 and has dropped again by
    r = 120, so neither end of the range is the bump.  Take the interior local
    maximum with the largest prominence and refine it with a parabola through
    the top three samples.
    """
    idx, props = find_peaks(y, prominence=0.0)
    if len(idx) == 0:
        return float(r[int(np.argmax(y))])
    i = int(idx[int(np.argmax(props["prominences"]))])
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    d = y0 - 2 * y1 + y2
    if d != 0:
        return float(r[i] - 0.5 * (r[i + 1] - r[i]) * (y2 - y0) / d)
    return float(r[i])


# A hard cut at kmax rings with period 2 pi / kmax in r; a Gaussian taper at
# k_damp = 3 h/Mpc removes it while smoothing xi only on ~0.3 Mpc/h scales,
# far below the ~10 Mpc/h width of the BAO bump.
K_DAMP = 3.0

r = np.linspace(50.0, 120.0, 1401)               # 0.05 Mpc/h sampling
xi_r = xi(r, k_damp=K_DAMP)
r_peak = peak_of(r, r ** 2 * xi_r)

# --- robustness of the upper limit and of the k sampling -------------------
print("convergence of the BAO peak position:")
print(f"{'kmax':>8} {'dk':>10} {'damping':>9} {'r_peak [Mpc/h]':>15} {'r^2 xi peak':>12}")
for kmax, dk, kd in [(1.0, DK_FINE, None), (2.0, DK_FINE, None), (5.0, DK_FINE, None),
                     (10.0, DK_FINE, None), (20.0, DK_FINE, None),
                     (10.0, 1.0e-4, K_DAMP), (10.0, 5.0e-4, K_DAMP),
                     (10.0, DK_FINE, 1.0), (10.0, DK_FINE, K_DAMP),
                     (20.0, DK_FINE, K_DAMP)]:
    y = r ** 2 * xi(r, kmax=kmax, dk=dk, k_damp=kd)
    rp = peak_of(r, y)
    print(f"{kmax:8.1f} {dk:10.1e} {str(kd):>9} {rp:15.2f} {np.interp(rp, r, y):12.4f}")

# --- figure ---------------------------------------------------------------
y = r ** 2 * xi_r
y_peak = float(np.interp(r_peak, r, y))

fig = plt.figure(figsize=(10.5, 4.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], hspace=0.55, wspace=0.28)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[:, 1])

ax1.loglog(kt, pt, lw=1.0)
ax1.set_ylabel(r"$P(k)$ [(Mpc/$h$)$^3$]")
ax1.set_xlabel(r"$k$ [$h$/Mpc]")
ax1.set_title("Tabulated power spectrum", fontsize=8)

# Ratio to a smooth fit makes the baryon wiggles visible.
mw = (kt > 0.01) & (kt < 0.6)
coef = np.polyfit(np.log(kt[mw]), np.log(pt[mw]), 6)
ax2.semilogx(kt[mw], pt[mw] / np.exp(np.polyval(coef, np.log(kt[mw]))), lw=1.0, color="C2")
ax2.axhline(1.0, color="0.5", lw=0.6)
ax2.set_xlabel(r"$k$ [$h$/Mpc]")
ax2.set_ylabel(r"$P/P_{\rm smooth}$")
ax2.set_title(r"Baryon wiggles near $k\sim0.1\,h$/Mpc", fontsize=8)

ax3.plot(r, y, lw=1.3)
ax3.axvline(r_peak, color="C3", ls="--", lw=1.0)
ax3.plot([r_peak], [y_peak], "o", color="C3", ms=4)
ax3.annotate(f"BAO peak\n$r = {r_peak:.1f}$ Mpc/$h$",
             xy=(r_peak, y_peak), xytext=(0.40, 0.90), textcoords="axes fraction",
             fontsize=8, color="C3", ha="left", va="top",
             bbox=dict(fc="white", ec="none", pad=1.5),
             arrowprops=dict(arrowstyle="->", color="C3", lw=0.8,
                             connectionstyle="arc3,rad=-0.2"))
ax3.set_xlabel(r"$r$ [Mpc/$h$]")
ax3.set_ylabel(r"$r^2\xi(r)$ [(Mpc/$h$)$^2$]")
ax3.set_title(r"Correlation function, $r^2\xi(r)$", fontsize=8)
ax3.set_xlim(50, 120)
ax3.set_ylim(0, 1.12 * float(y.max()))
fig.savefig("figs/problem3_bao.png")

np.savetxt("results/xi_of_r.txt", np.column_stack([r, xi_r]),
           header="r [Mpc/h]    xi(r)", fmt="%.6e")
print(f"\nBAO peak at r = {r_peak:.2f} Mpc/h  (r^2 xi = {y_peak:.3f} (Mpc/h)^2)")
