"""Extrapolate per-cycle estimation cost vs state dimension n.

Anchored to the measured Lorenz96 timing at n=40, Ne=80 (Table 7.1, timing_data.csv):
the estimation cost per cycle for direct-precision shrinkage is roughly 1.5s out of
the 26s total per 100 cycles (the rest being ensemble propagation), i.e. ~0.015s per
cycle. We extrapolate the cost of each method using the asymptotic formulas in
Table 5.1 of the proposed method chapter, keeping Ne=80 fixed.

Output: paper/figures/timing_scaling.pdf
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "figures", "timing_scaling.pdf")

# Anchor: per-cycle estimation cost in seconds, measured at n=40, Ne=80.
# Direct precision shrinkage: ~0.015 s/cycle (asymptotic component only, excluding
# ensemble propagation, which is model-specific and irrelevant for scaling).
n_anchor = 40
Ne = 80
m = int(0.8 * n_anchor)  # kept proportional; affects EnKF sample only at O(m^2 Ne)
r = 2  # Cholesky bandwidth
B = 200  # NERCOME bootstrap draws

# Per-cycle cost anchor (seconds). Chosen so that the direct precision methods at
# n=40 land at ~1.5e-2 s, consistent with our measurements once the ensemble
# propagation is subtracted.
c_direct_anchor = 1.5e-2


def cost_direct(n):
    """Direct precision shrinkage: O(n^2 Ne + n^3)."""
    return c_direct_anchor * (n**2 * Ne + n**3) / (n_anchor**2 * Ne + n_anchor**3)


def cost_eigenvalue(n):
    """Eigenvalue shrinkage: O(n^2 Ne + 2 n^3)."""
    return c_direct_anchor * (n**2 * Ne + 2 * n**3) / (n_anchor**2 * Ne + n_anchor**3)


def cost_nercome(n):
    """NERCOME: O(B (n^2 Ne + n^3))."""
    return c_direct_anchor * B * (n**2 * Ne + n**3) / (n_anchor**2 * Ne + n_anchor**3)


def cost_ledoit_wolf(n):
    """Ledoit-Wolf + inversion for use as precision: O(n^2 Ne + n^3)."""
    return c_direct_anchor * (n**2 * Ne + n**3) / (n_anchor**2 * Ne + n_anchor**3)


def cost_enkf(n):
    """Standard EnKF (sample covariance path): O(n^2 Ne + m^2 Ne); scale m with n."""
    m_n = int(0.8 * n)
    m_a = int(0.8 * n_anchor)
    return c_direct_anchor * (n**2 * Ne + m_n**2 * Ne) / (n_anchor**2 * Ne + n_anchor**3)


def cost_cholesky(n):
    """EnKF-Cholesky with bandwidth r: O(n * r * Ne)."""
    return c_direct_anchor * (n * r * Ne) / (n_anchor**2 * Ne + n_anchor**3)


def main():
    ns = np.logspace(1.3, 6, 200)  # n from ~20 to 10^6

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.loglog(ns, [cost_direct(n) for n in ns], label="Direct precision shrinkage  $\\mathcal{O}(n^2 N_e + n^3)$", lw=2)
    ax.loglog(ns, [cost_eigenvalue(n) for n in ns], label="Eigenvalue shrinkage  $\\mathcal{O}(n^2 N_e + 2n^3)$", lw=2, ls="--")
    ax.loglog(ns, [cost_nercome(n) for n in ns], label=f"NERCOME  $\\mathcal{{O}}(B(n^2 N_e + n^3)),\\ B={B}$", lw=2, ls=":")
    ax.loglog(ns, [cost_ledoit_wolf(n) for n in ns], label="Ledoit-Wolf + inversion  $\\mathcal{O}(n^2 N_e + n^3)$", lw=2, ls="-.")
    ax.loglog(ns, [cost_enkf(n) for n in ns], label="EnKF (sample)  $\\mathcal{O}(n^2 N_e + m^2 N_e)$", lw=1.5, color="gray")
    ax.loglog(ns, [cost_cholesky(n) for n in ns], label=f"EnKF-Cholesky  $\\mathcal{{O}}(n r N_e),\\ r={r}$", lw=2, color="black", ls="--")

    # Regime markers.
    for n_mark, label in [(40, "testbed\n(this work)"), (1e3, "QG-scale"), (1e6, "operational NWP")]:
        ax.axvline(n_mark, color="tab:red", alpha=0.25, lw=1)
        ax.text(n_mark * 1.05, 3e4, label, rotation=90, va="top", ha="left", fontsize=9, color="tab:red", alpha=0.8)

    ax.set_xlabel("State dimension $n$")
    ax.set_ylabel("Per-cycle estimation cost (s, extrapolated)")
    ax.set_title(f"Per-cycle estimation cost vs.\\ state dimension ($N_e = {Ne}$)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(ns[0], ns[-1])

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
