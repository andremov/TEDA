# -*- coding: utf-8 -*-
"""
Generate additional figures for the thesis that require running experiments.

This script produces figures not in the original paper:
  1. RMSE time series at Ne=50 on Lorenz96 (shows instability)
  2. RMSE time series at Ne=60 on Lorenz96
  3. RMSE time series for cosmological AR(1) model at Ne=30
  4. Sample covariance matrices at different Ne for Lorenz96
  5. Localization matrix visualization (Gaspari-Cohn at different radii)
  6. Per-cycle alpha* and beta* evolution over assimilation steps (Lorenz96)

Usage:
    cd d:/thesis
    python thesis/generate_additional_figures.py

WARNING: This script runs experiments and may take 30-60 minutes.
         NERCOME is excluded from time series plots to save time.
"""

import sys
import os
import warnings
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
logging.getLogger().setLevel(logging.WARNING)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyteda.models.lorenz96 import Lorenz96
from pyteda.background.background_core import Background
from pyteda.observation.observation_core import Observation
from pyteda.simulation.simulation_core import Simulation

from pyteda.analysis.analysis_enkf import AnalysisEnKF
from pyteda.analysis.analysis_enkf_modified_cholesky import AnalysisEnKFModifiedCholesky
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_identity import AnalysisEnKFDirectPrecisionShrinkageIdentity
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_identity_scaled import AnalysisEnKFDirectPrecisionShrinkageIdentityScaled
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_eigenvalues import AnalysisEnKFDirectPrecisionShrinkageEigenvalues
from pyteda.analysis.analysis_enkf_ledoit_wolf_shrinkage import AnalysisEnKFLedoitWolfShrinkage

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.figsize': (8, 5),
})

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')

# ── Configuration ────────────────────────────────────────────────────────────

N_VARIABLES = 40
FORCING = 8
N_OBS = 32
STD_OBS = 0.01
OBS_FREQ = 0.1
END_TIME = 10
INF_FACT = 1.05
RANDOM_SEED = 42

COLORS = {
    'EnKF': '#333333',
    'EnKF-Cholesky': '#8c564b',
    'Identity Shrinkage': '#d62728',
    'Scaled Shrinkage': '#ff7f0e',
    'Eigenvalue Shrinkage': '#9467bd',
    'Ledoit-Wolf': '#1f77b4',
}
LINESTYLES = {
    'EnKF': '--',
    'EnKF-Cholesky': '--',
    'Identity Shrinkage': '-',
    'Scaled Shrinkage': '-',
    'Eigenvalue Shrinkage': '-',
    'Ledoit-Wolf': '-.',
}


def run_single(model, Ne, analysis, seed):
    """Run one experiment, return (err_background, err_analysis)."""
    true_ic = model.get_initial_condition()
    model.get_initial_condition = lambda: true_ic
    np.random.seed(seed)
    background = Background(model, ensemble_size=Ne)
    observation = Observation(m=N_OBS, std_obs=STD_OBS)
    params = {'obs_freq': OBS_FREQ, 'end_time': END_TIME, 'inf_fact': INF_FACT}
    sim = Simulation(model, background, analysis, observation, params=params, log_level=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.run()
    err_b, err_a = sim.get_errors()
    return err_b, err_a


def get_methods(model):
    return {
        'EnKF': lambda: AnalysisEnKF(),
        'EnKF-Cholesky': lambda: AnalysisEnKFModifiedCholesky(model, r=2),
        'Identity Shrinkage': lambda: AnalysisEnKFDirectPrecisionShrinkageIdentity(model),
        'Scaled Shrinkage': lambda: AnalysisEnKFDirectPrecisionShrinkageIdentityScaled(model),
        'Eigenvalue Shrinkage': lambda: AnalysisEnKFDirectPrecisionShrinkageEigenvalues(model),
        'Ledoit-Wolf': lambda: AnalysisEnKFLedoitWolfShrinkage(model),
    }


# ── Figure 1 & 2: RMSE time series at Ne=50 and Ne=60 ───────────────────────

def generate_rmse_time_series(Ne):
    """Generate RMSE time series plot for a given ensemble size."""
    print(f"\n=== Generating RMSE time series for Ne={Ne} ===")
    model = Lorenz96(n=N_VARIABLES, F=FORCING)
    methods = get_methods(model)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, constructor in methods.items():
        print(f"  Running {name}...")
        analysis = constructor()
        _, err_a = run_single(model, Ne, analysis, RANDOM_SEED)
        ax.plot(err_a, label=name, color=COLORS[name],
                linestyle=LINESTYLES[name], linewidth=1.2)

    ax.set_xlabel('Assimilation cycle')
    ax.set_ylabel('Relative RMSE')
    ax.set_title(f'RMSE over 100 assimilation cycles (Lorenz 96, $N_e = {Ne}$)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    outpath = os.path.join(FIGURES_DIR, f'rmse_time_ne{Ne}.pdf')
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {outpath}")


# ── Figure 3: Sample covariance at different Ne ──────────────────────────────

def generate_sample_covariance_comparison():
    """Generate sample covariance matrices at different ensemble sizes."""
    print("\n=== Generating sample covariance comparison ===")
    model = Lorenz96(n=N_VARIABLES, F=FORCING)
    ne_values = [45, 60, 100]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, Ne in zip(axes, ne_values):
        np.random.seed(RANDOM_SEED)
        background = Background(model, ensemble_size=Ne)
        Xb = background.get_initial_ensemble(initial_perturbation=0.05,
                                              time=np.arange(0, 10, 0.01))
        # Run a few more propagation steps
        time = np.arange(0, OBS_FREQ, 0.01)
        for _ in range(20):
            Xb = background.forecast_step(Xb, time)
        S = background.get_covariance_matrix()
        im = ax.imshow(S, cmap='RdBu_r', aspect='equal',
                       vmin=-np.percentile(np.abs(S), 95),
                       vmax=np.percentile(np.abs(S), 95))
        ax.set_title(f'$N_e = {Ne}$')
        ax.set_xlabel('Variable index')
        if ax == axes[0]:
            ax.set_ylabel('Variable index')
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('Sample covariance $\\mathbf{S}$ at different ensemble sizes (Lorenz 96, $n=40$)',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    outpath = os.path.join(FIGURES_DIR, 'sample_cov_comparison.pdf')
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {outpath}")


# ── Figure 4: Localization matrix visualization ──────────────────────────────

def generate_localization_visualization():
    """Generate Gaspari-Cohn localization matrices at different radii."""
    print("\n=== Generating localization visualization ===")
    n = 40
    radii = [2, 5, 10]

    def gaspari_cohn(z):
        """Gaspari-Cohn correlation function."""
        result = np.zeros_like(z)
        mask1 = (z >= 0) & (z <= 1)
        mask2 = (z > 1) & (z <= 2)
        z1 = z[mask1]
        result[mask1] = -0.25*z1**5 + 0.5*z1**4 + 5/8*z1**3 - 5/3*z1**2 + 1
        z2 = z[mask2]
        result[mask2] = (1/12*z2**5 - 0.5*z2**4 + 5/8*z2**3 + 5/3*z2**2
                         - 5*z2 + 4 - 2/(3*z2))
        return result

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, r in zip(axes, radii):
        # Build localization matrix (cyclic distance for Lorenz96)
        L = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                d = min(abs(i - j), n - abs(i - j))  # cyclic distance
                L[i, j] = gaspari_cohn(np.array([d / r]))[0]
        im = ax.imshow(L, cmap='viridis', aspect='equal', vmin=0, vmax=1)
        ax.set_title(f'$r = {r}$')
        ax.set_xlabel('Variable index')
        if ax == axes[0]:
            ax.set_ylabel('Variable index')
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('Gaspari-Cohn localization matrix $\\hat{\\Lambda}$ at different radii (Lorenz 96, $n=40$)',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    outpath = os.path.join(FIGURES_DIR, 'localization_matrices.pdf')
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {outpath}")


# ── Figure 5: Target matrix comparison ───────────────────────────────────────

def generate_target_comparison():
    """Generate heatmaps of the four shrinkage target matrices."""
    print("\n=== Generating target matrix comparison ===")
    n = 18  # Use cosmological dimension for more informative targets

    # Simulate typical diagonal entries
    np.random.seed(42)
    variances = np.abs(np.random.randn(n)) * 1e-8 + 1e-9  # typical power spectrum variances

    # Identity target
    T_identity = np.eye(n)

    # Eigenvalue target (simulated sorted eigenvalues)
    eigenvalues = np.sort(np.abs(np.random.randn(n) * 5 + 10))
    T_eigenvalue = np.diag(eigenvalues)

    # Scaled identity target
    T_scaled = np.diag(1 / (2 * variances))

    # Cosmological target (proportional to inverse variance)
    N_ell = np.array([9, 13, 17, 21, 25, 29, 33, 37, 41] * 2)[:n]
    C_ell = np.abs(np.random.randn(n) * 1e-4 + 5e-4)
    T_cosmo_cov = np.diag(2 / N_ell * C_ell**2)
    T_cosmo = np.diag(1 / np.diag(T_cosmo_cov))

    targets = [
        (T_identity, r'Identity: $\mathbf{I}_n$'),
        (T_eigenvalue, r'Eigenvalue: $\mathrm{diag}(e_1, \ldots, e_n)$'),
        (T_scaled, r'Scaled: $\mathrm{diag}(1/2\sigma_i^2)$'),
        (T_cosmo, r'Cosmological: $(\mathbf{T}^{(2)})^{-1}$'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (T, title) in zip(axes, targets):
        # Normalize for visualization
        T_norm = T / np.max(np.abs(T)) if np.max(np.abs(T)) > 0 else T
        im = ax.imshow(T_norm, cmap='Blues', aspect='equal', vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('$j$')
        if ax == axes[0]:
            ax.set_ylabel('$i$')
        ax.set_xticks([0, n//2, n-1])
        ax.set_yticks([0, n//2, n-1])

    fig.suptitle('Shrinkage target matrices $\\Pi_0$ (normalized, $d=18$)', fontsize=12, y=1.02)
    fig.tight_layout()
    outpath = os.path.join(FIGURES_DIR, 'target_comparison.pdf')
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {outpath}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # These don't require experiments (fast)
    print("=" * 60)
    print("Phase 1: Figures from synthetic data (fast)")
    print("=" * 60)
    generate_localization_visualization()
    generate_target_comparison()

    # These require TEDA experiments (slower)
    print("\n" + "=" * 60)
    print("Phase 2: Figures from experiments (slower)")
    print("=" * 60)
    generate_sample_covariance_comparison()
    generate_rmse_time_series(Ne=50)
    generate_rmse_time_series(Ne=60)

    print("\n" + "=" * 60)
    print("All done!")
    print("=" * 60)
    print("\nGenerated figures:")
    print("  - localization_matrices.pdf")
    print("  - target_comparison.pdf")
    print("  - sample_cov_comparison.pdf")
    print("  - rmse_time_ne50.pdf")
    print("  - rmse_time_ne60.pdf")
