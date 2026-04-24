# -*- coding: utf-8 -*-
"""
Generate all figures and table data for the LNCS paper.

Usage:
    cd d:/thesis
    python paper/generate_figures.py

Outputs:
    paper/figures/rmse_time.pdf       - Fig 2: RMSE vs assimilation step
    paper/figures/rmse_vs_ne.pdf      - Fig 3: RMSE vs ensemble size
    paper/figures/precision_heatmap.pdf - Fig 4: Precision matrix heatmaps
    paper/figures/table_data.csv      - Table 1: RMSE values
"""

import sys
import os
import warnings
import logging

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Suppress verbose per-timestep logging from TEDA
logging.getLogger().setLevel(logging.WARNING)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pyteda.models.lorenz96 import Lorenz96
from pyteda.background.background_core import Background
from pyteda.observation.observation_core import Observation
from pyteda.simulation.simulation_core import Simulation

# Analysis methods
from pyteda.analysis.analysis_enkf import AnalysisEnKF
from pyteda.analysis.analysis_enkf_modified_cholesky import AnalysisEnKFModifiedCholesky
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_identity import AnalysisEnKFDirectPrecisionShrinkageIdentity
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_identity_scaled import AnalysisEnKFDirectPrecisionShrinkageIdentityScaled
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_eigenvalues import AnalysisEnKFDirectPrecisionShrinkageEigenvalues
from pyteda.analysis.analysis_enkf_ledoit_wolf_shrinkage import AnalysisEnKFLedoitWolfShrinkage
from pyteda.analysis.analysis_enkf_nercome_shrinkage import AnalysisEnKFNercomeShrinkage

# ── Configuration ──────────────────────────────────────────────────────────────

N_VARIABLES = 40          # Lorenz96 state dimension
FORCING = 8               # Lorenz96 forcing parameter
N_OBS = 32                # Number of observations
STD_OBS = 0.01            # Observation error std
OBS_FREQ = 0.1            # Observation frequency (time units)
END_TIME = 10             # Total simulation time (=> 100 analysis cycles)
INF_FACT = 1.05           # Inflation factor
ENSEMBLE_SIZES = [50, 60, 80, 100]  # Ne values (must be > n+2=42 for precision methods)
N_RUNS = 30               # Repetitions per configuration
TRANSIENT = 50            # Skip first N cycles when computing time-averaged RMSE
RANDOM_SEED_BASE = 42
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')

# ── Method definitions ─────────────────────────────────────────────────────────

def get_methods(model):
    """Return dict of method_name -> (constructor_fn, requires_invertible_S)."""
    return {
        'EnKF': (lambda: AnalysisEnKF(), False),
        'EnKF-Cholesky': (lambda: AnalysisEnKFModifiedCholesky(model, r=2), False),
        'Identity Shrinkage': (lambda: AnalysisEnKFDirectPrecisionShrinkageIdentity(model), True),
        'Scaled Shrinkage': (lambda: AnalysisEnKFDirectPrecisionShrinkageIdentityScaled(model), True),
        'Eigenvalue Shrinkage': (lambda: AnalysisEnKFDirectPrecisionShrinkageEigenvalues(model), True),
        'Ledoit-Wolf': (lambda: AnalysisEnKFLedoitWolfShrinkage(model), False),
        'NERCOME': (lambda: AnalysisEnKFNercomeShrinkage(model, max_draws=1000), False),
    }

# ── Experiment runner ──────────────────────────────────────────────────────────

def run_single_experiment(model, ensemble_size, analysis, seed):
    """Run one DA experiment, return (error_background, error_analysis) arrays."""
    # Pre-compute and cache the true initial condition (always uses seed=10
    # internally). We then override get_initial_condition to return the cached
    # value without resetting np.random state, so our per-run seed controls
    # ensemble generation and observation noise.
    true_ic = model.get_initial_condition()  # deterministic, seed=10 inside
    original_get_ic = model.get_initial_condition
    model.get_initial_condition = lambda: true_ic

    np.random.seed(seed)
    background = Background(model, ensemble_size=ensemble_size)
    observation = Observation(m=N_OBS, std_obs=STD_OBS)
    params = {'obs_freq': OBS_FREQ, 'end_time': END_TIME, 'inf_fact': INF_FACT}

    sim = Simulation(model, background, analysis, observation, params=params,
                      log_level=None)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.run()

    # Restore original method
    model.get_initial_condition = original_get_ic

    err_b, err_a = sim.get_errors()
    return err_b, err_a


def load_existing_csv():
    """Load existing table_data.csv, return dict of {method: {Ne: (mean, std)}} or empty."""
    import csv
    path = os.path.join(FIGURES_DIR, 'table_data.csv')
    existing = {}
    if not os.path.exists(path):
        return existing
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row['Method']
            existing[method] = {}
            for Ne in ENSEMBLE_SIZES:
                cell = row.get(f'Ne={Ne}', 'N/A')
                if cell and cell != 'N/A':
                    try:
                        mean_s, std_s = cell.split('+/-')
                        existing[method][Ne] = (float(mean_s), float(std_s))
                    except ValueError:
                        pass
    return existing


def save_partial_csv(results):
    """Save current results to CSV (called after each method completes)."""
    import csv
    path = os.path.join(FIGURES_DIR, 'table_data.csv')
    rows = []
    for method_name, ne_dict in results.items():
        row_data = {'Method': method_name}
        for Ne in ENSEMBLE_SIZES:
            runs = ne_dict.get(Ne, [])
            if len(runs) == 0:
                row_data[f'Ne={Ne}'] = 'N/A'
            else:
                avg_per_run = [np.mean(r[TRANSIENT:]) for r in runs]
                mean_val = np.mean(avg_per_run) * 100
                std_val = np.std(avg_per_run) * 100
                row_data[f'Ne={Ne}'] = f"{mean_val:.4f} +/- {std_val:.4f}"
        rows.append(row_data)
    with open(path, 'w', newline='') as f:
        fieldnames = ['Method'] + [f'Ne={Ne}' for Ne in ENSEMBLE_SIZES]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [checkpoint] Saved {path}")


def run_all_experiments():
    """Run all methods x ensemble sizes x repetitions. Returns results dict.

    Incremental: loads existing CSV, skips methods already completed (non-zero std),
    and saves CSV after each method finishes.
    """
    model = Lorenz96(n=N_VARIABLES, F=FORCING)
    methods = get_methods(model)

    # Load previously completed results from CSV
    existing = load_existing_csv()

    # results[method][Ne] = list of err_a arrays (one per run)
    results = {name: {Ne: [] for Ne in ENSEMBLE_SIZES} for name in methods}

    for method_name in methods:
        # Check if this method already has non-zero std devs for ALL Ne values
        if method_name in existing:
            all_done = all(
                existing[method_name].get(Ne, (0, 0))[1] > 0.0
                for Ne in ENSEMBLE_SIZES
            )
            if all_done:
                print(f"  [skip] {method_name} — already in CSV with non-zero std devs")
                # Re-populate results with placeholder so CSV save is correct
                # We'll use the existing mean/std to reconstruct dummy run lists
                for Ne in ENSEMBLE_SIZES:
                    mean_v, std_v = existing[method_name].get(Ne, (0.0, 0.0))
                    # Store as single synthetic value (mean only) — just for CSV output
                    results[method_name][Ne] = [np.full(100, mean_v / 100)]
                continue

        print(f"\n  --- Method: {method_name} ---")
        constructor, needs_inv = methods[method_name]

        for Ne in ENSEMBLE_SIZES:
            if needs_inv and Ne <= N_VARIABLES + 2:
                print(f"    Skipping Ne={Ne} (needs Ne > {N_VARIABLES+2})")
                continue

            for run_idx in range(N_RUNS):
                seed = RANDOM_SEED_BASE + run_idx * 1000 + Ne
                print(f"    Ne={Ne}, run {run_idx+1}/{N_RUNS} (seed={seed})")
                try:
                    analysis = constructor()
                    _, err_a = run_single_experiment(model, Ne, analysis, seed)
                    results[method_name][Ne].append(err_a)
                except Exception as e:
                    print(f"    FAILED: {e}")

        # Save checkpoint after each method completes
        save_partial_csv(results)

    return results


# ── Extract precision matrices for heatmap ─────────────────────────────────────

def extract_precision_matrices(Ne=60):
    """Run one step and extract precision matrices from each method."""
    model = Lorenz96(n=N_VARIABLES, F=FORCING)

    # Cache true IC to prevent seed reset inside sim.run()
    true_ic = model.get_initial_condition()
    original_get_ic = model.get_initial_condition
    model.get_initial_condition = lambda: true_ic

    np.random.seed(RANDOM_SEED_BASE)

    background = Background(model, ensemble_size=Ne)
    observation = Observation(m=N_OBS, std_obs=STD_OBS)

    # Run a few forecast-analysis cycles to get away from initial transient
    params = {'obs_freq': OBS_FREQ, 'end_time': 3.0, 'inf_fact': INF_FACT}

    # First, run standard EnKF to get a reasonable ensemble
    analysis_enkf = AnalysisEnKF()
    sim = Simulation(model, background, analysis_enkf, observation, params=params,
                     log_level=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.run()

    model.get_initial_condition = original_get_ic

    # Now get the current ensemble and compute DX
    Xb = analysis_enkf.get_ensemble()
    xb = np.mean(Xb, axis=1)
    DX = Xb - np.outer(xb, np.ones(Ne))

    # Reference: large-ensemble sample precision
    S_sample = np.cov(DX, bias=False)

    precision_matrices = {}
    precision_matrices['Sample $\\mathbf{S}^{-1}$'] = np.linalg.inv(S_sample)

    # Identity shrinkage
    method = AnalysisEnKFDirectPrecisionShrinkageIdentity(model)
    precision_matrices['Identity'] = method.get_precision_matrix(DX)

    # Scaled shrinkage
    method = AnalysisEnKFDirectPrecisionShrinkageIdentityScaled(model)
    precision_matrices['Scaled'] = method.get_precision_matrix(DX)

    # Eigenvalue shrinkage
    method = AnalysisEnKFDirectPrecisionShrinkageEigenvalues(model)
    precision_matrices['Eigenvalue'] = method.get_precision_matrix(DX)

    # Ledoit-Wolf
    method = AnalysisEnKFLedoitWolfShrinkage(model)
    precision_matrices['Ledoit-Wolf'] = method.get_precision_matrix(DX)

    # Modified Cholesky
    method = AnalysisEnKFModifiedCholesky(model, r=2)
    precision_matrices['Cholesky'] = method.get_precision_matrix(DX)

    # NERCOME (fewer draws for speed in heatmap)
    method = AnalysisEnKFNercomeShrinkage(model, max_draws=200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision_matrices['NERCOME'] = method.get_precision_matrix(DX)

    return precision_matrices


# ── Plotting ───────────────────────────────────────────────────────────────────

COLORS = {
    'EnKF': '#1f77b4',
    'EnKF-Cholesky': '#ff7f0e',
    'Identity Shrinkage': '#2ca02c',
    'Scaled Shrinkage': '#d62728',
    'Eigenvalue Shrinkage': '#9467bd',
    'Ledoit-Wolf': '#8c564b',
    'NERCOME': '#e377c2',
}

LINESTYLES = {
    'EnKF': '--',
    'EnKF-Cholesky': '--',
    'Identity Shrinkage': '-',
    'Scaled Shrinkage': '-',
    'Eigenvalue Shrinkage': '-',
    'Ledoit-Wolf': '-.',
    'NERCOME': ':',
}


def plot_rmse_time(results, Ne_plot=None):
    """Fig 2: RMSE vs assimilation step for a single ensemble size."""
    if Ne_plot is None:
        Ne_plot = max(ENSEMBLE_SIZES)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    for method_name in results:
        runs = results[method_name][Ne_plot]
        if len(runs) == 0:
            continue
        # Stack runs and compute mean
        stacked = np.array(runs)
        mean_err = np.mean(stacked, axis=0)
        steps = np.arange(len(mean_err))

        ax.semilogy(steps, mean_err,
                     label=method_name,
                     color=COLORS.get(method_name, 'gray'),
                     linestyle=LINESTYLES.get(method_name, '-'),
                     linewidth=1.5)

    ax.set_xlabel('Assimilation step')
    ax.set_ylabel('Relative RMSE')
    ax.set_title(f'RMSE over time ($N_e = {Ne_plot}$)')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, 'rmse_time.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f"Saved {path}")
    plt.close(fig)


def plot_rmse_vs_ne(results):
    """Fig 3: Time-averaged RMSE vs ensemble size."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    for method_name in results:
        ne_vals = []
        means = []
        stds = []

        for Ne in ENSEMBLE_SIZES:
            runs = results[method_name][Ne]
            if len(runs) == 0:
                continue
            # Compute time-averaged RMSE per run (skip transient)
            avg_per_run = [np.mean(r[TRANSIENT:]) for r in runs]
            ne_vals.append(Ne)
            means.append(np.mean(avg_per_run))
            stds.append(np.std(avg_per_run))

        if len(ne_vals) == 0:
            continue

        ne_vals = np.array(ne_vals)
        means = np.array(means)
        stds = np.array(stds)

        color = COLORS.get(method_name, 'gray')
        ax.errorbar(ne_vals, means, yerr=stds,
                     label=method_name,
                     color=color,
                     linestyle=LINESTYLES.get(method_name, '-'),
                     marker='o', markersize=5, capsize=3, linewidth=1.5)

    ax.set_xlabel('Ensemble size $N_e$')
    ax.set_ylabel('Time-averaged relative RMSE')
    ax.set_title('RMSE vs. ensemble size')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, 'rmse_vs_ne.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f"Saved {path}")
    plt.close(fig)


def plot_precision_heatmaps():
    """Fig 4: 2x4 grid of precision matrix heatmaps (7 matrices, 1 empty slot)."""
    prec = extract_precision_matrices(Ne=60)

    n_cols = 4
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 6.5))
    names = list(prec.keys())

    # Use symmetric log scale for better visualization
    for idx, name in enumerate(names):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        P = prec[name]

        # Clip extreme values for visualization
        vmax = np.percentile(np.abs(P), 98)
        im = ax.imshow(P, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       interpolation='nearest', aspect='equal')
        ax.set_title(name, fontsize=9)
        ax.set_xticks([0, 19, 39])
        ax.set_yticks([0, 19, 39])
        fig.colorbar(im, ax=ax, shrink=0.8)

    # Hide any unused subplot slots
    for idx in range(len(names), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle('Precision matrix estimates ($N_e = 60$, $n = 40$)', fontsize=11)
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, 'precision_heatmap.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f"Saved {path}")
    plt.close(fig)


def generate_table(results):
    """Generate Table 1 data: time-averaged RMSE for each method x Ne.

    Merges with existing CSV to preserve values for methods that were skipped
    (loaded from checkpoint with synthetic single-run data).
    """
    import csv

    print("\n" + "="*80)
    print("TABLE 1: Time-averaged relative RMSE (x 10^-2)")
    print("="*80)

    # Load existing CSV to preserve values for skipped methods
    existing = load_existing_csv()

    header = f"{'Method':<25s}"
    for Ne in ENSEMBLE_SIZES:
        header += f" | Ne={Ne:>3d}  "
    print(header)
    print("-" * len(header))

    rows = []
    for method_name in results:
        row = f"{method_name:<25s}"
        row_data = {'Method': method_name}

        for Ne in ENSEMBLE_SIZES:
            runs = results[method_name][Ne]
            # If only 1 run (synthetic from skip), use existing CSV value
            if len(runs) <= 1 and method_name in existing and Ne in existing[method_name]:
                mean_val, std_val = existing[method_name][Ne]
                cell = f"{mean_val:.4f}+/-{std_val:.4f}"
                row += f" | {cell:>16s}"
                row_data[f'Ne={Ne}'] = f"{mean_val:.4f} +/- {std_val:.4f}"
            elif len(runs) == 0:
                row += f" | {'N/A':>8s}"
                row_data[f'Ne={Ne}'] = 'N/A'
            else:
                avg_per_run = [np.mean(r[TRANSIENT:]) for r in runs]
                mean_val = np.mean(avg_per_run) * 100  # x10^-2
                std_val = np.std(avg_per_run) * 100
                cell = f"{mean_val:.4f}+/-{std_val:.4f}"
                row += f" | {cell:>16s}"
                row_data[f'Ne={Ne}'] = f"{mean_val:.4f} +/- {std_val:.4f}"

        print(row)
        rows.append(row_data)

    # Save as CSV
    path = os.path.join(FIGURES_DIR, 'table_data.csv')
    with open(path, 'w', newline='') as f:
        fieldnames = ['Method'] + [f'Ne={Ne}' for Ne in ENSEMBLE_SIZES]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("="*60)
    print("Generating figures for LNCS paper")
    print(f"Config: n={N_VARIABLES}, Ne={ENSEMBLE_SIZES}, "
          f"runs={N_RUNS}, obs_freq={OBS_FREQ}")
    print("="*60)

    # Run all experiments
    print("\n--- Running experiments ---")
    results = run_all_experiments()

    # Generate figures
    print("\n--- Generating figures ---")
    plot_rmse_time(results)
    plot_rmse_vs_ne(results)
    plot_precision_heatmaps()

    # Generate table data
    generate_table(results)

    print("\nDone! Figures saved to:", FIGURES_DIR)
