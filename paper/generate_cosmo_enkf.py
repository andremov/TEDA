# -*- coding: utf-8 -*-
"""
Part B: TEDA EnKF experiment with AR(1) cosmological model.

Runs the full TEDA data assimilation pipeline with an AR(1) model calibrated
to BOSS DR12 mock catalog statistics. This tests all 8 methods (including
the cosmological shrinkage target) through the EnKF pipeline.

Usage:
    cd d:/thesis
    python paper/generate_cosmo_enkf.py

Outputs:
    paper/figures/cosmo_enkf_table.csv     - RMSE table
    paper/figures/cosmo_enkf_rmse.pdf      - RMSE vs ensemble size
    paper/figures/cosmo_enkf_rmse_time.pdf - RMSE over time (one Ne)
"""

import sys
import os
import warnings
import logging
import csv
import io
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
logging.getLogger().setLevel(logging.WARNING)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyteda.models.ar1_power_spectrum import AR1PowerSpectrum
from pyteda.background.background_core import Background
from pyteda.observation.observation_core import Observation
from pyteda.simulation.simulation_core import Simulation

# Analysis methods
from pyteda.analysis.analysis_enkf import AnalysisEnKF
from pyteda.analysis.analysis_enkf_modified_cholesky import AnalysisEnKFModifiedCholesky
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_identity import AnalysisEnKFDirectPrecisionShrinkageIdentity
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_identity_scaled import AnalysisEnKFDirectPrecisionShrinkageIdentityScaled
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_eigenvalues import AnalysisEnKFDirectPrecisionShrinkageEigenvalues
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_identity_scaled_cosmo import AnalysisEnKFDirectPrecisionShrinkageIdentityScaledCosmo
from pyteda.analysis.analysis_enkf_ledoit_wolf_shrinkage import AnalysisEnKFLedoitWolfShrinkage
from pyteda.analysis.analysis_enkf_nercome_shrinkage import AnalysisEnKFNercomeShrinkage

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'looijmans')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
MOCKS_DIR = os.path.join(DATA_DIR, 'output', 'BOSS_DR12_NGC_z1', 'mocks')
TARGET_FILE = os.path.join(DATA_DIR, 'output', 'BOSS_DR12_NGC_z1', 'T_18_18.matrix')

P_DIM = 18              # Data vector dimension
N_OBS = 14              # Observe 14 of 18 bins (~ 78%, similar to 32/40 for Lorenz96)
STD_OBS = 100.0         # Observation error std (scaled to power spectrum magnitude)
OBS_FREQ = 0.1          # Observation frequency
END_TIME = 10.0         # Total simulation time (=> 100 analysis cycles)
INF_FACT = 1.02         # Lighter inflation for near-linear system
PHI = 0.95              # AR(1) persistence
ENSEMBLE_SIZES = [24, 30, 40, 50]  # Must be > d+2 = 20
N_RUNS = 30
TRANSIENT = 50          # Skip first 50 cycles
RANDOM_SEED_BASE = 42

COLORS = {
    'EnKF': '#333333',
    'EnKF-Cholesky': '#8c564b',
    'Identity Shrinkage': '#d62728',
    'Scaled Shrinkage': '#ff7f0e',
    'Eigenvalue Shrinkage': '#9467bd',
    'Cosmological Shrinkage': '#2ca02c',
    'NERCOME': '#e377c2',
    'Ledoit-Wolf': '#1f77b4',
}

LINESTYLES = {
    'EnKF': '--',
    'EnKF-Cholesky': '--',
    'Identity Shrinkage': '-',
    'Scaled Shrinkage': '-.',
    'Eigenvalue Shrinkage': ':',
    'Cosmological Shrinkage': '-',
    'NERCOME': '-',
    'Ledoit-Wolf': '-',
}


# ── Data loading and model setup ─────────────────────────────────────────────

def load_mock_statistics():
    """Load mock data statistics for AR(1) calibration."""
    P_all = np.loadtxt(os.path.join(MOCKS_DIR, 'n2048', 'P_18_2048_v1.matrix'))
    mean_spectrum = np.mean(P_all, axis=1)
    cov_mock = np.cov(P_all)
    print(f"Mock statistics: mean range [{mean_spectrum.min():.1f}, {mean_spectrum.max():.1f}]")
    print(f"  Cov diagonal range [{np.diag(cov_mock).min():.1f}, {np.diag(cov_mock).max():.1f}]")
    return mean_spectrum, cov_mock


def load_target_covariance():
    """Load the cosmological target T."""
    return np.loadtxt(TARGET_FILE)


def create_model(mean_spectrum, cov_mock):
    """Create AR(1) model calibrated to mock data."""
    model = AR1PowerSpectrum(mean_spectrum, cov_mock, phi=PHI)
    model.create_decorrelation_matrix(r=2)
    return model


def setup_cosmological_target(analysis_cosmo, T):
    """Configure cosmological target from T matrix (same as Part A)."""
    T_diag = np.diag(T)
    power_spectrum = {}
    multipole_sampling = {}
    data_vector_mapping = []
    survey_geometry = {
        'survey_area_deg2': 7500,
        'sky_fraction': 0.18,
        'redshift_bins': 1,
    }
    for i in range(P_DIM):
        ell = i + 2
        N_ell = 2
        C_ell = np.sqrt(T_diag[i])
        power_spectrum[ell] = C_ell
        multipole_sampling[ell] = N_ell
        data_vector_mapping.append((ell, 0))

    with redirect_stdout(io.StringIO()):
        analysis_cosmo.set_cosmological_parameters(
            power_spectrum=power_spectrum,
            survey_geometry=survey_geometry,
            multipole_sampling=multipole_sampling,
            data_vector_mapping=data_vector_mapping,
        )


# ── Method definitions ────────────────────────────────────────────────────────

def get_methods(model, T):
    """Return dict of method_name -> (constructor_fn, requires_Ne_gt_d_plus_2)."""
    def make_cosmo():
        c = AnalysisEnKFDirectPrecisionShrinkageIdentityScaledCosmo(model)
        setup_cosmological_target(c, T)
        return c

    return {
        'EnKF': (lambda: AnalysisEnKF(), False),
        'EnKF-Cholesky': (lambda: AnalysisEnKFModifiedCholesky(model, r=2), False),
        'Identity Shrinkage': (lambda: AnalysisEnKFDirectPrecisionShrinkageIdentity(model), True),
        'Scaled Shrinkage': (lambda: AnalysisEnKFDirectPrecisionShrinkageIdentityScaled(model), True),
        'Eigenvalue Shrinkage': (lambda: AnalysisEnKFDirectPrecisionShrinkageEigenvalues(model), True),
        'Cosmological Shrinkage': (make_cosmo, True),
        'NERCOME': (lambda: AnalysisEnKFNercomeShrinkage(model, max_draws=200), False),
        'Ledoit-Wolf': (lambda: AnalysisEnKFLedoitWolfShrinkage(model), False),
    }


# ── Experiment runner ─────────────────────────────────────────────────────────

def run_single(model, ensemble_size, analysis, seed):
    """Run one DA experiment, return (err_b, err_a) arrays."""
    true_ic = model.get_initial_condition()
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

    model.get_initial_condition = original_get_ic
    return sim.get_errors()


def save_csv(results, methods):
    """Save results to CSV."""
    path = os.path.join(FIGURES_DIR, 'cosmo_enkf_table.csv')
    rows = []
    for name in methods:
        row = {'Method': name}
        for Ne in ENSEMBLE_SIZES:
            runs = results.get(name, {}).get(Ne, [])
            if runs:
                avg_per_run = [np.mean(r[TRANSIENT:]) for r in runs]
                mean_val = np.mean(avg_per_run) * 100
                std_val = np.std(avg_per_run) * 100
                row[f'Ne={Ne}'] = f"{mean_val:.4f} +/- {std_val:.4f}"
            else:
                row[f'Ne={Ne}'] = 'N/A'
        rows.append(row)

    with open(path, 'w', newline='') as f:
        fieldnames = ['Method'] + [f'Ne={Ne}' for Ne in ENSEMBLE_SIZES]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved CSV: {path}")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_rmse_vs_ne(results, method_names):
    """Plot time-averaged RMSE vs ensemble size."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for name in method_names:
        means, stds = [], []
        for Ne in ENSEMBLE_SIZES:
            runs = results.get(name, {}).get(Ne, [])
            if runs:
                avg_per_run = [np.mean(r[TRANSIENT:]) for r in runs]
                means.append(np.mean(avg_per_run))
                stds.append(np.std(avg_per_run))
            else:
                means.append(np.nan)
                stds.append(0)

        ax.errorbar(ENSEMBLE_SIZES, means, yerr=stds,
                     label=name, color=COLORS.get(name, '#333'),
                     linestyle=LINESTYLES.get(name, '-'),
                     marker='o', markersize=4, capsize=3, linewidth=1.5)

    ax.set_xlabel('Ensemble size $N_e$', fontsize=11)
    ax.set_ylabel('Time-averaged relative RMSE', fontsize=11)
    ax.set_title('EnKF with AR(1) Cosmological Model (BOSS DR12)', fontsize=12)
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.set_xticks(ENSEMBLE_SIZES)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'cosmo_enkf_rmse.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print(f"  Saved figure: {path}")
    plt.close(fig)


def plot_rmse_time(results, method_names, Ne_plot=40):
    """Plot RMSE over assimilation cycles for a single Ne."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for name in method_names:
        runs = results.get(name, {}).get(Ne_plot, [])
        if not runs:
            continue
        # Average across runs
        min_len = min(len(r) for r in runs)
        arr = np.array([r[:min_len] for r in runs])
        mean_rmse = np.mean(arr, axis=0)
        cycles = np.arange(len(mean_rmse))
        ax.plot(cycles, mean_rmse, label=name,
                color=COLORS.get(name, '#333'),
                linestyle=LINESTYLES.get(name, '-'), linewidth=1.2)

    ax.set_xlabel('Assimilation cycle', fontsize=11)
    ax.set_ylabel('Relative RMSE', fontsize=11)
    ax.set_title(f'RMSE over time — AR(1) Cosmological Model ($N_e={Ne_plot}$)', fontsize=12)
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'cosmo_enkf_rmse_time.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print(f"  Saved figure: {path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Part B: TEDA EnKF with AR(1) Cosmological Model")
    print("=" * 60)

    mean_spectrum, cov_mock = load_mock_statistics()
    T = load_target_covariance()
    model = create_model(mean_spectrum, cov_mock)

    methods = get_methods(model, T)
    method_names = list(methods.keys())

    # results[method][Ne] = list of err_a arrays
    results = {name: {Ne: [] for Ne in ENSEMBLE_SIZES} for name in method_names}

    total = len(method_names) * len(ENSEMBLE_SIZES) * N_RUNS
    count = 0

    for method_name, (constructor, needs_invertible) in methods.items():
        print(f"\n--- {method_name} ---")
        for Ne in ENSEMBLE_SIZES:
            if needs_invertible and Ne <= P_DIM + 2:
                print(f"  Ne={Ne}: skipped (need Ne > d+2={P_DIM+2})")
                continue

            for run in range(N_RUNS):
                seed = RANDOM_SEED_BASE + run
                analysis = constructor()
                try:
                    err_b, err_a = run_single(model, Ne, analysis, seed)
                    results[method_name][Ne].append(err_a)
                except Exception as e:
                    print(f"  Ne={Ne} run {run}: FAILED - {e}")

                count += 1
                if count % 10 == 0:
                    print(f"  Progress: {count}/{total}")

        # Checkpoint after each method
        save_csv(results, method_names)

    # Final plots
    plot_rmse_vs_ne(results, method_names)
    plot_rmse_time(results, method_names)
    print("\nDone.")


if __name__ == '__main__':
    os.makedirs(FIGURES_DIR, exist_ok=True)
    main()
