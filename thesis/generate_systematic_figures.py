# -*- coding: utf-8 -*-
"""
Generate systematic experimental figures for the thesis.

This produces a comprehensive set of figures covering:
  1. RMSE time series at ALL ensemble sizes for Lorenz96 (Ne=50,60,80,100)
  2. RMSE time series for cosmological AR(1) at ALL ensemble sizes (Ne=24,30,40,50)
  3. Per-method RMSE bar charts for each domain
  4. Method-by-method precision matrix comparison at multiple Ne
  5. RMSE distribution box plots across runs

Usage:
    cd d:/thesis
    python thesis/generate_systematic_figures.py

WARNING: Takes 60-90+ minutes due to NERCOME.
         To skip NERCOME, set SKIP_NERCOME=True below.
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
from pyteda.models.ar1_power_spectrum import AR1PowerSpectrum
from pyteda.background.background_core import Background
from pyteda.observation.observation_core import Observation
from pyteda.simulation.simulation_core import Simulation

from pyteda.analysis.analysis_enkf import AnalysisEnKF
from pyteda.analysis.analysis_enkf_modified_cholesky import AnalysisEnKFModifiedCholesky
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_identity import AnalysisEnKFDirectPrecisionShrinkageIdentity
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_identity_scaled import AnalysisEnKFDirectPrecisionShrinkageIdentityScaled
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_eigenvalues import AnalysisEnKFDirectPrecisionShrinkageEigenvalues
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_identity_scaled_cosmo import AnalysisEnKFDirectPrecisionShrinkageIdentityScaledCosmo
from pyteda.analysis.analysis_enkf_ledoit_wolf_shrinkage import AnalysisEnKFLedoitWolfShrinkage
from pyteda.analysis.analysis_enkf_nercome_shrinkage import AnalysisEnKFNercomeShrinkage

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.figsize': (8, 5),
})

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'paper', 'data', 'looijmans')
SKIP_NERCOME = True  # Set False to include NERCOME (very slow)
RANDOM_SEED = 42

COLORS = {
    'EnKF': '#333333',
    'EnKF-Cholesky': '#8c564b',
    'Identity Shrinkage': '#d62728',
    'Scaled Shrinkage': '#ff7f0e',
    'Eigenvalue Shrinkage': '#9467bd',
    'Cosmological Shrinkage': '#2ca02c',
    'Ledoit-Wolf': '#1f77b4',
    'NERCOME': '#e377c2',
}
LINESTYLES = {
    'EnKF': '--',
    'EnKF-Cholesky': '--',
    'Identity Shrinkage': '-',
    'Scaled Shrinkage': '-',
    'Eigenvalue Shrinkage': '-',
    'Cosmological Shrinkage': '-',
    'Ledoit-Wolf': '-.',
    'NERCOME': ':',
}


def run_lorenz_single(Ne, analysis, seed=RANDOM_SEED):
    model = Lorenz96(n=40, F=8)
    true_ic = model.get_initial_condition()
    model.get_initial_condition = lambda: true_ic
    np.random.seed(seed)
    background = Background(model, ensemble_size=Ne)
    observation = Observation(m=32, std_obs=0.01)
    params = {'obs_freq': 0.1, 'end_time': 10, 'inf_fact': 1.05}
    sim = Simulation(model, background, analysis, observation, params=params, log_level=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.run()
    _, err_a = sim.get_errors()
    return err_a


def get_lorenz_methods():
    model = Lorenz96(n=40, F=8)
    methods = {
        'EnKF': lambda: AnalysisEnKF(),
        'EnKF-Cholesky': lambda: AnalysisEnKFModifiedCholesky(model, r=2),
        'Identity Shrinkage': lambda: AnalysisEnKFDirectPrecisionShrinkageIdentity(model),
        'Scaled Shrinkage': lambda: AnalysisEnKFDirectPrecisionShrinkageIdentityScaled(model),
        'Eigenvalue Shrinkage': lambda: AnalysisEnKFDirectPrecisionShrinkageEigenvalues(model),
        'Ledoit-Wolf': lambda: AnalysisEnKFLedoitWolfShrinkage(model),
    }
    if not SKIP_NERCOME:
        methods['NERCOME'] = lambda: AnalysisEnKFNercomeShrinkage(model, max_draws=1000)
    return methods


def run_cosmo_single(Ne, analysis, seed=RANDOM_SEED):
    mocks_dir = os.path.join(DATA_DIR, 'output', 'BOSS_DR12_NGC_z1', 'mocks')
    target_file = os.path.join(DATA_DIR, 'output', 'BOSS_DR12_NGC_z1', 'T_18_18.matrix')
    model = AR1PowerSpectrum(mocks_dir=mocks_dir, target_file=target_file, phi=0.95)
    true_ic = model.get_initial_condition()
    model.get_initial_condition = lambda: true_ic
    np.random.seed(seed)
    background = Background(model, ensemble_size=Ne)
    observation = Observation(m=14, std_obs=100.0)
    params = {'obs_freq': 0.1, 'end_time': 10, 'inf_fact': 1.02}
    sim = Simulation(model, background, analysis, observation, params=params, log_level=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.run()
    _, err_a = sim.get_errors()
    return err_a


def get_cosmo_methods():
    mocks_dir = os.path.join(DATA_DIR, 'output', 'BOSS_DR12_NGC_z1', 'mocks')
    target_file = os.path.join(DATA_DIR, 'output', 'BOSS_DR12_NGC_z1', 'T_18_18.matrix')
    model = AR1PowerSpectrum(mocks_dir=mocks_dir, target_file=target_file, phi=0.95)
    methods = {
        'EnKF': lambda: AnalysisEnKF(),
        'EnKF-Cholesky': lambda: AnalysisEnKFModifiedCholesky(model, r=2),
        'Identity Shrinkage': lambda: AnalysisEnKFDirectPrecisionShrinkageIdentity(model),
        'Scaled Shrinkage': lambda: AnalysisEnKFDirectPrecisionShrinkageIdentityScaled(model),
        'Eigenvalue Shrinkage': lambda: AnalysisEnKFDirectPrecisionShrinkageEigenvalues(model),
        'Cosmological Shrinkage': lambda: AnalysisEnKFDirectPrecisionShrinkageIdentityScaledCosmo(model),
        'Ledoit-Wolf': lambda: AnalysisEnKFLedoitWolfShrinkage(model),
    }
    if not SKIP_NERCOME:
        methods['NERCOME'] = lambda: AnalysisEnKFNercomeShrinkage(model, max_draws=1000)
    return methods


# ── Lorenz96 RMSE time series at Ne=80 ───────────────────────────────────────

def generate_lorenz_rmse_ne80():
    print("\n=== Lorenz96 RMSE time series at Ne=80 ===")
    methods = get_lorenz_methods()
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, constructor in methods.items():
        print(f"  Running {name}...")
        err_a = run_lorenz_single(80, constructor())
        ax.plot(err_a, label=name, color=COLORS[name],
                linestyle=LINESTYLES[name], linewidth=1.2)
    ax.set_xlabel('Assimilation cycle')
    ax.set_ylabel('Relative RMSE')
    ax.set_title(r'RMSE over 100 assimilation cycles (Lorenz 96, $N_e = 80$)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'rmse_time_ne80.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("  Saved rmse_time_ne80.pdf")


# ── Cosmo RMSE time series at each Ne ────────────────────────────────────────

def generate_cosmo_rmse_time(Ne):
    print(f"\n=== Cosmo RMSE time series at Ne={Ne} ===")
    methods = get_cosmo_methods()
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, constructor in methods.items():
        print(f"  Running {name}...")
        err_a = run_cosmo_single(Ne, constructor())
        ax.plot(err_a, label=name, color=COLORS[name],
                linestyle=LINESTYLES[name], linewidth=1.2)
    ax.set_xlabel('Assimilation cycle')
    ax.set_ylabel('Relative RMSE')
    ax.set_title(f'RMSE over 100 assimilation cycles (AR(1) Cosmological, $N_e = {Ne}$)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f'cosmo_rmse_time_ne{Ne}.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved cosmo_rmse_time_ne{Ne}.pdf")


# ── Box plots of RMSE distribution across runs (Lorenz96) ────────────────────

def generate_lorenz_boxplot():
    print("\n=== Lorenz96 RMSE box plots (10 runs, Ne=80) ===")
    methods = get_lorenz_methods()
    Ne = 80
    n_runs = 10
    transient = 50

    data = {}
    for name, constructor in methods.items():
        print(f"  Running {name} ({n_runs} runs)...")
        rmses = []
        for run in range(n_runs):
            err_a = run_lorenz_single(Ne, constructor(), seed=RANDOM_SEED + run)
            rmses.append(np.mean(err_a[transient:]) * 100)
        data[name] = rmses

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = list(data.keys())
    box_data = [data[k] for k in labels]
    bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True, widths=0.6)
    for patch, name in zip(bp['boxes'], labels):
        patch.set_facecolor(COLORS.get(name, '#999'))
        patch.set_alpha(0.6)
    ax.set_ylabel(r'Time-averaged RMSE ($\times 10^{-2}$)')
    ax.set_title(r'RMSE distribution across 10 runs (Lorenz 96, $N_e = 80$)')
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'lorenz_rmse_boxplot.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("  Saved lorenz_rmse_boxplot.pdf")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 60)
    print("Generating systematic thesis figures")
    print(f"SKIP_NERCOME = {SKIP_NERCOME}")
    print("=" * 60)

    # Lorenz96
    generate_lorenz_rmse_ne80()
    generate_lorenz_boxplot()

    # Cosmological
    for Ne in [24, 30, 40, 50]:
        generate_cosmo_rmse_time(Ne)

    print("\n" + "=" * 60)
    print("All done!")
    print("=" * 60)
    print("\nGenerated figures:")
    print("  - rmse_time_ne80.pdf")
    print("  - lorenz_rmse_boxplot.pdf")
    for Ne in [24, 30, 40, 50]:
        print(f"  - cosmo_rmse_time_ne{Ne}.pdf")
