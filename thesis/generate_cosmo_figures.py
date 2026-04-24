# -*- coding: utf-8 -*-
"""
Generate cosmological RMSE time series and box plots for the thesis.

Run this separately from the Lorenz96 scripts for parallelism.

Usage:
    cd d:/thesis
    python thesis/generate_cosmo_figures.py
"""
import sys, os, warnings, logging
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

from pyteda.analysis.analysis_enkf import AnalysisEnKF
from pyteda.analysis.analysis_enkf_modified_cholesky import AnalysisEnKFModifiedCholesky
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_identity import AnalysisEnKFDirectPrecisionShrinkageIdentity
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_identity_scaled import AnalysisEnKFDirectPrecisionShrinkageIdentityScaled
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_eigenvalues import AnalysisEnKFDirectPrecisionShrinkageEigenvalues
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_identity_scaled_cosmo import AnalysisEnKFDirectPrecisionShrinkageIdentityScaledCosmo
from pyteda.analysis.analysis_enkf_ledoit_wolf_shrinkage import AnalysisEnKFLedoitWolfShrinkage

plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'axes.labelsize': 12, 'legend.fontsize': 9})

FIG = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'paper', 'data', 'looijmans')
MOCKS_DIR = os.path.join(DATA_DIR, 'output', 'BOSS_DR12_NGC_z1', 'mocks')
TARGET_FILE = os.path.join(DATA_DIR, 'output', 'BOSS_DR12_NGC_z1', 'T_18_18.matrix')
SEED = 42
TRANSIENT = 50
PHI = 0.95

COLORS = {
    'EnKF': '#333333', 'EnKF-Cholesky': '#8c564b',
    'Identity Shrinkage': '#d62728', 'Scaled Shrinkage': '#ff7f0e',
    'Eigenvalue Shrinkage': '#9467bd', 'Cosmological Shrinkage': '#2ca02c',
    'Ledoit-Wolf': '#1f77b4',
}
LS = {
    'EnKF': '--', 'EnKF-Cholesky': '--',
    'Identity Shrinkage': '-', 'Scaled Shrinkage': '-',
    'Eigenvalue Shrinkage': '-', 'Cosmological Shrinkage': '-',
    'Ledoit-Wolf': '-.',
}


def load_data():
    P_all = np.loadtxt(os.path.join(MOCKS_DIR, 'n2048', 'P_18_2048_v1.matrix'))
    mean_spectrum = np.mean(P_all, axis=1)
    cov_mock = np.cov(P_all)
    T = np.loadtxt(TARGET_FILE)
    return mean_spectrum, cov_mock, T


def create_model(mean_spectrum, cov_mock):
    model = AR1PowerSpectrum(mean_spectrum, cov_mock, phi=PHI)
    model.create_decorrelation_matrix(r=2)
    return model


def get_methods(model, T):
    T_diag = np.diag(T)
    methods = {
        'EnKF': lambda: AnalysisEnKF(),
        'EnKF-Cholesky': lambda: AnalysisEnKFModifiedCholesky(model, r=2),
        'Identity Shrinkage': lambda: AnalysisEnKFDirectPrecisionShrinkageIdentity(model),
        'Scaled Shrinkage': lambda: AnalysisEnKFDirectPrecisionShrinkageIdentityScaled(model),
        'Eigenvalue Shrinkage': lambda: AnalysisEnKFDirectPrecisionShrinkageEigenvalues(model),
        'Cosmological Shrinkage': lambda: AnalysisEnKFDirectPrecisionShrinkageIdentityScaledCosmo(model),
        'Ledoit-Wolf': lambda: AnalysisEnKFLedoitWolfShrinkage(model),
    }
    return methods


def run_cosmo(model, Ne, analysis, seed=SEED):
    ic = model.get_initial_condition()
    model.get_initial_condition = lambda: ic
    np.random.seed(seed)
    bg = Background(model, ensemble_size=Ne)
    obs = Observation(m=14, std_obs=100.0)
    sim = Simulation(model, bg, analysis, obs,
                     params={'obs_freq': 0.1, 'end_time': 10, 'inf_fact': 1.02},
                     log_level=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.run()
    _, ea = sim.get_errors()
    return ea


if __name__ == '__main__':
    os.makedirs(FIG, exist_ok=True)
    mean_spectrum, cov_mock, T = load_data()

    # RMSE time series at each Ne
    for Ne in [24, 30, 40, 50]:
        print(f"\n=== Cosmo RMSE time series at Ne={Ne} ===")
        model = create_model(mean_spectrum, cov_mock)
        methods = get_methods(model, T)
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, ctor in methods.items():
            print(f"  {name}...")
            ea = run_cosmo(model, Ne, ctor())
            ax.plot(ea, label=name, color=COLORS[name], linestyle=LS[name], lw=1.2)
        ax.set_xlabel('Assimilation cycle')
        ax.set_ylabel('Relative RMSE')
        ax.set_title(f'RMSE over 100 cycles (AR(1) Cosmological, $N_e = {Ne}$)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f'cosmo_rmse_time_ne{Ne}.pdf'), bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved cosmo_rmse_time_ne{Ne}.pdf")

    # Box plots at Ne=30
    print(f"\n=== Cosmo box plots at Ne=30 (10 runs) ===")
    model = create_model(mean_spectrum, cov_mock)
    methods = get_methods(model, T)
    data = {}
    for name, ctor in methods.items():
        print(f"  {name} (10 runs)...")
        rmses = []
        for r in range(10):
            ea = run_cosmo(model, 30, ctor(), seed=SEED + r)
            rmses.append(np.mean(ea[TRANSIENT:]) * 100)
        data[name] = rmses

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = list(data.keys())
    bp = ax.boxplot([data[k] for k in labels], tick_labels=labels, patch_artist=True, widths=0.6)
    for patch, name in zip(bp['boxes'], labels):
        patch.set_facecolor(COLORS.get(name, '#999'))
        patch.set_alpha(0.6)
    ax.set_ylabel(r'Time-averaged RMSE ($\times 10^{-2}$)')
    ax.set_title(r'RMSE distribution across 10 runs (AR(1) Cosmological, $N_e = 30$)')
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'cosmo_rmse_boxplot.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("  Saved cosmo_rmse_boxplot.pdf")

    print("\nDone! Generated 5 figures.")
