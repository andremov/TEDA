# -*- coding: utf-8 -*-
"""
Time one representative run of each method on both domains.
Runs a single seed at one ensemble size per domain.

Usage:
    cd d:/thesis
    python paper/compute_timing.py
"""

import sys
import os
import time
import warnings
import logging
import io
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
logging.getLogger().setLevel(logging.WARNING)

import numpy as np

from pyteda.models.lorenz96 import Lorenz96
from pyteda.background.background_core import Background
from pyteda.observation.observation_core import Observation
from pyteda.simulation.simulation_core import Simulation

# Cosmo imports
from pyteda.models.ar1_power_spectrum import AR1PowerSpectrum
from cosmo_enkf_common import setup_model, setup_cosmological_target

# Analysis methods
from pyteda.analysis.analysis_enkf import AnalysisEnKF
from pyteda.analysis.analysis_enkf_modified_cholesky import AnalysisEnKFModifiedCholesky
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_identity import AnalysisEnKFDirectPrecisionShrinkageIdentity
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_identity_scaled import AnalysisEnKFDirectPrecisionShrinkageIdentityScaled
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_eigenvalues import AnalysisEnKFDirectPrecisionShrinkageEigenvalues
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_identity_scaled_cosmo import AnalysisEnKFDirectPrecisionShrinkageIdentityScaledCosmo
from pyteda.analysis.analysis_enkf_nercome_shrinkage import AnalysisEnKFNercomeShrinkage
from pyteda.analysis.analysis_enkf_ledoit_wolf_shrinkage import AnalysisEnKFLedoitWolfShrinkage

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
SEED = 42
N_REPEATS = 3  # average over a few runs for stability


def time_method(run_fn, n_repeats=N_REPEATS):
    """Time a function over n_repeats, return mean seconds."""
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        run_fn()
        times.append(time.perf_counter() - t0)
    return np.mean(times), np.std(times)


def run_lorenz_experiment(model, Ne, analysis):
    """Run one Lorenz96 DA experiment."""
    true_ic = model.get_initial_condition()
    original_get_ic = model.get_initial_condition
    model.get_initial_condition = lambda: true_ic
    np.random.seed(SEED)
    background = Background(model, ensemble_size=Ne)
    observation = Observation(m=32, std_obs=0.01)
    params = {'obs_freq': 0.1, 'end_time': 10, 'inf_fact': 1.05}
    sim = Simulation(model, background, analysis, observation, params=params, log_level=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.run()
    model.get_initial_condition = original_get_ic


def run_cosmo_experiment(model, Ne, analysis):
    """Run one cosmo AR(1) DA experiment."""
    true_ic = model.get_initial_condition()
    original_get_ic = model.get_initial_condition
    model.get_initial_condition = lambda: true_ic
    np.random.seed(SEED)
    background = Background(model, ensemble_size=Ne)
    observation = Observation(m=14, std_obs=100.0)
    params = {'obs_freq': 0.1, 'end_time': 10, 'inf_fact': 1.02}
    sim = Simulation(model, background, analysis, observation, params=params, log_level=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.run()
    model.get_initial_condition = original_get_ic


def main():
    results = {}

    # --- Lorenz96 (Ne=80, n=40) ---
    print("=" * 60)
    print("LORENZ96 TIMING (Ne=80, n=40, 100 cycles)")
    print("=" * 60)
    lorenz_model = Lorenz96(n=40, F=8)
    Ne_lorenz = 80

    lorenz_methods = {
        'EnKF': lambda: AnalysisEnKF(),
        'EnKF-Cholesky': lambda: AnalysisEnKFModifiedCholesky(lorenz_model),
        'Identity Shrinkage': lambda: AnalysisEnKFDirectPrecisionShrinkageIdentity(lorenz_model),
        'Scaled Shrinkage': lambda: AnalysisEnKFDirectPrecisionShrinkageIdentityScaled(lorenz_model),
        'Eigenvalue Shrinkage': lambda: AnalysisEnKFDirectPrecisionShrinkageEigenvalues(lorenz_model),
        'NERCOME': lambda: AnalysisEnKFNercomeShrinkage(lorenz_model, max_draws=200),
        'Ledoit-Wolf': lambda: AnalysisEnKFLedoitWolfShrinkage(lorenz_model),
    }

    for name, make_analysis in lorenz_methods.items():
        print(f"  Timing {name}...", end=' ', flush=True)
        mean_t, std_t = time_method(
            lambda ma=make_analysis: run_lorenz_experiment(lorenz_model, Ne_lorenz, ma())
        )
        results[('Lorenz96', name)] = (mean_t, std_t)
        print(f"{mean_t:.2f} +/- {std_t:.2f} s")

    # --- Cosmo AR(1) (Ne=30, d=18) ---
    print("\n" + "=" * 60)
    print("COSMO AR(1) TIMING (Ne=30, d=18, 100 cycles)")
    print("=" * 60)
    cosmo_model, T = setup_model()
    Ne_cosmo = 30

    def make_cosmo_analysis():
        a = AnalysisEnKFDirectPrecisionShrinkageIdentityScaledCosmo(cosmo_model)
        setup_cosmological_target(a, T)
        return a

    cosmo_methods = {
        'EnKF': lambda: AnalysisEnKF(),
        'EnKF-Cholesky': lambda: AnalysisEnKFModifiedCholesky(cosmo_model),
        'Identity Shrinkage': lambda: AnalysisEnKFDirectPrecisionShrinkageIdentity(cosmo_model),
        'Scaled Shrinkage': lambda: AnalysisEnKFDirectPrecisionShrinkageIdentityScaled(cosmo_model),
        'Eigenvalue Shrinkage': lambda: AnalysisEnKFDirectPrecisionShrinkageEigenvalues(cosmo_model),
        'Cosmological Shrinkage': make_cosmo_analysis,
        'NERCOME': lambda: AnalysisEnKFNercomeShrinkage(cosmo_model, max_draws=200),
        'Ledoit-Wolf': lambda: AnalysisEnKFLedoitWolfShrinkage(cosmo_model),
    }

    for name, make_analysis in cosmo_methods.items():
        print(f"  Timing {name}...", end=' ', flush=True)
        mean_t, std_t = time_method(
            lambda ma=make_analysis: run_cosmo_experiment(cosmo_model, Ne_cosmo, ma())
        )
        results[('Cosmo', name)] = (mean_t, std_t)
        print(f"{mean_t:.2f} +/- {std_t:.2f} s")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY: Relative cost (normalized to EnKF)")
    print("=" * 60)
    for domain in ['Lorenz96', 'Cosmo']:
        enkf_time = results.get((domain, 'EnKF'), (1, 0))[0]
        print(f"\n  {domain}:")
        print(f"  {'Method':<25s} {'Time (s)':>10s} {'Relative':>10s}")
        print(f"  {'-'*47}")
        for name in ['EnKF', 'EnKF-Cholesky', 'Identity Shrinkage', 'Scaled Shrinkage',
                      'Eigenvalue Shrinkage', 'Cosmological Shrinkage', 'NERCOME', 'Ledoit-Wolf']:
            key = (domain, name)
            if key in results:
                t, _ = results[key]
                print(f"  {name:<25s} {t:10.2f} {t/enkf_time:10.2f}x")

    # Save to CSV
    csv_path = os.path.join(FIGURES_DIR, 'timing_data.csv')
    with open(csv_path, 'w', newline='') as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(['Domain', 'Method', 'Time_s', 'Std_s'])
        for (domain, name), (t, s) in sorted(results.items()):
            writer.writerow([domain, name, f"{t:.4f}", f"{s:.4f}"])
    print(f"\nSaved {csv_path}")


if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    main()
