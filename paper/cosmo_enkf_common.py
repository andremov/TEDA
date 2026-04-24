# -*- coding: utf-8 -*-
"""
Shared setup for Part B cosmological EnKF per-method scripts.
"""

import sys
import os
import csv
import warnings
import logging
import io
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
logging.getLogger().setLevel(logging.WARNING)

import numpy as np

from pyteda.models.ar1_power_spectrum import AR1PowerSpectrum
from pyteda.background.background_core import Background
from pyteda.observation.observation_core import Observation
from pyteda.simulation.simulation_core import Simulation

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'looijmans')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
MOCKS_DIR = os.path.join(DATA_DIR, 'output', 'BOSS_DR12_NGC_z1', 'mocks')
TARGET_FILE = os.path.join(DATA_DIR, 'output', 'BOSS_DR12_NGC_z1', 'T_18_18.matrix')
CSV_PATH = os.path.join(FIGURES_DIR, 'cosmo_enkf_table.csv')

P_DIM = 18
N_OBS = 14
STD_OBS = 100.0
OBS_FREQ = 0.1
END_TIME = 10.0
INF_FACT = 1.02
PHI = 0.95
ENSEMBLE_SIZES = [24, 30, 40, 50]
N_RUNS = 30
TRANSIENT = 50
RANDOM_SEED_BASE = 42


def setup_model():
    """Load mock data, create AR(1) model, return (model, T)."""
    P_all = np.loadtxt(os.path.join(MOCKS_DIR, 'n2048', 'P_18_2048_v1.matrix'))
    mean_spectrum = np.mean(P_all, axis=1)
    cov_mock = np.cov(P_all)
    model = AR1PowerSpectrum(mean_spectrum, cov_mock, phi=PHI)
    model.create_decorrelation_matrix(r=2)
    T = np.loadtxt(TARGET_FILE)
    print(f"Model: d={P_DIM}, mean range [{mean_spectrum.min():.1f}, {mean_spectrum.max():.1f}]")
    return model, T


def setup_cosmological_target(analysis_cosmo, T):
    """Configure cosmological target from T matrix."""
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
        power_spectrum[ell] = np.sqrt(T_diag[i])
        multipole_sampling[ell] = 2
        data_vector_mapping.append((ell, 0))

    with redirect_stdout(io.StringIO()):
        analysis_cosmo.set_cosmological_parameters(
            power_spectrum=power_spectrum,
            survey_geometry=survey_geometry,
            multipole_sampling=multipole_sampling,
            data_vector_mapping=data_vector_mapping,
        )


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


def load_csv():
    """Load existing cosmo_enkf_table.csv into dict."""
    rows = {}
    if not os.path.exists(CSV_PATH):
        return rows
    with open(CSV_PATH, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row['Method']] = dict(row)
    return rows


def save_csv(rows):
    """Save rows dict back to CSV."""
    fieldnames = ['Method'] + [f'Ne={Ne}' for Ne in ENSEMBLE_SIZES]
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows.values():
            writer.writerow(row)
    print(f"  Saved {CSV_PATH}")


def run_method(method_name, constructor, needs_invertible=False):
    """Run a single method across all ensemble sizes and merge into CSV."""
    print(f"=== {method_name} ===")
    model, T = setup_model()
    row = {'Method': method_name}

    for Ne in ENSEMBLE_SIZES:
        if needs_invertible and Ne <= P_DIM + 2:
            print(f"  Ne={Ne}: skipped (need Ne > {P_DIM + 2})")
            row[f'Ne={Ne}'] = 'N/A'
            continue

        err_a_runs = []
        for run_idx in range(N_RUNS):
            seed = RANDOM_SEED_BASE + run_idx
            print(f"  Ne={Ne}, run {run_idx+1}/{N_RUNS}")
            try:
                analysis = constructor(model, T)
                _, err_a = run_single(model, Ne, analysis, seed)
                err_a_runs.append(err_a)
            except Exception as e:
                print(f"    FAILED: {e}")

        if err_a_runs:
            avg_per_run = [np.mean(r[TRANSIENT:]) for r in err_a_runs]
            mean_val = np.mean(avg_per_run) * 100
            std_val = np.std(avg_per_run) * 100
            row[f'Ne={Ne}'] = f"{mean_val:.4f} +/- {std_val:.4f}"
            print(f"  Ne={Ne}: {mean_val:.4f} +/- {std_val:.4f}")
        else:
            row[f'Ne={Ne}'] = 'N/A'

    existing_rows = load_csv()
    existing_rows[method_name] = row
    save_csv(existing_rows)
    print(f"Done. {method_name} merged into {CSV_PATH}")
