# -*- coding: utf-8 -*-
"""
Run ONLY Scaled Shrinkage on Lorenz96 and merge into table_data.csv.

Usage:
    cd d:/thesis
    python paper/run_lorenz_scaled.py
"""

import sys
import os
import csv
import warnings
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
logging.getLogger().setLevel(logging.WARNING)

import numpy as np

from pyteda.models.lorenz96 import Lorenz96
from pyteda.background.background_core import Background
from pyteda.observation.observation_core import Observation
from pyteda.simulation.simulation_core import Simulation
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_identity_scaled import AnalysisEnKFDirectPrecisionShrinkageIdentityScaled

N_VARIABLES = 40
FORCING = 8
N_OBS = 32
STD_OBS = 0.01
OBS_FREQ = 0.1
END_TIME = 10
INF_FACT = 1.05
ENSEMBLE_SIZES = [50, 60, 80, 100]
N_RUNS = 30
TRANSIENT = 50
RANDOM_SEED_BASE = 42
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
CSV_PATH = os.path.join(FIGURES_DIR, 'table_data.csv')

METHOD_NAME = 'Scaled Shrinkage'


def run_single(model, ensemble_size, analysis, seed):
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
    rows = {}
    if not os.path.exists(CSV_PATH):
        return rows
    with open(CSV_PATH, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row['Method']] = dict(row)
    return rows


def save_csv(rows):
    fieldnames = ['Method'] + [f'Ne={Ne}' for Ne in ENSEMBLE_SIZES]
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows.values():
            writer.writerow(row)
    print(f"  Saved {CSV_PATH}")


def main():
    print(f"=== {METHOD_NAME} ===")
    model = Lorenz96(n=N_VARIABLES, F=FORCING)
    row = {'Method': METHOD_NAME}

    for Ne in ENSEMBLE_SIZES:
        if Ne <= N_VARIABLES + 2:
            print(f"  Ne={Ne}: skipped (need Ne > {N_VARIABLES + 2})")
            row[f'Ne={Ne}'] = 'N/A'
            continue

        err_a_runs = []
        for run_idx in range(N_RUNS):
            seed = RANDOM_SEED_BASE + run_idx * 1000 + Ne
            print(f"  Ne={Ne}, run {run_idx+1}/{N_RUNS} (seed={seed})")
            try:
                analysis = AnalysisEnKFDirectPrecisionShrinkageIdentityScaled(model)
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
    existing_rows[METHOD_NAME] = row
    save_csv(existing_rows)
    print(f"\nDone. {METHOD_NAME} merged into {CSV_PATH}")


if __name__ == '__main__':
    os.makedirs(FIGURES_DIR, exist_ok=True)
    main()
