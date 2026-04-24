# -*- coding: utf-8 -*-
"""
Run experiments for a single DA method and checkpoint results to table_data.csv.

Each method saves its results immediately on completion, so the script can be
run independently per method and interrupted/resumed safely.

Usage (from d:/thesis):
    python paper/run_method.py "Identity Shrinkage"
    python paper/run_method.py "Scaled Shrinkage"
    python paper/run_method.py "Eigenvalue Shrinkage"
    python paper/run_method.py "NERCOME"

After all methods are done, run generate_figures.py to produce the plots:
    python paper/generate_figures.py
"""

import sys
import os
import csv
import warnings
import logging

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.getLogger().setLevel(logging.WARNING)

import numpy as np

from generate_figures import (
    get_methods, run_single_experiment, load_existing_csv,
    ENSEMBLE_SIZES, N_RUNS, RANDOM_SEED_BASE, TRANSIENT,
    N_VARIABLES, FORCING, FIGURES_DIR,
)
from pyteda.models.lorenz96 import Lorenz96


def save_method_to_csv(method_name, run_lists):
    """Write one method's results into table_data.csv, preserving all other rows."""
    path = os.path.join(FIGURES_DIR, 'table_data.csv')

    # Compute fresh row for this method
    new_row = {'Method': method_name}
    for Ne in ENSEMBLE_SIZES:
        runs = run_lists.get(Ne, [])
        if not runs:
            new_row[f'Ne={Ne}'] = 'N/A'
        else:
            avg_per_run = [np.mean(r[TRANSIENT:]) for r in runs]
            mean_val = np.mean(avg_per_run) * 100
            std_val  = np.std(avg_per_run)  * 100
            new_row[f'Ne={Ne}'] = f"{mean_val:.4f} +/- {std_val:.4f}"

    # Load existing rows right before saving to avoid race conditions
    existing_rows = {}
    if os.path.exists(path):
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows[row['Method']] = dict(row)

    existing_rows[method_name] = new_row

    # Write all rows back (order: existing order, new method appended if absent)
    with open(path, 'w', newline='') as f:
        fieldnames = ['Method'] + [f'Ne={Ne}' for Ne in ENSEMBLE_SIZES]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing_rows.values():
            writer.writerow(row)

    print(f"  [saved] {path}")


def run_one_method(method_name):
    model = Lorenz96(n=N_VARIABLES, F=FORCING)
    methods = get_methods(model)

    if method_name not in methods:
        print(f"Unknown method: '{method_name}'")
        print(f"Available methods: {list(methods.keys())}")
        sys.exit(1)

    # Skip if already completed
    existing = load_existing_csv()
    if method_name in existing:
        all_done = all(
            existing[method_name].get(Ne, (0, 0))[1] > 0.0
            for Ne in ENSEMBLE_SIZES
        )
        if all_done:
            print(f"[skip] '{method_name}' already complete in CSV. Nothing to do.")
            return

    constructor, needs_inv = methods[method_name]
    run_lists = {Ne: [] for Ne in ENSEMBLE_SIZES}

    print(f"\n{'='*60}")
    print(f"Method: {method_name}")
    print(f"Config: Ne={ENSEMBLE_SIZES}, runs={N_RUNS}")
    print(f"{'='*60}")

    for Ne in ENSEMBLE_SIZES:
        if needs_inv and Ne <= N_VARIABLES + 2:
            print(f"  Skipping Ne={Ne} (needs Ne > {N_VARIABLES+2})")
            continue

        for run_idx in range(N_RUNS):
            seed = RANDOM_SEED_BASE + run_idx * 1000 + Ne
            print(f"  Ne={Ne}, run {run_idx+1}/{N_RUNS} (seed={seed})", flush=True)
            try:
                analysis = constructor()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, err_a = run_single_experiment(model, Ne, analysis, seed)
                run_lists[Ne].append(err_a)
            except Exception as e:
                print(f"  FAILED: {e}")

    save_method_to_csv(method_name, run_lists)
    print(f"\nDone: '{method_name}' saved to table_data.csv")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    run_one_method(sys.argv[1])
