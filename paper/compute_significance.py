# -*- coding: utf-8 -*-
"""
Compute pairwise Welch's t-tests from existing CSV results.
No experiments needed — uses mean, std, and n from the CSVs.

Usage:
    cd d:/thesis
    python paper/compute_significance.py
"""

import os
import csv
import numpy as np
from scipy import stats

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
N_RUNS = 30  # number of independent seeds used in all experiments


def load_csv(path):
    """Load CSV into {method: {key: (mean, std)}}."""
    data = {}
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['Method']
            data[name] = {}
            for key, val in row.items():
                if key == 'Method':
                    continue
                if val and val != 'N/A':
                    mean_s, std_s = val.split('+/-')
                    data[name][key] = (float(mean_s.strip()), float(std_s.strip()))
    return data


def welch_t_test(mean1, std1, n1, mean2, std2, n2):
    """Welch's t-test for difference in means. Returns (t_stat, p_value)."""
    se1 = std1 / np.sqrt(n1)
    se2 = std2 / np.sqrt(n2)
    se_diff = np.sqrt(se1**2 + se2**2)
    if se_diff == 0:
        return 0.0, 1.0
    t_stat = (mean1 - mean2) / se_diff
    # Welch-Satterthwaite degrees of freedom
    nu = (se1**2 + se2**2)**2 / (se1**4 / (n1 - 1) + se2**4 / (n2 - 1))
    p_value = 2 * stats.t.sf(abs(t_stat), df=nu)
    return t_stat, p_value


def compare_methods(data, reference, Ne_key, n_runs=N_RUNS):
    """Compare all methods against a reference method at a given Ne."""
    if reference not in data or Ne_key not in data[reference]:
        return
    ref_mean, ref_std = data[reference][Ne_key]
    print(f"\n  {Ne_key} — Reference: {reference} ({ref_mean:.4f} +/- {ref_std:.4f})")
    print(f"  {'Method':<25s} {'Mean':>8s} {'Std':>8s} {'t-stat':>8s} {'p-value':>10s} {'Sig?':>6s}")
    print(f"  {'-'*65}")
    for name in sorted(data.keys()):
        if name == reference or Ne_key not in data[name]:
            continue
        m, s = data[name][Ne_key]
        t, p = welch_t_test(ref_mean, ref_std, n_runs, m, s, n_runs)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {name:<25s} {m:8.4f} {s:8.4f} {t:8.3f} {p:10.4f} {sig:>6s}")


def main():
    # --- Lorenz96 ---
    lorenz_path = os.path.join(FIGURES_DIR, 'table_data.csv')
    print("=" * 70)
    print("LORENZ96: Pairwise comparisons vs Ledoit-Wolf")
    print("=" * 70)
    lorenz = load_csv(lorenz_path)
    for Ne_key in ['Ne=50', 'Ne=60', 'Ne=80', 'Ne=100']:
        compare_methods(lorenz, 'Ledoit-Wolf', Ne_key)

    # --- Cosmo EnKF ---
    cosmo_path = os.path.join(FIGURES_DIR, 'cosmo_enkf_table.csv')
    print("\n" + "=" * 70)
    print("COSMO EnKF (Part B): Pairwise comparisons vs Ledoit-Wolf")
    print("=" * 70)
    cosmo = load_csv(cosmo_path)
    for Ne_key in ['Ne=24', 'Ne=30', 'Ne=40', 'Ne=50']:
        compare_methods(cosmo, 'Ledoit-Wolf', Ne_key)


if __name__ == '__main__':
    main()
