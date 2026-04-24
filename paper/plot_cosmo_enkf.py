# -*- coding: utf-8 -*-
"""
Generate Part B plots from cosmo_enkf_table.csv (no experiments, just plotting).

Usage:
    cd d:/thesis
    python paper/plot_cosmo_enkf.py
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
CSV_PATH = os.path.join(FIGURES_DIR, 'cosmo_enkf_table.csv')
ENSEMBLE_SIZES = [24, 30, 40, 50]

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

# Desired display order (top methods last so they draw on top)
ORDER = [
    'EnKF',
    'Eigenvalue Shrinkage',
    'Cosmological Shrinkage',
    'EnKF-Cholesky',
    'Scaled Shrinkage',
    'Identity Shrinkage',
    'NERCOME',
    'Ledoit-Wolf',
]


def load_data():
    """Load CSV and parse into {method: {Ne: (mean, std)}}."""
    data = {}
    with open(CSV_PATH, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['Method']
            data[name] = {}
            for Ne in ENSEMBLE_SIZES:
                cell = row.get(f'Ne={Ne}', 'N/A')
                if cell and cell != 'N/A':
                    mean_s, std_s = cell.split('+/-')
                    data[name][Ne] = (float(mean_s.strip()), float(std_s.strip()))
    return data


def plot_rmse_vs_ne(data):
    """Plot time-averaged RMSE vs ensemble size."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for name in ORDER:
        if name not in data:
            continue
        means, stds, nes = [], [], []
        for Ne in ENSEMBLE_SIZES:
            if Ne in data[name]:
                m, s = data[name][Ne]
                means.append(m / 100)  # Convert back from ×10^-2
                stds.append(s / 100)
                nes.append(Ne)

        ax.errorbar(nes, means, yerr=stds,
                     label=name, color=COLORS.get(name, '#333'),
                     linestyle=LINESTYLES.get(name, '-'),
                     marker='o', markersize=4, capsize=3, linewidth=1.5)

    ax.set_xlabel('Ensemble size $N_e$', fontsize=11)
    ax.set_ylabel('Time-averaged relative RMSE', fontsize=11)
    ax.set_xticks(ENSEMBLE_SIZES)
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'cosmo_enkf_rmse.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


def plot_bar_comparison(data):
    """Bar chart comparing all methods at Ne=30 (best separation)."""
    Ne_plot = 30
    fig, ax = plt.subplots(figsize=(7, 4))

    names = [n for n in ORDER if n in data and Ne_plot in data[n]]
    means = [data[n][Ne_plot][0] for n in names]
    stds = [data[n][Ne_plot][1] for n in names]
    colors = [COLORS.get(n, '#333') for n in names]

    bars = ax.barh(range(len(names)), means, xerr=stds,
                    color=colors, capsize=3, alpha=0.85, height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Time-averaged relative RMSE ($\\times 10^{-2}$)', fontsize=10)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'cosmo_enkf_bar.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


if __name__ == '__main__':
    data = load_data()
    print(f"Loaded {len(data)} methods from {CSV_PATH}")
    plot_rmse_vs_ne(data)
    plot_bar_comparison(data)
    print("Done.")
