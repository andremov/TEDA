# -*- coding: utf-8 -*-
"""
Plot Lorenz96 results from individual method JSON files.

Run AFTER all run_lorenz_method.py jobs finish:
    python thesis/plot_lorenz_results.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'axes.labelsize': 12, 'legend.fontsize': 9})

FIG = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')
TRACE_DIR = os.path.join(FIG, 'lorenz_traces')
TRANSIENT = 50

METHOD_NAMES = {
    'EnKF': 'EnKF', 'Cholesky': 'EnKF-Cholesky',
    'Identity': 'Identity Shrinkage', 'Scaled': 'Scaled Shrinkage',
    'Eigenvalue': 'Eigenvalue Shrinkage', 'LedoitWolf': 'Ledoit-Wolf',
}
COLORS = {
    'EnKF': '#333333', 'EnKF-Cholesky': '#8c564b',
    'Identity Shrinkage': '#d62728', 'Scaled Shrinkage': '#ff7f0e',
    'Eigenvalue Shrinkage': '#9467bd', 'Ledoit-Wolf': '#1f77b4',
}
LS = {
    'EnKF': '--', 'EnKF-Cholesky': '--',
    'Identity Shrinkage': '-', 'Scaled Shrinkage': '-',
    'Eigenvalue Shrinkage': '-', 'Ledoit-Wolf': '-.',
}

def load_all():
    data = {}
    for key, display in METHOD_NAMES.items():
        path = os.path.join(TRACE_DIR, f'{key}.json')
        if os.path.exists(path):
            with open(path) as f:
                data[display] = json.load(f)
        else:
            print(f"  WARNING: {path} not found, skipping {display}")
    return data

if __name__ == '__main__':
    data = load_all()
    if not data:
        print("No data found. Run run_lorenz_method.py first.")
        exit(1)

    # RMSE time series (first run) at Ne=50, 60, 80
    for Ne in [50, 60, 80]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, ne_data in data.items():
            if str(Ne) in ne_data and len(ne_data[str(Ne)]) > 0:
                ax.plot(ne_data[str(Ne)][0], label=name, color=COLORS[name],
                        linestyle=LS[name], lw=1.2)
        ax.set_xlabel('Assimilation cycle')
        ax.set_ylabel('Relative RMSE')
        ax.set_title(f'RMSE over 100 cycles (Lorenz 96, $N_e = {Ne}$)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f'rmse_time_ne{Ne}.pdf'), bbox_inches='tight')
        plt.close(fig)
        print(f"Saved rmse_time_ne{Ne}.pdf")

    # Box plots at Ne=50, 80, 100
    for Ne in [50, 80, 100]:
        box_data = {}
        for name, ne_data in data.items():
            if str(Ne) in ne_data:
                rmses = [np.mean(trace[TRANSIENT:]) * 100 for trace in ne_data[str(Ne)]]
                box_data[name] = rmses
        if not box_data:
            continue

        suffix = f'_ne{Ne}' if Ne != 80 else ''
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = list(box_data.keys())
        bp = ax.boxplot([box_data[k] for k in labels], tick_labels=labels, patch_artist=True, widths=0.6)
        for patch, name in zip(bp['boxes'], labels):
            patch.set_facecolor(COLORS.get(name, '#999'))
            patch.set_alpha(0.6)
        ax.set_ylabel(r'Time-averaged RMSE ($\times 10^{-2}$)')
        ax.set_title(f'RMSE distribution across 10 runs (Lorenz 96, $N_e = {Ne}$)')
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f'lorenz_rmse_boxplot{suffix}.pdf'), bbox_inches='tight')
        plt.close(fig)
        print(f"Saved lorenz_rmse_boxplot{suffix}.pdf")

    print("\nDone!")
