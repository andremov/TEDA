"""Generate timing comparison figure from existing timing_data.csv.

Produces:
  - figures/timing_comparison.pdf: grouped bar chart comparing methods on both domains
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.figsize': (8, 5),
})

DATA = os.path.join('..', 'paper', 'figures', 'timing_data.csv')
OUT = os.path.join('..', 'paper', 'figures')

df = pd.read_csv(DATA)

# Separate domains
cosmo = df[df['Domain'] == 'Cosmo'].sort_values('Time_s')
lorenz = df[df['Domain'] == 'Lorenz96'].sort_values('Time_s')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- Cosmological domain ---
colors_map = {
    'EnKF': '#1f77b4', 'EnKF-Cholesky': '#ff7f0e',
    'Identity Shrinkage': '#d62728', 'Scaled Shrinkage': '#2ca02c',
    'Eigenvalue Shrinkage': '#9467bd', 'Cosmological Shrinkage': '#8c564b',
    'Ledoit-Wolf': '#e377c2', 'NERCOME': '#7f7f7f'
}

bars1 = ax1.barh(range(len(cosmo)), cosmo['Time_s'],
                 xerr=cosmo['Std_s'],
                 color=[colors_map.get(m, '#333') for m in cosmo['Method']],
                 capsize=3, edgecolor='white', linewidth=0.5)
ax1.set_yticks(range(len(cosmo)))
ax1.set_yticklabels(cosmo['Method'], fontsize=9)
ax1.set_xlabel('Time (seconds per 100 cycles)')
ax1.set_title(r'AR(1) Cosmological ($d=18$, $N_e=30$)')
ax1.set_xscale('log')
ax1.grid(True, alpha=0.3, axis='x')

# --- Lorenz96 domain ---
bars2 = ax2.barh(range(len(lorenz)), lorenz['Time_s'],
                 xerr=lorenz['Std_s'],
                 color=[colors_map.get(m, '#333') for m in lorenz['Method']],
                 capsize=3, edgecolor='white', linewidth=0.5)
ax2.set_yticks(range(len(lorenz)))
ax2.set_yticklabels(lorenz['Method'], fontsize=9)
ax2.set_xlabel('Time (seconds per 100 cycles)')
ax2.set_title(r'Lorenz 96 ($n=40$, $N_e=80$)')
ax2.set_xscale('log')
ax2.grid(True, alpha=0.3, axis='x')

fig.suptitle('Computational cost comparison', fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'timing_comparison.pdf'), bbox_inches='tight')
print('Saved timing_comparison.pdf')
plt.close('all')
