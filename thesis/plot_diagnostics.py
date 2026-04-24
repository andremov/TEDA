"""Generate diagnostic figures from existing diagnostics.csv data.

Produces:
  - figures/alpha_vs_ne.pdf: alpha* coefficient vs ensemble size
  - figures/beta_vs_ne.pdf: unclamped beta* vs ensemble size (log scale)
  - figures/condition_vs_ne.pdf: condition number vs ensemble size (log scale)
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.figsize': (7, 4.5),
})

DATA = os.path.join('..', 'paper', 'figures', 'diagnostics.csv')
OUT = os.path.join('..', 'paper', 'figures')

df = pd.read_csv(DATA)

colors = {'Identity': '#d62728', 'Scaled': '#2ca02c', 'Eigenvalue': '#1f77b4'}
markers = {'Identity': 'o', 'Scaled': 's', 'Eigenvalue': '^'}

# --- Figure 1: alpha* vs Ne ---
fig, ax = plt.subplots()
for method in ['Identity', 'Scaled', 'Eigenvalue']:
    sub = df[df['method'] == method]
    ax.plot(sub['Ne'], sub['alpha_uncl_mean'], marker=markers[method],
            color=colors[method], label=f'{method} target', linewidth=1.5, markersize=5)

# Theoretical bound: 1 - n/Ne
ne_range = np.arange(42, 105, 1)
ax.plot(ne_range, 1 - 40/ne_range, 'k--', alpha=0.5, label=r'$1 - n/N_e$ bound')

ax.set_xlabel(r'Ensemble size $N_e$')
ax.set_ylabel(r'$\alpha^*$')
ax.set_title(r'Shrinkage coefficient $\alpha^*$ vs ensemble size (Lorenz 96, $n=40$)')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'alpha_vs_ne.pdf'), bbox_inches='tight')
print('Saved alpha_vs_ne.pdf')

# --- Figure 2: unclamped beta* vs Ne (log scale) ---
fig, ax = plt.subplots()
for method in ['Identity', 'Scaled', 'Eigenvalue']:
    sub = df[df['method'] == method]
    vals = sub['beta_uncl_mean'].values
    # Use absolute value for log scale (eigenvalue has tiny positive values)
    ax.semilogy(sub['Ne'], np.abs(vals), marker=markers[method],
                color=colors[method], label=f'{method} target', linewidth=1.5, markersize=5)

ax.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Clamping threshold')
ax.set_xlabel(r'Ensemble size $N_e$')
ax.set_ylabel(r'$|\beta^*|$ (unclamped)')
ax.set_title(r'Unclamped $\beta^*$ vs ensemble size (Lorenz 96, $n=40$)')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'beta_vs_ne.pdf'), bbox_inches='tight')
print('Saved beta_vs_ne.pdf')

# --- Figure 3: condition number vs Ne (log scale) ---
fig, ax = plt.subplots()
for method in ['Identity', 'Scaled', 'Eigenvalue']:
    sub = df[df['method'] == method]
    ax.semilogy(sub['Ne'], sub['cond_Theta_mean'], marker=markers[method],
                color=colors[method], label=f'{method} target (mean)', linewidth=1.5, markersize=5)
    ax.fill_between(sub['Ne'], sub['cond_Theta_mean'], sub['cond_Theta_max'],
                     color=colors[method], alpha=0.15)

ax.set_xlabel(r'Ensemble size $N_e$')
ax.set_ylabel(r'Condition number of $\Pi_{\mathrm{LS}}$')
ax.set_title(r'Condition number of shrinkage estimator (Lorenz 96, $n=40$)')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'condition_vs_ne.pdf'), bbox_inches='tight')
print('Saved condition_vs_ne.pdf')

plt.close('all')
print('Done.')
