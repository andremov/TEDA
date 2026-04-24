# Generates: paper/figures/lorenz_per_method_panels.pdf
# Uses existing Lorenz trace data (no experiments).
import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')
TRACE_DIR = os.path.join(FIG, 'lorenz_traces')

METHOD_NAMES = {
    'EnKF': 'EnKF', 'Cholesky': 'EnKF-Cholesky',
    'Identity': 'Identity Shrinkage', 'Scaled': 'Scaled Shrinkage',
    'Eigenvalue': 'Eigenvalue Shrinkage', 'LedoitWolf': 'Ledoit-Wolf',
}
COLORS_NE = {50: '#d62728', 60: '#ff7f0e', 80: '#2ca02c', 100: '#1f77b4'}

data = {}
for key, display in METHOD_NAMES.items():
    path = os.path.join(TRACE_DIR, f'{key}.json')
    if os.path.exists(path):
        with open(path) as f:
            data[display] = json.load(f)

fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
axes = axes.flatten()

for ax, (name, ne_data) in zip(axes, data.items()):
    for Ne in [50, 60, 80, 100]:
        if str(Ne) in ne_data and len(ne_data[str(Ne)]) > 0:
            ax.plot(ne_data[str(Ne)][0], color=COLORS_NE[Ne],
                    lw=1, label=f'$N_e={Ne}$', alpha=0.8)
    ax.set_title(name, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='upper right')

for ax in axes[3:]:
    ax.set_xlabel('Assimilation cycle')
for ax in [axes[0], axes[3]]:
    ax.set_ylabel('Relative RMSE')

fig.suptitle('Per-method RMSE time series across ensemble sizes (Lorenz 96)', fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'lorenz_per_method_panels.pdf'), bbox_inches='tight')
plt.close()
print('Saved lorenz_per_method_panels.pdf')
