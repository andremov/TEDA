# Generates: paper/figures/eigenvalue_spectrum.pdf
# Uses Lorenz96 model, fast (~30 seconds).
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyteda.models.lorenz96 import Lorenz96
from pyteda.background.background_core import Background

FIG = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')
model = Lorenz96(n=40, F=8)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
ne_values = [45, 60, 100]

for ax, Ne in zip(axes, ne_values):
    np.random.seed(42)
    bg = Background(model, ensemble_size=Ne)
    bg.get_initial_ensemble(initial_perturbation=0.05, time=np.arange(0, 10, 0.01))
    time = np.arange(0, 0.1, 0.01)
    Xb = bg.get_ensemble()
    for _ in range(20):
        Xb = bg.forecast_step(Xb, time)
    S = bg.get_covariance_matrix()

    eig_sample = np.sort(np.linalg.eigvalsh(S))[::-1]

    # Also get a "reference" from a large ensemble
    np.random.seed(42)
    bg_large = Background(model, ensemble_size=500)
    bg_large.get_initial_ensemble(initial_perturbation=0.05, time=np.arange(0, 10, 0.01))
    Xb_large = bg_large.get_ensemble()
    for _ in range(20):
        Xb_large = bg_large.forecast_step(Xb_large, time)
    S_large = bg_large.get_covariance_matrix()
    eig_true = np.sort(np.linalg.eigvalsh(S_large))[::-1]

    ax.semilogy(range(1, 41), eig_true, 'k-', lw=2, label=f'Reference ($N_e=500$)', alpha=0.7)
    ax.semilogy(range(1, 41), eig_sample, 'o-', color='#d62728', lw=1, markersize=4,
                label=f'Sample ($N_e={Ne}$)')
    ax.set_xlabel('Eigenvalue index')
    if ax == axes[0]:
        ax.set_ylabel('Eigenvalue magnitude')
    ax.set_title(f'$N_e = {Ne}$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle('Eigenvalue spectrum of sample covariance vs reference (Lorenz 96, $n=40$)',
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'eigenvalue_spectrum.pdf'), bbox_inches='tight')
plt.close()
print('Saved eigenvalue_spectrum.pdf')
