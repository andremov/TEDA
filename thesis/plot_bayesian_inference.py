# Generates: paper/figures/bayesian_inference.pdf
# No experiments needed.
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

FIG = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')
x = np.linspace(0, 500, 1000)
prior = norm.pdf(x, 250, 80)
likelihood = norm.pdf(x, 180, 50)
posterior_mean = (250/80**2 + 180/50**2) / (1/80**2 + 1/50**2)
posterior_std = np.sqrt(1 / (1/80**2 + 1/50**2))
posterior = norm.pdf(x, posterior_mean, posterior_std)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x, likelihood, 'b-', lw=2, label='Likelihood $\\mathcal{P}(\\mathbf{y}|\\mathbf{x})$')
ax.plot(x, prior, color='orange', lw=2, label='Prior $\\mathcal{P}(\\mathbf{x})$')
ax.plot(x, posterior, 'g-', lw=2, label='Posterior $\\mathcal{P}(\\mathbf{x}|\\mathbf{y})$')
ax.set_xlabel('State value')
ax.set_ylabel('Probability density')
ax.set_title('Bayesian inference: combining prior and likelihood')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'bayesian_inference.pdf'), bbox_inches='tight')
plt.close()
print('Saved bayesian_inference.pdf')
