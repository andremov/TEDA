# Generates: paper/figures/da_cycle.pdf
# No experiments needed. Schematic diagram.
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

FIG = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')

fig, ax = plt.subplots(figsize=(10, 4))
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-1.5, 2.5)
ax.set_aspect('equal')
ax.axis('off')

# Boxes
box_style = dict(boxstyle='round,pad=0.4', facecolor='#e8e8e8', edgecolor='black', lw=1.5)
analysis_style = dict(boxstyle='round,pad=0.4', facecolor='#c8e6c9', edgecolor='#2e7d32', lw=1.5)
obs_style = dict(boxstyle='round,pad=0.3', facecolor='#bbdefb', edgecolor='#1565c0', lw=1.5)

for k, xpos in enumerate([0, 4, 8]):
    # Forecast box
    ax.text(xpos, 1, f'$\\mathbf{{x}}^b_{k+1}$\nForecast', ha='center', va='center',
            fontsize=10, bbox=box_style)
    # Analysis box
    ax.text(xpos + 2, 1, f'$\\mathbf{{x}}^a_{k+1}$\nAnalysis', ha='center', va='center',
            fontsize=10, bbox=analysis_style)
    # Observation
    ax.text(xpos + 2, -0.5, f'$\\mathbf{{y}}_{k+1}$', ha='center', va='center',
            fontsize=11, bbox=obs_style)
    # Arrows
    ax.annotate('', xy=(xpos + 1.1, 1), xytext=(xpos + 0.9, 1),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.annotate('', xy=(xpos + 2, 0.4), xytext=(xpos + 2, -0.1),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#1565c0'))
    if xpos < 8:
        ax.annotate('$\\mathcal{M}$', xy=(xpos + 3.1, 1), xytext=(xpos + 2.9, 1),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='#d32f2f'),
                    fontsize=11, color='#d32f2f', ha='center', va='bottom')

ax.text(5, 2.2, 'Sequential Data Assimilation Cycle', ha='center', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'da_cycle.pdf'), bbox_inches='tight')
plt.close()
print('Saved da_cycle.pdf')
