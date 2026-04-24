# -*- coding: utf-8 -*-
"""Generate TEDA class diagram for the paper (Figure 1)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')


def draw_class_box(ax, x, y, name, methods=None, width=2.4, height=None,
                   facecolor='#E8F0FE', edgecolor='#4285F4', fontsize=8,
                   bold_name=True):
    """Draw a UML-style class box."""
    if methods is None:
        methods = []
    line_h = 0.22
    if height is None:
        height = 0.4 + len(methods) * line_h

    rect = mpatches.FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.05", facecolor=facecolor,
        edgecolor=edgecolor, linewidth=1.5)
    ax.add_patch(rect)

    # Class name
    name_y = y + height/2 - 0.25
    weight = 'bold' if bold_name else 'normal'
    ax.text(x, name_y, name, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, family='monospace')

    # Divider line
    if methods:
        div_y = name_y - 0.18
        ax.plot([x - width/2 + 0.05, x + width/2 - 0.05], [div_y, div_y],
                color=edgecolor, linewidth=0.8)

        # Methods
        for i, m in enumerate(methods):
            ax.text(x - width/2 + 0.12, div_y - 0.08 - i * line_h, m,
                    ha='left', va='center', fontsize=6.5, family='monospace')

    return (x, y, width, height)


def draw_arrow(ax, x1, y1, x2, y2, style='->'):
    """Draw an inheritance arrow from child to parent."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color='#555555',
                                linewidth=1.2, shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.5, 7)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── Row 1: Core framework classes ──
    draw_class_box(ax, 1.5, 6, 'Model', ['propagate()', 'get_initial_condition()'],
                   facecolor='#FFF3E0', edgecolor='#E65100', width=2.6)
    draw_class_box(ax, 4.5, 6, 'Background', ['get_ensemble()', 'forecast_step()'],
                   facecolor='#FFF3E0', edgecolor='#E65100', width=2.6)
    draw_class_box(ax, 7.5, 6, 'Observation', ['generate_observation()', 'get_observation()'],
                   facecolor='#FFF3E0', edgecolor='#E65100', width=2.6)

    # ── Row 2: Simulation ──
    draw_class_box(ax, 4.5, 4.3, 'Simulation', ['run()', 'get_errors()'],
                   facecolor='#F3E5F5', edgecolor='#7B1FA2', width=2.6)

    # Arrows from Simulation to core classes
    ax.annotate('', xy=(1.5, 5.4), xytext=(3.5, 4.6),
                arrowprops=dict(arrowstyle='->', color='#999', linewidth=1, linestyle='--'))
    ax.annotate('', xy=(4.5, 5.4), xytext=(4.5, 4.6),
                arrowprops=dict(arrowstyle='->', color='#999', linewidth=1, linestyle='--'))
    ax.annotate('', xy=(7.5, 5.4), xytext=(5.5, 4.6),
                arrowprops=dict(arrowstyle='->', color='#999', linewidth=1, linestyle='--'))

    # ── Row 2 right: Analysis (abstract) ──
    draw_class_box(ax, 8, 4.3, 'Analysis (ABC)',
                   ['perform_assimilation()', 'get_precision_matrix()'],
                   facecolor='#E8F0FE', edgecolor='#1565C0', width=2.8)

    # Arrow from Simulation to Analysis
    ax.annotate('', xy=(8, 4.6), xytext=(5.8, 4.3),
                arrowprops=dict(arrowstyle='->', color='#999', linewidth=1, linestyle='--'))

    # ── Row 3: Existing analysis methods ──
    draw_class_box(ax, 1.2, 2.5, 'AnalysisEnKF', [],
                   facecolor='#E8F0FE', edgecolor='#1565C0', width=2.2, height=0.5)
    draw_class_box(ax, 3.8, 2.5, 'EnKF-Cholesky', [],
                   facecolor='#E8F0FE', edgecolor='#1565C0', width=2.2, height=0.5)

    draw_arrow(ax, 1.2, 2.75, 8, 3.85)
    draw_arrow(ax, 3.8, 2.75, 8, 3.85)

    # ── Row 3: New shrinkage methods (highlighted) ──
    shrinkage_color = '#E8F5E9'
    shrinkage_edge = '#2E7D32'

    draw_class_box(ax, 6.5, 2.5, 'Shrinkage-Identity', [],
                   facecolor=shrinkage_color, edgecolor=shrinkage_edge, width=2.2, height=0.5)
    draw_class_box(ax, 9, 2.5, 'Ledoit-Wolf', [],
                   facecolor=shrinkage_color, edgecolor=shrinkage_edge, width=1.8, height=0.5)

    draw_arrow(ax, 6.5, 2.75, 8, 3.85)
    draw_arrow(ax, 9, 2.75, 8, 3.85)

    # ── Row 4: More shrinkage methods ──
    draw_class_box(ax, 1.5, 1.0, 'Shrinkage-Scaled', [],
                   facecolor=shrinkage_color, edgecolor=shrinkage_edge, width=2.4, height=0.5)
    draw_class_box(ax, 4.2, 1.0, 'Shrinkage-Eigenvalue', [],
                   facecolor=shrinkage_color, edgecolor=shrinkage_edge, width=2.6, height=0.5)
    draw_class_box(ax, 7.0, 1.0, 'Shrinkage-Cosmo', [],
                   facecolor=shrinkage_color, edgecolor=shrinkage_edge, width=2.4, height=0.5)
    draw_class_box(ax, 9.5, 1.0, 'NERCOME', [],
                   facecolor=shrinkage_color, edgecolor=shrinkage_edge, width=1.6, height=0.5)

    draw_arrow(ax, 1.5, 1.25, 8, 3.85)
    draw_arrow(ax, 4.2, 1.25, 8, 3.85)
    draw_arrow(ax, 7.0, 1.25, 8, 3.85)
    draw_arrow(ax, 9.5, 1.25, 8, 3.85)

    # ── Legend ──
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FFF3E0', edgecolor='#E65100', label='Core framework'),
        Patch(facecolor='#F3E5F5', edgecolor='#7B1FA2', label='Orchestration'),
        Patch(facecolor='#E8F0FE', edgecolor='#1565C0', label='Existing analysis'),
        Patch(facecolor='#E8F5E9', edgecolor='#2E7D32', label='New (this work)'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=8,
              framealpha=0.9, edgecolor='#ccc')

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'class_diagram.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(FIGURES_DIR, exist_ok=True)
    main()
