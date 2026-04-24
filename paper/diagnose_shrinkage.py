# -*- coding: utf-8 -*-
"""
Diagnostic script to investigate why direct precision shrinkage methods
show non-monotonic degradation with increasing ensemble size.

Logs alpha*, beta* (unclamped and clamped), denominator values,
and condition numbers at each assimilation step.

Usage:
    cd d:/thesis
    python paper/diagnose_shrinkage.py
"""

import sys
import os
import warnings
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from pyteda.models.lorenz96 import Lorenz96
from pyteda.background.background_core import Background
from pyteda.observation.observation_core import Observation
from pyteda.simulation.simulation_core import Simulation

from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_identity import AnalysisEnKFDirectPrecisionShrinkageIdentity
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_identity_scaled import AnalysisEnKFDirectPrecisionShrinkageIdentityScaled
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_eigenvalues import AnalysisEnKFDirectPrecisionShrinkageEigenvalues

# ── Configuration ─────────────────────────────────────────────────────────────

N_VARIABLES = 40
FORCING = 8
N_OBS = 32
STD_OBS = 0.01
OBS_FREQ = 0.1
END_TIME = 10
INF_FACT = 1.05
ENSEMBLE_SIZES = [45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
RANDOM_SEED = 42
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')


# ── Diagnostic wrapper classes ────────────────────────────────────────────────

class DiagnosticMixin:
    """Mixin that logs shrinkage weight diagnostics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagnostics = []

    def compute_shrinkage_weights(self, S_inv, Pi0, data_vector_length, ensemble_size):
        """Override to log diagnostics before returning."""
        # Compute raw values (same math as parent)
        singular_values = np.linalg.svd(S_inv, compute_uv=False)
        squared_trace_norm = np.sum(singular_values) ** 2
        frobenius_norm_sq_Pi0 = np.linalg.norm(Pi0, 'fro') ** 2
        trace_S_inv_Pi0 = np.trace(S_inv @ Pi0)

        numerator_alpha = (ensemble_size ** -1) * squared_trace_norm * frobenius_norm_sq_Pi0
        denominator_alpha = (squared_trace_norm * frobenius_norm_sq_Pi0) - (trace_S_inv_Pi0 ** 2)

        if np.isclose(denominator_alpha, 0):
            alpha_unclamped = 0.0
        else:
            alpha_unclamped = 1 - (data_vector_length / ensemble_size) - (numerator_alpha / denominator_alpha)

        first_term_beta = 1 - (data_vector_length / ensemble_size) - alpha_unclamped
        if np.isclose(frobenius_norm_sq_Pi0, 0):
            beta_unclamped = 0.0
        else:
            beta_unclamped = first_term_beta * (trace_S_inv_Pi0 / frobenius_norm_sq_Pi0)

        alpha_clamped = max(0, alpha_unclamped)
        beta_clamped = max(0, beta_unclamped)
        if alpha_clamped + beta_clamped > 1:
            beta_clamped = 1 - alpha_clamped

        # Condition number of S_inv
        cond_S_inv = np.linalg.cond(S_inv)

        # Cauchy-Schwarz ratio: how close is tr(A·B)² to ||A||²_tr·||B||²_F
        cs_product = squared_trace_norm * frobenius_norm_sq_Pi0
        cs_ratio = (trace_S_inv_Pi0 ** 2) / cs_product if cs_product > 0 else float('nan')

        self.diagnostics.append({
            'alpha_unclamped': alpha_unclamped,
            'alpha_clamped': alpha_clamped,
            'beta_unclamped': beta_unclamped,
            'beta_clamped': beta_clamped,
            'denominator': denominator_alpha,
            'numerator': numerator_alpha,
            'cs_ratio': cs_ratio,
            'cond_S_inv': cond_S_inv,
            'squared_trace_norm': squared_trace_norm,
            'frobenius_Pi0': frobenius_norm_sq_Pi0,
            'trace_product': trace_S_inv_Pi0,
        })

        return alpha_clamped, beta_clamped

    def get_precision_matrix(self, DX):
        """Override to also log condition number of final Theta_hat."""
        Theta_hat = super().get_precision_matrix(DX)
        if len(self.diagnostics) > 0:
            self.diagnostics[-1]['cond_Theta'] = np.linalg.cond(Theta_hat)
        return Theta_hat


class DiagIdentity(DiagnosticMixin, AnalysisEnKFDirectPrecisionShrinkageIdentity):
    pass

class DiagScaled(DiagnosticMixin, AnalysisEnKFDirectPrecisionShrinkageIdentityScaled):
    pass

class DiagEigenvalue(DiagnosticMixin, AnalysisEnKFDirectPrecisionShrinkageEigenvalues):
    pass


# ── Run diagnostics ──────────────────────────────────────────────────────────

def run_diagnostic(model, Ne, analysis_cls):
    """Run one experiment and return the diagnostics list + error array."""
    np.random.seed(RANDOM_SEED)
    analysis = analysis_cls(model)
    background = Background(model, ensemble_size=Ne)
    observation = Observation(m=N_OBS, std_obs=STD_OBS)
    params = {'obs_freq': OBS_FREQ, 'end_time': END_TIME, 'inf_fact': INF_FACT}

    sim = Simulation(model, background, analysis, observation, params=params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.run()

    _, err_a = sim.get_errors()
    return analysis.diagnostics, err_a


def summarize_diagnostics(diagnostics, Ne, method_name, err_a):
    """Print summary statistics for one method/Ne combination."""
    n = len(diagnostics)
    if n == 0:
        print(f"  {method_name} Ne={Ne}: NO DATA")
        return {}

    # Skip first 50 (transient)
    transient = min(50, n // 2)
    post = diagnostics[transient:]

    alphas_u = [d['alpha_unclamped'] for d in post]
    betas_u = [d['beta_unclamped'] for d in post]
    alphas_c = [d['alpha_clamped'] for d in post]
    betas_c = [d['beta_clamped'] for d in post]
    denoms = [d['denominator'] for d in post]
    cs_ratios = [d['cs_ratio'] for d in post]
    cond_thetas = [d.get('cond_Theta', float('nan')) for d in post]

    # Clamping frequency
    alpha_clamped_pct = sum(1 for a in alphas_u if a < 0 or a > 1) / len(alphas_u) * 100
    beta_clamped_pct = sum(1 for b in betas_u if b < 0 or b > 1) / len(betas_u) * 100

    rmse = np.mean(err_a[transient:]) * 100

    summary = {
        'method': method_name,
        'Ne': Ne,
        'rmse_x100': f"{rmse:.4f}",
        'alpha_uncl_mean': f"{np.mean(alphas_u):.4f}",
        'alpha_uncl_min': f"{np.min(alphas_u):.4f}",
        'alpha_uncl_max': f"{np.max(alphas_u):.4f}",
        'alpha_clamp_pct': f"{alpha_clamped_pct:.1f}%",
        'beta_uncl_mean': f"{np.mean(betas_u):.4f}",
        'beta_uncl_min': f"{np.min(betas_u):.4f}",
        'beta_uncl_max': f"{np.max(betas_u):.4f}",
        'beta_clamp_pct': f"{beta_clamped_pct:.1f}%",
        'denom_mean': f"{np.mean(denoms):.2e}",
        'denom_min': f"{np.min(denoms):.2e}",
        'cs_ratio_mean': f"{np.mean(cs_ratios):.6f}",
        'cs_ratio_max': f"{np.max(cs_ratios):.6f}",
        'cond_Theta_mean': f"{np.nanmean(cond_thetas):.2e}",
        'cond_Theta_max': f"{np.nanmax(cond_thetas):.2e}",
    }

    print(f"  {method_name} Ne={Ne}:")
    print(f"    RMSE(x100)={rmse:.4f}")
    print(f"    alpha* unclamped: mean={np.mean(alphas_u):.4f} min={np.min(alphas_u):.4f} max={np.max(alphas_u):.4f} | clamped {alpha_clamped_pct:.1f}% of steps")
    print(f"    beta*  unclamped: mean={np.mean(betas_u):.4f} min={np.min(betas_u):.4f} max={np.max(betas_u):.4f} | clamped {beta_clamped_pct:.1f}% of steps")
    print(f"    denominator: mean={np.mean(denoms):.2e} min={np.min(denoms):.2e}")
    print(f"    Cauchy-Schwarz ratio: mean={np.mean(cs_ratios):.6f} max={np.max(cs_ratios):.6f} (1.0 = singular)")
    print(f"    cond(Theta): mean={np.nanmean(cond_thetas):.2e} max={np.nanmax(cond_thetas):.2e}")

    return summary


def main():
    model = Lorenz96(n=N_VARIABLES, F=FORCING)

    methods = {
        'Identity': DiagIdentity,
        'Scaled': DiagScaled,
        'Eigenvalue': DiagEigenvalue,
    }

    all_summaries = []

    print("=" * 70)
    print("SHRINKAGE COEFFICIENT DIAGNOSTICS")
    print(f"n={N_VARIABLES}, F={FORCING}, seed={RANDOM_SEED}")
    print(f"Hartlap factor gamma = (Ne - {N_VARIABLES} - 2) / (Ne - 1)")
    print("=" * 70)

    for Ne in ENSEMBLE_SIZES:
        gamma = (Ne - N_VARIABLES - 2) / (Ne - 1)
        print(f"\n--- Ne = {Ne}  (Ne/n = {Ne/N_VARIABLES:.2f}, Hartlap gamma = {gamma:.4f}) ---")

        for method_name, cls in methods.items():
            try:
                diags, err_a = run_diagnostic(model, Ne, cls)
                summary = summarize_diagnostics(diags, Ne, method_name, err_a)
                all_summaries.append(summary)
            except Exception as e:
                print(f"  {method_name} Ne={Ne}: FAILED - {e}")
                all_summaries.append({'method': method_name, 'Ne': Ne, 'error': str(e)})

    # Save to CSV
    os.makedirs(FIGURES_DIR, exist_ok=True)
    csv_path = os.path.join(FIGURES_DIR, 'diagnostics.csv')
    if all_summaries:
        keys = all_summaries[0].keys()
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in all_summaries:
                writer.writerow(row)
        print(f"\nSaved diagnostics to {csv_path}")

    # Print RMSE summary table
    print("\n" + "=" * 70)
    print("RMSE SUMMARY (x10^-2, post-transient)")
    print("=" * 70)
    header = f"{'Ne':>5s}"
    for m in methods:
        header += f" | {m:>15s}"
    print(header)
    print("-" * len(header))
    for Ne in ENSEMBLE_SIZES:
        row = f"{Ne:>5d}"
        for m in methods:
            match = [s for s in all_summaries if s.get('method') == m and s.get('Ne') == Ne]
            if match and 'rmse_x100' in match[0]:
                row += f" | {match[0]['rmse_x100']:>15s}"
            else:
                row += f" | {'ERR':>15s}"
        print(row)


if __name__ == '__main__':
    main()
