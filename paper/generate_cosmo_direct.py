# -*- coding: utf-8 -*-
"""
Part A: Direct precision matrix comparison using Looijmans et al. BOSS mock data.

Tests precision matrix estimation quality WITHOUT the EnKF pipeline.
Uses 2048 Patchy mock catalogs from BOSS DR12, following Looijmans et al. (2025).

Usage:
    cd d:/thesis
    python paper/generate_cosmo_direct.py

Outputs:
    paper/figures/cosmo_matrix_loss.pdf   - Matrix loss vs subsample size
    paper/figures/cosmo_table_data.csv    - Numerical results
"""

import sys
import os
import warnings
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
logging.getLogger().setLevel(logging.WARNING)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_identity import AnalysisEnKFDirectPrecisionShrinkageIdentity
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_identity_scaled import AnalysisEnKFDirectPrecisionShrinkageIdentityScaled
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_eigenvalues import AnalysisEnKFDirectPrecisionShrinkageEigenvalues
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_identity_scaled_cosmo import AnalysisEnKFDirectPrecisionShrinkageIdentityScaledCosmo
from pyteda.analysis.analysis_enkf_ledoit_wolf_shrinkage import AnalysisEnKFLedoitWolfShrinkage
from pyteda.analysis.analysis_enkf_nercome_shrinkage import AnalysisEnKFNercomeShrinkage
from pyteda.analysis.analysis_enkf_modified_cholesky import AnalysisEnKFModifiedCholesky

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'looijmans')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
MOCKS_DIR = os.path.join(DATA_DIR, 'output', 'BOSS_DR12_NGC_z1', 'mocks')
TARGET_FILE = os.path.join(DATA_DIR, 'output', 'BOSS_DR12_NGC_z1', 'T_18_18.matrix')

SUBSAMPLE_SIZES = [24, 30, 40, 50]
N_DRAWS = 100  # Random subsamples per size
P_DIM = 18     # Data vector dimension (9 k-bins × 2 multipoles)

COLORS = {
    'Identity Shrinkage': '#d62728',
    'Scaled Shrinkage': '#ff7f0e',
    'Eigenvalue Shrinkage': '#9467bd',
    'Cosmological Shrinkage': '#2ca02c',
    'NERCOME': '#e377c2',
    'Ledoit-Wolf': '#1f77b4',
    'EnKF-Cholesky': '#8c564b',
    'Sample (Hartlap)': '#7f7f7f',
}

LINESTYLES = {
    'Identity Shrinkage': '--',
    'Scaled Shrinkage': '-.',
    'Eigenvalue Shrinkage': ':',
    'Cosmological Shrinkage': '-',
    'NERCOME': '-',
    'Ledoit-Wolf': '-',
    'EnKF-Cholesky': '--',
    'Sample (Hartlap)': ':',
}


# ── Dummy model for analysis class constructors ──────────────────────────────

class DummyModel:
    """Minimal model stub — analysis classes need a model in __init__ but
    get_precision_matrix(DX) doesn't use it."""
    def get_number_of_variables(self):
        return P_DIM

dummy_model = DummyModel()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_mocks():
    """Load the full P_18_2048 data matrix from Looijmans' pre-generated output."""
    path = os.path.join(MOCKS_DIR, 'n2048', 'P_18_2048_v1.matrix')
    P = np.loadtxt(path)  # shape: (18, 2048)
    print(f"Loaded all mocks: shape {P.shape}")
    return P


def load_target_covariance():
    """Load the cosmological target covariance matrix T (diagonal, 18×18)."""
    T = np.loadtxt(TARGET_FILE)
    print(f"Loaded target covariance T: shape {T.shape}")
    return T


def compute_true_precision(P_all):
    """Compute 'true' precision from all 2048 mocks (with Hartlap correction)."""
    S_full = np.cov(P_all)  # 18×18
    n_mocks = P_all.shape[1]
    hartlap = (n_mocks - P_DIM - 2) / (n_mocks - 1)
    Pi_true = hartlap * np.linalg.inv(S_full)
    print(f"True precision: Hartlap factor = {hartlap:.4f}, cond(Pi_true) = {np.linalg.cond(Pi_true):.2f}")
    return Pi_true


# ── Estimator setup ──────────────────────────────────────────────────────────

def setup_cosmological_target(analysis_cosmo, T):
    """Configure the cosmological analysis class with real BOSS parameters.

    We directly provide the target precision Pi0 = T^{-1} through the
    cosmological parameter interface, mapping each of the 18 data vector
    elements to a unique (ell, z_bin) pair with the correct C_l and N_l.
    """
    # T is diagonal: T_ii = (2/N_l) * C_l^2
    # We need to reverse-engineer C_l and N_l for each element.
    # Since T is diagonal and we only need Pi0 = T^{-1}, we can construct
    # synthetic C_l and N_l that reproduce the exact T values.

    # Use synthetic mapping: each data element gets a unique "multipole"
    T_diag = np.diag(T)
    power_spectrum = {}
    multipole_sampling = {}
    data_vector_mapping = []
    survey_geometry = {
        'survey_area_deg2': 7500,
        'sky_fraction': 0.18,
        'redshift_bins': 1,
    }

    for i in range(P_DIM):
        ell = i + 2  # Use ℓ = 2, 3, ..., 19
        # T_ii = (2/N_l) * C_l^2
        # Choose N_l = 2 so that T_ii = C_l^2, hence C_l = sqrt(T_ii)
        N_ell = 2
        C_ell = np.sqrt(T_diag[i])

        power_spectrum[ell] = C_ell
        multipole_sampling[ell] = N_ell
        data_vector_mapping.append((ell, 0))

    # Suppress verbose output
    import io
    from contextlib import redirect_stdout
    with redirect_stdout(io.StringIO()):
        analysis_cosmo.set_cosmological_parameters(
            power_spectrum=power_spectrum,
            survey_geometry=survey_geometry,
            multipole_sampling=multipole_sampling,
            data_vector_mapping=data_vector_mapping,
        )

    # Verify the target matches
    test_S_inv = np.eye(P_DIM)  # dummy
    Pi0_check = analysis_cosmo.compute_target_precision_matrix(test_S_inv)
    Pi0_expected = np.linalg.inv(T)
    if not np.allclose(Pi0_check, Pi0_expected, rtol=1e-10):
        print("WARNING: Cosmological target does not match expected T^{-1}!")
    else:
        print("Cosmological target verified: Pi0 = T^{-1}")


def create_estimators(T):
    """Create all estimator instances."""
    estimators = {}

    # Direct precision shrinkage methods
    estimators['Identity Shrinkage'] = AnalysisEnKFDirectPrecisionShrinkageIdentity(dummy_model)
    estimators['Scaled Shrinkage'] = AnalysisEnKFDirectPrecisionShrinkageIdentityScaled(dummy_model)
    estimators['Eigenvalue Shrinkage'] = AnalysisEnKFDirectPrecisionShrinkageEigenvalues(dummy_model)

    cosmo = AnalysisEnKFDirectPrecisionShrinkageIdentityScaledCosmo(dummy_model)
    setup_cosmological_target(cosmo, T)
    estimators['Cosmological Shrinkage'] = cosmo

    # Covariance-based methods
    estimators['Ledoit-Wolf'] = AnalysisEnKFLedoitWolfShrinkage(dummy_model)
    estimators['NERCOME'] = AnalysisEnKFNercomeShrinkage(dummy_model, max_draws=200)

    return estimators


def compute_sample_precision(DX, n_samples):
    """Compute Hartlap-corrected sample precision (baseline)."""
    S = np.cov(DX, rowvar=True, bias=False)
    S_inv = np.linalg.inv(S)
    hartlap = (n_samples - P_DIM - 2) / (n_samples - 1)
    return hartlap * S_inv


# ── Matrix loss metric ────────────────────────────────────────────────────────

def frobenius_loss(Pi_est, Pi_true):
    """Relative Frobenius loss: ||Pi_est - Pi_true||_F / ||Pi_true||_F"""
    return np.linalg.norm(Pi_est - Pi_true, 'fro') / np.linalg.norm(Pi_true, 'fro')


# ── Main experiment ───────────────────────────────────────────────────────────

def run_experiment():
    print("=" * 60)
    print("Part A: Direct Precision Matrix Comparison")
    print("=" * 60)

    P_all = load_all_mocks()
    T = load_target_covariance()
    Pi_true = compute_true_precision(P_all)
    estimators = create_estimators(T)

    # Method names in display order (EnKF-Cholesky excluded: needs decorrelation matrix)
    method_names = [
        'Sample (Hartlap)',
        'Identity Shrinkage',
        'Scaled Shrinkage',
        'Eigenvalue Shrinkage',
        'Cosmological Shrinkage',
        'NERCOME',
        'Ledoit-Wolf',
    ]

    # Results: {method: {n: [losses]}}
    results = {m: {n: [] for n in SUBSAMPLE_SIZES} for m in method_names}

    csv_path = os.path.join(FIGURES_DIR, 'cosmo_table_data.csv')

    for n in SUBSAMPLE_SIZES:
        print(f"\n--- Subsample size n = {n} ---")

        for draw in range(N_DRAWS):
            if (draw + 1) % 20 == 0:
                print(f"  Draw {draw + 1}/{N_DRAWS}")

            # Random subsample of n mocks
            rng = np.random.RandomState(seed=draw * 1000 + n)
            idx = rng.choice(P_all.shape[1], size=n, replace=False)
            P_sub = P_all[:, idx]  # (18, n)

            # Compute deviation matrix (mean-subtracted)
            mean_vec = np.mean(P_sub, axis=1, keepdims=True)
            DX = P_sub - mean_vec  # (18, n)

            # Sample precision (Hartlap-corrected baseline)
            try:
                Pi_sample = compute_sample_precision(DX, n)
                results['Sample (Hartlap)'][n].append(frobenius_loss(Pi_sample, Pi_true))
            except Exception as e:
                results['Sample (Hartlap)'][n].append(np.nan)

            # Each estimator
            for name, est in estimators.items():
                try:
                    Pi_est = est.get_precision_matrix(DX)
                    loss = frobenius_loss(Pi_est, Pi_true)
                    results[name][n].append(loss)
                except Exception as e:
                    if draw == 0:
                        print(f"  WARNING: {name} failed at n={n}: {e}")
                    results[name][n].append(np.nan)

    # Save CSV
    save_csv(results, method_names, csv_path)

    # Plot
    plot_matrix_loss(results, method_names)

    return results


def save_csv(results, method_names, csv_path):
    """Save results to CSV."""
    header = "Method," + ",".join([f"n={n}" for n in SUBSAMPLE_SIZES])
    lines = [header]
    for m in method_names:
        vals = []
        for n in SUBSAMPLE_SIZES:
            losses = [x for x in results[m][n] if not np.isnan(x)]
            if losses:
                mean = np.mean(losses)
                std = np.std(losses)
                vals.append(f"{mean:.4f} +/- {std:.4f}")
            else:
                vals.append("N/A")
        lines.append(f"{m},{','.join(vals)}")

    with open(csv_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\nSaved CSV: {csv_path}")


def plot_matrix_loss(results, method_names):
    """Plot relative Frobenius loss vs subsample size."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for m in method_names:
        means = []
        stds = []
        for n in SUBSAMPLE_SIZES:
            losses = [x for x in results[m][n] if not np.isnan(x)]
            if losses:
                means.append(np.mean(losses))
                stds.append(np.std(losses))
            else:
                means.append(np.nan)
                stds.append(0)

        color = COLORS.get(m, '#333333')
        ls = LINESTYLES.get(m, '-')
        ax.errorbar(SUBSAMPLE_SIZES, means, yerr=stds,
                     label=m, color=color, linestyle=ls,
                     marker='o', markersize=4, capsize=3, linewidth=1.5)

    ax.set_xlabel('Number of mock realizations $n$', fontsize=11)
    ax.set_ylabel('Relative Frobenius loss', fontsize=11)
    ax.set_title('Precision Matrix Estimation on BOSS DR12 Mock Data', fontsize=12)
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.set_xticks(SUBSAMPLE_SIZES)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'cosmo_matrix_loss.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print(f"Saved figure: {path}")
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(FIGURES_DIR, exist_ok=True)
    run_experiment()
