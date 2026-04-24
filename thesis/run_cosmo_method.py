# -*- coding: utf-8 -*-
"""
Run a SINGLE method on the cosmological AR(1) model and save its RMSE traces.

Usage:
    cd d:/thesis
    python thesis/run_cosmo_method.py <method_name>

Methods: EnKF, Cholesky, Identity, Scaled, Eigenvalue, Cosmo, LedoitWolf

Run ALL methods in parallel:
    python thesis/run_cosmo_method.py EnKF &
    python thesis/run_cosmo_method.py Cholesky &
    python thesis/run_cosmo_method.py Identity &
    python thesis/run_cosmo_method.py Scaled &
    python thesis/run_cosmo_method.py Eigenvalue &
    python thesis/run_cosmo_method.py Cosmo &
    python thesis/run_cosmo_method.py LedoitWolf &

After all finish, run:
    python thesis/plot_cosmo_results.py
"""
import sys, os, warnings, logging, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
logging.getLogger().setLevel(logging.WARNING)

import numpy as np
from pyteda.models.ar1_power_spectrum import AR1PowerSpectrum
from pyteda.background.background_core import Background
from pyteda.observation.observation_core import Observation
from pyteda.simulation.simulation_core import Simulation
from pyteda.analysis.analysis_enkf import AnalysisEnKF
from pyteda.analysis.analysis_enkf_modified_cholesky import AnalysisEnKFModifiedCholesky
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_identity import AnalysisEnKFDirectPrecisionShrinkageIdentity
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_identity_scaled import AnalysisEnKFDirectPrecisionShrinkageIdentityScaled
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_eigenvalues import AnalysisEnKFDirectPrecisionShrinkageEigenvalues
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_identity_scaled_cosmo import AnalysisEnKFDirectPrecisionShrinkageIdentityScaledCosmo
from pyteda.analysis.analysis_enkf_ledoit_wolf_shrinkage import AnalysisEnKFLedoitWolfShrinkage

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'paper', 'data', 'looijmans')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures', 'cosmo_traces')
MOCKS_DIR = os.path.join(DATA_DIR, 'output', 'BOSS_DR12_NGC_z1', 'mocks')
TARGET_FILE = os.path.join(DATA_DIR, 'output', 'BOSS_DR12_NGC_z1', 'T_18_18.matrix')
SEED = 42
NE_VALUES = [24, 30, 40, 50]
N_RUNS = 10

def load_data():
    P_all = np.loadtxt(os.path.join(MOCKS_DIR, 'n2048', 'P_18_2048_v1.matrix'))
    return np.mean(P_all, axis=1), np.cov(P_all), np.loadtxt(TARGET_FILE)

def create_model(mean_spectrum, cov_mock):
    model = AR1PowerSpectrum(mean_spectrum, cov_mock, phi=0.95)
    model.create_decorrelation_matrix(r=2)
    return model

def setup_cosmo_target(analysis, T, d=18):
    """Configure cosmological target parameters."""
    import io
    from contextlib import redirect_stdout
    T_diag = np.diag(T)
    power_spectrum = {}
    multipole_sampling = {}
    data_vector_mapping = []
    for i in range(d):
        ell = i + 2
        power_spectrum[ell] = np.sqrt(T_diag[i])
        multipole_sampling[ell] = 2
        data_vector_mapping.append((ell, 0))
    with redirect_stdout(io.StringIO()):
        analysis.set_cosmological_parameters(
            power_spectrum=power_spectrum,
            survey_geometry={'survey_area_deg2': 7500, 'sky_fraction': 0.18, 'redshift_bins': 1},
            multipole_sampling=multipole_sampling,
            data_vector_mapping=data_vector_mapping,
        )

def make_analysis(method_key, model, T=None):
    a = {
        'EnKF': lambda: AnalysisEnKF(),
        'Cholesky': lambda: AnalysisEnKFModifiedCholesky(model, r=2),
        'Identity': lambda: AnalysisEnKFDirectPrecisionShrinkageIdentity(model),
        'Scaled': lambda: AnalysisEnKFDirectPrecisionShrinkageIdentityScaled(model),
        'Eigenvalue': lambda: AnalysisEnKFDirectPrecisionShrinkageEigenvalues(model),
        'Cosmo': lambda: AnalysisEnKFDirectPrecisionShrinkageIdentityScaledCosmo(model),
        'LedoitWolf': lambda: AnalysisEnKFLedoitWolfShrinkage(model),
    }[method_key]()
    if method_key == 'Cosmo' and T is not None:
        setup_cosmo_target(a, T)
    return a

def run_one(model, Ne, analysis, seed):
    ic = model.get_initial_condition()
    model.get_initial_condition = lambda: ic
    np.random.seed(seed)
    bg = Background(model, ensemble_size=Ne)
    obs = Observation(m=14, std_obs=100.0)
    sim = Simulation(model, bg, analysis, obs,
                     params={'obs_freq': 0.1, 'end_time': 10, 'inf_fact': 1.02},
                     log_level=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.run()
    _, ea = sim.get_errors()
    return ea.tolist()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run_cosmo_method.py <EnKF|Cholesky|Identity|Scaled|Eigenvalue|Cosmo|LedoitWolf>")
        sys.exit(1)

    method_key = sys.argv[1]
    os.makedirs(OUT_DIR, exist_ok=True)
    mean_spectrum, cov_mock, T = load_data()

    results = {}
    for Ne in NE_VALUES:
        print(f"[{method_key}] Ne={Ne}...")
        traces = []
        model = create_model(mean_spectrum, cov_mock)
        for run in range(N_RUNS):
            analysis = make_analysis(method_key, model, T)
            ea = run_one(model, Ne, analysis, SEED + run)
            traces.append(ea)
            print(f"  run {run+1}/{N_RUNS} done")
        results[str(Ne)] = traces

    outpath = os.path.join(OUT_DIR, f'{method_key}.json')
    with open(outpath, 'w') as f:
        json.dump(results, f)
    print(f"Saved {outpath}")
