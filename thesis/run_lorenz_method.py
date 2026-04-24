# -*- coding: utf-8 -*-
"""
Run a SINGLE method on Lorenz96 and save its RMSE traces.

Usage:
    python thesis/run_lorenz_method.py <method_name>

Methods: EnKF, Cholesky, Identity, Scaled, Eigenvalue, LedoitWolf
"""
import sys, os, warnings, logging, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
logging.getLogger().setLevel(logging.WARNING)

import numpy as np
from pyteda.models.lorenz96 import Lorenz96
from pyteda.background.background_core import Background
from pyteda.observation.observation_core import Observation
from pyteda.simulation.simulation_core import Simulation
from pyteda.analysis.analysis_enkf import AnalysisEnKF
from pyteda.analysis.analysis_enkf_modified_cholesky import AnalysisEnKFModifiedCholesky
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_identity import AnalysisEnKFDirectPrecisionShrinkageIdentity
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_identity_scaled import AnalysisEnKFDirectPrecisionShrinkageIdentityScaled
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_eigenvalues import AnalysisEnKFDirectPrecisionShrinkageEigenvalues
from pyteda.analysis.analysis_enkf_ledoit_wolf_shrinkage import AnalysisEnKFLedoitWolfShrinkage

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures', 'lorenz_traces')
SEED = 42
NE_VALUES = [50, 60, 80, 100]
N_RUNS = 10

def make_analysis(method_key):
    model = Lorenz96(n=40, F=8)
    return {
        'EnKF': lambda: AnalysisEnKF(),
        'Cholesky': lambda: AnalysisEnKFModifiedCholesky(model, r=2),
        'Identity': lambda: AnalysisEnKFDirectPrecisionShrinkageIdentity(model),
        'Scaled': lambda: AnalysisEnKFDirectPrecisionShrinkageIdentityScaled(model),
        'Eigenvalue': lambda: AnalysisEnKFDirectPrecisionShrinkageEigenvalues(model),
        'LedoitWolf': lambda: AnalysisEnKFLedoitWolfShrinkage(model),
    }[method_key]()

def run_one(Ne, analysis, seed):
    model = Lorenz96(n=40, F=8)
    ic = model.get_initial_condition()
    model.get_initial_condition = lambda: ic
    np.random.seed(seed)
    bg = Background(model, ensemble_size=Ne)
    obs = Observation(m=32, std_obs=0.01)
    sim = Simulation(model, bg, analysis, obs,
                     params={'obs_freq': 0.1, 'end_time': 10, 'inf_fact': 1.05},
                     log_level=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.run()
    _, ea = sim.get_errors()
    return ea.tolist()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run_lorenz_method.py <EnKF|Cholesky|Identity|Scaled|Eigenvalue|LedoitWolf>")
        sys.exit(1)

    method_key = sys.argv[1]
    os.makedirs(OUT_DIR, exist_ok=True)

    results = {}
    for Ne in NE_VALUES:
        print(f"[{method_key}] Ne={Ne}...")
        traces = []
        for run in range(N_RUNS):
            analysis = make_analysis(method_key)
            ea = run_one(Ne, analysis, SEED + run)
            traces.append(ea)
            print(f"  run {run+1}/{N_RUNS} done")
        results[str(Ne)] = traces

    outpath = os.path.join(OUT_DIR, f'{method_key}.json')
    with open(outpath, 'w') as f:
        json.dump(results, f)
    print(f"Saved {outpath}")
