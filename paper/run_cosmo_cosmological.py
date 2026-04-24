# -*- coding: utf-8 -*-
"""Run Cosmological Shrinkage on AR(1) cosmological model. Usage: cd d:/thesis && python paper/run_cosmo_cosmological.py"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cosmo_enkf_common import run_method, setup_cosmological_target, FIGURES_DIR
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_identity_scaled_cosmo import AnalysisEnKFDirectPrecisionShrinkageIdentityScaledCosmo

def make_cosmo(model, T):
    c = AnalysisEnKFDirectPrecisionShrinkageIdentityScaledCosmo(model)
    setup_cosmological_target(c, T)
    return c

if __name__ == '__main__':
    os.makedirs(FIGURES_DIR, exist_ok=True)
    run_method('Cosmological Shrinkage', make_cosmo, needs_invertible=True)
