# -*- coding: utf-8 -*-
"""Run Scaled Shrinkage on AR(1) cosmological model. Usage: cd d:/thesis && python paper/run_cosmo_scaled.py"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cosmo_enkf_common import run_method, FIGURES_DIR
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_identity_scaled import AnalysisEnKFDirectPrecisionShrinkageIdentityScaled

if __name__ == '__main__':
    os.makedirs(FIGURES_DIR, exist_ok=True)
    run_method('Scaled Shrinkage', lambda model, T: AnalysisEnKFDirectPrecisionShrinkageIdentityScaled(model), needs_invertible=True)
