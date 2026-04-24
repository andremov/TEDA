# -*- coding: utf-8 -*-
"""Run Identity Shrinkage on AR(1) cosmological model. Usage: cd d:/thesis && python paper/run_cosmo_identity.py"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cosmo_enkf_common import run_method, FIGURES_DIR
from pyteda.analysis.analysis_enkf_direct_precision_shrinkage_identity import AnalysisEnKFDirectPrecisionShrinkageIdentity

if __name__ == '__main__':
    os.makedirs(FIGURES_DIR, exist_ok=True)
    run_method('Identity Shrinkage', lambda model, T: AnalysisEnKFDirectPrecisionShrinkageIdentity(model), needs_invertible=True)
