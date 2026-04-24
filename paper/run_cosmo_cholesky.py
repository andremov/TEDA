# -*- coding: utf-8 -*-
"""Run EnKF-Cholesky on AR(1) cosmological model. Usage: cd d:/thesis && python paper/run_cosmo_cholesky.py"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cosmo_enkf_common import run_method, FIGURES_DIR
from pyteda.analysis.analysis_enkf_modified_cholesky import AnalysisEnKFModifiedCholesky

if __name__ == '__main__':
    os.makedirs(FIGURES_DIR, exist_ok=True)
    run_method('EnKF-Cholesky', lambda model, T: AnalysisEnKFModifiedCholesky(model, r=2))
