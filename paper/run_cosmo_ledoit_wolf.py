# -*- coding: utf-8 -*-
"""Run Ledoit-Wolf on AR(1) cosmological model. Usage: cd d:/thesis && python paper/run_cosmo_ledoit_wolf.py"""
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cosmo_enkf_common import run_method, FIGURES_DIR
from pyteda.analysis.analysis_enkf_ledoit_wolf_shrinkage import AnalysisEnKFLedoitWolfShrinkage

if __name__ == '__main__':
    os.makedirs(FIGURES_DIR, exist_ok=True)
    run_method('Ledoit-Wolf', lambda model, T: AnalysisEnKFLedoitWolfShrinkage(model))
