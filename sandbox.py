# -*- coding: utf-8 -*-

import numpy as np
import scipy as sci
from analysis.analysis import Analysis
from sklearn.linear_model import Ridge


regularization_factor=0.01
# Xb = background.get_ensemble()
# n, ensemble_size = Xb.shape
# DX = Xb - np.outer(xb, np.ones(ensemble_size))~
n = 5
lr = Ridge(fit_intercept=False, alpha=regularization_factor)
L = np.eye(n)
D = np.zeros((n, n))
# D[0, 0] = 1 / np.var(DX[0, :])  # We are estimating D^{-1}
print(L)
print(L.T)
print(D)
print(L.T @ (D @ L))