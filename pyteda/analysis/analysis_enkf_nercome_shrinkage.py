# -*- coding: utf-8 -*-

import numpy as np
from .analysis_core import Analysis

class AnalysisEnKFNercomeShrinkage(Analysis):
    """Analysis EnKF NERCOME (Non-parametric Eigenvalue-Regularized COvariance Matrix Estimator)

    Implements the NERCOME algorithm from Joachimi (2017), as described in
    Looijmans et al. (2024) Section 2.3. The algorithm:
    1. Splits data columns into two subsets of size s and (m-s)
    2. Computes diagonalized sample estimates from each subset
    3. Minimizes a distance function Q(s) over the split parameter s
    4. Averages over random splits to reduce variance

    Attributes:
        model (Model object): An object that has all the methods and attributes of the model
        max_draws (int): Maximum number of random splits to average over
    """

    def __init__(self, model, max_draws=1000, **kwargs):
        """
        Initialize an instance of AnalysisEnKFNercomeShrinkage.

        Parameters:
            model (Model object): An object that has all the methods and attributes of the model given
            max_draws (int, optional): Maximum number of random split draws per s value. Default 1000.
        """
        self.model = model
        self.max_draws = max_draws

    def _compute_nercome_for_split(self, DX, s):
        """
        Compute the NERCOME estimator for a given split parameter s.

        Per Looijmans et al. Section 2.3:
        (i)   Split columns of DX into X1 (d x s) and X2 (d x (m-s))
        (ii)  Compute sample covariances S1, S2 from each
        (iii) Diagonalize: S_a = U_a D_a U_a^T for a=1,2
        (iv)  Z = U1 diag(U1^T S2 U1) U1^T
        (v)   Average Z over N_draw random splits
        (vi)  Q(s) = ||Z_avg - S2_avg||_F is minimized

        Parameters:
            DX (ndarray): Deviation matrix (d x m)
            s (int): Split parameter, number of columns in first subset

        Returns:
            Z_avg (ndarray): Averaged NERCOME matrix estimate
            S2_avg (ndarray): Averaged S2 for distance computation
        """
        d, m = DX.shape

        # Determine number of draws: min of (n choose s) and max_draws
        try:
            from math import comb
            n_possible = comb(m, s)
        except (ImportError, OverflowError):
            n_possible = self.max_draws + 1
        n_draws = min(n_possible, self.max_draws)

        Z_sum = np.zeros((d, d))
        S2_sum = np.zeros((d, d))

        for _ in range(n_draws):
            # Random split
            perm = np.random.permutation(m)
            idx1 = perm[:s]
            idx2 = perm[s:]

            X1 = DX[:, idx1]
            X2 = DX[:, idx2]

            # Compute sample covariances (unbiased)
            S1 = np.cov(X1, bias=False) if s > 1 else np.outer(X1.ravel(), X1.ravel())
            S2 = np.cov(X2, bias=False) if (m - s) > 1 else np.outer(X2.ravel(), X2.ravel())

            # Ensure 2D
            S1 = np.atleast_2d(S1)
            S2 = np.atleast_2d(S2)

            # Eigendecomposition of S1
            _, U1 = np.linalg.eigh(S1)

            # Compute Z = U1 diag(U1^T S2 U1) U1^T
            projected = U1.T @ S2 @ U1
            Z = U1 @ np.diag(np.diag(projected)) @ U1.T

            Z_sum += Z
            S2_sum += S2

        return Z_sum / n_draws, S2_sum / n_draws

    def get_precision_matrix(self, DX):
        """
        Compute the NERCOME precision matrix estimate.

        Finds the optimal split parameter s* by minimizing Q(s) = ||Z(s) - S2(s)||_F,
        then returns the inverse of the NERCOME covariance estimate Z(s*).
        No Hartlap factor is applied per Looijmans et al. Section 2.3.

        Parameters:
            DX (ndarray): Deviation matrix (d x m)

        Returns:
            precision_matrix (ndarray): Precision matrix
        """
        d, m = DX.shape

        # Search over split parameters s in [2, m-2]
        s_values = list(range(2, m - 1))

        if len(s_values) == 0:
            # Fallback: not enough ensemble members for splitting
            S = np.cov(DX, bias=False)
            return np.linalg.inv(S + 1e-10 * np.eye(d))

        best_Q = np.inf
        best_Z = None

        for s in s_values:
            Z_avg, S2_avg = self._compute_nercome_for_split(DX, s)
            Q = np.linalg.norm(Z_avg - S2_avg, 'fro')

            if Q < best_Q:
                best_Q = Q
                best_Z = Z_avg

        precision_matrix = np.linalg.inv(best_Z)

        return precision_matrix

    def perform_assimilation(self, background, observation):
        """Perform assimilation step of ensemble Xa given the background and the observations

        Parameters:
            background (Background Object): The background object defined in the class background
            observation (Observation Object): The observation object defined in the class observation

        Returns:
            Xa (Matrix): Matrix of ensemble
        """
        Xb = background.get_ensemble()
        y = observation.get_observation()
        H = observation.get_observation_operator()
        R = observation.get_data_error_covariance()
        n, ensemble_size = Xb.shape
        Ys = np.random.multivariate_normal(y, R, size=ensemble_size).T
        xb = np.mean(Xb, axis=1)
        DX = Xb - np.outer(xb, np.ones(ensemble_size))
        Binv = self.get_precision_matrix(DX)
        D = Ys - H @ Xb
        Rinv = np.diag(np.reciprocal(np.diag(R)))
        IN = Binv + H.T @ (Rinv @ H)
        max_diag_IN = np.max(np.abs(np.diag(IN)))
        epsilon_IN = max(max_diag_IN * 1e-6, 1e-6)
        IN_regularized = IN + epsilon_IN * np.eye(IN.shape[0])
        Z = np.linalg.solve(IN_regularized, H.T @ (Rinv @ D))
        self.Xa = Xb + Z
        return self.Xa

    def get_analysis_state(self):
        """Compute column-wise mean vector of Matrix of ensemble Xa"""
        return np.mean(self.Xa, axis=1)

    def get_ensemble(self):
        """Returns ensemble Xa"""
        return self.Xa

    def get_error_covariance(self):
        """Returns the computed covariance matrix of the ensemble Xa"""
        return np.cov(self.Xa)

    def inflate_ensemble(self, inflation_factor):
        """Computes ensemble Xa given the inflation factor"""
        n, ensemble_size = self.Xa.shape
        xa = self.get_analysis_state()
        DXa = self.Xa - np.outer(xa, np.ones(ensemble_size))
        self.Xa = np.outer(xa, np.ones(ensemble_size)) + inflation_factor * DXa
