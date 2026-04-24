# -*- coding: utf-8 -*-
"""
AR(1) Power Spectrum Model for cosmological data assimilation experiments.

Simulates evolution of a d-dimensional state representing power spectrum bins:
    x_{t+1} = phi * x_t + (1 - phi) * x_mean + epsilon,  epsilon ~ N(0, Q)

The process noise Q is calibrated so that the stationary distribution
matches the empirical covariance of the BOSS mock catalogs:
    Sigma_stationary = Q / (1 - phi^2)  =>  Q = (1 - phi^2) * Sigma_mock

Since the cosmological power spectrum covariance is approximately diagonal,
this model's precision matrix is near-diagonal — matching the assumptions
of direct precision shrinkage methods.
"""

import numpy as np
from .model import Model


class AR1PowerSpectrum(Model):
    """AR(1) model calibrated to cosmological power spectrum mock data."""

    def __init__(self, mean_spectrum, cov_mock, phi=0.95, dt=0.1):
        """
        Parameters
        ----------
        mean_spectrum : ndarray (d,)
            Mean power spectrum from mock catalogs.
        cov_mock : ndarray (d, d)
            Sample covariance from mock catalogs.
        phi : float
            Persistence parameter (0 < phi < 1). Controls how quickly
            the state reverts to the mean.
        dt : float
            Base time step for propagation.
        """
        self.n = len(mean_spectrum)
        self.mean_spectrum = mean_spectrum.copy()
        self.phi = phi
        self.dt = dt
        self._L = None

        # Process noise: Q = (1 - phi^2) * Sigma_mock
        # So the stationary distribution has covariance = Sigma_mock
        self.Q = (1 - phi ** 2) * cov_mock
        # Cholesky factor for sampling noise
        self.Q_chol = np.linalg.cholesky(self.Q)

    def get_number_of_variables(self):
        return self.n

    def get_initial_condition(self, seed=10, T=None):
        """Draw from the stationary distribution.

        We use the mock covariance as the stationary distribution,
        so initial conditions are: x ~ N(mean_spectrum, Sigma_mock).
        Then propagate for a spinup period to decorrelate.
        """
        np.random.seed(seed=seed)
        x0 = self.mean_spectrum + self.Q_chol @ np.random.randn(self.n) / np.sqrt(1 - self.phi ** 2)
        # Spinup: 100 steps
        for _ in range(100):
            x0 = self._step(x0)
        return x0

    def _step(self, x):
        """Single AR(1) step."""
        noise = self.Q_chol @ np.random.randn(self.n)
        return self.phi * x + (1 - self.phi) * self.mean_spectrum + noise

    def propagate(self, x0, T, just_final_state=True):
        """Propagate the AR(1) model over time array T.

        Parameters
        ----------
        x0 : ndarray (d,)
            Initial state.
        T : array-like
            Time array. Number of AR(1) steps = len(T) - 1.
        just_final_state : bool
            If True, return only the final state.

        Returns
        -------
        ndarray
            Final state (d,) or trajectory (len(T), d).
        """
        n_steps = max(len(T) - 1, 1)
        if just_final_state:
            x = x0.copy()
            for _ in range(n_steps):
                x = self._step(x)
            return x
        else:
            trajectory = np.zeros((len(T), self.n))
            trajectory[0] = x0
            x = x0.copy()
            for i in range(1, len(T)):
                x = self._step(x)
                trajectory[i] = x
            return trajectory

    def create_decorrelation_matrix(self, r):
        """Create decorrelation matrix.

        For the AR(1) model with near-diagonal covariance, we use an
        identity-like structure (no spatial correlations to remove).
        """
        self._L = np.eye(self.n)

    def get_decorrelation_matrix(self):
        return self._L

    def get_ngb(self, i, r):
        """Get neighbors for modified Cholesky (not really applicable)."""
        return np.arange(max(0, i - r), min(self.n, i + r + 1))

    def get_pre(self, i, r):
        """Get predecessors for modified Cholesky."""
        ngb = self.get_ngb(i, r)
        return ngb[ngb < i]
