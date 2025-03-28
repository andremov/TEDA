# -*- coding: utf-8 -*-

import numpy as np
import scipy as sci
from analysis.analysis import Analysis
from sklearn.linear_model import Ridge

class AnalysisEnKFDirectPrecisionShrinkage1(Analysis):
    """Analysis EnKF Direct Precision Shrinkage 1
    
    Attributes:
        model (Model object): An object that has all the methods and attributes of the model
        r (int): Value used in the process of removing correlations

    Methods:
        get_precision_matrix(DX, regularization_factor=0.01): Returns the computed precision matrix
        perform_assimilation(background, observation): Perform assimilation step given background and observations
        get_analysis_state(): Returns the computed column mean of ensemble Xa
        get_ensemble(): Returns ensemble Xa
        get_error_covariance(): Returns the computed covariance matrix of the ensemble Xa
        inflate_ensemble(inflation_factor): Computes new ensemble Xa given the inflation factor
    """

    def __init__(self, model, r=1, **kwargs):
        """
        Initialize an instance of AnalysisEnKFDirectPrecisionShrinkage1.

        Parameters:
            model (Model object): An object that has all the methods and attributes of the model given
            r (int, optional): Value used in the process of removing correlations
        """
        self.model = model
        self.r = r

    def compute_shrinkage_weights(self, S_inv, Pi0, n):
        """
        Compute the optimal shrinkage weights alpha* and beta* for the precision matrix.

        Parameters:
            S_inv (numpy.ndarray): Inverse sample covariance matrix.
            Pi0 (numpy.ndarray): Target precision matrix.
            n (int): Number of samples.

        Returns:
            tuple: (alpha_star, beta_star)
        """
        d = S_inv.shape[0]  # Dimensionality

        # Compute squared trace norm of S_inv
        trace_norm_sq_S_inv = np.trace(S_inv @ S_inv)

        # Compute squared Frobenius norm of Pi0
        frobenius_norm_sq_Pi0 = np.linalg.norm(Pi0, 'fro') ** 2

        # Compute trace of (S_inv * Pi0)
        trace_S_inv_Pi0 = np.trace(S_inv @ Pi0)

        # Compute α*
        numerator = (1 / n) * trace_norm_sq_S_inv * frobenius_norm_sq_Pi0
        denominator = (trace_norm_sq_S_inv * frobenius_norm_sq_Pi0) - (trace_S_inv_Pi0 ** 2)
        alpha_star = 1 - (d / n) - (numerator / denominator)

        # Compute β*
        beta_star = (1 - (d / n) - alpha_star) * (trace_S_inv_Pi0 / frobenius_norm_sq_Pi0)

        return alpha_star, beta_star

    def get_precision_matrix(self, DX, regularization_factor=0.01):
        """
        Perform calculations to get the precision matrix given the deviation matrix.

        Parameters:
            DX (ndarray): Deviation matrix
            regularization_factor (float, optional): Value used as alpha in the ridge model

        Returns:
            precision_matrix (ndarray): Precision matrix
        """
        n, ensemble_size = DX.shape

        # Compute sample covariance matrix S
        S = np.cov(DX, rowvar=True, bias=True)
        
        # Compute empirical precision matrix (inverse of S)
        S_inv = np.linalg.inv(S)

        # Define a default target precision matrix (e.g., identity matrix)
        Pi0 = np.eye(S.shape[0])

        alpha_star, beta_star = self.compute_shrinkage_weights(S_inv, Pi0, n)

        # Compute the shrinkage precision matrix
        Theta_hat = alpha_star * S_inv + beta_star * Pi0

        return Theta_hat

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
        Binv = self.get_precision_matrix(DX, self.r)
        D = Ys - H @ Xb
        Rinv = np.diag(np.reciprocal(np.diag(R)))
        IN = Binv + H.T @ (Rinv @ H)
        Z = np.linalg.solve(IN, H.T @ (Rinv @ D))
        self.Xa = Xb + Z
        return self.Xa

    def get_analysis_state(self):
        """Compute column-wise mean vector of Matrix of ensemble Xa

        Parameters:
            None

        Returns:
            mean_vector (ndarray): Mean vector
        """
        return np.mean(self.Xa, axis=1)

    def get_ensemble(self):
        """Returns ensemble Xa

        Parameters:
            None

        Returns:
            ensemble_matrix (ndarray): Ensemble matrix
        """
        return self.Xa

    def get_error_covariance(self):
        """Returns the computed covariance matrix of the ensemble Xa

        Parameters:
            None

        Returns:
            covariance_matrix (ndarray): Covariance matrix of the ensemble Xa
        """
        return np.cov(self.Xa)

    def inflate_ensemble(self, inflation_factor):
        """Computes ensemble Xa given the inflation factor

        Parameters:
            inflation_factor (int): Double number indicating the inflation factor

        Returns:
            None
        """
        n, ensemble_size = self.Xa.shape
        xa = self.get_analysis_state()
        DXa = self.Xa - np.outer(xa, np.ones(ensemble_size))
        self.Xa = np.outer(xa, np.ones(ensemble_size)) + inflation_factor * DXa
