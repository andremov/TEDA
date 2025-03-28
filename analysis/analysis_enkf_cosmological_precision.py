# -*- coding: utf-8 -*-

import numpy as np
import scipy as sci
from analysis.analysis import Analysis
from sklearn.linear_model import Ridge


class AnalysisEnKFCosmologicalPrecision(Analysis):
    """Analysis EnKF Cosmological Precision decomposition
    
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
        Initialize an instance of AnalysisEnKFCosmologicalPrecision.

        Parameters:
            model (Model object): An object that has all the methods and attributes of the model given
            r (int, optional): Value used in the process of removing correlations
        """
        self.model = model
        self.r = r

    def get_precision_matrix(self, DX, regularization_factor=0.01):
        """
        Perform calculations to get the precision matrix given the deviation matrix.

        Parameters:
            DX (ndarray): Deviation matrix
            regularization_factor (float, optional): Value used as alpha in the ridge model

        Returns:
            precision_matrix (ndarray): Precision matrix
        """

        # TODO: 1. calcular la inverse sample covariance matrix (cosmologica)
            # A. implementar las forumlas de silla^1 o silla^2 (y T^2 en este caso)
        # TODO: 2. aproximar la inverse background error covariance matrix (usando la sample cov. mat.)
        # TODO: 3. profit

        n, ensemble_size = DX.shape
        lr = Ridge(fit_intercept=False, alpha=regularization_factor)
        L = np.eye(n)
        D = np.zeros((n, n))
        D[0, 0] = 1 / np.var(DX[0, :])  # We are estimating D^{-1}
        for i in range(1, n):
            ind_prede = self.model.get_pre(i, self.r)
            y = DX[i, :]
            X = DX[ind_prede, :].T
            lr_fit = lr.fit(X, y)
            err_i = y - lr_fit.predict(X)
            D[i, i] = 1 / np.var(err_i) # ecuacion 13b de paper1 pp10
            L[i, ind_prede] = -lr_fit.coef_
        
        # inverse background error covariance matrix
        return L.T @ (D @ L) # ecuacion 14a de paper1 pp10

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

        xb = np.mean(Xb, axis=1) # ensemble mean, x_bar
        DX = Xb - np.outer(xb, np.ones(ensemble_size))
        P = self.get_precision_matrix(DX, self.r)

        # Ys = np.random.multivariate_normal(y, R, size=ensemble_size).T
        # D = Ys - H @ Xb
        # Rinv = np.diag(np.reciprocal(np.diag(R)))
        # IN = Binv + H.T @ (Rinv @ H)
        # Z = np.linalg.solve(IN, H.T @ (Rinv @ D))

        # Compute Kalman gain using a direct method
        # Step 1: Compute the term H * P
        HP = H @ P

        # Step 2: Compute S = H * P * H^T + R
        S = HP @ H.T + R

        # Step 3: Compute Kalman gain K
        K = HP.T @ np.linalg.inv(S)

        # Compute the updated state
        self.Xa = xb + K @ (y - H @ xb)
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
