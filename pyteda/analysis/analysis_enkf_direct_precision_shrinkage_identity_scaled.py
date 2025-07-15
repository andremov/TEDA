# -*- coding: utf-8 -*-

import numpy as np
from .analysis_core import Analysis

class AnalysisEnKFDirectPrecisionShrinkageIdentityScaled(Analysis):
    """Analysis EnKF Direct Precision Shrinkage Identity Scaled
    This class implements the EnKF Direct Precision Shrinkage method with the second target precision matrix 
    choice from Looijmans et al. (2024). Following equations (14) and (16), it computes Pi0^(2) = (T^(2))^(-1)
    where T^(2) is a diagonal covariance matrix target. For general applications beyond cosmology, we use
    the diagonal variances of the sample covariance matrix as an analogy to the cosmological power spectrum.
    The shrinkage weights are computed based on the empirical precision matrix and the target precision matrix.
    The method is designed to improve the performance of the ensemble Kalman filter by reducing the impact 
    of noise in the covariance estimation.
    The method is particularly useful in high-dimensional state spaces where the number of observations is small
    compared to the number of state variables.
    
    Attributes:
        model (Model object): An object that has all the methods and attributes of the model

    Methods:
        get_precision_matrix(DX, regularization_factor=0.01): Returns the computed precision matrix
        perform_assimilation(background, observation): Perform assimilation step given background and observations
        get_analysis_state(): Returns the computed column mean of ensemble Xa
        get_ensemble(): Returns ensemble Xa
        get_error_covariance(): Returns the computed covariance matrix of the ensemble Xa
        inflate_ensemble(inflation_factor): Computes new ensemble Xa given the inflation factor
    """

    def __init__(self, model, **kwargs):
        """
        Initialize an instance of AnalysisEnKFDirectPrecisionShrinkageIdentityScaled.

        Parameters:
            model (Model object): An object that has all the methods and attributes of the model given
        """
        self.model = model

    def compute_target_precision_matrix(self, S_inv):
        """
        Compute the target precision matrix Pi0^(2) = (T^(2))^(-1), following equations (14) and (16)
        from Looijmans et al. (2024). For general applications, we use the diagonal variances
        of the sample covariance matrix as an analogy to the cosmological power spectrum.
        
        Parameters:
        S_inv (numpy.ndarray): The inverse of the sample covariance matrix (d x d).
        
        Returns:
        numpy.ndarray: The target precision matrix Pi0^(2) (d x d).
        """
        d = S_inv.shape[0]
        
        # Get the sample covariance matrix S from S_inv
        S = np.linalg.inv(S_inv)
        
        # Use diagonal variances as our "power spectrum" analogy
        variances = np.diag(S)
        
        # Use a scaling factor analogous to 2/N_l from equation (14)
        # For general applications, we use a simple constant scaling
        scale_factor = 2.0
        
        # Create diagonal covariance target T^(2) following equation (14) structure
        T2_diag = scale_factor * variances
        T2 = np.diag(T2_diag)
        
        # Return the precision matrix Pi0^(2) = (T^(2))^(-1) following equation (16)
        Pi0 = np.linalg.inv(T2)
        
        return Pi0

    def compute_shrinkage_weights(self, S_inv, Pi0, data_vector_length, ensemble_size):
        """
        Compute the optimal shrinkage weights alpha* and beta* for the precision matrix.

        Parameters:
            S_inv (numpy.ndarray): Inverse sample covariance matrix.
            Pi0 (numpy.ndarray): Target precision matrix.
            data_vector_length (int):  The dimension of the data vector.
            ensemble_size (int): The ensemble size.

        Returns:
            tuple: (alpha_star, beta_star)
        """

        # Compute squared trace norm of S_inv
        singular_values = np.linalg.svd(S_inv, compute_uv=False)
        squared_trace_norm = np.sum(singular_values) ** 2
        
        # Compute squared Frobenius norm of Pi0
        frobenius_norm_sq_Pi0 = np.linalg.norm(Pi0, 'fro') ** 2

        # Compute trace of (S_inv * Pi0)
        trace_S_inv_Pi0 = np.trace(S_inv @ Pi0)

        # Compute α* and β*
        numerator_alpha = (ensemble_size ** -1) * squared_trace_norm * frobenius_norm_sq_Pi0
        denominator_alpha = (squared_trace_norm * frobenius_norm_sq_Pi0) - (trace_S_inv_Pi0 ** 2)

        # Handle potential division by zero
        if np.isclose(denominator_alpha, 0):
            print("Warning: Denominator for alpha_star is close to zero. Defaulting to full shrinkage (alpha=0).")
            alpha_star_unclamped = 0.0 # Or consider alternative handling based on paper's edge cases
        else:
            alpha_star_unclamped = 1 - (data_vector_length / ensemble_size) - (numerator_alpha / denominator_alpha)

        first_term_beta = 1 - (data_vector_length / ensemble_size) - alpha_star_unclamped

        # Handle potential division by zero for beta_star's denominator
        if np.isclose(frobenius_norm_sq_Pi0, 0):
             print("Warning: Frobenius norm of Pi0 is close to zero. Defaulting to 0 for beta_star contribution.")
             beta_star_unclamped = 0.0
        else:
             beta_star_unclamped = first_term_beta * (trace_S_inv_Pi0 / frobenius_norm_sq_Pi0)

        # Clamping alpha* and beta* to be within [0, 1] and sum <= 1
        alpha_star = max(0, alpha_star_unclamped)
        beta_star = max(0, beta_star_unclamped)

        # Ensure that alpha_star + beta_star <= 1. If sum > 1, adjust beta_star.
        if alpha_star + beta_star > 1:
            beta_star = 1 - alpha_star

        return alpha_star, beta_star

    def get_precision_matrix(self, DX):
        """
        Perform calculations to get the precision matrix given the deviation matrix.

        Parameters:
            DX (ndarray): Deviation matrix

        Returns:
            precision_matrix (ndarray): Precision matrix
        """
        n, ensemble_size = DX.shape

        # Compute sample covariance matrix S
        S = np.cov(DX, rowvar=True, bias=True)
        
        # Compute empirical precision matrix (inverse of S)
        Pi_S_unscaled = np.linalg.inv(S)

        # Apply Hartlap factor to Pi_S (if applicable)
        # This factor is crucial for optimal performance.
        # It requires n_val > d_dim + 1. If not, the inverse is singular, and Hartlap factor might be ill-defined.
        if (ensemble_size - 1) > (n + 1): # Check for validity of the Hartlap factor
            S_inv = ((ensemble_size - 1 - n) / (ensemble_size - 1)) * Pi_S_unscaled
        else:
            # If n_val is small, Pi_S_unscaled might be singular.
            # You might need to use a pseudo-inverse or add Tikhonov regularization here
            # before passing to compute_shrinkage_weights, as np.linalg.inv will fail.
            print(f"Warning: Ensemble size (n={ensemble_size}) is too small relative to dimension (d={n}) for standard inverse/Hartlap factor. Consider alternative regularization.")
            S_inv = Pi_S_unscaled # Or use np.linalg.pinv(S) or add a small diagonal loading

        # Define the target precision matrix Pi0^(2) = (T^(2))^(-1) following equations (14) and (16)
        Pi0 = self.compute_target_precision_matrix(S_inv)

        alpha_star, beta_star = self.compute_shrinkage_weights(S_inv, Pi0, n, ensemble_size)

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

        Binv = self.get_precision_matrix(DX)

        D = Ys - H @ Xb
        Rinv = np.diag(np.reciprocal(np.diag(R)))
        IN = Binv + H.T @ (Rinv @ H)

        # Calculate a dynamic epsilon based on IN's trace or max diagonal element
        # Use absolute values for robustness against negative diagonals (shouldn't happen here, but good practice)
        diag_IN = np.diag(IN)
        max_diag_IN = np.max(np.abs(diag_IN))
        
        # If IN is essentially a zero matrix or has tiny values, max_diag_IN could be zero
        if np.isclose(max_diag_IN, 0):
            epsilon_IN = 1e-6 # Fallback if max_diag is zero
        else:
            epsilon_IN = max_diag_IN * 1e-6
        
        IN_regularized = IN + epsilon_IN * np.eye(IN.shape[0])
        
        Z = np.linalg.solve(IN_regularized, H.T @ (Rinv @ D))
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
