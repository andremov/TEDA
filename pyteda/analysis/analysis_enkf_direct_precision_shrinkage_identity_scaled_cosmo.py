# -*- coding: utf-8 -*-

import numpy as np
import scipy as sci
from analysis.analysis import Analysis
from sklearn.linear_model import Ridge

class AnalysisEnKFDirectPrecisionShrinkageIdentityScaledCosmo(Analysis):
    """Analysis EnKF Direct Precision Shrinkage Identity Scaled Cosmological
    This class implements the exact cosmological precision matrix shrinkage method from 
    Looijmans et al. (2024), equations (14) and (16), in its original cosmological context.
    
    This implementation requires:
    1. Cosmological power spectrum C_ℓ data
    2. Survey-specific multipole moment sampling N_ℓ 
    3. Proper cosmological survey geometry
    4. Real cosmological data vectors mapped to (ℓ, redshift bin) pairs
    
    The target precision matrix follows: T⁽²⁾ᵢᵢ = (2/N_ℓ) × C_ℓ, then Pi0^(2) = (T^(2))^(-1)
    
    Attributes:
        model (Model object): An object that has all the methods and attributes of the model
        power_spectrum (dict): Cosmological power spectrum C_ℓ indexed by multipole ℓ
        survey_geometry (dict): Survey parameters including sky coverage, redshift bins, etc.
        multipole_sampling (dict): Number of modes N_ℓ available per multipole ℓ
        data_vector_mapping (list): Mapping of data vector indices to (ℓ, z_bin) pairs

    Methods:
        set_cosmological_parameters(...): Set cosmological context and data mapping
        get_precision_matrix(DX): Returns the computed precision matrix
        perform_assimilation(background, observation): Perform assimilation step given background and observations
        get_analysis_state(): Returns the computed column mean of ensemble Xa
        get_ensemble(): Returns ensemble Xa
        get_error_covariance(): Returns the computed covariance matrix of the ensemble Xa
        inflate_ensemble(inflation_factor): Computes new ensemble Xa given the inflation factor
    """

    def __init__(self, model, **kwargs):
        """
        Initialize an instance of AnalysisEnKFDirectPrecisionShrinkageIdentityScaledCosmo.

        Parameters:
            model (Model object): An object that has all the methods and attributes of the model given
        """
        self.model = model
        self.power_spectrum = None
        self.survey_geometry = None
        self.multipole_sampling = None
        self.data_vector_mapping = None
        self.cosmological_parameters_set = False

    def set_cosmological_parameters(self, power_spectrum, survey_geometry, multipole_sampling, data_vector_mapping):
        """
        Set the cosmological parameters required for the precision matrix computation.
        
        Parameters:
            power_spectrum (dict): Dictionary with multipole ℓ as keys and C_ℓ power spectrum values
                                 e.g., {2: 1.2e-9, 3: 8.5e-10, ...}
            survey_geometry (dict): Survey parameters including:
                                  - 'sky_fraction': Fraction of sky covered (0-1)
                                  - 'redshift_bins': Number of redshift bins
                                  - 'survey_area_deg2': Survey area in square degrees
            multipole_sampling (dict): Number of modes N_ℓ per multipole ℓ
                                     e.g., {2: 5, 3: 7, 4: 9, ...}
            data_vector_mapping (list): List of (ℓ, z_bin) tuples mapping each data vector element
                                      e.g., [(2, 0), (2, 1), (3, 0), (3, 1), ...]
        """
        self.power_spectrum = power_spectrum
        self.survey_geometry = survey_geometry
        self.multipole_sampling = multipole_sampling
        self.data_vector_mapping = data_vector_mapping
        self.cosmological_parameters_set = True
        
        print("Cosmological parameters set successfully.")
        print(f"Power spectrum range: ℓ = {min(power_spectrum.keys())} to {max(power_spectrum.keys())}")
        print(f"Survey area: {survey_geometry.get('survey_area_deg2', 'N/A')} deg²")
        print(f"Sky fraction: {survey_geometry.get('sky_fraction', 'N/A')}")
        print(f"Data vector dimension: {len(data_vector_mapping)}")

    def compute_target_precision_matrix(self, S_inv):
        """
        Compute the cosmological target precision matrix Pi0^(2) = (T^(2))^(-1) following 
        equations (14) and (16) from Looijmans et al. (2024).
        
        For each data vector element i mapped to (ℓ, z_bin):
        T⁽²⁾ᵢᵢ = (2/N_ℓ) × C_ℓ
        
        Parameters:
        S_inv (numpy.ndarray): The inverse of the sample covariance matrix (d x d).
        
        Returns:
        numpy.ndarray: The cosmological target precision matrix Pi0^(2) (d x d).
        """
        if not self.cosmological_parameters_set:
            raise RuntimeError(
                "Cosmological parameters not set. Call set_cosmological_parameters() first."
            )
        
        d = S_inv.shape[0]
        
        if len(self.data_vector_mapping) != d:
            raise ValueError(
                f"Data vector mapping length ({len(self.data_vector_mapping)}) "
                f"must match data vector dimension ({d})"
            )
        
        # Compute T⁽²⁾ diagonal elements following equation (14)
        T2_diagonal = np.zeros(d)
        
        for i, (ell, z_bin) in enumerate(self.data_vector_mapping):
            # Get C_ℓ from power spectrum
            if ell not in self.power_spectrum:
                raise ValueError(f"Multipole ℓ={ell} not found in power spectrum")
            C_ell = self.power_spectrum[ell]
            
            # Get N_ℓ from multipole sampling
            if ell not in self.multipole_sampling:
                raise ValueError(f"Multipole ℓ={ell} not found in multipole sampling")
            N_ell = self.multipole_sampling[ell]
            
            # Compute T⁽²⁾ diagonal element using equation (14)
            # T⁽²⁾ᵢᵢ = (2/N_ℓ) × C_ℓ
            T2_diagonal[i] = (2.0 / N_ell) * C_ell
        
        # Create diagonal covariance target T⁽²⁾
        T2 = np.diag(T2_diagonal)
        
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
        This method requires cosmological parameters to be set first.

        Parameters:
            DX (ndarray): Deviation matrix

        Returns:
            precision_matrix (ndarray): Precision matrix
        """
        if not self.cosmological_parameters_set:
            raise RuntimeError(
                "Cannot compute precision matrix without cosmological parameters. "
                "Call set_cosmological_parameters() first."
            )
        
        n, ensemble_size = DX.shape

        # Compute sample covariance matrix S
        S = np.cov(DX, rowvar=True, bias=True)
        
        # Compute empirical precision matrix (inverse of S)
        Pi_S_unscaled = np.linalg.inv(S)

        # Apply Hartlap factor to Pi_S (if applicable)
        if (ensemble_size - 1) > (n + 1):
            S_inv = ((ensemble_size - 1 - n) / (ensemble_size - 1)) * Pi_S_unscaled
        else:
            print(f"Warning: Ensemble size (n={ensemble_size}) is too small relative to dimension (d={n}) for standard inverse/Hartlap factor.")
            S_inv = Pi_S_unscaled

        # Compute the cosmological target precision matrix Pi0^(2)
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
        if not self.cosmological_parameters_set:
            raise RuntimeError(
                "Cannot perform assimilation without cosmological parameters. "
                "Call set_cosmological_parameters() first."
            )
            
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
        diag_IN = np.diag(IN)
        max_diag_IN = np.max(np.abs(diag_IN))
        
        if np.isclose(max_diag_IN, 0):
            epsilon_IN = 1e-6
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

def create_example_cosmological_parameters_for_lorenz96():
    """
    Create example cosmological parameters adapted for a 40-dimensional Lorenz96 system.
    This is for demonstration purposes only - real cosmological surveys would have 
    much more complex data vector structures.
    
    Returns:
        tuple: (power_spectrum, survey_geometry, multipole_sampling, data_vector_mapping)
    """
    # Example power spectrum (temperature anisotropies)
    # These values are illustrative - real values depend on cosmological parameters
    power_spectrum = {
        2: 1.2e-9,   # Large scale fluctuations  
        3: 8.5e-10,
        4: 6.8e-10,
        5: 5.9e-10,
        10: 2.1e-10,
        20: 8.5e-11,
        50: 1.2e-11,
        100: 3.4e-12,
    }
    
    # Example survey geometry (similar to Euclid or LSST)
    survey_geometry = {
        'survey_area_deg2': 15000,  # 15,000 square degrees
        'sky_fraction': 0.36,       # ~36% of sky
        'redshift_bins': 5,         # 5 tomographic redshift bins
        'galaxy_density': 30,       # galaxies per arcmin²
    }
    
    # Example multipole sampling (depends on survey area and ℓ)
    # N_ℓ ≈ (2ℓ + 1) × f_sky for full-sky surveys
    multipole_sampling = {}
    f_sky = survey_geometry['sky_fraction']
    
    for ell in power_spectrum.keys():
        # Simplified calculation - real surveys have more complex mode counting
        N_ell = int((2 * ell + 1) * f_sky)
        multipole_sampling[ell] = max(N_ell, 1)  # Ensure at least 1 mode
    
    # Create a mapping for 40-dimensional Lorenz96 system
    # Map each of the 40 variables to (ℓ, z_bin) pairs
    data_vector_mapping = []
    multipoles = list(power_spectrum.keys())
    n_redshift_bins = survey_geometry['redshift_bins']
    
    for i in range(40):  # Lorenz96 has 40 variables
        # Cycle through multipoles and redshift bins
        ell_idx = i % len(multipoles)
        z_bin = (i // len(multipoles)) % n_redshift_bins
        ell = multipoles[ell_idx]
        data_vector_mapping.append((ell, z_bin))
    
    return power_spectrum, survey_geometry, multipole_sampling, data_vector_mapping

"""
# Create example parameters for Lorenz96
power_spectrum, survey_geometry, multipole_sampling, data_vector_mapping = create_example_cosmological_parameters_for_lorenz96()

# Initialize the cosmological precision analysis
cosmo_analysis = AnalysisEnKFDirectPrecisionShrinkageIdentityScaledCosmo(model)

# Set cosmological parameters
cosmo_analysis.set_cosmological_parameters(
    power_spectrum=power_spectrum,
    survey_geometry=survey_geometry,
    multipole_sampling=multipole_sampling,
    data_vector_mapping=data_vector_mapping
)

# Now the analysis can be used with proper cosmological context
# result = cosmo_analysis.perform_assimilation(background, observation)
"""
