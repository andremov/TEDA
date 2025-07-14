# Choose the desired method from the pool of available methods:
from analysis.analysis_enkf_direct_precision_shrinkage_eigenvalues import AnalysisEnKFDirectPrecisionShrinkageEigenvalues
from analysis.analysis_enkf_direct_precision_shrinkage_identity import AnalysisEnKFDirectPrecisionShrinkageIdentity
from analysis.analysis_enkf_direct_precision_shrinkage_identity_scaled import AnalysisEnKFDirectPrecisionShrinkageIdentityScaled
from analysis.analysis_enkf_nercome_shrinkage import AnalysisEnKFNercomeShrinkage
from analysis.analysis_enkf_ledoit_wolf_shrinkage import AnalysisEnKFLedoitWolfShrinkage
from analysis.analysis_enkf_modified_cholesky import AnalysisEnKFModifiedCholesky
from background.background import Background
from models.lorenz96 import Lorenz96
from observation.observation import Observation
from simulation.simulation import Simulation
import numpy as np

# Select a model to perform the simulations:
model = Lorenz96()

# Create a background object with the desired parameters:
background = Background(model, ensemble_size=50)

# Create an analysis object with the desired method and its parameters + the chosen model:
analysis_direct_identity = AnalysisEnKFDirectPrecisionShrinkageIdentity(model)
analysis_direct_identity_scaled = AnalysisEnKFDirectPrecisionShrinkageIdentityScaled(model)
analysis_direct_eigen = AnalysisEnKFDirectPrecisionShrinkageEigenvalues(model)
analysis_nercome = AnalysisEnKFNercomeShrinkage(model)
analysis_ledoit_wolf = AnalysisEnKFLedoitWolfShrinkage(model)
analysis_cholesky = AnalysisEnKFModifiedCholesky(model, r=2)

# Define the observation parameters:
observation = Observation(m=32, std_obs=0.01)

# Set up the parameters for the simulation:
params = {'obs_freq': 0.1, 'obs_times': 10, 'inf_fact': 1.04}
simulation_direct_identity = Simulation(model, background, analysis_direct_identity, observation, params=params)
simulation_direct_identity_scaled = Simulation(model, background, analysis_direct_identity_scaled, observation, params=params)
simulation_direct_eigen = Simulation(model, background, analysis_direct_eigen, observation, params=params)
simulation_nercome = Simulation(model, background, analysis_nercome, observation, params=params)
simulation_ledoit_wolf = Simulation(model, background, analysis_ledoit_wolf, observation, params=params)
simulation_cholesky = Simulation(model, background, analysis_cholesky, observation, params=params)

# and then, run the simulation!
simulation_direct_identity.run()
simulation_direct_identity_scaled.run()
simulation_direct_eigen.run()
simulation_nercome.run()
simulation_ledoit_wolf.run()
simulation_cholesky.run()

# You can then request the backgound and analysis errors per assimilation step for plotting purposes or statistical computations:
import matplotlib.pyplot as plt

errb_direct_identity, erra_direct_identity = simulation_direct_identity.get_errors()
errb_direct_identity_scaled, erra_direct_identity_scaled = simulation_direct_identity_scaled.get_errors()
errb_direct_eigen, erra_direct_eigen = simulation_direct_eigen.get_errors()
errb_nercome, erra_nercome = simulation_nercome.get_errors()
errb_ledoit_wolf, erra_ledoit_wolf = simulation_ledoit_wolf.get_errors()
errb_cholesky, erra_cholesky = simulation_cholesky.get_errors()

plt.figure(figsize=(12, 10))
plt.plot(np.log10(errb_direct_identity), color="orange", label="Direct Identity (Background)")
plt.plot(np.log10(erra_direct_identity), color="orange", linewidth=2, linestyle=":", label="Direct Identity (Analysis)")

plt.plot(np.log10(errb_direct_identity_scaled), color="tomato", label="Direct Identity Scaled (Background)")
plt.plot(np.log10(erra_direct_identity_scaled), color="tomato", linewidth=2, linestyle=":", label="Direct Identity Scaled (Analysis)")

plt.plot(np.log10(errb_direct_eigen), color="firebrick", label="Direct Eigen (Background)")
plt.plot(np.log10(erra_direct_eigen), color="firebrick", linewidth=2, linestyle=":", label="Direct Eigen (Analysis)")

plt.plot(np.log10(errb_nercome), color="steelblue", label="Nercome (Background)")
plt.plot(np.log10(erra_nercome), color="steelblue", linewidth=2, linestyle=":", label="Nercome (Analysis)")

plt.plot(np.log10(errb_ledoit_wolf), color="aqua", label="Ledoit Wolf (Background)")
plt.plot(np.log10(erra_ledoit_wolf), color="aqua", linewidth=2, linestyle=":", label="Ledoit Wolf (Analysis)")

plt.plot(np.log10(errb_cholesky), color="fuchsia", label="Cholesky (Background)")
plt.plot(np.log10(erra_cholesky), color="fuchsia", linewidth=2, linestyle=":", label="Cholesky (Analysis)")

plt.legend()
plt.xlabel("Assimilation Step")
plt.ylabel("Log10(Error)")
plt.title("Comparison of EnKF Methods")
plt.show()
