# Choose the desired method from the pool of available methods:
from analysis.analysis_enkf_direct_precision_shrinkage_1 import AnalysisEnKFDirectPrecisionShrinkage1
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
background = Background(model, ensemble_size=20)

# Create an analysis object with the desired method and its parameters + the chosen model:
analysis_direct = AnalysisEnKFDirectPrecisionShrinkage1(model, r=2)
analysis_nercome = AnalysisEnKFNercomeShrinkage(model, r=2)
analysis_ledoit_wolf = AnalysisEnKFLedoitWolfShrinkage(model, r=2)
analysis_cholesky = AnalysisEnKFModifiedCholesky(model, r=2)

# Define the observation parameters:
observation = Observation(m=32, std_obs=0.01)

# Set up the parameters for the simulation:
params = {'obs_freq': 0.1, 'obs_times': 10, 'inf_fact': 1.04}
simulation_direct = Simulation(model, background, analysis_direct, observation, params=params)
simulation_nercome = Simulation(model, background, analysis_nercome, observation, params=params)
simulation_ledoit_wolf = Simulation(model, background, analysis_ledoit_wolf, observation, params=params)
simulation_cholesky = Simulation(model, background, analysis_cholesky, observation, params=params)

# and then, run the simulation!
simulation_direct.run()
simulation_nercome.run()
simulation_ledoit_wolf.run()
simulation_cholesky.run()

# You can then request the backgound and analysis errors per assimilation step for plotting purposes or statistical computations:
import matplotlib.pyplot as plt

errb_direct, erra_direct = simulation_direct.get_errors()
errb_nercome, erra_nercome = simulation_nercome.get_errors()
errb_ledoit_wolf, erra_ledoit_wolf = simulation_ledoit_wolf.get_errors()
errb_cholesky, erra_cholesky = simulation_cholesky.get_errors()

plt.figure(figsize=(12, 10))
plt.plot(np.log10(errb_direct), color="green")
plt.plot(np.log10(erra_direct), color="green", linewidth=2, linestyle=":")

plt.plot(np.log10(errb_nercome), color="blue")
plt.plot(np.log10(erra_nercome), color="blue", linewidth=2, linestyle=":")

plt.plot(np.log10(errb_ledoit_wolf), color="red")
plt.plot(np.log10(erra_ledoit_wolf), color="red", linewidth=2, linestyle=":")

plt.plot(np.log10(errb_cholesky), color="black")
plt.plot(np.log10(erra_cholesky), color="black", linewidth=2, linestyle=":")

plt.show()

print(erra_direct, errb_direct)