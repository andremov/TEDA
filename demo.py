# Choose the desired method from the pool of available methods:
from analysis.analysis_enkf_cosmological_precision import AnalysisEnKFCosmologicalPrecision
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
analysis = AnalysisEnKFCosmologicalPrecision(model, r=2)

# Define the observation parameters:
observation = Observation(m=32, std_obs=0.01)

# Set up the parameters for the simulation:
params = {'obs_freq': 0.1, 'obs_times': 10, 'inf_fact': 1.04}
simulation = Simulation(model, background, analysis, observation, params=params)

# and then, run the simulation!
simulation.run()

# You can then request the backgound and analysis errors per assimilation step for plotting purposes or statistical computations:
import matplotlib.pyplot as plt

errb, erra = simulation.get_errors()

plt.figure(figsize=(12, 10))
plt.plot(np.log10(errb),'-ob')
plt.plot(np.log10(erra),'-or')
plt.show()