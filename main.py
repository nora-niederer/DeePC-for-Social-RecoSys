import numpy as np
from deepc_for_recommender import npDeePC as DeePC
from deepc_for_recommender import npMPC as MPC
from data_generation import generate_data 
from simulator import runsim
import matplotlib.pyplot as plt
import math

# how many datasets we want to try
datasets = 1

# simulation range
simN = 10

# number of simulations
sims = 10
show_state_plt = False 
show_cost_plt = True

# Define system parameters (only relevant if new_data == True)
num_users = 20 
num_steps = 300 # How many data points collected
sparsity_factor = 0.5 # % of max connections
bias_factor = 1.0 # % of max bias

m = 1    # Dimension of input (always 1)
p = num_users   # Dimension of output

N = 10   # Prediction horizon
Tini = 1   # Initial time 

print(f"Trying {datasets} different datasets")

for i in range(datasets):
    runsim(True, num_users, num_steps, sparsity_factor, bias_factor, simN,
           sims, show_state_plt, show_cost_plt, N, Tini, f"data{i}")

