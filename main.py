import numpy as np
from deepc_for_recommender import npDeePC as DeePC
from deepc_for_recommender import npMPC as MPC
from data_generation import generate_data 
from simulator import runsim
import matplotlib.pyplot as plt
import os

# how many datasets we want to try
datasets = 100

# simulation range
simN = 25

# number of simulations
sims = 1
show_state_plt = False 
show_cost_plt = False
show_acc_plt = True

# Define system parameters (only relevant if new_data == True)
new_data = True
num_users = 20 
num_steps = 250 # How many data points collected
sparsity_factor = 0.5 # % of max connections
bias_factor = 1.0 # % of max bias

m = 1    # Dimension of input (always 1)
p = num_users   # Dimension of output

N = 5   # Prediction horizon
Tini = 1   # Initial time 
noise = None

print(f"Trying {datasets} different datasets: ")

errors_system1 = [] 
errors_system2 = [] 

for i in range(datasets):
    print(f"Trying Dataset#{i}")
    DeePC_error, MPC_error = runsim(new_data, num_users, num_steps, sparsity_factor, bias_factor, simN,
           sims, show_state_plt, show_cost_plt, show_acc_plt, N, Tini, noise, None, None, None, f"data{i}")
#                                                                              ^g1    ^g2   ^y

    errors_system1.append(DeePC_error)
    errors_system2.append(MPC_error)


# Calculate the median of the errors for each system
median_system1 = np.median(errors_system1)
median_system2 = np.median(errors_system2)

# Create a figure and axis for plotting
fig, ax = plt.subplots()

# X positions for the bars
x = np.arange(2)

# Plot bars representing the median errors for each system
bars = ax.bar(x, [median_system1, median_system2], color=['red', 'blue'], alpha=0.3, label='Median Error', 
              edgecolor='black', linewidth=2)

# Highlight the top edge of the bars by adjusting the edgecolor and linewidth
for bar in bars:
    bar.set_edgecolor('black')  # Set the top edge color
    bar.set_linewidth(2)        # Set the thickness of the edge line

jitter = 0.05  # Adjust the spread of the dots

# For System 1 (add jitter to the x positions)
ax.scatter(np.full_like(errors_system1, x[0]) + np.random.uniform(-jitter, jitter, size=len(errors_system1)), 
           errors_system1, color='black', zorder=5)

# For System 2 (add jitter to the x positions)
ax.scatter(np.full_like(errors_system2, x[1]) + np.random.uniform(-jitter, jitter, size=len(errors_system2)), 
           errors_system2, color='black', zorder=5)

# Customize labels, title, and legend
ax.set_xticks(x)
ax.set_xticklabels(['DeePC', 'MPC'])
ax.set_ylabel('Steady-State Error')
ax.set_title('Comparison of Steady-State Errors: Median with Spread')
#ax.legend(['Steady-State Errors'], loc='upper right')

plt.savefig(f"Test_Users{num_users}_Steps{simN}_DataPoints{num_steps}_Horizon{N}_Sparsity{sparsity_factor}_Bias{bias_factor}_Tini{Tini}.pdf", format='pdf')

# Show plot
plt.tight_layout()
plt.show()

for i in range(datasets):
    os.remove(f"data{i}.npz")