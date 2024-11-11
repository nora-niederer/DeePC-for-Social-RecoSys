import numpy as np
import torch
from deepc_for_recommender import npDeePC as DeePC
from deepc_for_recommender import npMPC as MPC
from data_generation import generate_data  # replace with your actual file name
import matplotlib.pyplot as plt

# if we want new data for deepc
new_data = False

# simulation range
simN = 20

# how many sims?
sims = 0 # does currently nothing

# Define system parameters (only relevant if new_data == True)
num_users = 20 
num_steps = 300 # How many data points collected
sparsity_factor = 0.5 # % of max connections
bias_factor = 1.0 # % of max bias

m = 1    # Dimension of input (always 1)
p = num_users   # Dimension of output

# Initialize DeePC
N = 5    # Prediction horizon
Tini = 1   # Initial time horizon

#Define cost matrizes
Q = np.eye(num_steps*num_users)
R = np.eye(num_steps)

if new_data:
    #call skript to generate data
    print("Starting Data generation")
    generate_data(num_users, num_steps, sparsity_factor, bias_factor)
    print("Finished Data collection")
else: 
    print("Skipping Data generation")

# # Load the data from the .npz file
data = np.load('data.npz')

# # Access the individual arrays by their names
A = data['A']
B = data['B']
Lambda = data['Lambda']
ud = data['ud']
yd = data['xd']

# Define constraints
y_constraints = (np.array([0]), np.array([1]))  # Output must be between 0 and 1
u_constraints = (np.array([0]), np.array([1]))  # Input must be between 0 and 1

# Create the DeePC model
deepc = DeePC(ud, yd, y_constraints, u_constraints, N, Tini, num_users, p, m)
mpc = MPC(A, np.expand_dims(B, axis=1), N, u_constraints, y_constraints)

# Setup model (this exists because you would give it the reference values and Q and R but we don't need any of those)
deepc.setup()
mpc.setup()

# Initialize matrices and vectors
x0 = np.random.rand(num_users)
xevo_deepc = np.zeros((num_users, simN+Tini))
xevo_deepc[:, 0] = x0  # initial state for all users
xevo_mpc = np.zeros((num_users, simN+Tini))
xevo_mpc[:, 0] = x0  # initial state for all users
uevo_deepc = np.zeros(simN-1+Tini)
uevo_deepc[0:Tini] = np.random.rand(Tini)  # Initial input values (scalar)
uevo_mpc = np.zeros(simN-1+Tini)
uevo_mpc[0:Tini] = uevo_deepc[0:Tini]  # Initial input value (scalar)

#Simulate Tini steps
for k in range(Tini):
    x_deepc = xevo_deepc[:,k ]
    x_deepc = A @ x_deepc + B * uevo_deepc[k - 1] + Lambda @ x0
    xevo_deepc[:, k+1] = x_deepc

    x_mpc = xevo_mpc[:,k ]
    x_mpc = A @ x_mpc + B * uevo_mpc[k - 1] + Lambda @ x0
    xevo_mpc[:, k+1] = x_mpc


# For loop for all the time steps
for k in range(Tini, simN-1+Tini):
    # Ensure dimensional consistency
    # A is (num_users, num_users), xevo[:, k] is (num_users,)
    # B is (num_users,), and uevo[k] is scalar

    x_deepc = xevo_deepc[:,k ]
    x_deepc = A @ x_deepc + B * uevo_deepc[k - 1] + Lambda @ x0
    xevo_deepc[:, k+1] = x_deepc

    x_mpc = xevo_mpc[:,k ]
    x_mpc = A @ x_mpc + B * uevo_mpc[k - 1] + Lambda @ x0
    xevo_mpc[:, k+1] = x_mpc

    # Calculate optimal inputs
    # Assuming model.solve() returns a tuple or array where the first element is the optimal input
    optimal_behaviour_deepc = deepc.solve(uevo_deepc[k-Tini:k].flatten(), xevo_deepc[:, k-Tini:k].flatten())
    optimal_behaviour_mpc = mpc.solve(xevo_mpc[:,k].flatten())

    # Update the input vector
    uevo_deepc[k] = optimal_behaviour_deepc[0]
    uevo_mpc[k] = optimal_behaviour_mpc[0]

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot deepc results on the first subplot
for k in range(num_users):
    axes[0].plot(xevo_deepc[k, :], linewidth=0.5, color=[0, 0, 0])  # Plot each user's trajectory in black

axes[0].plot(np.mean(xevo_deepc, axis=0), linewidth=1.0, color='magenta', label='Mean Opinion (DeepC)')
axes[0].plot(uevo_deepc, linewidth=1.0, color='b', label='Input (DeepC)')
axes[0].set_title('DeepC Opinion Trajectories')
axes[0].set_ylabel('Opinion')
axes[0].legend(loc='upper right')
axes[0].grid(True)

# Plot mpc results on the second subplot
for k in range(num_users):
    axes[1].plot(xevo_mpc[k, :], linewidth=0.5, color=[0, 0, 0])  # Plot each user's trajectory in black

axes[1].plot(np.mean(xevo_mpc, axis=0), linewidth=1.0, color='magenta', label='Mean Opinion (MPC)')
axes[1].plot(uevo_mpc, linewidth=1.0, color='b', label='Input (MPC)')
axes[1].set_title('MPC Opinion Trajectories')
axes[1].set_xlabel('Timestep')
axes[1].set_ylabel('Opinion')
axes[1].legend(loc='upper right')
axes[1].grid(True)

# Show the plot
plt.tight_layout()
plt.show()