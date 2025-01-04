import numpy as np
from deepc_for_recommender import npDeePC as DeePC
from deepc_for_recommender import npMPC as MPC
from data_generation import generate_data 
import matplotlib.pyplot as plt
import math


def runsim(new_data, num_users, num_steps, sparsity_factor, bias_factor,
           simrange, sims, plt_state, plt_cost, show_acc_cost,
           N, Tini, noise=False, lam_g1=None, lam_g2=None, lam_y=None, data_name="data"):
    
    #print(f"Running {sims} Simulations with {N} prediction Horizon and Tini={Tini}")

    m = 1    # Dimension of input (always 1)
    p = num_users   # Dimension of output

    if new_data:
        # Call skript to generate data
        #print(f"Starting Data generation for {num_users} users, {num_steps} steps, sparsity={sparsity_factor} and bias={bias_factor}")
        generate_data(num_users, num_steps, sparsity_factor, bias_factor, noise,data_name)
        #print("Finished Data collection")
    else: 
        print("Skipping new Data generation")

    # Load the data from the .npz file
    data = np.load(f"{data_name}.npz")

    # Access the individual arrays by their names
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
    mpc = MPC(A, np.expand_dims(B, axis=1), Lambda, N, u_constraints, y_constraints)

    xevo_deepc = np.zeros((sims, num_users, simrange+Tini))
    xevo_mpc = np.zeros((sims, num_users, simrange+Tini))
    uevo_deepc = np.zeros((sims, simrange+Tini-1))
    uevo_mpc = np.zeros((sims, simrange+Tini-1))

    for s in range(sims):
        #print(f"Running simulation #{s+1}..")
        # Initialize matrices and vectors
        x0 = yd[:, 0]
        # x0.sort()
        xevo_deepc[s, :, 0] = x0  # initial state for all users
        xevo_mpc[s, :, 0] = x0  # initial state for all users
        #uevo_deepc[s, 0] = np.mean(xevo_deepc[s, :, 0])
        uevo_deepc[s, 0] = ud[0]
        uevo_mpc[s, 0] = uevo_deepc[s, 0]

        if(Tini > 1):
            uevo_deepc[s, 1:Tini] = np.random.rand(Tini-1)  
            uevo_mpc[s, 1:Tini] = uevo_deepc[s, 1:Tini]  

        # Setup model
        deepc.setup(lam_g1, lam_g2, lam_y)
        mpc.setup(x0)

        #Simulate Tini steps
        for k in range(Tini):
            x_deepc = xevo_deepc[s, :, k]
            x_deepc = A @ x_deepc + B * uevo_deepc[s, k] + Lambda @ x0
            xevo_deepc[s, :, k+1] = x_deepc

            x_mpc = xevo_mpc[s, :, k]
            x_mpc = A @ x_mpc + B * uevo_mpc[s, k] + Lambda @ x0
            xevo_mpc[s, :, k+1] = x_mpc


        # For loop for all the time steps
        for k in range(Tini, simrange-1+Tini):
            # A is (num_users, num_users), xevo[:, k] is (num_users,)
            # B is (num_users,), and uevo[k] is scalar
            if noise:
                noise_vector = np.random.uniform(low=-noise, high=noise, size=xevo_mpc[s, :, k].flatten().size)
                optimal_behaviour_deepc = deepc.solve(uevo_deepc[s, k-Tini:k], np.clip(noise_vector+xevo_deepc[s,:,k], 0, 1))
                optimal_behaviour_mpc = mpc.solve(np.clip(noise_vector+xevo_mpc[s,:,k], 0, 1))
            else:
            # Calculate optimal inputs
                optimal_behaviour_deepc = deepc.solve(uevo_deepc[s, k-Tini:k], xevo_deepc[s, :, k-Tini:k].flatten())
                optimal_behaviour_mpc = mpc.solve(xevo_mpc[s, :, k].flatten())

            # Update the input vector
            uevo_deepc[s, k] = optimal_behaviour_deepc[0][0]
            uevo_mpc[s, k] = optimal_behaviour_mpc[0][0]

            # Applying input
            x_deepc = xevo_deepc[s, :, k]
            x_deepc = A @ x_deepc + B * uevo_deepc[s, k] + Lambda @ x0
            xevo_deepc[s, :, k+1] = x_deepc

            x_mpc = xevo_mpc[s, :, k]
            x_mpc = A @ x_mpc + B * uevo_mpc[s, k] + Lambda @ x0
            xevo_mpc[s, :, k+1] = x_mpc

    if plt_state:
        # Calculate the number of columns (3 rows per column, or adjust as needed)
        max_rows_per_column = 3
        num_columns = 2*math.ceil(sims / max_rows_per_column)

        # Calculate the number of rows
        num_rows = sims
        if sims > max_rows_per_column: 
            num_rows = 3

        # Create a figure and axes for the grid layout
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(4 * num_columns, 4 * num_rows), sharex=True)

        # Plot for each subplot (DeepC on left side, MPC on right side)
        for i in range(sims):
            # Plot DeepC on the left column
            if sims > 1:
                ax_deepc = axes[i%3, math.floor(i/3)]  # Subplot for DeepC
            else:
                ax_deepc = axes[i]
            for k in range(num_users):
                handle0, = ax_deepc.plot(xevo_deepc[i, k, :], linewidth=0.5, color=[0, 0, 0])  # Plot each user's trajectory in black
            handle1, = ax_deepc.plot(np.mean(xevo_deepc[i, :, :], axis=0), linewidth=1.0, color='magenta', label='Mean Opinion')
            handle2, = ax_deepc.plot(uevo_deepc[i, :], linewidth=1.0, color='b', label='Input (DeepC)')
            ax_deepc.set_title(f"DeePC - Plot {i+1}", pad=15)
            ax_deepc.set_xlabel('Timestep')
            ax_deepc.set_ylabel('Opinion')
            ax_deepc.grid(True)

            # Plot MPC on the right column
            if sims > 1:
                ax_mpc = axes[i%3, math.floor(num_columns/2) + math.floor(i/3)]  # Subplot for MPC
            else:
                ax_mpc = axes[sims+i]
            for k in range(num_users):
                ax_mpc.plot(xevo_mpc[i, k, :], linewidth=0.5, color=[0, 0, 0])  # Plot each user's trajectory in black
            ax_mpc.plot(np.mean(xevo_mpc[i, :, :], axis=0), linewidth=1.0, color='magenta', label='Mean Opinion')
            ax_mpc.plot(uevo_mpc[i, :], linewidth=1.0, color='b', label='Input (MPC)')
            ax_mpc.set_title(f"MPC - Plot {i+1}", pad=15)
            ax_mpc.set_xlabel('Timestep')
            ax_mpc.set_ylabel('Opinion')
            ax_mpc.grid(True)

        # Add legend and title
        fig.legend([handle0, handle1, handle2], ["State i's opinion", "Mean opinion", "Proposed Input by Algo"], loc='upper center', ncol=3, fontsize=12, bbox_to_anchor=(0.5, 0.95), frameon=False)
        fig.suptitle(f"Comparison of DeePC and MPC Opinion Trajectories") # (num_steps={num_steps}, Tini={Tini})", fontsize=16, fontweight='bold')



        # Adjust the layout and spacing between subplots
        plt.tight_layout(pad = 3 + sims/2)  # Add more padding between subplots

        plt.savefig(f"StatePlot_Users{num_users}_Steps{simrange}_DataPoints{num_steps}_Horizon{N}_Sparsity{sparsity_factor}_Bias{bias_factor}_Tini{Tini}.pdf", format='pdf')

        plt.show()

    if plt_cost:

        # Calculate costs
        cost_deepc_sort = np.zeros((sims, simrange+Tini-1))
        #cost_deepc_unsort = np.zeros((sims-20, simrange+Tini-1))
        cost_mpc = np.zeros((sims, simrange+Tini-1))
        
        for i in range(sims):
            #if i < 20:
            cost_deepc_sort[i, :] = np.sum((xevo_deepc[i, :, :-1] - uevo_deepc[i, :]) ** 2, axis=0)
            #else:
            #    cost_deepc_unsort[i-20, :] = np.sum((xevo_deepc[i, :, :-1] - uevo_deepc[i, :]) ** 2, axis=0)
            cost_mpc[i, :] = np.sum((xevo_mpc[i, :, :-1] - uevo_mpc[i, :]) ** 2, axis=0)

        cost_deepc_sort = np.mean(cost_deepc_sort, axis=0)
        #cost_deepc_unsort = np.mean(cost_deepc_unsort, axis=0)
        cost_mpc = np.mean(cost_mpc, axis=0)

        # Plot the first line
        plt.plot(cost_deepc_sort[Tini:-1], label=f"DeePC", color='blue')
        #plt.plot(cost_deepc_unsort[Tini:-1], label=f"DeePC Unsorted", color='green')

        # Plot the second line
        plt.plot(cost_mpc[Tini:-1], label=f"MPC", color='red')

        # Customize the plot
        plt.title(f"Mean cost for {sims} simulations and Tini={Tini}")  # Title of the plot
        plt.xlabel("Timestep")  # Label for the x-axis
        plt.ylabel("Cost")  # Label for the y-axis
        plt.legend()  # Show a legend to identify the lines
        plt.grid(True)  # Add grid lines for better readability

        #plt.yscale('log')

        plt.savefig('evo1.pdf', format='pdf')

        # Display the plot
        plt.show(block=True)
        #plt.pause(0.2)

    return [np.sum((xevo_deepc[sims-1, :, -2] - uevo_deepc[sims-1, -1]) ** 2, axis=0), np.sum((xevo_mpc[sims-1, :, -2] - uevo_mpc[sims-1, -1]) ** 2, axis=0)]