import numpy as np
from deepc_for_recommender import npDeePC as DeePC
from deepc_for_recommender import npMPC as MPC
from data_generation import generate_data 
from simulator import runsim
import matplotlib.pyplot as plt
import os
import itertools
import pandas as pd


# simulation range
simN = 15

# number of simulations
sims = 1 #artifact from old code, always leave on 1.

#what to plot
show_state_plt = False #State trajectories
show_cost_plt = False #Cost trajectory 
show_acc_plt = False #Steady state spectrum
show_paramcompare_plt = True #Parameter tuning

# Define system parameter to check
new_data = True #artifact from old code, always leave on true.

#Different testing parameters
num_users = (15,) 
num_datapoints = (200,) #for DeePC
sparsity_factor = (0.5,) # floating point ratio of max connections
bias_factor = (1.0,) # floating point ratio of max bias 
horizon = (5,)   # Prediction horizon
noise_var = (0, 0.1, 0.25,1, 4) #variance floating point value of random normal distr. noise

m = 1    # Dimension of input (always 1)
p = num_users   # Dimension of output

#Don't put too high otherwise takes forever to run!
datasets =  3000 #of data sets to try

#Technically testing parameter but saw no result difference would need to be added to parameter combination.
Tini = 1   # Initial time 

totalcost = True #Collect total cost or just final cost?

# Set regularization parameters for DeePC
DeePC_g1 = None
DeePC_g2 = None
DeePC_y = 0.1

#Set regularization parameters for MPC

MPC_y = 0.1

# Generate all combinations of the parameters
parameter_combinations = itertools.product(num_users, num_datapoints, sparsity_factor, bias_factor, horizon, noise_var)
 
# Initialize an empty list to store the results
results = []

for users, datapoints, sparsity, bias, N, noise in parameter_combinations:
    print(f"Trying {datasets} different datasets: ")
    errors_system1 = [] 
    errors_system2 = [] 
    print(users, datapoints, sparsity, bias, N, noise)
    for i in range(datasets):
        print(f"Trying Dataset#{i}")
        try:
            DeePC_error, MPC_error = runsim(new_data, users, datapoints, sparsity, bias, simN,
                sims, show_state_plt, show_cost_plt, show_acc_plt, N, Tini, totalcost, noise, 1, DeePC_g1, DeePC_g2, DeePC_y, MPC_y,f"data{i}")
        except Exception as e:
            # Catch any error that occurs during the simulation
            print(f"Solver failed during simulation {i}. Skipping this dataset.")

            
        errors_system1.append(DeePC_error)
        errors_system2.append(MPC_error)

     # Save the results for this parameter combination
    results.append({
        'users': users,
        'datapoints': datapoints,
        'sparsity': sparsity,
        'bias': bias,
        'horizon': N,
        'noise': noise,
        'errors_system1': errors_system1,
        'errors_system2': errors_system2
    })


    # Calculate different metrix of the cost for each system
    median_system1 = np.median(errors_system1)
    median_system2 = np.median(errors_system2)
    variance_system1 = np.var(errors_system1)
    variance_system2 = np.var(errors_system2)
    std_dev_system1 = np.std(errors_system1)
    std_dev_system2 = np.std(errors_system2)

    if(show_acc_plt):
        # Create a figure and axis for plotting
        fig, ax = plt.subplots()

        # X positions for the bars
        x = np.arange(2)

        jitter = 0.05  # Adjust the spread of the dots

        # For System 1 (add jitter to the x positions)
        ax.scatter(np.full_like(errors_system1, x[0]) + np.random.uniform(-jitter, jitter, size=len(errors_system1)), 
                errors_system1, color='red', zorder=5, s=15, alpha=0.4)

        # For System 2 (add jitter to the x positions)
        ax.scatter(np.full_like(errors_system2, x[1]) + np.random.uniform(-jitter, jitter, size=len(errors_system2)), 
                errors_system2, color='blue', zorder=5, s=15, alpha=0.4)

        # Plot bars representing the median errors for each system
        bars = ax.bar(x, [median_system1, median_system2], color=['red', 'blue'], alpha=0.4, label='Median Cost', 
              edgecolor='black', linewidth=2, yerr=[std_dev_system1, std_dev_system2], capsize=40, ecolor='black')

        # Highlight the top edge of the bars by adjusting the edgecolor and linewidth
        for bar in bars:
            bar.set_edgecolor('black')  # Set the top edge color
            bar.set_linewidth(2)        # Set the thickness of the edge line

        # Customize labels, title, and legend
        ax.set_xticks(x)
        ax.set_xticklabels([f"DeePC (var={variance_system1:.2e})", f"MPC (var={variance_system2:.2e})"])
        ax.set_ylabel('Final Cost')
        ax.set_title('Comparison of Final Cost: Median with Spread')
        

        plt.savefig(f"Test_Users{num_users}_Steps{simN}_DataPoints{num_datapoints}_Horizon{noise}_Sparsity{sparsity_factor}_Bias{bias_factor}_Tini{Tini}.pdf", format='pdf')

        # Show plot
        plt.tight_layout()
        plt.show()

if show_paramcompare_plt:

    plt.rcParams.update({'font.size': 16})  # Adjust the number for desired size

    df = pd.DataFrame(results)

    # Flatten the results to make it suitable for plotting
    flat_results = []

    for index, row in df.iterrows():
        for i in range(datasets):
            flat_results.append({
                'users': row['users'],
                'datapoints': row['datapoints'],
                'sparsity': row['sparsity'],
                'bias': row['bias'],
                'horizon': row['horizon'],
                'noise': row['noise'],
                'error_system1': row['errors_system1'][i],  # DeePC error
                'error_system2': row['errors_system2'][i]   # MPC error
            })

    # Convert the flat results to a DataFrame
    flat_df = pd.DataFrame(flat_results)

    # Group by the parameter combinations and store all errors for each combination
    grouped_df = flat_df.groupby(
        ['users', 'datapoints', 'sparsity', 'bias', 'horizon', 'noise'], as_index=False
    ).agg({
        'error_system1': list,  # Store all errors for System 1 (DeePC)
        'error_system2': list   # Store all errors for System 2 (MPC)
    })

    grouped_df['parameter_combination'] = grouped_df.apply(
        lambda row: ' | '.join([
            f"Users: {row['users']}" if len(num_users) > 1 else "",
            f"Data: {row['datapoints']}" if len(num_datapoints) > 1 else "",
            f"Sparsity: {row['sparsity']}" if len(sparsity_factor) > 1 else "",
            f"Bias: {row['bias']}" if len(bias_factor) > 1 else "",
            f"Horizon: {row['horizon']}" if len(horizon) > 1 else "",
            f"Noise: {row['noise']}" if len(noise_var) > 1 else ""
        ]).strip(' | '),  # Strip any trailing " | "
        axis=1
    )

    # Calculate the median errors for both systems
    grouped_df['median_system1'] = grouped_df['error_system1'].apply(np.median)
    grouped_df['median_system2'] = grouped_df['error_system2'].apply(np.median)
    grouped_df['std_dev_system1'] = grouped_df['error_system1'].apply(np.std)
    grouped_df['std_dev_system2'] = grouped_df['error_system2'].apply(np.std)
    grouped_df['var_system1'] = grouped_df['error_system1'].apply(np.var)
    grouped_df['var_system2'] = grouped_df['error_system2'].apply(np.var)
    
    # Print key values
    print(f"DeePC: [med={grouped_df['median_system1']}, std={grouped_df['std_dev_system1']}, var={grouped_df['var_system1']}]")
    print(f"MPC: [med={grouped_df['median_system2']}, std={grouped_df['std_dev_system2']}, var={grouped_df['var_system2']}]")


    # Set up the plot
    fig, ax = plt.subplots(figsize=(15, 10))

    # Define the positions for the bars
    width = 0.4  # Width of the bars
    y = np.arange(len(grouped_df))  # The y locations for the bars (instead of x)

    # Plot both DeePC and MPC as grouped horizontal bars
    ax.barh(y - width/2, grouped_df['median_system1'], width, alpha = 0.4,label='DeePC', color='red', xerr=grouped_df['std_dev_system1'], capsize=15, ecolor='black')
    ax.barh(y + width/2, grouped_df['median_system2'], width, alpha=0.4,label='MPC', color='blue', xerr=grouped_df['std_dev_system2'], capsize=15, ecolor='black')

    jitter = 0  # Adjust the spread of the dots

    # Scatter points for DeePC errors (flattening the list of errors)
    for idx, row in grouped_df.iterrows():
        # Scatter each individual error value
        ax.scatter(
            row['error_system1'],  # The x values are the error values themselves
            np.array([idx] * len(row['error_system1'])) - (width/2) + np.random.uniform(-jitter, jitter, size=len(errors_system1)),  # y values are the same for all errors in a parameter combination
            color='red', alpha=0.1, label='_nolegend_'  # Scatter for DeePC
        )
        
    # Scatter points for MPC errors (flattening the list of errors)
    for idx, row in grouped_df.iterrows():
        # Scatter each individual error value
        ax.scatter(
            row['error_system2'],  # The x values are the error values themselves
            np.array([idx] * len(row['error_system2'])) + (width/2) + np.random.uniform(-jitter, jitter, size=len(errors_system2)),  # y values are the same for all errors in a parameter combination
            color='blue', alpha=0.1, label='_nolegend_'  # Scatter for MPC
        )

    # Labeling and styling
    ax.set_ylabel('Parameter Combinations')
    if totalcost:
        ax.set_xlabel('Total Cost')
        ax.set_title('Comparison of Total Cost: DeePC vs MPC')
    else: 
        ax.set_xlabel('Final Cost')
        ax.set_title('Comparison of Final Cost: DeePC vs MPC')
    ax.set_yticks(y)
    ax.set_yticklabels(grouped_df['parameter_combination'])

    # Add a legend
    ax.legend()



    # Adjust layout for better spacing
    plt.tight_layout()

    # Save figure as PDF
    plt.savefig(f"TotalInternalNoiseParamComp_withreg_Sets{datasets}_Users{num_users}_Steps{simN}_DataPoints{num_datapoints}_Horizon{N}_Sparsity{sparsity_factor}_Bias{bias_factor}_Tini{Tini}_Noise{noise}.pdf", format='pdf')

    # Show the plot
    plt.show()


for i in range(datasets):
    os.remove(f"data{i}.npz")