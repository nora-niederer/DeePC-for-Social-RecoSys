import numpy as np

def generate_data(num_users, num_steps, sparsity_factor, bias_factor, data_name="data"):
    # Define constants
    max_connections = num_users * (num_users + 1)
    connections = round(sparsity_factor * max_connections)

    # Network adjacency matrix
    W = generate_sparse_row_stochastic_matrix(num_users, num_users + 1, connections)

    # Biases
    Lambda = np.diag(np.random.rand(num_users) * bias_factor)

    # Dynamics matrices
    A = (np.eye(num_users) - Lambda) @ W[:, :-1]
    B = (np.eye(num_users) - Lambda) @ W[:, -1]

    # Initial opinion
    x0 = np.random.rand(num_users)
    # x0.sort()

    # Random inputs
    ud = np.random.rand(num_steps)

    # State evolution matrix
    xd = np.zeros((num_users, num_steps))
    xd[:, 0] = x0

    # Let model run
    x = x0
    for k in range(1, num_steps):
        x = A @ x + B * ud[k - 1] + Lambda @ x0
        xd[:, k] = x

    # Save data
    np.savez(f"{data_name}.npz", A=A, B=B, Lambda=Lambda, ud=ud, xd=xd)

def generate_sparse_row_stochastic_matrix(m, n, non_zero_entries):
    # Generates an m x n sparse row stochastic matrix with a specified number of non-zero entries

    # Check if inputs are valid
    if m <= 0 or n <= 0 or not (m.is_integer() and n.is_integer()) or non_zero_entries < m or non_zero_entries > m * n:
        raise ValueError('Invalid inputs. Ensure positive integers, and non_zero_entries is between m and m*n.')

    # Initialize the matrix with zeros
    R = np.zeros((m, n))

    # Ensure at least one non-zero entry per row for row stochastic property
    for i in range(m):
        col_index = np.random.randint(n)  # Random column index for the non-zero entry
        R[i, col_index] = np.random.rand()  # Assign a random value

    # Fill the rest of the non-zero entries randomly across the matrix
    additional_entries = non_zero_entries - m  # Subtract the already filled entries
    while additional_entries > 0:
        row_index = np.random.randint(m)
        col_index = np.random.randint(n)
        if R[row_index, col_index] == 0:  # Check if the position is already filled
            R[row_index, col_index] = np.random.rand()  # Assign a random value
            additional_entries -= 1

    # Normalize each row to sum to 1
    row_sums = R.sum(axis=1)  # Calculate the sum of elements in each row
    R = R / row_sums[:, np.newaxis]  # Divide each element by its row sum to normalize
    return R