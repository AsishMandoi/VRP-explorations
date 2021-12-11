import numpy as np


def generate_vrp_instance(n, seed=None):

    """Generate a random VRP instance.
    Args:
        n: No. of nodes exclusing depot.
        seed: Seed value for random number generator. Defaults to None, which sets a random seed.
    Returns:
        A list of (n + 1) x coordinates, a list of (n + 1) y coordinates and an (n + 1) x (n + 1) numpy array as the
        cost matrix.
    """

    # Set seed
    if seed is not None:
        np.random.seed(seed)

    # Generate VRP instance
    xc = (np.random.rand(n + 1) - 0.5) * 10
    yc = (np.random.rand(n + 1) - 0.5) * 10
    instance = np.zeros((n + 1, n + 1))
    for ii in range(n + 1):
        for jj in range(ii + 1, n + 1):
            instance[ii, jj] = (xc[ii] - xc[jj]) ** 2 + (yc[ii] - yc[jj]) ** 2
            instance[jj, ii] = instance[ii, jj]

    # Return output
    return instance, xc, yc


def generate_cvrp_instance(n, m, seed=None):

    """Generate a random CVRP instance.
    Args:
        n: No. of nodes exclusing depot.
        m: No. of vehicles in the problem.
        seed: Seed value for random number generator. Defaults to None, which sets a random seed.
    Returns:
        A list of (n + 1) x coordinates, a list of (n + 1) y coordinates, an (n + 1) x (n + 1) numpy array as the
        cost matrix, a list of m capacities for the vehicles and a list of n demads for the nodes.
    """

    # Set seed
    if seed is not None:
        np.random.seed(seed)

    # Acquire vrp instance
    instance, xc, yc = generate_vrp_instance(n)

    # Generate capacity and demand
    demands = np.random.rand(n) * 10
    capacities = np.random.rand(m)
    capacities = 4 * capacities * sum(demands) / sum(capacities)

    # Floor data
    demands = np.floor(demands).astype(int)
    capacities = np.floor(capacities).astype(int)

    # Return output
    return instance, xc, yc, capacities, demands
