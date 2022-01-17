import warnings
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from itertools import product
from matplotlib.colors import rgb2hex
from collections import Counter
from .vehicle_routing import VehicleRouter
from qiskit_optimization import QuadraticProgram


class SolutionPartitionSolver(VehicleRouter):

    """SPS Solver implementation."""

    def __init__(self, n_clients, n_vehicles, cost_matrix, **params):

        """Initializes any required variables and calls init of super class."""

        # Initialize cluster data
        self.route = None
        self.start_indices = None
        self.end_indices = None
        self.partition_cost = None

        # Call parent initializer
        super().__init__(n_clients, n_vehicles, cost_matrix, **params)

    def build_quadratic_program(self):

        """Builds the required quadratic program and sets the names of variables in self.variables."""

        # Initialization
        self.qp = QuadraticProgram(name='Vehicle Routing Problem')

        # Designate variable names
        self.variables = np.array([[f'x.{i}.{j}' for i in range(1, self.n + 1)] for j in range(1, self.n + 1)])

        # Add variables to quadratic program
        for var in self.variables.reshape(-1):
            self.qp.binary_var(name=var)

        # Build objective function
        edgelist = [(i, j) for i, j in product(range(1, self.n + 1), repeat=2) if i != j]
        obj_linear_a = {f'x.{i}.{1}': self.cost[0, i] for i in range(1, self.n + 1)}
        obj_linear_b = {f'x.{i}.{self.n}': self.cost[i, 0] for i in range(1, self.n + 1)}
        obj_quadratic = {(f'x.{i}.{t}', f'x.{j}.{t + 1}'): self.cost[i, j] for i, j in edgelist
                         for t in range(1, self.n)}

        # Add objective to quadratic program
        self.qp.minimize(linear=dict(Counter(obj_linear_a) + Counter(obj_linear_b)), quadratic=obj_quadratic)

        # Add constraints - single delivery per client
        for i in range(1, self.n + 1):
            constraint_linear = {f'x.{i}.{j}': 1 for j in range(1, self.n + 1)}
            self.qp.linear_constraint(linear=constraint_linear, sense='==', rhs=1, name=f'single_delivery_{i}')

        # Add constraints - vehicle at one place at one time
        for j in range(1, self.n + 1):
            constraint_linear = {f'x.{i}.{j}': 1 for i in range(1, self.n + 1)}
            self.qp.linear_constraint(linear=constraint_linear, sense='==', rhs=1, name=f'single_location_{j}')

    def evaluate_vrp_cost(self):

        """Evaluate the optimized VRP cost under the optimized solution stored in self.solution.
        Returns:
            Optimized VRP cost as a float value.
        """

        # Return optimized energy
        return super().evaluate_vrp_cost() + self.partition_cost

    def solve(self, **params):

        """Add additional functionality to the parent solve function to be able to classically partition the TSP
        solution after quantum sampling.
        Args:
            params: Parameters to send to the selected backend solver. You may also specify the solver to select a
                different solver and override the specified self.solver.
        """

        # Solve TSP
        super().solve(**params)

        # Evaluate route
        var_list = self.variables.reshape(-1)
        sol_list = self.solution.reshape(-1)
        active_vars = [var_list[k] for k in range(len(var_list)) if sol_list[k] == 1]
        self.route = [int(var.split('.')[1]) for var in active_vars]

        # Evaluate partition costs
        partition_costs = np.zeros(self.n - 1)
        for i in range(self.n - 1):
            partition_costs[i] = self.cost[self.route[i], 0] + self.cost[0, self.route[i + 1]] - \
                                 self.cost[self.route[i], self.route[i + 1]]

        # Evaluate minimum cost partition
        cut_indices = np.argsort(partition_costs)[:(self.m - 1)]
        self.start_indices = np.sort(cut_indices) + 1
        self.start_indices = [0] + list(self.start_indices)
        self.end_indices = np.sort(cut_indices)
        self.end_indices = list(self.end_indices) + [self.n - 1]
        self.partition_cost = sum(partition_costs[cut_indices])

    def visualize(self, xc=None, yc=None):

        """Visualizes solution.
        Args:
            xc: x coordinates of nodes. Defaults to random values.
            yc: y coordinates of nodes. Defaults to random values.
        """

        # Resolve coordinates
        if xc is None:
            xc = (np.random.rand(self.n + 1) - 0.5) * 10
        if yc is None:
            yc = (np.random.rand(self.n + 1) - 0.5) * 10

        # Initialize figure
        plt.figure()
        ax = plt.gca()
        ax.set_title(f'Vehicle Routing Problem - {self.n} Clients & {self.m} Cars')
        cmap = plt.cm.get_cmap('Accent')
        
        # Build graph
        G = nx.MultiDiGraph()
        G.add_nodes_from(range(self.n + 1))

        # Plot nodes
        pos = {i: (xc[i], yc[i]) for i in range(self.n + 1)}
        labels = {i: str(i) for i in range(self.n + 1)}
        nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_color='b', node_size=500, alpha=0.8)
        nx.draw_networkx_labels(G, pos=pos, labels=labels, font_size=16)

        # Loop through cars
        for i in range(len(self.start_indices)):

            # Extract edge list
            route = [self.route[j] for j in range(self.start_indices[i], self.end_indices[i] + 1)]
            edgelist = [(0, route[0])] + [(route[j], route[j + 1]) for j in range(len(route) - 1)] + [(route[-1], 0)]

            # Plot edges
            G.add_edges_from(edgelist)
            nx.draw_networkx_edges(G, pos=pos, edgelist=edgelist, width=2, edge_color=rgb2hex(cmap(i)))

        # Show plot
        plt.grid(True)
        plt.show()


class CapcSolutionPartitionSolver(SolutionPartitionSolver):

    """Capacitated SPS Solver implementation."""

    def __init__(self, n_clients, n_vehicles, cost_matrix, capacity, demand, **params):

        """Initializes any required variables and calls init of super class."""

        # Store capacity data
        self.capacity = capacity
        self.demand = demand

        # Call parent initializer
        super().__init__(n_clients, n_vehicles, cost_matrix, **params)

    def build_partition_graph(self):

        """Build partition graph for post TSP partitioning."""

        # Initialize graph
        G = nx.DiGraph()
        G.add_nodes_from(range(self.n + 1))

        # Loop over nodes
        for i in range(self.n + 1):

            # Initialize
            j = i + 1
            demand = self.demand[self.route[j - 1] - 1] if j <= self.n else None
            cost = self.cost[0, self.route[j - 1]] if j <= self.n else None

            # Loop over target nodes
            while j <= self.n and demand <= self.capacity:
                trip_cost = cost + self.cost[self.route[j - 1], 0]
                G.add_edge(i, j, weight=trip_cost)
                j += 1
                if j <= self.n:
                    demand += self.demand[self.route[j - 1] - 1]
                    cost += self.cost[self.route[j - 2], self.route[j - 1]]

        # Return graph
        return G
    
    @staticmethod
    def shortest_walk(dG, sink, m):

        """Function to find shortest path with at most m edges.
        Args:
            dG: Input graph.
            sink: Destination Node.
            m: Max number of edges.
        Returns:
            Boolean value indicating successful pathfinding, the path length and the path. None returned for last 2
            fields if path not found.
        """

        # d(v,m) is the shortest walk length from source(0) to v using at most m edges
        n = dG.number_of_nodes() - 1
        d = [float('inf')] * (n + 1)
        path = [None] * (n + 1)         # store path to v with atmost m edges
        path[0] = [0]
        d[0] = 0                        # d(0,0) = 0

        # Initialization
        d_nextround = [float('inf')] * (n + 1)
        d_nextround[0] = 0
        path_nextround = [None] * (n + 1)
        path_nextround[0] = [0]

        # Main loop
        for i in range(m):
            for v in range(1, n + 1):
                path_dists = [(u, d[u]+dG.edges[u, v]['weight']) for u in dG.predecessors(v)]
                min_path_dists = min(dist for (vertex, dist) in path_dists)
                if min_path_dists < d[v]:
                    d_nextround[v] = min_path_dists
                    for (vertex, dist) in path_dists:
                        if dist == min_path_dists:
                            path_nextround[v] = path[int(vertex)] + [v]
                else:
                    d_nextround[v] = d[v]
                    path_nextround[v] = path[v]
            d = d_nextround.copy()
            path = path_nextround.copy()

        # Return output
        if d[sink] == float('inf'):
            return False, None, None
        else:
            path_length = d[sink]
            return True, path_length, path[sink]

    def solve(self, **params):

        """Add additional functionality to the parent solve function to be able to classically partition the TSP
        solution after quantum sampling.
        Args:
            params: Parameters to send to the selected backend solver. You may also specify the solver to select a
                different solver and override the specified self.solver.
        """

        # Solve TSP
        VehicleRouter.solve(self, **params)

        # Evaluate route
        var_list = self.variables.reshape(-1)
        sol_list = self.solution.reshape(-1)
        active_vars = [var_list[k] for k in range(len(var_list)) if sol_list[k] == 1]
        self.route = [int(var.split('.')[1]) for var in active_vars]

        # Evaluate minimum cost partition
        G = self.build_partition_graph()
        success, path_length, path = self.shortest_walk(G, self.n, self.m)
        if not success:
            warnings.warn('Unable to find route with given number of vehicles. Extending fleet...')
            path_length, path = nx.single_source_dijkstra(G, source=0, target=self.n)

        # Extract cuts from partition
        self.start_indices = path[:-1]
        self.end_indices = [j - 1 for j in path[1:]]
        self.partition_cost = path_length - VehicleRouter.evaluate_vrp_cost(self)
