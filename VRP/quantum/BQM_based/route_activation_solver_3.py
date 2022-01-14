import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from itertools import product
from matplotlib.colors import rgb2hex
from vehicle_routing import VehicleRouter
from qiskit_optimization import QuadraticProgram


class RouteActivationSolver(VehicleRouter):

    """RAS Solver implementation."""

    def __init__(self, n_clients, n_vehicles, cost_matrix, **params):

        """Initializes any required variables and calls init of super class."""

        # Call parent initializer
        super().__init__(n_clients, n_vehicles, cost_matrix, **params)

    def build_quadratic_program(self):

        """Builds the required quadratic program and sets the names of variables in self.variables."""

        # Initialization
        self.qp = QuadraticProgram(name='Vehicle Routing Problem')

        # Designate variable names
        edgelist = [(i, j) for i, j in product(range(self.n + 1), repeat=2) if i != j]
        self.variables = np.array([f'x.{i}.{j}' for i, j in edgelist])

        # Add variables to quadratic program
        for var in self.variables:
            self.qp.binary_var(name=var)

        # U = max(self.n-self.m+1, 1)

        # Add MTZ (integer) variables to quadratic program
        for i in range(1, self.n + 1):
            self.qp.integer_var(name=f't.{i}', lowerbound=1, upperbound=self.n)
        
        # Add objective to quadratic program
        obj_linear = {f'x.{i}.{j}': self.cost[i, j] for i, j in edgelist}
        self.qp.minimize(linear=obj_linear)

        # Add constraints - single delivery per client
        for i in range(self.n + 1):
            constraint_linear_a = {f'x.{j}.{i}': 1 for j in range(self.n + 1) if j != i}
            constraint_linear_b = {f'x.{i}.{j}': 1 for j in range(self.n + 1) if j != i}
            self.qp.linear_constraint(linear=constraint_linear_a, sense='==', rhs=1, name=f'single_delivery_to_{i}_a')
            self.qp.linear_constraint(linear=constraint_linear_b, sense='==', rhs=1, name=f'single_delivery_to_{i}_b')
        
        # # Add constraints - eliminate subtours
        # #  1. time starts at t_0 = 0 at node 0
        # self.qp.linear_constraint(linear={f't.{0}': 1}, sense='==', rhs=0, name=f'start_time=0')
        
        #  2. t_(i+1) is greater than t_i (a node will be visited only after the previous nodes have been visisted)
        for i, j in product(range(1, self.n + 1), repeat=2):
            if j != i:
                constraint_linear = {f't.{i}': 1, f't.{j}': -1, f'x.{i}.{j}': 2*self.n}
                self.qp.linear_constraint(linear=constraint_linear, sense='<=', rhs=2*self.n-1, name=f'eliminate_subtour_{i}_{j}')
    
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

        # Remove after debugging
        print(active_vars)
        # ------------------------

        from_nodes = [int(var.split('.')[1]) for var in active_vars]
        to_nodes = [int(var.split('.')[2]) for var in active_vars]

        edges = dict()
        for i in range(len(from_nodes)):
            edges[from_nodes[i]] = to_nodes[i]
        edges

        self.route = []
        i = edges[0]
        while i != 0:
            self.route.append(i)
            i = edges[i]

        # Evaluate partition costs
        partition_costs = np.zeros(len(self.route) - 1)
        for i in range(len(self.route) - 1):
            partition_costs[i] = self.cost[self.route[i], 0] + self.cost[0, self.route[i + 1]] - self.cost[self.route[i], self.route[i + 1]]

        # Evaluate minimum cost partition
        cut_indices = np.argsort(partition_costs)[:(self.m - 1)]
        self.start_indices = np.sort(cut_indices) + 1
        self.start_indices = [0] + list(self.start_indices)
        self.end_indices = np.sort(cut_indices)
        self.end_indices = list(self.end_indices) + [len(self.route) - 1]
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

        # Plot figure 0 - TSP solution ---------------------------------------------------------------------------
        # Initialize figure 0
        plt.figure()
        ax = plt.gca()
        ax.set_title(f'Vehicle Routing Problem - {self.n} Clients & {self.m} Cars')

        # Build graph
        G = nx.MultiDiGraph()
        G.add_nodes_from(range(self.n + 1))

        # Plot nodes
        pos = {i: (xc[i], yc[i]) for i in range(self.n + 1)}
        labels = {i: str(i) for i in range(self.n + 1)}
        nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_color='b', node_size=500, alpha=0.8)
        nx.draw_networkx_labels(G, pos=pos, labels=labels, font_size=16)

        # Plot edges
        edgelist = [self.variables[i] for i in range(len(self.variables)) if self.solution[i] == 1]
        edgelist = [(int(var.split('.')[1]), int(var.split('.')[2])) for var in edgelist]
        G.add_edges_from(edgelist)
        nx.draw_networkx_edges(G, pos=pos, edgelist=edgelist, width=2, edge_color='r')

        # Show plot
        plt.grid(True)
        plt.show()
        # ---------------------------------------------------------------------------------------------------------

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

    # def visualize(self, xc=None, yc=None):

    #     """Visualizes solution.
    #     Args:
    #         xc: x coordinates of nodes. Defaults to random values.
    #         yc: y coordinates of nodes. Defaults to random values.
    #     """

    #     # Resolve coordinates
    #     if xc is None:
    #         xc = (np.random.rand(self.n + 1) - 0.5) * 10
    #     if yc is None:
    #         yc = (np.random.rand(self.n + 1) - 0.5) * 10

    #     # Initialize figure
    #     plt.figure()
    #     ax = plt.gca()
    #     ax.set_title(f'Vehicle Routing Problem - {self.n} Clients & {self.m} Cars')

    #     # Build graph
    #     G = nx.MultiDiGraph()
    #     G.add_nodes_from(range(self.n + 1))

    #     # Plot nodes
    #     pos = {i: (xc[i], yc[i]) for i in range(self.n + 1)}
    #     labels = {i: str(i) for i in range(self.n + 1)}
    #     nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_color='b', node_size=500, alpha=0.8)
    #     nx.draw_networkx_labels(G, pos=pos, labels=labels, font_size=16)

    #     # Plot edges
    #     edgelist = [self.variables[i] for i in range(len(self.variables)) if self.solution[i] == 1]
    #     edgelist = [(int(var.split('.')[1]), int(var.split('.')[2])) for var in edgelist]
    #     G.add_edges_from(edgelist)
    #     nx.draw_networkx_edges(G, pos=pos, edgelist=edgelist, width=2, edge_color='r')

    #     # Show plot
    #     plt.grid(True)
    #     plt.show()