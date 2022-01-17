import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from itertools import product
from matplotlib.colors import rgb2hex
from .vehicle_routing import VehicleRouter
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.infinity import INFINITY


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

        self.variables = []

        # Designate variable names
        for k in range(1, self.m + 1):
            for i in range(self.n + 1):
                for j in range(self.n + 1):
                    if i != j:
                        self.qp.binary_var(name=f'x.{i}.{j}.{k}')
                        self.variables.append(f'x.{i}.{j}.{k}')
        
        self.variables = np.array(self.variables).reshape(self.m, self.n + 1, -1)
        
        # Add objective to quadratic program
        obj_linear = {f'x.{i}.{j}.{k}': self.cost[i, j] for k in range(1, self.m + 1) for i in range(self.n + 1) for j in range(self.n + 1) if i != j}
        self.qp.minimize(linear=obj_linear)

        # Add constraints - single delivery per client:
        #  1. Each client node must have exactly one edge directed towards it
        #  2. Each client node must have exactly one edge directed away from it
        #  3. Each client must be visited and left by exactly one vehicle
        for i in range(1, self.n + 1):
            constraint_linear_a = {f'x.{j}.{i}.{k}': 1 for k in range(1, self.m + 1) for j in range(self.n + 1) if j != i}
            constraint_linear_b = {f'x.{i}.{j}.{k}': 1 for k in range(1, self.m + 1) for j in range(self.n + 1) if j != i}
            self.qp.linear_constraint(linear=constraint_linear_a, sense='==', rhs=1, name=f'single_delivery_to_{i}_a')
            self.qp.linear_constraint(linear=constraint_linear_b, sense='==', rhs=1, name=f'single_delivery_to_{i}_b')
            for k in range(1, self.m + 1):
                vehicle_visiting = {f'x.{j}.{i}.{k}': 1 for j in range(self.n + 1) if j != i}
                vehicle_leaving = {f'x.{i}.{j}.{k}': -1 for j in range(self.n + 1) if j != i}
                self.qp.linear_constraint(linear={**vehicle_visiting, **vehicle_leaving}, sense='==', rhs=0, name=f'single_delivery_to_{i}_by_{k}')

        # Add constraints - Each vehicle is used at most once
        for k in range(1, self.m + 1):
            constraint_linear_a = {f'x.{0}.{i}.{k}': 1 for i in range(1, self.n + 1)}
            constraint_linear_b = {f'x.{i}.{0}.{k}': 1 for i in range(1, self.n + 1)}
            self.qp.linear_constraint(linear=constraint_linear_a, sense='<=', rhs=1, name=f'vehicle_{k}_leaves_once')
            self.qp.linear_constraint(linear=constraint_linear_b, sense='<=', rhs=1, name=f'vehicle_{k}_returns_once')

        # Add constraints - m vehicles leaving and returning to depot
        constraint_linear_a = {f'x.{0}.{i}.{k}': 1 for i in range(1, self.n + 1) for k in range(1, self.m + 1)}
        constraint_linear_b = {f'x.{i}.{0}.{k}': 1 for i in range(1, self.n + 1) for k in range(1, self.m + 1)}
        self.qp.linear_constraint(linear=constraint_linear_a, sense='==', rhs=min(self.m, self.n), name=f'{self.m}_vehicles_leaving_depot')
        self.qp.linear_constraint(linear=constraint_linear_b, sense='==', rhs=min(self.m, self.n), name=f'{self.m}_vehicles_returning_to_depot')

        # print(self.qp.export_as_lp_string())

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

        # Loop over cars
        for i in range(self.solution.shape[0]):

            # Get edges
            var_list = np.transpose(self.variables[i]).reshape(-1)
            sol_list = np.transpose(self.solution[i]).reshape(-1)
            active_vars = [var_list[k] for k in range(len(var_list)) if sol_list[k] == 1]
            
            # Remove this line after debugging
            print(active_vars)
            # ---------------------------------
            
            n1 = [int(var.split('.')[1]) for var in active_vars]
            n2 = [int(var.split('.')[2]) for var in active_vars]

            # Plot edges
            edgelist = [(n1[k], n2[k]) for k in range(len(n1))]
            G.add_edges_from(edgelist)
            nx.draw_networkx_edges(G, pos=pos, edgelist=edgelist, width=2, edge_color=rgb2hex(cmap(i)))

        # Show plot
        plt.grid(True)
        plt.show()
