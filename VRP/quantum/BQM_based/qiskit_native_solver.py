import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from itertools import product
from .vehicle_routing import VehicleRouter
from qiskit_optimization.applications import VehicleRouting


class QiskitNativeSolver(VehicleRouter):

    """QNS Solver implementation."""

    def __init__(self, n_clients, n_vehicles, cost_matrix, **params):

        """Initializes any required variables and calls init of super class."""

        # Call parent initializer
        super().__init__(n_clients, n_vehicles, cost_matrix, **params)

    def build_quadratic_program(self):

        """Builds the required quadratic program and sets the names of variables in self.variables."""

        # Build graph
        G = nx.Graph()
        G.add_nodes_from(list(range(self.n + 1)))
        edgelist = [(i, j) for i, j in product(range(self.n + 1), repeat=2) if i != j]
        for i, j in edgelist:
            G.add_edge(i, j, weight=self.cost[i, j])

        # Vehicle routing default quadratic program
        vrp = VehicleRouting(graph=G, num_vehicles=self.m, depot=0)
        self.qp = vrp.to_quadratic_program()
        self.variables = np.array(list(self.qp.variables_index.keys()))

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
        edgelist = [(int(var.split('_')[1]), int(var.split('_')[2])) for var in edgelist]
        G.add_edges_from(edgelist)
        nx.draw_networkx_edges(G, pos=pos, edgelist=edgelist, width=2, edge_color='r')

        # Show plot
        plt.grid(True)
        plt.show()
