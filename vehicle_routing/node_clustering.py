import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import dimod

from itertools import combinations
from matplotlib.colors import rgb2hex
from dwave.system import LeapHybridDQMSampler


class NodeClustering:

    """Class for performing node clustering using multilevel maxcut performed via Leap Hybrid DQM Sampler."""

    def __init__(self, n_nodes, n_clusters, cost_matrix):

        """Initializes the required variables and stores inputs.
        Args:
            n_nodes: No. of nodes.
            n_clusters: No. of clusters.
            cost_matrix: n x n matrix describing the cost of moving from node i to node j.
        """

        # Store critical inputs
        self.n = n_nodes
        self.k = n_clusters
        self.c = cost_matrix

        # Initialize quadratic structures
        self.dqm = None
        self.variables = None

        # Initialize result containers
        self.result = None
        self.solution = None

        # Build dqm
        self.rebuild()

    def rebuild(self):

        """Builds DQM for multi level maxcut."""

        # Set variables
        self.variables = np.array([f'x.{i}' for i in range(self.n)])

        # Initialize DQM
        self.dqm = dimod.DiscreteQuadraticModel()
        for var in self.variables:
            self.dqm.add_variable(self.k, label=var)

        # Set DQM biases
        for i, j in combinations(range(self.n), r=2):
            bias = {(p, p): self.c[i, j] + self.c[j, i] for p in range(self.k)}
            self.dqm.set_quadratic(f'x.{i}', f'x.{j}', bias)

    def solve(self):

        """Solves DQM using Leap Hybrid DQM Sampler."""

        # Solve DQM
        sampler = LeapHybridDQMSampler()
        self.result = sampler.sample_dqm(self.dqm)

        # Extract solution
        result_dict = self.result.first.sample
        self.solution = np.array([result_dict[var] for var in self.variables])

    def visualize(self, xc=None, yc=None):

        """Visualizes output as nodes coloured according to their assigned cluster.
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
        ax.set_title(f'Node Clustering - {self.n} Nodes & {self.k} Clusters')
        cmap = plt.cm.get_cmap('Accent')

        # Build graph
        G = nx.MultiDiGraph()
        G.add_nodes_from(range(self.n))
        node_colors = [rgb2hex(cmap(i)) for i in self.solution]

        # Plot nodes
        pos = {i: (xc[i], yc[i]) for i in range(self.n)}
        labels = {i: str(i) for i in range(self.n)}
        nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_color=node_colors, node_size=500, alpha=0.8)
        nx.draw_networkx_labels(G, pos=pos, labels=labels, font_size=16)

        # Show plot
        plt.grid(True)
        plt.show()


class CapcNodeClustering(NodeClustering):

    """Class for performing caoacitated node clustering using multilevel maxcut performed via Leap
    Hybrid DQM Sampler."""

    def __init__(self, n_nodes, n_clusters, cost_matrix, capacity, value, **params):

        """Initializes the required variables and stores inputs.
        Args:
            n_nodes: No. of nodes.
            n_clusters: No. of clusters.
            cost_matrix: n x n matrix describing the cost of moving from node i to node j.
            capacity: List of cluster capacities.
            value: List of node values.
            penalty: Lagrange multiplier for capacity constraints.
        """

        # Store capacity data
        self.capacity = capacity
        self.value = value

        # Store parameters
        self.penalty = params.setdefault('penalty', 1e5)

        # Call parent initializer
        super().__init__(n_nodes, n_clusters, cost_matrix)

    def rebuild(self):

        """Builds DQM for capacitated multi level maxcut."""

        # Build multilevel maxcut DQM
        super().rebuild()

        # Add capacity slack variables
        for i in range(self.k):
            self.dqm.add_variable(self.capacity[i], label=f'c.{i}')

        # Add capacity constraints
        for i in range(self.k):
            terms = [(f'x.{j}', i, self.value[j]) for j in range(self.n)]
            terms += [(f'c.{i}', j, j) for j in range(self.capacity[i])]
            constant = -self.capacity[i]
            self.dqm.add_linear_equality_constraint(terms, self.penalty, constant)
