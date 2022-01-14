import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from itertools import product
from matplotlib.colors import rgb2hex
from vehicle_routing import VehicleRouter
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

        # U = max(self.n-self.m+1, 1)

        # # Designate variable names
        # for i in range(self.n + 1):
        #     for j in range(self.n + 1):
        #         if i != j:
        #             self.qp.integer_var(name=f'x.{i}.{j}', lowerbound=0, upperbound=self.m)
        #             self.qp.binary_var(name=f'y.{i}.{j}')
        #     for k in range(self.m + 1):
        #         if i != 0:
        #             self.qp.integer_var(name=f't.{i}.{k}', lowerbound=0, upperbound=U)

        # # Add objective to quadratic program
        # obj_linear = {f'y.{i}.{j}': self.cost[i, j] for i in range(self.n + 1) for j in range(self.n + 1) if i != j}
        # self.qp.minimize(linear=obj_linear)


        # '''
        # Designate variable names
        edgelist = [(i, j) for i, j in product(range(self.n + 1), repeat=2) if i != j]
        self.variables = np.array([f'x.{i}.{j}' for i, j in edgelist])
        
        # Add binary variables 'x' to quadratic program
        for var in self.variables:
            self.qp.binary_var(name=var)
        
        # Add binary variables 'y', MTZ integer variables 't' and integer variables 'w' to reduce quadratic constraints into linear
        for i in range(1, self.n + 1):
            for k in range(1, self.m + 1):
                self.qp.binary_var(name=f'y.{i}.{k}')
                self.qp.integer_var(name=f't.{i}.{k}', lowerbound=1, upperbound=self.n)
                self.qp.integer_var(name=f'w.{i}.{k}', lowerbound=0, upperbound=self.n)

        # Add objective to quadratic program
        obj_linear = {self.variables[k]: self.cost[i, j] for k, (i, j) in enumerate(edgelist)}
        self.qp.minimize(linear=obj_linear)

        # Add constraints - single delivery per client:
        #  - Each client node must have exactly one edge directed towards it
        #  - Each client node must have exactly one edge directed away from it
        #  - Each client must be visited by exactly one vehicle
        for i in range(1, self.n + 1):
            constraint_linear_a = {f'x.{j}.{i}': 1 for j in range(self.n + 1) if j != i}
            constraint_linear_b = {f'x.{i}.{j}': 1 for j in range(self.n + 1) if j != i}
            constraint_linear_c = {f'y.{i}.{k}': 1 for k in range(1, self.m + 1)}
            self.qp.linear_constraint(linear=constraint_linear_a, sense='==', rhs=1, name=f'single_delivery_a_{i}')
            self.qp.linear_constraint(linear=constraint_linear_b, sense='==', rhs=1, name=f'single_delivery_b_{i}')
            self.qp.linear_constraint(linear=constraint_linear_c, sense='==', rhs=1, name=f'single_delivery_c_{i}')

        # Add constraints - m vehicles at depot
        constraint_linear_a = {f'x.{0}.{i}': 1 for i in range(1, self.n + 1)}
        constraint_linear_b = {f'x.{i}.{0}': 1 for i in range(1, self.n + 1)}
        self.qp.linear_constraint(linear=constraint_linear_a, sense='==', rhs=self.m, name=f'depot_a')
        self.qp.linear_constraint(linear=constraint_linear_b, sense='==', rhs=self.m, name=f'depot_b')

        # Add constraints - eliminate subtours
        for k in range(1, self.m + 1):
            for e, (i, j) in enumerate(product(range(1, self.n + 1), repeat=2)):
                if j != i:
                    constraint_linear = {f'w.{i}.{k}': 1, f'y.{i}.{k}': 1, f'w.{j}.{k}': -1, f'x.{i}.{j}': 2*self.n}
                    self.qp.linear_constraint(linear=constraint_linear, sense='<=', rhs=2*self.n, name=f'eliminate_subtour_{i}_{j}_{k}')
                    
                    constraint_linear_l1_i = {f'w.{i}.{k}': 1, f'y.{i}.{k}': -1}
                    self.qp.linear_constraint(linear=constraint_linear_l1_i, sense='>=', rhs=0, name=f'lower_bound_1_i_w_{i}_{k}_{e}')
                    constraint_linear_l2_i = {f'w.{i}.{k}': 1, f't.{i}.{k}': -1, f'y.{i}.{k}': -self.n}
                    self.qp.linear_constraint(linear=constraint_linear_l2_i, sense='>=', rhs=-self.n, name=f'lower_bound_2_i_w_{i}_{k}_{e}')
                    
                    constraint_linear_u1_i = {f'w.{i}.{k}': 1, f'y.{i}.{k}': -self.n}
                    self.qp.linear_constraint(linear=constraint_linear_u1_i, sense='<=', rhs=0, name=f'upper_bound_1_i_w_{i}_{k}_{e}')
                    constraint_linear_u2_i = {f'w.{i}.{k}': 1, f't.{i}.{k}': -1, f'y.{i}.{k}': -1}
                    self.qp.linear_constraint(linear=constraint_linear_u2_i, sense='<=', rhs=-1, name=f'upper_bound_2_i_w_{i}_{k}_{e}')
                    
                    constraint_linear_l1_j = {f'w.{j}.{k}': 1, f'y.{j}.{k}': -1}
                    self.qp.linear_constraint(linear=constraint_linear_l1_j, sense='>=', rhs=0, name=f'lower_bound_1_j_w_{j}_{k}_{e}')
                    constraint_linear_l2_j = {f'w.{j}.{k}': 1, f't.{j}.{k}': -1, f'y.{j}.{k}': -self.n}
                    self.qp.linear_constraint(linear=constraint_linear_l2_j, sense='>=', rhs=-self.n, name=f'lower_bound_2_j_w_{j}_{k}_{e}')

                    constraint_linear_u1_j = {f'w.{j}.{k}': 1, f'y.{j}.{k}': -self.n}
                    self.qp.linear_constraint(linear=constraint_linear_u1_j, sense='<=', rhs=0, name=f'upper_bound_1_j_w_{j}_{k}_{e}')
                    constraint_linear_u2_j = {f'w.{j}.{k}': 1, f't.{j}.{k}': -1, f'y.{j}.{k}': -1}
                    self.qp.linear_constraint(linear=constraint_linear_u2_j, sense='<=', rhs=-1, name=f'upper_bound_2_j_w_{j}_{k}_{e}')
        
        # for i in range(1, self.n + 1):
        #     for k in range(1, self.m + 1):
        #         constraint_linear = {f'y.{i}.{k}': 1, f'w.{i}.{k}': -1}
        #         self.qp.linear_constraint(linear=constraint_linear, sense='<=', rhs=0, name=f'y.{i}.{k}=bool(w.{i}.{k})__1')
        #         constraint_linear = {f'y.{i}.{k}': max(self.n-self.m+1, 1), f'w.{i}.{k}': -1}
        #         self.qp.linear_constraint(linear=constraint_linear, sense='>=', rhs=0, name=f'y.{i}.{k}=bool(w.{i}.{k})__2')

        # for i in range(1, self.n + 1):
        #     constraint_linear = {f'y.{i}.{k}': 1 for k in range(1, self.m + 1)}
        #     self.qp.linear_constraint(linear=constraint_linear, sense='==', rhs=1, name=f'single_delivery{i}')
        # print(self.qp.export_as_lp_string())
        # '''

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
        # cmap = plt.cm.get_cmap('Accent')

        # Build graph
        G = nx.MultiDiGraph()
        G.add_nodes_from(range(self.n + 1))

        # Plot nodes
        pos = {i: (xc[i], yc[i]) for i in range(self.n + 1)}
        labels = {i: str(i) for i in range(self.n + 1)}
        nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_color='b', node_size=500, alpha=0.8)
        nx.draw_networkx_labels(G, pos=pos, labels=labels, font_size=16)

        # # Loop over cars
        # for i in range(self.solution.shape[0]):

        #     # Get route
        #     var_list = np.transpose(self.variables[i]).reshape(-1)
        #     sol_list = np.transpose(self.solution[i]).reshape(-1)
        #     active_vars = [var_list[k] for k in range(len(var_list)) if sol_list[k] == 1]
        #     route = [int(var.split('.')[2]) for var in active_vars]

        #     # Plot edges
        #     edgelist = [(0, route[0])] + [(route[j], route[j + 1]) for j in range(len(route) - 1)] + [(route[-1], 0)]
        #     G.add_edges_from(edgelist)
        #     nx.draw_networkx_edges(G, pos=pos, edgelist=edgelist, width=2, edge_color=rgb2hex(cmap(i)))
        
        # Plot edges
        edgelist = [self.variables[i] for i in range(len(self.variables)) if self.solution[i] == 1]
        edgelist = [(int(var.split('.')[1]), int(var.split('.')[2])) for var in edgelist]
        G.add_edges_from(edgelist)
        nx.draw_networkx_edges(G, pos=pos, edgelist=edgelist, width=2, edge_color='r')

        # Show plot
        plt.grid(True)
        plt.show()


class CapcRouteActivationSolver(RouteActivationSolver):

    """Capacitated RAS Solver implementation."""

    def __init__(self, n_clients, n_vehicles, cost_matrix, capacity, demand, **params):

        """Initializes any required variables and calls init of super class."""

        # Store capacity data
        self.capacity = capacity
        self.demand = demand

        # Call parent initializer
        super().__init__(n_clients, n_vehicles, cost_matrix, **params)

    def build_quadratic_program(self):

        """Builds the required quadratic program and sets the names of variables in self.variables."""

        # Build quadratic program without capacity
        super().build_quadratic_program()

        # Add MTZ variables
        for i in range(1, self.n + 1):
            self.qp.integer_var(name=f'u.{i}', lowerbound=self.demand[i - 1], upperbound=self.capacity)

        # Add mtz capacity constraints
        edgelist = [(i, j) for i, j in product(range(1, self.n + 1), repeat=2) if i != j]
        for i, j in edgelist:
            constraint = {f'u.{i}': 1, f'u.{j}': -1, f'x.{i}.{j}': self.capacity}
            rhs = self.capacity - self.demand[j - 1]
            self.qp.linear_constraint(linear=constraint, sense='<=', rhs=rhs, name=f'mtz_{i}_{j}')

    def evaluate_qubo_feasibility(self, data=None):

        """This function is not supported due to additional MTZ variables."""
        print("Not Supported due to integer valued MTZ variables!")
