from ortools.sat.python import cp_model
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import time

'''
Solving VRP using a CP-SAT solver
CP-SAT solver: A constraint programming solver that uses SAT (satisfiability) methods.
'''

class RAS():
    '''Classical Implementation of Route Activation Solver (Method-1)'''

    def __init__(self, n, m, cost, xc, yc):
        self.n = n
        self.m = m
        self.cost = cost
        self.xc = xc
        self.yc = yc
        self.T_max = max(self.n-self.m+1, 1)
        self.B = 2*self.n
        self.sol = None
        self.model = cp_model.CpModel()
        # self.formulate_and_solve()
        # self.visualize()
    
    def formulate_and_solve(self):
        '''Formulate and solve the problem'''

        x = [[None] * (self.n+1) for _ in range(self.n+1)]
        y = [[None] * (self.m+1) for _ in range(self.n+1)]
        t = [[None] * (self.m+1) for _ in range(self.n+1)]
        w = [[None] * (self.m+1) for _ in range(self.n+1)]


        ##### Designate variable names
        for i in range(self.n+1):
            for j in range(self.n+1):
                if i != j:
                    x[i][j] = self.model.NewBoolVar(f'x.{i}.{j}')

        for i in range(1, self.n+1):
            for k in range(1, self.m+1):
                y[i][k] = self.model.NewBoolVar(f'y.{i}.{k}')
                t[i][k] = self.model.NewIntVar(1, self.T_max, f't.{i}.{k}')
                w[i][k] = self.model.NewIntVar(0, self.T_max, f'w.{i}.{k}')


        ##### CONSTRAINTS
        for i in range(1, self.n+1):

            # Each client node must have exactly one edge directed away from it
            self.model.Add(sum(x[i][j] for j in range(self.n+1) if j != i) == 1)

            # Each client node must have exactly one edge directed towards it
            self.model.Add(sum(x[j][i] for j in range(self.n+1) if j != i) == 1)

            # Each client must be visited by exactly one vehicle
            self.model.Add(sum(y[i][k] for k in range(1, self.m+1)) == 1)

        # m vehicles coming from depot
        self.model.Add(sum(x[0][i] for i in range(1, self.n+1)) == self.m)

        # m vehicles going to depot
        self.model.Add(sum(x[i][0] for i in range(1, self.n+1)) == self.m)

        # If there is an edge from node i to node j, then they must be visited by the same vehicle
        for i in range(1, self.n+1):
            for j in range(1, self.n+1):
                if i != j:
                    for k in range(1, self.m+1):
                        self.model.Add(y[i][k] >= y[j][k] - (1 - x[i][j]))
                        self.model.Add(y[j][k] >= y[i][k] - (1 - x[i][j]))

        # Each vehicle must be used at least once
        for k in range(1, self.m+1):
            self.model.Add(sum(y[i][k] for i in range(1, self.n+1)) >= 1)

        # Eliminate subtours - MTZ constraint
        for i in range(1, self.n+1):
            for j in range(1, self.n+1):
                if i != j:
                    for k in range(1, self.m+1):
                        self.model.Add(w[j][k] >= w[i][k] + y[i][k] - self.B*(1 - x[i][j]))
                        self.model.Add(w[i][k] >= y[i][k])
                        self.model.Add(w[i][k] >= self.T_max*y[i][k] + t[i][k] - self.T_max)
                        self.model.Add(w[i][k] <= self.T_max*y[i][k])
                        self.model.Add(w[i][k] <= t[i][k] + y[i][k] - 1)


        ##### OBJECTIVE FUNCTION
        self.model.Minimize(sum(self.cost[i][j]*x[i][j] for i in range(self.n+1) for j in range(self.n+1) if i != j))


        ##### SOLVE
        # Call the solver
        solver = cp_model.CpSolver()
        t = time.time()
        status = solver.Solve(self.model)
        t = time.time() - t


        # Display the solution
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            sol = {
                'min_cost': solver.ObjectiveValue(),
                'runtime': t*1000
            }

            print('\nEXACT (CLASSICAL) SOLVER')
            print(f'Minimum cost: {sol["min_cost"]}')
            self.sol = [[[None] * (self.n+1) for _ in range(self.n+1)]] + [[[None] * (self.m+1) for _ in range(self.n+1)]]
            for i in range(self.n+1):
                for j in range(self.n+1):
                    if j != i:
                        self.sol[0][i][j] = solver.Value(x[i][j])
            for i in range(1, self.n+1):
                for k in range(1, self.m+1):
                    self.sol[1][i][k] = solver.Value(y[i][k])
            print(f'Time taken to solve: {t*1000:.3f} ms')
            return sol
        else:
            print('No solution found.')

    def visualize(self):
        '''Visualize solution'''

        if self.sol is None:
            print('No solution to show!')
            return
        
        if self.xc is None or self.yc is None:
            self.xc = (np.random.rand(self.n + 1) - 0.5) * 20
            self.yc = (np.random.rand(self.n + 1) - 0.5) * 20

        # Initialize figure
        mpl.style.use('seaborn')
        plt.figure()
        ax = plt.gca()
        ax.set_title(f'Vehicle Routing Problem - {self.n} Clients & {self.m} Cars')

        # Build graph
        G = nx.MultiDiGraph()
        G.add_nodes_from(range(self.n + 1))
        
        pos = {i: (self.xc[i], self.yc[i]) for i in range(self.n + 1)}
        labels = {i: str(i) for i in range(self.n + 1)}
        
        # Plot depot node
        nx.draw_networkx_nodes(G, pos=pos, nodelist=[0], ax=ax, alpha=0.75, node_size=300, node_color=f'C0', node_shape='s')    # shapes - 'so^>v<dph8'

        # Loop over cars and plot other nodes and edges
        for k in range(1, self.m+1):

            # Get edges and nodes for this car
            edgelist = []
            nodelist = []
            for i in range(self.n+1):
                for j in range(self.n+1):
                    if self.sol[0][i][j] == 1 and (self.sol[1][i][k] == 1 or self.sol[1][j][k] == 1):
                        edgelist.append((i, j))
                        if i:
                            nodelist.append(i)
            
            nx.draw_networkx_nodes(G, pos=pos, nodelist=nodelist, ax=ax, alpha=0.75, node_size=300, node_color=f'C{k}')
            
            G.add_edges_from(edgelist)
            nx.draw_networkx_edges(G, pos=pos, edgelist=edgelist, width=1.5, edge_color=f'C{k}')

        nx.draw_networkx_labels(G, pos=pos, labels=labels, font_size=12)

        # Show plot
        plt.grid(True)
        plt.show()
