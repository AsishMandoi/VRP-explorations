from ortools.sat.python import cp_model
import time

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx

'''
Solving VRP using a CP-SAT solver
CP-SAT solver: A constraint programming solver that uses SAT (satisfiability) methods.
'''

class FQSTW():
	'''Classical Implementation of Full Qubo Solver for Vehicle Routing Problem with Time Windows'''

	def __init__(self, n, m, cost, xc, yc, tw):
		self.n = n
		self.m = m
		self.cost = cost
		self.xc = xc
		self.yc = yc
		self.tw = tw
		self.A_max = 50
		self.D_max = 50
		self.sol = None
		self.model = cp_model.CpModel()
		# self.formulate_and_solve()
		# self.visualize()


	def formulate_and_solve(self):
		'''Formulate and solve the problem'''

		x = [[None] * (self.n+self.m+1) for _ in range(self.n+1)]
		a = [None] * (self.n+self.m+1)
		d = [None] * (self.n+self.m+1)
		y = [[[None] * (self.n+self.m) for _ in range(self.n+1)] for __ in range(self.n+1)]
		
		##### Declare variables #####
		for t in range(self.n+self.m+1):
			for i in range(self.n+1):
				if t==0 or t==self.n+self.m:
					x[i][t] = 0
				else:
					x[i][t] = self.model.NewBoolVar(f'x.{i}.{t}')
			
			if t!=0:
				d[t] = self.model.NewIntVar(0, self.D_max, f'd.{t}')
				a[t] = self.model.NewIntVar(0, self.A_max, f'a.{t}')
		
		x[0][0] = 1								# At the zero-th time step the first vehicle is at the depot
		x[0][self.m+self.n] = 1		# At the final time step the last vehicle is at the depot
		a[0] = -1									# Arrival time at the depot before the routing starts
		d[0] = 0									# Departure time at the depot (THE START OF THE ROUTING)

		for t in range(self.n+self.m):
			for i in range(self.n+1):
				for j in range(self.n+1):
					y[i][j][t] = self.model.NewBoolVar(f'y.{i}.{j}.{t}')
		
		##### OBJECTIVE #####
		self.model.Minimize(sum(self.cost[i][j] * y[i][j][t] for i in range(self.n+1) for j in range(self.n+1) for t in range(self.n+self.m)))

		##### CONSTRAINTS #####
		for t in range(self.n+self.m):
			for i in range(self.n+1):
				for j in range(self.n+1):
					self.model.Add(y[i][j][t] <= x[i][t])
					self.model.Add(y[i][j][t] <= x[j][t+1])
					self.model.Add(y[i][j][t] >= x[i][t] + x[j][t+1] - 1)
		
		# 1. Exactly one client/depot is visited by exactly one vehicle at any given instant.
		for t in range(1, self.n+self.m):
			self.model.Add(sum(x[i][t] for i in range(self.n+1)) == 1)

		# 2. Each client is visited by one vehicle throughout all the time steps during the delivery.
		for i in range(1, self.n+1):
			self.model.Add(sum(x[i][t] for t in range(1, self.n+self.m)) == 1)

		# 3. The depot is visited (returned to) by m vehicles throughout the delivery process.
		self.model.Add(sum(x[0][t] for t in range(1, self.n+self.m+1)) == self.m)

		# 4. Each vehicle visits at least one client. (No vehicle starts from the depot and ends up in the depot in the next immediate step)
		for t in range(self.m+self.n):
			self.model.Add(x[0][t] + x[0][t+1] <= 1)

		## Time Window constraints
		# 5. Departure time at the depot is always set to 0
		for t in range(1, self.m+self.n+1):
			self.model.Add(d[t] + self.D_max * (1-x[0][t]) >= 0)
			self.model.Add(d[t] - self.D_max * (1-x[0][t]) <= 0)

			# Departure time at any client node is always greater than the arrival time
			for i in range(1, self.n+1):
				self.model.Add(d[t] - a[t] - self.A_max * (x[i][t]-1) >= 0)
		
		# 6. Arrival time at next node is the sum of departure time at current node, and time to travel from current node to next node
		for t in range(self.n+self.m):
			for j in range(self.n+1):
				self.model.Add(a[t+1] - d[t] - sum(self.cost[i][j] * x[i][t] for i in range(self.n+1)) + self.A_max * (1-x[j][t+1]) >= 0)
				self.model.Add(a[t+1] - d[t] - sum(self.cost[i][j] * x[i][t] for i in range(self.n+1)) - self.A_max * (1-x[j][t+1]) <= 0)
		
		# 7. Arrival time at any node lies within the time window (departure time need not)
		for t in range(1, self.m+self.n+1):
			self.model.Add(a[t] - sum(self.tw[i][0] * x[i][t] for i in range(self.n+1)) >= 0)
			self.model.Add(a[t] - sum(self.tw[i][1] * x[i][t] for i in range(self.n+1)) <= 0)
		
		solver = cp_model.CpSolver()
		t = time.time()
		status = solver.Solve(self.model)
		t = time.time() - t

		print('\nFULL QUBO SOLVER FOR TIME WINDOWS - EXACT (CLASSICAL) SOLVER')
		
		if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
			sol = {
				'min_cost': solver.ObjectiveValue(),
				'runtime': t*1000
			}

			print('\n')
			print(f'Minimum cost: {sol["min_cost"]}')
			self.sol = [[None] * (self.n+self.m+1) for _ in range(self.n+1)]
			for i in range(self.n+1):
				for j in range(self.n+self.m+1):
					self.sol[i][j] = solver.Value(x[i][j])
			print(f'Time taken to solve: {t*1000:.3f} ms')
			return self.sol
		else:
			print('No solution found.')


	def visualize(self):
		'''Visualize solution'''

		if self.sol is None:
			print('No solution to show!')
			return
		
		activeVars = []
		for i in range(self.n+1):
			for j in range(self.n+self.m+1):
				if self.sol[i][j] == 1:
					activeVars.append((i, j))
		activeVars = sorted(activeVars, key=lambda p: p[1])

		visitedNodesByVehicle = [[] for _ in range(self.m+1)]
		i=0
		for p in activeVars:
			if p[0] == 0: i+=1
			else: visitedNodesByVehicle[i].append(p[0])

		if self.xc is None or self.yc is None:
			np.random.seed(0)
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
		nx.draw_networkx_nodes(G, pos=pos, nodelist=[0], ax=ax, alpha=0.75, node_size=300, node_color=f'C0', node_shape='s')
		
		nodelist = [[] for _ in range(self.m+1)]
		edgelist = [[] for _ in range(self.m+1)]
		for i in range(1, self.m+1):
			prev = 0
			for node in visitedNodesByVehicle[i]:
				nodelist[i].append(node)
				edgelist[i].append((prev, node))
				prev = node
			edgelist[i].append((prev, 0))
		
		# Loop over cars and plot other nodes and edges
		for k in range(1, self.m+1):

			# Get edges and nodes for this car
			nx.draw_networkx_nodes(G, pos=pos, nodelist=nodelist[k], ax=ax, alpha=0.75, node_size=300, node_color=f'C{k}')
			
			G.add_edges_from(edgelist[k])
			nx.draw_networkx_edges(G, pos=pos, edgelist=edgelist[k], width=1.5, edge_color=f'C{k}')
			
		nx.draw_networkx_labels(G, pos=pos, labels=labels, font_size=12)

		# Show plot
		plt.grid(True)
		plt.show()
