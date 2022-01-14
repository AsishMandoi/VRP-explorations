from ortools.sat.python import cp_model
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import time


'''
Solving a custom formulation (GPS) of TSP using Google's OR-Tools
'''


class GPS():
	'''Guillermo Alonso-Linaje, Parfait Atchade-Adelomou, and Saúl Gonzalez-Bermejo'''

	def __init__(self, n, cost, xc, yc):
		self.n = n
		self.cost = cost
		self.xc = xc
		self.yc = yc
		self.sol = None
		self.model = cp_model.CpModel()
		self.formulate_and_solve()
		self.visualize()
	

	def formulate_and_solve(self):
		'''Formulate and solve the problem'''
		
		##### Declare Variables #####
		x = [[[None] * 3 for _ in range(self.n+1)] for __ in range(self.n+1)]
		w = [[[[None] * 3 for _ in range(self.n+1)] for __ in range(self.n+1)] for ___ in range(self.n+1)]

		for i in range(self.n+1):
			for j in range(self.n+1):
				if i != j:
					for r in range(3):
						x[i][j][r] = self.model.NewBoolVar(f'x.{i}.{j}.{r}')

		for i in range(self.n+1):
			for j in range(self.n+1):
				for k in range(self.n+1):
					if i != j and i != k and j != k:
						for r in range(1, 3):
							w[i][j][k][r] = self.model.NewBoolVar(f'w.{i}.{j}.{k}.{r}')


		##### CONSTRAINTS #####
		# Constraint 1: For each i, j one and only one of the possibilities must be met for r.
		for i in range(self.n+1):
			for j in range(self.n+1):
				if i != j:
					self.model.Add(sum(x[i][j][r] for r in range(3)) == 1)

		# Constraint 2a: Each city (including depot) must be reached only once.
		for j in range(self.n+1):
			self.model.Add(sum(x[i][j][1] for i in range(self.n+1) if i != j) == 1)

		# Constraint 2b: Each city (including depot) must be exited only once.
		for i in range(self.n+1):
			self.model.Add(sum(x[i][j][1] for j in range(self.n+1) if i != j) == 1)

		# Constraint 3: A city i can either be reached earlier or later than a city j.
		for i in range(1, self.n+1):
			for j in range(1, self.n+1):
				if i != j:
					self.model.Add(x[i][j][0] + x[j][i][0] == 1)

		# Constraint 4: If city i is reached before city j, and city j is reached before city k, then city i must be reached before city k.
		for i in range(1, self.n+1):
			for j in range(1, self.n+1):
				for k in range(1, self.n+1):
					if i != j and j != k and k != i:
						self.model.Add(x[i][j][0] + x[j][k][0] <= 2 * x[i][k][0] + w[i][j][k][1])
						self.model.Add(x[i][j][0] + x[j][k][0] >= 2 * x[i][k][0] - w[i][j][k][2])
		
		
		##### OBJECTIVE #####
		self.model.Minimize(sum(self.cost[i][j] * x[i][j][1] for i in range(self.n+1) for j in range(self.n+1) if i != j))


		##### SOLVE
		# Call the solver
		solver = cp_model.CpSolver()
		t = time.time()
		status = solver.Solve(self.model)
		t = time.time() - t


		# Display the solution
		if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
			print(f'\nMinimum cost: {solver.ObjectiveValue()}')
			self.sol = [[None for _ in range(self.n+1)] for __ in range(self.n+1)]
			for i in range(self.n+1):
				for j in range(self.n+1):
					if j != i:
						self.sol[i][j] = solver.Value(x[i][j][1])
		else:
			print('No solution found.')

		print(f'\nTime taken to solve: {t*1000:.3f} ms')


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
		ax.set_title(f'Travelling Salesman Problem - {self.n} Clients')

		# Build graph
		G = nx.MultiDiGraph()
		G.add_nodes_from(range(self.n + 1))
		
		pos = {i: (self.xc[i], self.yc[i]) for i in range(self.n + 1)}
		labels = {i: str(i) for i in range(self.n + 1)}
		
		# Plot depot node
		nx.draw_networkx_nodes(G, pos=pos, nodelist=[0], ax=ax, alpha=0.75, node_size=300, node_color='C1', node_shape='s')

		edgelist = []
		nodelist = []
		for i in range(self.n + 1):
			for j in range(self.n + 1):
				if i != j and self.sol[i][j] == 1:
					if i:
						nodelist.append(i)
					edgelist.append((i, j))
		
		# Plot other nodes and edges
		nx.draw_networkx_nodes(G, pos=pos, nodelist=nodelist, ax=ax, alpha=0.75, node_size=300, node_color='C0')
		
		G.add_edges_from(edgelist)
		nx.draw_networkx_edges(G, pos=pos, edgelist=edgelist, width=1.5, edge_color='C2')

		nx.draw_networkx_labels(G, pos=pos, labels=labels, font_size=12)

		# Show plot
		plt.grid(True)
		plt.show()
