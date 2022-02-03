import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import dimod
from neal import SimulatedAnnealingSampler
from dwave.system import LeapHybridCQMSampler

'''
Solving a custom formulation (GPS) of TSP using LeapHybridCQMSampler
'''

class GPS():
	'''Guillermo Alonso-Linaje, Parfait Atchade-Adelomou, and Saúl Gonzalez-Bermejo'''

	def __init__(self, n, cost, xc, yc):
		self.n = n
		self.cost = cost
		self.xc = xc
		self.yc = yc
		self.sol = None
		self.model = dimod.ConstrainedQuadraticModel()
		self.formulate()
		# self.solve()
		# self.visualize()
	

	def formulate(self):
		'''Formulate the problem'''
		
		##### Declare Variables #####
		x = [[[None] * 3 for _ in range(self.n+1)] for __ in range(self.n+1)]
		# a = [[None] * (self.n+1) for _ in range(self.n+1)]
		for i in range(self.n+1):
			for j in range(self.n+1):
				if i != j:
					# a[i][j] = dimod.Binary(f'a.{i}.{j}')
					for r in range(3):
						x[i][j][r] = dimod.Binary(f'x.{i}.{j}.{r}')
		

		##### OBJECTIVE #####
		self.model.set_objective(sum(self.cost[i][j] * x[i][j][1] for i in range(self.n+1) for j in range(self.n+1) if i != j)
														
														# Objective function for Constraint-4
														#  + sum(x[i][j][0] * x[j][k][0] - x[i][j][0] * x[i][k][0] - x[j][k][0] * x[i][k][0] + x[i][k][0] for i in range(1, self.n+1) for j in range(1, self.n+1) for k in range(1, self.n+1) if i != j and j != k and k != i)
		)

		
		##### CONSTRAINTS #####
		# Constraint 1: For each i, j, one and only one of the possibilities must be met for r.
		for i in range(self.n+1):
			for j in range(self.n+1):
				if i != j:
					self.model.add_constraint(sum(x[i][j][r] for r in range(3)) == 1)

		# Constraint 2a: Each city (including depot) must be reached only once.
		for j in range(self.n+1):
			self.model.add_constraint(sum(x[i][j][1] for i in range(self.n+1) if i != j) == 1)

		# Constraint 2b: Each city (including depot) must be exited only once.
		for i in range(self.n+1):
			self.model.add_constraint(sum(x[i][j][1] for j in range(self.n+1) if i != j) == 1)

		# Constraint 3: A city i can either be reached earlier or later than a city j.
		for i in range(1, self.n+1):
			for j in range(1, self.n+1):
				if i != j:
					self.model.add_constraint(x[i][j][2] + x[j][i][2] == 1)

		# Constraint 4: If city i is reached before city j, and city j is reached before city k, then city i must be reached before city k.
		for i in range(1, self.n+1):
			for j in range(1, self.n+1):
				for k in range(1, self.n+1):
					if i != j and j != k and k != i:
						self.model.add_constraint(x[j][i][2] * x[k][j][2] - x[j][i][2] * x[k][i][2] - x[k][j][2] * x[k][i][2] + x[k][i][2] == 0)


	def solve(self):
		'''Solve the problem'''
		
		sampler = LeapHybridCQMSampler()
		sampleset = sampler.sample_cqm(self.model, label=f"Travelling Salesman Problem ({self.n} Clients) - GPS")
		feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)

		if len(feasible_sampleset):
			print("{} feasible solutions of {}.".format(len(feasible_sampleset), len(sampleset)))
			self.sol = feasible_sampleset.first
			print(f'Minimum total cost: {self.sol.energy}')
		
			### DONE ABOVE ###
			# # Evaluate cost of the solution
			# x = [[None] * (self.n+1) for _ in range(self.n+1)]
			# varList = [var for var in self.sol.sample if var.split('.')[3] == '1']
			# for var in varList:
			# 	i=int(var.split('.')[1])
			# 	j=int(var.split('.')[2])
			# 	x[i][j] = self.sol.sample[var]
			# tot_cost = sum(self.cost[i][j] * x[i][j] for i in range(self.n+1) for j in range(self.n+1) if i != j)
			# print(f'Minimum total cost: {tot_cost}')
		
		print(f"Number of variables: {len(sampleset.variables)}")
		print(f"Runtime: {sampleset.info['run_time']/1000:.3f} ms")

		return sampleset


	def visualize(self):
		'''Visualize solution'''

		if self.sol is None:
			print('No solution to show!')
			return

		activeVarList = [i for i in self.sol.sample if self.sol.sample[i] == 1]
		activeVarList = [(int(i.split('.')[1]), int(i.split('.')[2])) for i in activeVarList if i.split('.')[3] == '1']
		
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
		nx.draw_networkx_nodes(G, pos=pos, nodelist=[0], ax=ax, alpha=0.75, node_size=300, node_color=f'C1', node_shape='s')

		edgelist = []
		nodelist = []
		for var in activeVarList:
			if var[0]:
				nodelist.append(var[0])
			edgelist.append((var[0], var[1]))
		
		# Plot other nodes and edges
		nx.draw_networkx_nodes(G, pos=pos, nodelist=nodelist, ax=ax, alpha=0.75, node_size=300, node_color=f'C0')
		
		G.add_edges_from(edgelist)
		nx.draw_networkx_edges(G, pos=pos, edgelist=edgelist, width=1.5, edge_color=f'C2')

		nx.draw_networkx_labels(G, pos=pos, labels=labels, font_size=12)

		# Show plot
		plt.grid(True)
		plt.show()
