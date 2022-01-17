import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import dimod
from neal import SimulatedAnnealingSampler
from dwave.system import LeapHybridCQMSampler

'''
Solving a custom formulation (GPS) of VRP using LeapHybridCQMSampler
'''

class GPS():
	'''Guillermo Alonso-Linaje, Parfait Atchade-Adelomou, and Saúl Gonzalez-Bermejo'''

	def __init__(self, n, m, cost, xc, yc):
		self.n = n
		self.m = m
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
		x = [[[[None] * (self.m+1) for _ in range(3)] for __ in range(self.n+1)] for ___ in range(self.n+1)]
		a = [[None] * (self.n+1) for _ in range(self.n+1)]
		for i in range(self.n+1):
			for j in range(self.n+1):
				if i != j:
					a[i][j] = dimod.Binary(f'a.{i}.{j}')
					for r in range(3):
						for k in range(1, self.m+1):
							x[i][j][r][k] = dimod.Binary(f'x.{i}.{j}.{r}.{k}')
		

		##### OBJECTIVE #####
		self.model.set_objective(sum(self.cost[i][j] * x[i][j][1][k] for i in range(self.n+1) for j in range(self.n+1) if i != j for k in range(1, self.m+1))
														
														# Objective function for Constraint-7
														#  + sum(a[i][j]*a[j][k] - a[i][j]*a[i][k] - a[j][k]*a[i][k] + a[i][k]*a[i][k] for i in range(1, self.n+1) for j in range(1, self.n+1) for k in range(1, self.n+1) if i != j and j != k and k != i)
		)

		
		##### CONSTRAINTS #####
		# Constraint 1: For each i, j, q, one and only one of the possibilities must be met for r.
		for i in range(self.n+1):
			for j in range(self.n+1):
				if i != j:
					for k in range(1, self.m+1):
						self.model.add_constraint(sum(x[i][j][r][k] for r in range(3)) == 1)

		# Constraint 2a: Each vehicle must leave the depot in the beginning.
		for k in range(1, self.m+1):
			self.model.add_constraint(sum(x[0][j][1][k] for j in range(1, self.n+1)) == 1)

		# Constraint 2b: Each vehicle must reach the depot in the end.
		for k in range(1, self.m+1):
			self.model.add_constraint(sum(x[i][0][1][k] for i in range(1, self.n+1)) == 1)
		
		# Constraint 3a: Every vehicle must arrive at each city once and only once.
		for j in range(1, self.n+1):
			self.model.add_constraint(sum(x[i][j][1][k] for i in range(self.n+1) if i != j for k in range(1, self.m+1)) == 1)

		# Constraint 3b: Every vehicle must leave each city once and only once.
		for i in range(1, self.n+1):
			self.model.add_constraint(sum(x[i][j][1][k] for j in range(self.n+1) if i != j for k in range(1, self.m+1)) == 1)

		# Constraint 4: A city i can either be reached earlier or later than a city j.
		for i in range(1, self.n+1):
			for j in range(1, self.n+1):
				if i != j:
					self.model.add_constraint(sum(x[i][j][r][k] for r in range(2) for k in range(1, self.m+1)) - a[i][j] * self.m == 0)
					self.model.add_constraint(a[i][j] + a[j][i] == 1)

		# Constraint 5: If the vehicle q arrives in the city j, then the vehicle q must leave the city j
		for i in range(self.n+1):
			for j in range(1, self.n+1):
				if i != j:
					for l in range(1, self.m+1):
						self.model.add_constraint(x[i][j][1][l] * (1 - sum(x[j][k][1][l] for k in range(self.n+1) if k != j)) == 0)

		# Constraint 6: A vehicle can either arrive at city i before city j or arrive at city j before city i.
		for i in range(1, self.n+1):
			for j in range(1, self.n+1):
				if i != j:
					for k in range(1, self.m+1):
						self.model.add_constraint(sum(x[i][j][r][k] for r in range(2)) + sum(x[j][i][r][k] for r in range(2)) == 1)

		# Constraint 7: If city i is reached before city j, and city j is reached before city k, then city i must be reached before city k.
		for i in range(1, self.n+1):
			for j in range(1, self.n+1):
				for k in range(1, self.n+1):
					if i != j and j != k and k != i:
						self.model.add_constraint(a[i][j]*a[j][k] - a[i][j]*a[i][k] - a[j][k]*a[i][k] + a[i][k]*a[i][k] == 0)


	def solve(self):
		'''Solve the problem'''
		
		sampler = LeapHybridCQMSampler()
		sampleset = sampler.sample_cqm(self.model, label=f"Vehicle Routing Problem ({self.n} Clients, {self.m} Vehicles) - GPS")
		feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)

		print('\nGUILLERMO, PARFAIT, SAÚL SOLVER (Constrained Quadratic Model)')
		if len(feasible_sampleset):
			print("{} feasible solutions of {}.".format(len(feasible_sampleset), len(sampleset)))
			self.sol = feasible_sampleset.first
			
			# ! Does not give the correct minimum cost, since the objective function is not exactly right.
			# print(f'Minimum total cost: {self.sol.energy}')
		
			### DONE ABOVE ###
			# Evaluate cost of the solution
			x = [[[None] * (self.m+1) for __ in range(self.n+1)] for ___ in range(self.n+1)]
			varList = [var for var in self.sol.sample if var.split('.')[0] == 'x' and var.split('.')[3] == '1']
			for var in varList:
				i=int(var.split('.')[1])
				j=int(var.split('.')[2])
				k=int(var.split('.')[4])
				x[i][j][k] = self.sol.sample[var]
			tot_cost = sum(self.cost[i][j] * x[i][j][k] for i in range(self.n+1) for j in range(self.n+1) if i != j for k in range(1, self.m+1))
			print(f'Minimum total cost: {tot_cost}')
		
		print(f"Number of variables: {len(sampleset.variables)}")
		print(f"Runtime: {sampleset.info['run_time']/1000:.3f} ms")

		return {
			'min_cost': self.sol.energy,
			'runtime': sampleset.info['run_time']/1000,
			'num_vars': len(sampleset.variables)
		}


	def visualize(self):
		'''Visualize solution'''

		if self.sol is None:
			print('No solution to show!')
			return

		activeVarList = [i for i in self.sol.sample if self.sol.sample[i] == 1]
		activeVarList = [(int(i.split('.')[4]), int(i.split('.')[1]), int(i.split('.')[2])) for i in activeVarList if i.split('.')[0] == 'x' and i.split('.')[3] == '1']
		
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
		nx.draw_networkx_nodes(G, pos=pos, nodelist=[0], ax=ax, alpha=0.75, node_size=300, node_color=f'C0', node_shape='s')

		edgelist = [[] for _ in range(self.m+1)]
		nodelist = [[] for _ in range(self.m+1)]
		for var in activeVarList:
			if var[1]:
				nodelist[var[0]].append(var[1])
			edgelist[var[0]].append((var[1], var[2]))
		
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
