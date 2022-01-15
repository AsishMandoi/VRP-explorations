import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import dimod
from neal import SimulatedAnnealingSampler
from dwave.system import LeapHybridCQMSampler

'''
Solving a custom formulation (FQS) of TSP using LeapHybridCQMSampler
'''

class FQS():
	'''Full Qubo Solver'''

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

		x = [[None] * (self.n+1) for _ in range(self.n+1)]


		##### Declare variables #####
		for i in range(self.n+1):
			for t in range(1, self.n+1):
				x[i][t] = dimod.Binary(label=f'x.{i}.{t}')


		##### OBJECTIVE #####
		self.model.set_objective(
														 	sum(self.cost[0][i] * x[i][1] for i in range(1, self.n+1)) +
								 						 	sum(self.cost[i][0] * x[i][self.n] for i in range(1, self.n+1)) +
								 					 	 	sum(self.cost[i][j] * x[i][t] * x[j][t+1] for i in range(1, self.n+1) for j in range(1, self.n+1) if i != j for t in range(1, self.n))
		)


		##### CONSTRAINTS #####
		# Exactly one client node is visited at every time step.
		for t in range(1, self.n+1):
			self.model.add_constraint(sum(x[i][t] for i in range(1, self.n+1)) == 1)

		# Each client node is visited once throughout all the time steps.
		for i in range(1, self.n+1):
			self.model.add_constraint(sum(x[i][t] for t in range(1, self.n+1)) == 1)


	def solve(self):
		'''Solve the problem'''
		
		sampler = LeapHybridCQMSampler()
		sampleset = sampler.sample_cqm(self.model, label=f'Travelling Salesman Problem ({self.n} Clients) - FQS')
		feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)
		
		if len(feasible_sampleset):
			self.sol = feasible_sampleset.first
			print("{} feasible solutions of {}.".format(len(feasible_sampleset), len(sampleset)))
			print(f'Minimum total cost: {self.sol.energy}')

			### DONE ABOVE ###
			# # Evaluate cost of the solution
			# x = [[None] * (self.n+1) for _ in range(self.n+1)]
			# varList = [var for var in self.sol.sample]
			# for var in varList:
			# 	i=int(var.split('.')[1])
			# 	t=int(var.split('.')[2])
			# 	x[i][t] = self.sol.sample[var]
			# tot_cost = sum(self.cost[0][i] * x[i][1] for i in range(1, self.n+1)) + sum(self.cost[i][0] * x[i][self.n] for i in range(1, self.n+1)) + sum(self.cost[i][j] * x[i][t] * x[j][t+1] for i in range(1, self.n+1) for j in range(1, self.n+1) if i != j for t in range(1, self.n))
			# print(f'Minimum total cost: {tot_cost}')
		
		print(f"Number of variables: {len(sampleset.variables)}")
		print(f"Runtime: {sampleset.info['run_time']/1000:.3f} ms")

		return sampleset


	def visualize(self):
		'''Visualize solution'''

		if self.sol is None:
			print('No solution to show!')
			return
		
		activeVarList = []
		for i in self.sol.sample:
			if self.sol.sample[i] == 1:
				activeVarList.append((int(i.split('.')[1]), int(i.split('.')[2])))
		
		activeVarList = sorted(activeVarList, key=lambda x: x[1])

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
		
		nodelist = []
		edgelist = []
		tmp = 0
		for var in activeVarList:
			if var[0] != tmp:
				if var[0]:
					nodelist.append(var[0])
				edgelist.append((tmp, var[0]))
				tmp = var[0]
		edgelist.append((tmp, 0))
		
		# Plot other nodes and edges
		nx.draw_networkx_nodes(G, pos=pos, nodelist=nodelist, ax=ax, alpha=0.75, node_size=300, node_color=f'C0')
		
		G.add_edges_from(edgelist)
		nx.draw_networkx_edges(G, pos=pos, edgelist=edgelist, width=1.5, edge_color=f'C2')
			
		nx.draw_networkx_labels(G, pos=pos, labels=labels, font_size=12)

		# Show plot
		plt.grid(True)
		plt.show()
