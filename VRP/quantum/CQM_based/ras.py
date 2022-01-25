from dimod.reference.samplers.exact_solver import ExactCQMSolver
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import dimod
from neal import SimulatedAnnealingSampler
from dwave.system import LeapHybridCQMSampler

'''
Solving a custom formulation (RAS) of VRP using LeapHybridCQMSampler
'''

class RAS():
	'''Route Activation Solver (Method-1)'''

	def __init__(self, n, m, cost, xc, yc):
		self.n = n
		self.m = m
		self.cost = cost
		self.xc = xc
		self.yc = yc
		self.T_max = max(self.n-self.m+1, 1)
		self.B = 2*self.n
		self.sol = None
		self.model = dimod.ConstrainedQuadraticModel()
		self.formulate()
		# self.solve()
		# self.visualize()


	def formulate(self):
		'''Formulate the problem'''

		x = [[None] * (self.n+1) for _ in range(self.n+1)]
		y = [[None] * (self.m+1) for _ in range(self.n+1)]
		t = [None] * (self.n+1)


		##### Declare variables #####
		for i in range(self.n+1):
			for j in range(self.n+1):
				if i != j:
					x[i][j] = dimod.Binary(label=f'x.{i}.{j}')

		for i in range(1, self.n+1):
			t[i] = dimod.Integer(lower_bound=1, upper_bound=self.T_max, label=f't.{i}')
			for k in range(1, self.m+1):
				y[i][k] = dimod.Binary(label=f'y.{i}.{k}')


		##### OBJECTIVE #####
		self.model.set_objective(sum(self.cost[i][j]*x[i][j] for i in range(self.n+1) for j in range(self.n+1) if i != j))


		##### CONSTRAINTS #####
		for i in range(1, self.n+1):

			# Each client node must have exactly one edge directed away from it
			self.model.add_constraint(sum(x[i][j] for j in range(self.n+1) if j != i) == 1)

			# Each client node must have exactly one edge directed towards it
			self.model.add_constraint(sum(x[j][i] for j in range(self.n+1) if j != i) == 1)

			# Each client must be visited by exactly one vehicle
			self.model.add_constraint(sum(y[i][k] for k in range(1, self.m+1)) == 1)

		# m vehicles coming from depot
		self.model.add_constraint(sum(x[0][i] for i in range(1, self.n+1)) == self.m)

		# m vehicles going to depot
		self.model.add_constraint(sum(x[i][0] for i in range(1, self.n+1)) == self.m)

		# If there is an edge from node i to node j, then both ith and jth node must be visited by the same vehicle
		for i in range(1, self.n+1):
			for j in range(1, self.n+1):
				if i != j:
					for k in range(1, self.m+1):
						self.model.add_constraint(y[i][k] - y[j][k] + (1 - x[i][j]) >= 0)
						self.model.add_constraint(y[j][k] - y[i][k] + (1 - x[i][j]) >= 0)

		# Each vehicle must be used at least once
		for k in range(1, self.m+1):
			self.model.add_constraint(sum(y[i][k] for i in range(1, self.n+1)) >= 1)

		# Eliminate subtours - MTZ constraint
		for i in range(1, self.n+1):
			for j in range(1, self.n+1):
				if i != j:
					for k in range(1, self.m+1):
						self.model.add_constraint(t[j] * y[j][k] - (t[i] + 1) * y[i][k] + self.B*(1 - x[i][j])>= 0)


	def solve(self, **params):
		'''Solve the problem'''
		
		sampler = LeapHybridCQMSampler()
		sampleset = sampler.sample_cqm(self.model, label=f"Vehicle Routing Problem ({self.n} Clients, {self.m} Vehicles) - RAS", time_limit=params['time_limit'])
		feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)
		
		print('\nROUTE ACTIVATION SOLVER (Constrained Quadratic Model)')
		if len(feasible_sampleset):
			self.sol = feasible_sampleset.first
			print("{} feasible solutions of {}.".format(len(feasible_sampleset), len(sampleset)))
			print(f'Minimum total cost: {self.sol.energy}')
			print(f"Number of variables: {len(sampleset.variables)}")
			print(f"Runtime: {sampleset.info['run_time']/1000:.3f} ms")
		
			return {
				'min_cost': self.sol.energy,
				'runtime': sampleset.info['run_time']/1000,
				'num_vars': len(sampleset.variables)
			}	
		
		print("No feasible solutions.")
		print(f"Number of variables: {len(sampleset.variables)}")
		print(f"Runtime: {sampleset.info['run_time']/1000:.3f} ms")
		return {
			'min_cost': None,
			'runtime': sampleset.info['run_time']/1000,
			'num_vars': len(sampleset.variables)
		}


	def visualize(self):
		'''Visualize solution'''

		if self.sol is None:
			print('No solution to show!')
			return
		
		activeVarList = [i for i in self.sol.sample if self.sol.sample[i] == 1]
		activeXList = [(int(i.split('.')[1]), int(i.split('.')[2])) for i in activeVarList if i.split('.')[0] == 'x']
		activeYList = [(int(i.split('.')[1]), int(i.split('.')[2])) for i in activeVarList if i.split('.')[0] == 'y']
		
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

		next_node = [[]] + [None for _ in range(1, self.n + 1)]
		node_id = [None] * (self.n + 1)
		for from_node, to_node in activeXList:
			if from_node:
				next_node[from_node] = to_node
			else:
				next_node[from_node].append(to_node)
		
		nodelist = [[] for _ in range(self.m+1)]
		edgelist = [[] for _ in range(self.m+1)]
		for i, k in activeYList:
			nodelist[k].append(i)
			node_id[i] = k
			edgelist[k].append((i, next_node[i]))

		for i in next_node[0]:
			edgelist[node_id[i]].append((0, i))
		
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
