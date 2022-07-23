import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import dimod
from neal import SimulatedAnnealingSampler
from dwave.system import LeapHybridCQMSampler

'''
Solving a custom formulation (Full Qubo Solver) of VRPTW using LeapHybridCQMSampler
'''

class FQSTW():
	'''Full Qubo Solver for Vehicle Routing Problem with Time Windows'''

	def __init__(self, n, m, cost, xc, yc, tw):
		self.n = n
		self.m = m
		self.cost = cost
		self.xc = xc
		self.yc = yc
		self.tw = tw
		self.A_max = 30
		self.W_max = 30
		self.sol = None
		self.model = dimod.ConstrainedQuadraticModel()
		self.formulate()


	def formulate(self):
		'''Formulate the problem'''

		x = [[None] * (self.n+self.m+1) for _ in range(self.n+1)]
		a = [None] * (self.n+self.m+1)
		w = [None] * (self.n+self.m+1)

		##### Declare variables #####
		for t in range(self.n+self.m+1):
			for i in range(self.n+1):
				if t==0 or t==self.n+self.m:
					x[i][t] = 0
				else:
					x[i][t] = dimod.Binary(label=f'x.{i}.{t}')
			
			if t!=0 and t!=self.n+self.m:
				w[t] = dimod.Integer(lower_bound=0, upper_bound=self.W_max, label=f'w.{t}')
				a[t] = dimod.Integer(lower_bound=0, upper_bound=self.A_max, label=f'a.{t}')
		
		x[0][0] = 1								# At the zero-th time step the first vehicle is at the depot
		x[0][self.m+self.n] = 1		# At the final time step the last vehicle is at the depot
		a[0] = 0
		w[0] = 0
		a[self.m+self.n] = 0
		w[self.m+self.n] = 0

		##### OBJECTIVE #####
		self.model.set_objective(sum(self.cost[i][j] * x[i][t] * x[j][t+1] for i in range(self.n+1) for j in range(self.n+1) for t in range(self.n+self.m)))

		##### CONSTRAINTS #####
		# 1. Exactly one client/depot is visited by exactly one vehicle at any given instant.
		for t in range(1, self.n+self.m):
			self.model.add_constraint(sum(x[i][t] for i in range(self.n+1)) == 1)

		# 2. Each client is visited by one vehicle throughout all the time steps during the delivery.
		for i in range(1, self.n+1):
			self.model.add_constraint(sum(x[i][t] for t in range(1, self.n+self.m)) == 1)

		# 3. The depot is visited (returned to) by m vehicles throughout the delivery process.
		self.model.add_constraint(sum(x[0][t] for t in range(1, self.n+self.m+1)) == self.m)

		# 4. Each vehicle visits at least one client. (No vehicle starts from the depot and ends up in the depot in the next immediate step)
		for t in range(self.m+self.n):
			self.model.add_constraint(x[0][t] + x[0][t+1] <= 1)

		## Time Window constraints
		# 5. Arrival time and waiting time at the depot is always set to 0
		for t in range(1, self.m+self.n):
			self.model.add_constraint(x[0][t] * a[t] == 0)
			self.model.add_constraint(x[0][t] * w[t] == 0)
		
		# 6. Recursively defining the arrival time w.r.t. the arrival times, waiting times and time to travel
		# for t in range(self.n+self.m):
			# self.model.add_constraint(a[t+1] - a[t] - w[t] - sum(self.cost[i][j] * x[i][t] * x[j][t+1] for i in range(self.n+1) for j in range(self.n+1)) == 0)
			# self.model.add_constraint(a[t+1] - a[t] - w[t]
			# 													- sum(self.cost[i][j] * x[i][t] * x[j][t+1] for i in range(self.n+1) for j in range(self.n+1))
			# 													+ self.W_max * (1 - sum(x[i][t] for i in range(1, self.n+1)))
			# 													>= 0
			# )
			# self.model.add_constraint(a[t+1] - a[t] - w[t]
			# 													- sum(self.cost[i][j] * x[i][t] * x[j][t+1] for i in range(self.n+1) for j in range(self.n+1))
			# 													- self.W_max * (1 - sum(x[i][t] for i in range(1, self.n+1)))
			# 													<= 0
			# )
			# self.model.add_constraint(
			# 	a[t+1]
			# 	- sum(w[tau] for tau in range(t+1))
			# 	- sum(self.cost[i][j] * x[i][tau] * x[j][tau+1] for i in range(self.n+1) for j in range(self.n+1) for tau in range(t+1))
			# 	+ self.W_max * (1 - sum(x[i][t] for i in range(1, self.n+1)))
			# 	>= 0
			# )
			# self.model.add_constraint(
			# 	a[t+1]
			# 	- sum(w[tau] for tau in range(t+1))
			# 	- sum(self.cost[i][j] * x[i][tau] * x[j][tau+1] for i in range(self.n+1) for j in range(self.n+1) for tau in range(t+1))
			# 	- self.W_max * (1 - sum(x[i][t] for i in range(1, self.n+1)))
			# 	<= 0
			# )
		
		for t in range(1, self.m+self.n):
			self.model.add_constraint(a[t] + w[t] - sum(self.tw[i][0] * x[i][t] for i in range(self.n+1)) >= 0)
			self.model.add_constraint(a[t] + w[t] - sum(self.tw[i][1] * x[i][t] for i in range(self.n+1)) <= 0)


	def solve(self, **params):
		'''Solve the problem'''
		
		sampler = LeapHybridCQMSampler()
		sampleset = sampler.sample_cqm(self.model, label=f'Vehicle Routing Problem ({self.n} Clients, {self.m} Vehicles) - FQS', time_limit=params['time_limit'])
		feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)
		
		print('\nFULL QUBO SOLVER (Constrained Quadratic Model)')
		
		tot_cost = None
		if len(feasible_sampleset):
			self.sol = feasible_sampleset.first
			print("{} feasible solutions of {}.".format(len(feasible_sampleset), len(sampleset)))

			tot_cost = self.sol.energy
			print(f'Minimum total cost: {tot_cost}')
		else:
			print("No feasible solutions.")
		
		print(f"Number of variables: {len(sampleset.variables)}")
		print(f"Runtime: {sampleset.info['run_time']/1000:.3f} ms")

		return dict({
			'min_cost': tot_cost,
			'runtime': sampleset.info['run_time']/1000,
			'num_vars': len(sampleset.variables),
			'sampleset': sampleset
		})


	def visualize(self):
		'''Visualize solution'''

		if self.sol is None:
			print('No solution to show!')
			return
		
		activeVars = []
		for x in self.sol.sample:
			if self.sol.sample[x] == 1:
				activeVars.append((int(x.split('.')[1]), int(x.split('.')[2])))
		activeVars = sorted(activeVars, key=lambda p: p[1])

		visitedNodesByVehicle = [[] for _ in range(self.m+1)]
		i=1
		for p in activeVars:
			if p[0] == 0: i+=1
			else: visitedNodesByVehicle[i].append(p[0])

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
