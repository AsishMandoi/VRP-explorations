import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import dimod
from neal import SimulatedAnnealingSampler
from dwave.system import LeapHybridCQMSampler

'''
Solving a custom formulation (FQS) of VRP using LeapHybridCQMSampler
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
		# self.solve()
		# self.visualize()


	def formulate(self):
		'''Formulate the problem'''

		x = [[[None] * (self.m+1) for _ in range(self.n+self.m+1)] for __ in range(self.n+1)]
		a = [[None] * (self.m+1) for _ in range(self.n+self.m+1)]
		w = [[None] * (self.m+1) for _ in range(self.n+self.m+1)]


		##### Declare variables #####
		for k in range(1, self.m+1):
			for t in range(1, self.n+self.m+1):
				for i in range(self.n+1):
					x[i][t][k] = dimod.Binary(label=f'x.{i}.{t}.{k}')
				
				w[t][k] = dimod.Integer(lower_bound=0, upper_bound=self.W_max, label=f'w.{t}.{k}')
				a[t][k] = dimod.Integer(lower_bound=0, upper_bound=self.A_max, label=f'a.{t}.{k}')
			
		##### OBJECTIVE #####
		self.model.set_objective(
															sum(self.cost[0][i] * x[i][1][k] for i in range(1, self.n+1) for k in range(1, self.m+1))
								 						 	+ sum(self.cost[i][0] * x[i][self.n+self.m][k] for i in range(1, self.n+1) for k in range(1, self.m+1))
								 					 	 	+ sum(self.cost[i][j] * x[i][t][k] * x[j][t+1][k] for i in range(self.n+1) for j in range(self.n+1) if i != j for t in range(1, self.n+self.m) for k in range(1, self.m+1))
															
															# If constraint 5 does not give a feasible solution, uncomment the following line and comment the above line and constraint 5
															# + sum(self.cost[i][j] * x[i][ti][k] * x[j][tj][k] for i in range(1, self.n+1) for j in range(1, self.n+1) if i != j for ti in range(1, self.n) for tj in range(1, self.n) if tj > ti for k in range(1, self.m+1))
		)


		##### CONSTRAINTS #####
		# 1. Exactly one client/depot is visited by exactly one vehicle at any given instant.
		for t in range(1, self.n+self.m+1):
			self.model.add_constraint(sum(x[i][t][k] for i in range(self.n+1) for k in range(1, self.m+1)) == 1)

		# 2. Each client is visited by one vehicle throughout all the time steps.
		for i in range(1, self.n+1):
			self.model.add_constraint(sum(x[i][t][k] for t in range(1, self.n+self.m+1) for k in range(1, self.m+1)) == 1)

		# 3. The depot is visited by each vehicle at some time step.
		for k in range(1, self.m+1):
			self.model.add_constraint(sum(x[0][t][k] for t in range(1, self.n+self.m+1)) == 1)

		# 4. Each vehicle visits at least one client.
		for k in range(1, self.m+1):
			self.model.add_constraint(sum(x[i][t][k] for i in range(1, self.n+1) for t in range(1, self.n+self.m+1)) >= 1)

		# 5. No other vehicle is allowed to start its journey when another vehicle is already in its journey.
		for k in range(1, self.m+1):
			for t in range(1, self.n+self.m):
				for i in range(1, self.n+1):
					#	5a. If a vehicle has visited a client in time step t, then it must visit another client or the depot in the next time step.
					self.model.add_constraint(x[i][t][k] * (1 - sum(x[j][t+1][k] for j in range(self.n+1) if j != i)) == 0)

				#	5b. If a vehicle has visited the depot in time step t, then the journey for that vehicle is over.
				self.model.add_constraint(x[0][t][k] * sum(x[i][tau][k] for tau in range(t+1, self.n+self.m+1) for i in range(1, self.n+1)) == 0)

		## Time Window constraints
		for k in range(1, self.m+1):
			for t in range(1, self.m+self.n):
				self.model.add_constraint(x[0][t][k] * a[t][k] == 0)
				self.model.add_constraint(x[0][t][k] * w[t][k] == 0)
			self.model.add_constraint(a[self.m+self.n][k] == 0)
			self.model.add_constraint(w[self.m+self.n][k] == 0)
		
		for k in range(1, self.m+1):
			# self.model.add_constraint(a[1][k] - sum(self.cost[0][i] * x[i][1][k] for i in range(1, self.n+1)) == 0)
			self.model.add_constraint(a[1][k] - sum(self.cost[0][i] * x[i][1][k] for i in range(1, self.n+1)) + self.W_max >= 0)
			self.model.add_constraint(a[1][k] - sum(self.cost[0][i] * x[i][1][k] for i in range(1, self.n+1)) - self.W_max <= 0)
		
		for t in range(2, self.m+self.n+1):
			for k in range(1, self.m+1):
				# self.model.add_constraint(
				# 	a[t][k]
				# 	- a[t-1][k]
				# 	- w[t-1][k]
				# 	- sum(self.cost[i][j] * x[i][t-1][k] * x[j][t][k] for i in range(self.n+1) for j in range(self.n+1) if i != j)
				# 	== 0
				# )
				self.model.add_constraint(
					a[t][k]
					- a[t-1][k]
					- w[t-1][k]
					- sum(self.cost[i][j] * x[i][t-1][k] * x[j][t][k] for i in range(self.n+1) for j in range(self.n+1) if i != j)
					+ self.W_max * (1 - sum(x[i][t-1][k] for i in range(1, self.n+1)))
					>= 0
				)
				self.model.add_constraint(
					a[t][k]
					- a[t-1][k]
					- w[t-1][k]
					- sum(self.cost[i][j] * x[i][t-1][k] * x[j][t][k] for i in range(self.n+1) for j in range(self.n+1) if i != j)
					- self.W_max * (1 - sum(x[i][t-1][k] for i in range(1, self.n+1)))
					<= 0
				)

		for k in range(1, self.m+1):
			for t in range(1, self.m+self.n+1):
				self.model.add_constraint(a[t][k] + w[t][k] - sum(self.tw[i][0] * x[i][t][k] for i in range(self.n+1)) >= 0)
				self.model.add_constraint(a[t][k] + w[t][k] - sum(self.tw[i][1] * x[i][t][k] for i in range(self.n+1)) <= 0)


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

			# !Does not give the correct minimum cost for some reason.
			# print(f'Minimum total cost: {self.sol.energy} ____(X)')

			# Evaluate cost of the solution ___________________________________________________________________________(1)
			# x = [[[None] * (self.m+1) for _ in range(self.n+1)] for __ in range(self.n+1)]
			# varList = [var for var in self.sol.sample]
			# for var in varList:
			# 	i=int(var.split('.')[1])
			# 	t=int(var.split('.')[2])
			# 	k=int(var.split('.')[3])
			# 	x[i][t][k] = self.sol.sample[var]
			# tot_cost = sum(self.cost[0][i] * x[i][1][k] for i in range(1, self.n+1) for k in range(1, self.m+1)) + sum(self.cost[i][0] * x[i][self.n][k] for i in range(1, self.n+1) for k in range(1, self.m+1)) + sum(self.cost[i][j] * x[i][t][k] * x[j][t+1][k] for i in range(1, self.n+1) for j in range(1, self.n+1) if i != j for t in range(1, self.n) for k in range(1, self.m+1))
			# tot_cost = sum(self.cost[0][i] * x[i][1][k] for i in range(1, self.n+1) for k in range(1, self.m+1)) + sum(self.cost[i][0] * x[i][self.n][k] for i in range(1, self.n+1) for k in range(1, self.m+1)) + sum(self.cost[i][j] * x[i][ti][k] * x[j][tj][k] for i in range(1, self.n+1) for j in range(1, self.n+1) if i != j for ti in range(1, self.n) for tj in range(1, self.n) if tj > ti for k in range(1, self.m+1))

			# Evaluate cost of the solution ___________________________________________________________________________(2)
			activeVarList = [[] for _ in range(self.m+1)]
			for i in self.sol.sample:
				if self.sol.sample[i] == 1:
					activeVarList[int(i.split('.')[3])].append((int(i.split('.')[1]), int(i.split('.')[2])))
			
			for i in range(self.m+1):
				activeVarList[i] = sorted(activeVarList[i], key=lambda x: x[1])

			tot_cost = 0.0
			for i in range(1, self.m+1):
				tmp = 0
				for var in activeVarList[i]:
					if var[0] != tmp:
						tot_cost += self.cost[tmp][var[0]]
						tmp = var[0]
				tot_cost += self.cost[tmp][0]	
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
		
		activeVarList = [[] for _ in range(self.m+1)]
		for i in self.sol.sample:
			if self.sol.sample[i] == 1:
				activeVarList[int(i.split('.')[3])].append((int(i.split('.')[1]), int(i.split('.')[2])))
		
		for i in range(self.m+1):
			activeVarList[i] = sorted(activeVarList[i], key=lambda x: x[1])

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
			tmp = 0
			for var in activeVarList[i]:
				if var[0] != tmp:
					if var[0]:
						nodelist[i].append(var[0])
					edgelist[i].append((tmp, var[0]))
					tmp = var[0]
			edgelist[i].append((tmp, 0))
		
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
