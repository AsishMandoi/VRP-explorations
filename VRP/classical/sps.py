from ortools.sat.python import cp_model
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

'''
Solving VRP using a CP-SAT solver
CP-SAT solver: A constraint programming solver that uses SAT (satisfiability) methods.
'''

class SPS():
    '''Solution Partitioning Solver'''

    def __init__(self, n, m, cost, xc, yc):
        self.n = n
        self.m = m
        self.cost = cost
        self.xc = xc
        self.yc = yc
        self.model = cp_model.CpModel()
        self.sol = None

        # Instantiate the data problem.
        self.data = self.create_data_model()

        # Create the routing index manager.
        self.manager = pywrapcp.RoutingIndexManager(len(self.data['distance_matrix']), self.data['num_vehicles'], self.data['depot'])
        
        # Create Routing Model.
        self.routing = pywrapcp.RoutingModel(self.manager)

        self.main()

        if self.sol:
            nodelists = self.partition()
            self.print_solution(nodelists=nodelists)
            self.visualize(nodelists=nodelists)
        else:
            print('No solution found!')

    
    def create_data_model(self):
        """Stores the data for the problem."""
        data = {}
        data['distance_matrix'] = self.cost  # yapf: disable
        data['num_vehicles'] = 1 # self.m
        data['depot'] = 0
        return data


    def print_solution(self, nodelists):
        """Prints solution on console."""
        print(f'Objective: {self.sol.ObjectiveValue()}')
        total_route_distance = 0
        for i in range(self.m):
            route_distance = 0
            plan_output = f'Route for vehicle {i}:\n'
            for j in range(len(nodelists[i][:-1])):
                plan_output += f' {nodelists[i][j]} ->'
                route_distance += self.routing.GetArcCostForVehicle(nodelists[i][j], nodelists[i][j+1], 0)
            plan_output += f' {nodelists[i][-1]}\n'
            plan_output += f'Route distance: {route_distance}\n'
            print(plan_output)
            total_route_distance += route_distance
        
        print(f"Total Distance: {total_route_distance}")


    # Create and register a transit callback.
    def distance_callback(self, from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = self.manager.IndexToNode(from_index)
        to_node = self.manager.IndexToNode(to_index)
        return self.data['distance_matrix'][from_node][to_node]
    

    def main(self):
        transit_callback_index = self.routing.RegisterTransitCallback(self.distance_callback)

        # Define cost of each arc.
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Distance constraint.
        dimension_name = 'Distance'
        self.routing.AddDimension(
            transit_callback_index,
            0,     # no slack
            3000,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name)
        distance_dimension = self.routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        # Solve the problem.
        self.sol = self.routing.SolveWithParameters(search_parameters)

    def partition(self):
        depot = index = self.routing.Start(0)
        indexList = []
        while not self.routing.IsEnd(index):
            indexList.append(index)
            index = self.sol.Value(self.routing.NextVar(index))
        
        indexList.append(depot)

        partitionCost = []
        # Get the cost partitioning
        for i, idx in enumerate(indexList):
            prev_idx = idx
            if i == self.n:
                break
            idx = indexList[i+1]
            if prev_idx != depot and idx != depot:
                cost = self.routing.GetArcCostForVehicle(prev_idx, depot, 0) + self.routing.GetArcCostForVehicle(depot, idx, 0) - self.routing.GetArcCostForVehicle(prev_idx, idx, 0)
                partitionCost.append((prev_idx, idx, cost))
        partitionCost.sort(key=lambda x: x[2])
        
        nodelists = []
        j, k = 0, 0
        for i in range(len(partitionCost)):
            if j == self.m-1:
                nodelists.append(indexList[k:])
                break
            if partitionCost[i][0] != depot and partitionCost[i][1] != depot:
                j += 1
                idx = indexList.index(partitionCost[i][0])
                indexList[idx:idx+1] = partitionCost[i][0], depot
                nodelists.append(indexList[k:idx+2])
                k = idx+1
        
        return nodelists

    def visualize(self, nodelists=None):
        '''Visualize solution'''
        
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

        # Loop over cars and plot other nodes and edges
        for vehicle_id in range(self.m):
            nodelist = nodelists[vehicle_id]
            edgelist = []
            for i in range(len(nodelist) - 1):
                edgelist.append((nodelist[i], nodelist[i+1]))
            
            # Plot nodes
            nx.draw_networkx_nodes(G, pos=pos, nodelist=nodelist[1:][:-1], ax=ax, alpha=0.75, node_size=300, node_color=f'C{vehicle_id + 1}')

            # Plot edges
            G.add_edges_from(edgelist)
            nx.draw_networkx_edges(G, pos=pos, edgelist=edgelist, width=1.5, edge_color=f'C{vehicle_id + 1}')
        
        nx.draw_networkx_labels(G, pos=pos, labels=labels, font_size=12)

        # Show plot
        plt.grid(True)
        plt.show()