import numpy as np

from VRP.classical.ras import RAS as ExactSolver
from VRP.classical.sps import SPS as SPS_c
from VRP.quantum.CQM_based.fqs import FQS as FQS_cqm
from VRP.quantum.CQM_based.ras import RAS as RAS_cqm
from VRP.quantum.CQM_based.gps import GPS as GPS_cqm
from VRP.quantum.BQM_based.full_qubo_solver import FullQuboSolver as FQS_bqm
from VRP.quantum.BQM_based.route_activation_solver import RouteActivationSolver as RAS_bqm
from VRP.quantum.BQM_based.dbscan_solver import DBSCANSolver as DBSS_bqm
from VRP.quantum.BQM_based.solution_partition_solver import SolutionPartitionSolver as SPS_bqm

class VRPSolver:
    '''Class to solve VRP'''

    def __init__(self, n, m, cost, xc, yc, **params):
        
        params.setdefault('model', 'CQM')
        params.setdefault('solver', 'ras')
        params.setdefault('time_limit', 5)
        
        self.n = n
        self.m = m
        self.cost = cost
        self.xc = xc
        self.yc = yc
        self.params = params

        self.solver = None
        self.sol = None

    def solve_vrp(self):

        time_limit = self.params['time_limit']
        model = self.params['model']

        solver = {
            'CQM': {
                'fqs': FQS_cqm,
                'ras': RAS_cqm,
                'gps': GPS_cqm,
                # 'dbss': DBSS_cqm,
                # 'sps': SPS_cqm
            },
            'BQM': {
                'fqs': FQS_bqm,
                'ras': RAS_bqm,
                'dbss': DBSS_bqm,
                'sps': SPS_bqm
            }
        }[model][self.params['solver']]

        self.solver = solver(self.n, self.m, self.cost, xc=self.xc, yc=self.yc)
        if model == 'BQM':
            self.solver.solve(solver='neal')
        else:
            self.solver.solve(time_limit=time_limit)

    def plot_solution(self):
        if self.solver != None:
            if self.params['model'] == 'BQM':
                self.solver.visualize(xc=self.xc, yc=self.yc)
            else:
                self.solver.visualize()
        else:
            print('No solution to plot!')


def compare_solvers(n, m, cost, xc, yc, **params):
    params.setdefault('n_iter', 1)
    params.setdefault('time_limit', 5)

    n_iter = params['n_iter']
    time_limit = params['time_limit']
    

    sol = ExactSolver(n, m, cost, xc, yc).formulate_and_solve()
    exact_min_cost = sol['min_cost']
    
    solversList = [RAS_cqm, FQS_cqm, GPS_cqm]   # RAS_bqm, FQS_bqm, DBSS_bqm, SPS_bqm
    
    sum_min_costs = [0] * len(solversList)
    sum_runtimes = [0] * len(solversList)
    num_vars = [None] * len(solversList)
    for i in range(n_iter):
        for j, solver in enumerate(solversList):
            sol = solver(n, m, cost, xc, yc).solve(time_limit=time_limit)
            sum_min_costs[j] += sol['min_cost'] if sol['min_cost'] is not None else 0
            sum_runtimes[j] += sol['runtime']
            num_vars[j] = sol['num_vars']
    
    avg_min_costs = [sum_min_costs[i] / n_iter if sum_min_costs[i] != 0 else None for i in range(len(solversList))]
    avg_runtimes = [sum_runtimes[i] / n_iter for i in range(len(solversList))]
    approximation_ratios = [avg_min_costs[i] / exact_min_cost if avg_min_costs[i] is not None else None for i in range(len(solversList))]

    comparison_table = {solversList[i].__name__: {'avg_min_cost': avg_min_costs[i], 'avg_runtime': avg_runtimes[i], 'num_vars': num_vars[i], 'approximation_ratio': approximation_ratios[i]} for i in range(len(solversList))}

    comparison_table = [{'exact_min_cost': exact_min_cost}, comparison_table]
    return comparison_table


def random_routing_instance(n, seed=None):
    """
    Generate a random instance (n+1 random coordinates and a cost matrix which is same as the distance between them).
    Args:
        n: No. of nodes exclusing depot.
        seed: Seed value for random number generator. Defaults to None, which sets a random seed.
    Returns:
        A list of (n + 1) x coordinates, a list of (n + 1) y coordinates and an (n + 1) x (n + 1) numpy array as the
        cost matrix.
    """
    
    # Set seed
    if seed is not None:
        np.random.seed(seed)

    # Generate distance_matrix
    xc = (np.random.rand(n + 1) - 0.5) * 20
    yc = (np.random.rand(n + 1) - 0.5) * 20
    xc[0], yc[0] = 0, 0
    dist_mat = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            dist_mat[i, j] = np.sqrt((xc[i] - xc[j]) ** 2 + (yc[i] - yc[j]) ** 2) * 100
            dist_mat[j, i] = dist_mat[i, j]

    # Return output
    return xc, yc, dist_mat.astype(int)


def random_routing_instance_with_time_windows(n, seed=None):
    """
    Generate a random instance (n+1 random coordinates and a cost matrix which is same as the distance between them).
    Args:
        n: No. of nodes exclusing depot.
        seed: Seed value for random number generator. Defaults to None, which sets a random seed.
    Returns:
        x: a list of (n + 1) x coordinates,
        y: a list of (n + 1) y coordinates,
        cost: an (n + 1) x (n + 1) numpy array as the cost matrix,
        tw: a list of pair of numbers as the time windows for each coordinate
    """
    
    # Set seed
    if seed is not None:
        np.random.seed(seed)

    # Generate distance_matrix
    xc = (np.random.rand(n + 1) - 0.5) * 20
    yc = (np.random.rand(n + 1) - 0.5) * 20
    xc[0], yc[0] = 0, 0
    dist_mat = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            dist_mat[i, j] = np.sqrt((xc[i] - xc[j]) ** 2 + (yc[i] - yc[j]) ** 2) * 100
            dist_mat[j, i] = dist_mat[i, j]

    MAX_EARLIEST_ARRIVAL_TIME = 5   # The maximum possible "earliest arrival time"
    MAX_TIME_WINDOW_RANGE = 30      # The largest possible time window size

    time_windows = [[0, 30]]
    for node in range(n):
        ti = 0  # round(np.random.rand() * MAX_EARLIEST_ARRIVAL_TIME)
        tf = 30 # round(np.random.rand() * MAX_TIME_WINDOW_RANGE + ti)
        time_windows.append([ti, tf])

    return xc, yc, dist_mat.astype(int), time_windows


def custom_routing_instance(n):
    """
    Generate a custom TSP instance (n+1 coordinates and a cost matrix which is same as the distance between them).
    Args:
        n: No. of nodes exclusing depot.
    Returns:
        A list of (n + 1) x coordinates, a list of (n + 1) y coordinates and an (n + 1) x (n + 1) numpy array as the
        cost matrix.
    """
    
    custom_coords = [
        ([0,   9.68397635, -21.91415037,    8.33057986,   2.96044684, -20.74446885], [0, -5.37904717, -0.14612363, -8.39254627, -7.88307738, -6.71815846]),
        ([0, -14.62374849,   0.64433429,  -22.06068663, -24.17317981,  12.94189313], [0,  7.79525625, -2.17971286, -9.13602211, 19.52634242, 17.35703871]),
        ([0, -20.547731,     0.21868276,   22.94555096, -21.60056947, -12.77923658], [0, -21.68070314,   3.85064948,  10.24842767,   5.33432128, -23.79246391]),
        ([0, -23.95905211,  -1.81608017,    5.17390474,  14.4451113,    7.92617635], [0,  11.01641074, -11.25120004, -23.69568939,   2.42716362,  11.41399686]),
        ([0,  15.56202638,  -8.1563987,   -14.75819874, -16.40754363,   0.99306327], [0,  2.10506766, 14.74870063,  4.10311324, 20.76978131, -9.60709729]),
        ([0,  17.53090505, -10.37294117,  -13.84557367, -15.11426717,  -6.10549414], [0, -18.07329384, -24.45858156, -20.871561,    22.97596238,  11.09926968]),
        ([0,  17.39440017, -24.6850199,   -19.22633574, -21.8862367,   10.20874553], [0,  -2.42883951, -23.66801217,   9.19644146,  15.24107506, -24.45443753]),
        ([0, -24.23999789,  24.44285518,    8.72046533,  -2.01580788,  -6.19607689], [0, -11.48069452, -17.60094231,   2.67137847, -13.23332478,  -8.5483955 ]),
        ([0,  -8.72775711,  19.45745275,   -4.44703119, -16.1182592,  -12.93482615], [0,  19.96848055,   2.78356029,  -3.79202314, -24.46107761,  23.34860146]),
        ([0,  12.79365238, -12.45608602,   10.29188926,  -6.11647882,  11.14503212], [0,   5.35068747,   8.78709806, -18.40157431,  23.6464964,  -13.89138151])
    ]
    
    xc, yc = custom_coords[0][0], custom_coords[0][1]
    dist_mat = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            dist_mat[i, j] = np.sqrt((xc[i] - xc[j]) ** 2 + (yc[i] - yc[j]) ** 2) * 100
            dist_mat[j, i] = dist_mat[i, j]
    
    return dist_mat.astype(int), xc, yc
