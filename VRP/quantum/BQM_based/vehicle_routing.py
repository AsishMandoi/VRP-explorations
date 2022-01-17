import numpy as np
import dimod
import time

from functools import partial
from .solver_backend import SolverBackend
from dwave.embedding.chain_strength import uniform_torque_compensation
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import OptimizationResult


class VehicleRouter:

    """Abstract Class for solving the Vehicle Routing Problem. To build a VRP solver, simply inherit from this class
    and overide the build_quadratic_program function in this class."""

    def __init__(self, n_clients, n_vehicles, cost_matrix, **params):

        """Initializes the VRP by storing all inputs, initializing variables for storing the quadratic structures and
        results and calls the rebuild function to build all quadratic structures.
        Args:
            n_clients: No. of nodes in the problem (excluding the depot).
            n_vehicles: No. of vehicles available for delivery.
            cost_matrix: (n_clients + 1) x (n_clients + 1) matrix describing the cost of moving from node i to node j.
            penalty: Penalty value to use for constraints in the QUBO. Defaults to automatic calculation by qiskit
                converters.
            chain_strength: Chain strength to be used for D-Wave sampler. Defaults to automatic chain strength
                calculation via uniform torque compensation.
            num_reads: Number of samples to read. Defaults to 1000.
            solver: Select a backend solver. Defaults to 'dwave'.
        """

        # Store critical inputs
        self.n = n_clients
        self.m = n_vehicles
        self.cost = np.array(cost_matrix)

        # Extract parameters
        self.penalty = params.setdefault('constraint_penalty', None)
        self.chain_strength = params.setdefault('chain_strength', partial(uniform_torque_compensation, prefactor=2))
        self.num_reads = params.setdefault('num_reads', 1000)
        self.solver = params.setdefault('solver', 'dwave')

        # Initialize quadratic structures
        self.qp = None
        self.qubo = None
        self.bqm = None
        self.variables = None

        # Initialize result containers
        self.result = None
        self.solution = None

        # Initialize timer
        self.clock = None
        self.timing = {}

        # Initialize backend
        self.backend = SolverBackend(self)

        # Build quadratic models
        self.rebuild()

    def build_quadratic_program(self):

        """Dummy function to be overriden in child class. Required to set self.variables to contain the names of all
        variables in the form of a numpy array and self.qp to contain the quadratic program to be solved."""

        # Dummy. Override in child class.
        pass

    def build_bqm(self):

        """Converts the quadratic program in self.qp to a QUBO by appending all constraints to the objective function
        in the form of penalties and then builds a BQM from the QUBO for solving by D-Wave."""

        # Convert to QUBO
        converter = QuadraticProgramToQubo(penalty=self.penalty)
        self.qubo = converter.convert(self.qp)

        # Extract qubo data
        Q = self.qubo.objective.quadratic.to_dict(use_name=True)
        g = self.qubo.objective.linear.to_dict(use_name=True)
        c = self.qubo.objective.constant

        # Build BQM
        self.bqm = dimod.BQM(g, Q, c, dimod.BINARY)

    def rebuild(self):

        """Builds the quadratic program by calling build_quadratic_program and then the QUBO and BQM by calling
        build_bqm."""

        # Begin stopwatch
        self.clock = time.time()

        # Rebuild quadratic models
        self.build_quadratic_program()
        self.build_bqm()

        # Record build time
        self.timing['qubo_build_time'] = (time.time() - self.clock) * 1e6

    def extract_solution(self, result_dict):

        """Uses a result dictionary mapping variable names to the solved solution to build the self.solution variable
        in the same shape as self.variables and containing the corresponding solutions.
        Args:
            result_dict: Dictionary mapping variable names to solved values for these variables.
        """

        # Extract solution from result dictionary
        var_list = self.variables.reshape(-1)
        self.solution = np.zeros(var_list.shape)
        for i in range(len(var_list)):
            self.solution[i] = result_dict[var_list[i]]

        # Reshape result
        self.solution = self.solution.reshape(self.variables.shape)

    def evaluate_vrp_cost(self):

        """Evaluate the optimized VRP cost under the optimized solution stored in self.solution.
        Returns:
            Optimized VRP cost as a float value.
        """

        # Return optimized energy
        if type(self.result) == OptimizationResult:
            return self.result.fval
        else:
            return self.result.first.energy

    def evaluate_qubo_feasibility(self, data=None):

        """Evaluates whether the QUBO is feasible under the supplied data as inputs. If this data is not
        supplied, the self.solution variable is used instead.
        Args:
            data: Values of the variables in the solution to be tested. Defaults to self.solution.
        Returns:
            A 3-tuple containing a boolean value indicating whether the QUBO is feasible or not, a list of variables
            that violate constraints, and the list of violated constraints. If feasible, (True, [], []) is returned.
        """

        # Resolve data
        if data is None:
            data = self.solution.reshape(-1)
        else:
            data = np.array(data).reshape(-1)

        # Get constraint violation data
        return self.qp.get_feasibility_info(data)

    def solve(self, **params):

        """Solve the QUBO using the selected solver.
        Args:
            params: Parameters to send to the selected backend solver. You may also specify the solver to select a
                different solver and override the specified self.solver.
        """

        # Resolve solver
        params.setdefault('solver', self.solver)

        # Solve
        self.backend.solve(**params)
