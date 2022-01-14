import time
from dwave_qbsolv.dimod_wrapper import QBSolv
import hybrid
import dwave.inspector

from greedy import SteepestDescentSolver
from dwave.system import LeapHybridSampler, DWaveSampler, EmbeddingComposite
from neal import SimulatedAnnealingSampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit import Aer


class SolverBackend:

    """Class containing all backend solvers that may be used to solve the Vehicle Routing Problem."""

    def __init__(self, vrp):

        """Initializes required variables and stores the supplied instance of the VehicleRouter object."""

        # Store relevant data
        self.vrp = vrp

        # Solver dictionary
        self.solvers = {'dwave': self.solve_dwave,
                        'leap': self.solve_leap,
                        'hybrid': self.solve_hybrid,
                        'neal': self.solve_neal,
                        'qbsolv': self.solve_qbsolv,
                        'qaoa': self.solve_qaoa,
                        'npme': self.solve_npme}

        # Initialize necessary variables
        self.dwave_result = None
        self.result_dict = None

        self.solver_limit = 4

    def solve(self, solver, **params):

        """Takes the solver as input and redirects control to the corresponding solver.
        Args:
            solver: The selected solver.
            params: Parameters to send to the selected backend solver..
        """

        # Select solver and solve
        solver = self.solvers[solver]
        solver(**params)

    def solve_dwave(self, **params):

        """Solve using DWaveSampler and EmbeddingComposite.
        Args:
            params: inspect: Defaults to False. Set to True to run D-Wave inspector for the sampled solution.
            params: post_process: Defaults to False. Set to True to run classical post processing for improving the
                D-Wave solution.
        """

        # Resolve parameters
        params['solver'] = 'dwave'
        inspect = params.setdefault('inspect', False)
        post_process = params.setdefault('post_process', False)

        # Solve
        sampler = EmbeddingComposite(DWaveSampler())
        result = sampler.sample(self.vrp.bqm, num_reads=self.vrp.num_reads, chain_strength=self.vrp.chain_strength)

        # Post process
        if not post_process:
            self.vrp.result = result
        else:
            post_processor = SteepestDescentSolver()
            self.vrp.result = post_processor.sample(self.vrp.bqm, num_reads=self.vrp.num_reads, initial_states=result)

        # Extract solution
        self.vrp.timing.update(result.info["timing"])
        self.result_dict = self.vrp.result.first.sample
        self.vrp.extract_solution(self.result_dict)

        # Inspection
        self.dwave_result = result
        if inspect:
            dwave.inspector.show(result)

    def solve_hybrid(self, **params):

        """Solve using dwave-hybrid.
        Args:
            params: Additional parameters that may be required by a solver. Not required here.
        """

        # Resolve parameters
        params['solver'] = 'hybrid'

        # Build sampler workflow
        workflow = hybrid.Loop(
            hybrid.RacingBranches(
                hybrid.InterruptableTabuSampler(),
                hybrid.EnergyImpactDecomposer(size=30, rolling=True, rolling_history=0.75)
                | hybrid.QPUSubproblemAutoEmbeddingSampler()
                | hybrid.SplatComposer()) | hybrid.ArgMin(), convergence=3)

        # Solve
        sampler = hybrid.HybridSampler(workflow)
        self.vrp.result = sampler.sample(self.vrp.bqm, num_reads=self.vrp.num_reads,
                                         chain_strength=self.vrp.chain_strength)

        # Extract solution
        self.result_dict = self.vrp.result.first.sample
        self.vrp.extract_solution(self.result_dict)

    def solve_leap(self, **params):

        """Solve using Leap Hybrid Sampler.
        Args:
            params: Additional parameters that may be required by a solver. Not required here.
        """

        # Resolve parameters
        params['solver'] = 'leap'

        # Solve
        sampler = LeapHybridSampler()
        self.vrp.result = sampler.sample(self.vrp.bqm)

        # Extract solution
        self.vrp.timing.update(self.vrp.result.info)
        self.result_dict = self.vrp.result.first.sample
        self.vrp.extract_solution(self.result_dict)

    def solve_neal(self, **params):

        """Solve using Simulated Annealing Sampler.
        Args:
            params: Additional parameters that may be required by a solver. Not required here.
        """

        # Resolve parameters
        params['solver'] = 'neal'

        # Solve
        sampler = SimulatedAnnealingSampler()
        self.vrp.result = sampler.sample(self.vrp.bqm)

        # Extract solution
        self.vrp.timing.update(self.vrp.result.info)
        self.result_dict = self.vrp.result.first.sample
        self.vrp.extract_solution(self.result_dict)

    def solve_qbsolv(self, **params):

        """Solve using Simulated Annealing Sampler.
        Args:
            params: Additional parameters that may be required by a solver. Not required here.
        """

        # Resolve parameters
        params['solver'] = 'qbsolv'

        # Solve
        self.vrp.result = QBSolv().sample(self.vrp.bqm, solver_limit=self.solver_limit)

        # Extract solution
        self.vrp.timing.update(self.vrp.result.info)
        self.result_dict = self.vrp.result.first.sample
        self.vrp.extract_solution(self.result_dict)

    def solve_qaoa(self, **params):

        """Solve using qiskit Minimum Eigen Optimizer based on a QAOA backend.
        Args:
            params: Additional parameters that may be required by a solver. Not required here.
        """

        # Resolve parameters
        params['solver'] = 'qaoa'
        self.vrp.clock = time.time()

        # Build optimizer and solve
        solver = QAOA(quantum_instance=Aer.get_backend('qasm_simulator'))
        optimizer = MinimumEigenOptimizer(min_eigen_solver=solver)
        self.vrp.result = optimizer.solve(self.vrp.qp)
        self.vrp.timing['qaoa_solution_time'] = (time.time() - self.vrp.clock) * 1e6

        # Build result dictionary
        self.result_dict = {self.vrp.result.variable_names[i]: self.vrp.result.x[i]
                            for i in range(len(self.vrp.result.variable_names))}

        # Extract solution
        self.vrp.extract_solution(self.result_dict)

    def solve_npme(self, **params):

        """Solve using qiskit Minimum Eigen Optimizer based on NumPyMinimumEigensolver().
        Args:
            params: Additional parameters that may be required by a solver. Not required here.
        """

        # Resolve parameters
        params['solver'] = 'npme'
        self.vrp.clock = time.time()

        # Build optimizer and solve
        solver = NumPyMinimumEigensolver()
        optimizer = MinimumEigenOptimizer(min_eigen_solver=solver)
        self.vrp.result = optimizer.solve(self.vrp.qp)
        self.vrp.timing['npme_solution_time'] = (time.time() - self.vrp.clock) * 1e6

        # Build result dictionary
        self.result_dict = {self.vrp.result.variable_names[i]: self.vrp.result.x[i]
                            for i in range(len(self.vrp.result.variable_names))}

        # Extract solution
        self.vrp.extract_solution(self.result_dict)
