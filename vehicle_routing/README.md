# Vehicle Routing Problem - Core Solvers
This directory contains the core solvers for solving the Vehicle Routing Problem. API usage instructions may be found in the code docstrings. Example usage is shown in the ***Vehicle Routing.ipynb*** notebook.

## Core Solvers
Following core solvers have been implemented.

 - ***FQS*** - Full Qubo Solver
 - ***APS*** - Average Partitioning Solver
 - ***RAS*** - Route Activation Solver
 - ***QNS*** - Qiskit Native Solver
 - ***CTS*** - Clustered Tsp Solver
 - ***SPS*** - Solution Partitioning Solver

Descriptions of individual solvers may be found in the ***Vehicle Routing.ipynb*** notebook.

## Solver Backends
Following backend solvers may be used.

 - D-Wave Sampler
 - D-Wave Hybrid Sampler
 - Leap Hybrid Sampler
 - Qiskit QAOA Backend

## Python Files
Each solver is implemented as a seperate python class in a seperate python file. The solver classes inherit from a base **VehicleRouter** class defined in **vehicle_routing.py**. There are three other files:
 - **node_clustering.py:** This file implements a **NodeClustering** class for clustering in CTS via Leap's Hybrid DQM Sampler.
 - **solver_backend.py:** This file contains the backend solvers listed above.
 - **utility.py:** This file implements general utility functions such as generatinga random VRP instance.
