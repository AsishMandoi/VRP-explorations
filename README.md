# **Solving the Vehicle Routing Problem**
*Exploring various quantum annealing-based approaches to solve the vehicle routing problem.*

This project was part of the **QOSF Quantum Computing Mentorship Program 2021 Cohort 4**, mentored by [**Dr. Vesselin G. Gueorguiev**](https://www.linkedin.com/in/vgg-consulting/)<br>
Mentees: *The Qubit Players*
  - **Asish Kumar Mandoi**<br>
    *Junior Undergraduate at Indian Institute of Technology Kanpur, Department of Electrical Engineering*<br>
    [`Website`](https://asishmandoi.github.io/) ~ [`LinkedIn`](https://www.linkedin.com/in/asish-mandoi-4178581b4/) ~ [`GitHub`](https://github.com/AsishMandoi)
  - **Arya Bhatta**<br>
    *Junior Undergraduate at Indian Institute of Technology Kanpur, Department of Electrical Engineering*<br>
    [`LinkedIn`](https://www.linkedin.com/in/arya-bhatta-26a877200/)

## Introduction
Vehicle routing is a challenging logistics management problem. More precisely, it is an **NP-hard** combinatorial optimization problem.
- Problem statement: ***What is the optimal set of routes for a fleet of vehicles to traverse in order to deliver to a given set of customers?***
- Generalises **The Travelling Salesman Problem (TSP)**.

## Overview
The primary focus of the project has been on improving the applicability of quantum annealing-based solvers for the Vehicle Routing Problem (VRP) for large numbers of customers and vehicles by using a minimal number of qubits. It includes implementations of three solvers:
  - Route Activation Solver
  - GPS Solver (Guillermo, Parfait, Saúl) ~ [[4](https://github.com/AsishMandoi/VRP-explorations#references)]
  - DBSCAN Solver ~ [[7](https://github.com/AsishMandoi/VRP-explorations#references)]

The first two are non-clustering solvers that give exact solutions for the simplest variant of the VRP for small datasets and the third one is a clustering-based solver that gives approximate solutions for large datasets. The project also demonstrates the usage of the [Binary Quadratic Model](https://docs.ocean.dwavesys.com/en/stable/concepts/bqm.html) and [Constrained Quadratic Model](https://docs.ocean.dwavesys.com/en/stable/concepts/cqm.html) provided by D-Wave to formulate various other solvers including the above. The accuracy and runtimes of these solvers, calculated against the exact solutions obtained by classical implementations using [Google's OR-Tools](https://developers.google.com/optimization), are compared with each other.
<br>

*The project was inspired from [this video](https://youtu.be/GK8IT0C9Upk) and [this paper](https://link.springer.com/chapter/10.1007/978-3-030-50433-5_42) by [**Paweł Gora**](https://www.mimuw.edu.pl/~pawelg/) and others. A lot of initial ideas were burrowed from [a project on the same topic](https://github.com/VGGatGitHub/QOSF-cohort3) by **Shantom Borah** and others.*

## Get started
You may start with the `vrp_results.ipynb` notebook where you can choose to run any solver on any model and get the solutions the Vehicle Routing Problem for any number of clients and vehicles. You can play with the `time_limit` parameter to see how the solver performs on different datasets, but make sure not to set it too high as you may exhaust your limited monthly resources on your D-Wave Leap account.

The `quantum` subfolder in VRP folder further contains two more subfolders `CQM_based` and `BQM_based`. The first one contains the implementations of the solvers using the Constrained Quadratic Model (CQM) and the second one contains the implementations of the solvers using the Binary Quadratic Model (BQM).:
- Constrained Quadratic Model is a new model recently released by D-Wave Systems that is capable of encoding Quadratically Constrained Quadratic Programs (QCQPs)
- Binary Quadratic Model is a model that encodes Ising or QUBO problems.

<br>

<details>
  <summary><code><b>Directory Structure</b></code></summary>
  
  ```bash
  VRP-explorations
  ├── _venv_                # (virtual python environment)
  ├── presentations
  │   ├── notebook_1.ipynb
  │   ├── .
  │   ├── .
  │   └── notebook_9.ipynb
  ├── TSP
  │   ├── classical
  │   │   └── gps.py
  │   └── quantum
  │       ├── fqs.py
  │       └── gps.py
  ├── VRP
  │   ├── classical
  │   │   ├── ras.py
  │   │   └── sps.py
  │   └── quantum
  │       ├── BQM_based
  │       │   ├── full_qubo_solver.py
  │       │   ├── route_activation_solver.py
  │       │   ├── solution_partition_solver.py
  │       │   ├── dbscan_solver.py
  │       │   └── ...
  │       └── CQM_based
  │           ├── fqs.py
  │           ├── gps.py
  │           └── ras.py
  ├── utils.py
  ├── tsp_results.py
  ├── vrp_results.py
  ├── requirements.txt
  ├── setup.sh
  ├── .gitignore
  └── README.md
  ```
</details>
<br>

## Run this repo

### Locally
Requirements:
- A bash terminal
- Python version >= 3.9
- D-Wave Leap account

```bash
# Clone this repo
git clone https://github.com/AsishMandoi/VRP-explorations.git

# Go into the project's main directory and setup the environment to run all codes
cd VRP-explorations && bash setup.sh

### The `setup.sh` script consists of the following steps:
# 1. Create a virtual environment (named '_venv_') if it doesn't exist
# 2. Install the required packages in the virtual environment
# 3. Run `dwave setup` to get access to the D-Wave backend solvers
```
*The above process will also run the `dwave setup` command. This will require your authentication token from your account on D-Wave Leap. Learn more about how to [set up your environment](https://docs.ocean.dwavesys.com/en/latest/overview/install.html#set-up-your-environment) for using `dwave-ocean-sdk`.*

### On D-Wave Leap platform
*Coming soon*

## Next steps
- [ ] Incorporate solutions for other more general variants of the problem (especially for the CVRPTW)
- [ ] Make a circuit based algorithm from scratch (that does not use any application modules from any libraries) to solve VRP
- [ ] Find potential real life applications for VRP (for e.g. Supply Chain, Traffic Flow Optimization)

## References
1. Borowski, Michal, Gora, Pawel, *"New Hybrid Quantum Annealing Algorithms for Solving Vehicle Routing Problem"* (2020)
2. Feld, Sebastian, et al. *"A Hybrid Solution Method for the Capacitated Vehicle Routing Problem Using a Quantum Annealer"* (2019)
3. Fisher, Marshall L., Jaikumar, Ramchandran, *"A generalized assignment heuristic for vehicle routing"* (1981)
4. Gonzalez-Bermejo, Saul, et al. *"GPS: Improvement in the formulation of the TSP for its generalizations type QUBO"*
5. The [QOSF cohort - 3 project](https://github.com/VGGatGitHub/QOSF-cohort3) by Shantom, Aniket and Avneesh based on this topic
6. [Vehicle routing problem - Wikipedia](https://en.wikipedia.org/wiki/Vehicle_routing_problem)
7. [DBSCAN - Wikipedia](https://en.wikipedia.org/wiki/DBSCAN)
8. D-Wave's documentations on [Binary Quadratic Model](https://docs.ocean.dwavesys.com/en/stable/concepts/bqm.html) and [Constrained Quadratic Model](https://docs.ocean.dwavesys.com/en/stable/concepts/cqm.html)
11. [Google's OR-Tools](https://developers.google.com/optimization/routing/vrp)