# **Solving the Vehicle Routing Problem**
*The Qubit Players*
- QOSF Quantum Computing Mentorship Program 2021 Cohort - 4
- Mentor: Dr. Vesselin G. Gueorguiev
- Mentees:
    - **Asish Kumar Mandoi**, *Junior Undergraduate at Indian Institute of Technology Kanpur, Department of Electrical Engineering*
      - [LinkedIn](https://www.linkedin.com/in/asish-mandoi-4178581b4/)
      - [GitHub](https://github.com/AsishMandoi)
      - [Resume](https://drive.google.com/file/d/1J34OVkYKVrQjxndY_oPV-kW02WMCqfgg/view?usp=sharing)
    - **Arya Bhatta**, *Junior Undergraduate at Indian Institute of Technology Kanpur, Department of Electrical Engineering*
      - [Resume](https://drive.google.com/file/d/1MSddzwGTJxjNEGhw2eMKC9z8VLGYU6vR/view?usp=sharing)

## Introduction
Vehicle routing is a challenging logistics management problem. More precisely, it is an NP-hard combinatorial optimization problem.
- Problem statement: ***What is the optimal set of routes for a fleet of vehicles to traverse in order to deliver to a given set of customers?***
- Generalises **The Travelling Salesman Problem (TSP)**.

## Project Description
1. We started from the simplest version of the VRP, implemented two different solvers for it:
    - Route Activation Solver (RAS)
    - Density-Based Spatial Clustering Of Applications with Noise (DBSCAN) Solver (DBSS)
2. Kept generalizing the solvers by enabling them to solve other variants like
    - Capacitated Vehicle Routing Problem (CVRP) can be solved by all the solvers
    - Multi-Depot Vehicle Routing Problem (MDVRP) can be solved by DBSCAN Solver
3. Tackled large sets of customers by first clustering them using DBSCAN approach and then solving smaller instances of problems

## Run this repo

### Locally
Requirements:
- Python version >= 3.9.8
- D-Wave Leap account

```bash
# Clone this repo
git clone https://github.com/AsishMandoi/VRP-explorations.git

# Install all required packages
cd VRP-explorations && pip install -r vehicle_routing/requirements.txt

```

- Install Ocean Tools
  - [Install Ocean Software](https://docs.ocean.dwavesys.com/en/latest/overview/install.html#install-ocean-software) (`dwave-ocean-sdk`)
  - [Set Up Your Environment](https://docs.ocean.dwavesys.com/en/latest/overview/install.html#set-up-your-environment)

### On D-Wave Leap platform

## Next steps
  - Use techniques to achieve better accuracy for larger QUBO problems
  - Incorporate solutions for other more general variants of the problem
  - Make a circuit based algorithm from scratch (that does not use any application modules from any libraries) to solve VRP
  - Find potential real life applications for VRP (for e.g. Supply Chain, Traffic Flow Optimization)

## References
1. Borowski, Michal, Gora, Pawel, *New Hybrid Quantum Annealing Algorithms for Solving Vehicle Routing Problem* (2020)
2. Feld, Sebastian, et al. *A Hybrid Solution Method for the Capacitated Vehicle Routing Problem Using a Quantum Annealer* (2019)
3. Fisher, Marshall L., Jaikumar, Ramchandran, *A generalized assignment heuristic for vehicle routing* (1981)
4. The [QOSF cohort - 3 project](https://github.com/VGGatGitHub/QOSF-cohort3) by Shantom, Aniket and Avneesh based on this topic
5. [Misha Lavrov's lecture](https://faculty.math.illinois.edu/~mlavrov/slides/482-spring-2020/slides35.pdf) on "Travelling Salesman Problem"
6. [Vehicle routing problem - Wikipedia](https://en.wikipedia.org/wiki/Vehicle_routing_problem)
7. [DBSCAN - Wikipedia](https://en.wikipedia.org/wiki/DBSCAN)