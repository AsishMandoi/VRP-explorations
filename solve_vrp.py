from VRP.classical.ras import RAS as ExactSolver
from VRP.quantum.CQM_based.fqs_v2 import FQS
from utils import random_routing_instance

n=7     # number of clients
m=3     # number of vehicles

def vrp_solver(n, m, **params):
  xc, yc, cost = random_routing_instance(n, 0)

  c_solver = ExactSolver(n, m, cost, xc, yc)
  c_sol = c_solver.formulate_and_solve()

  solver = FQS(n, m, cost, xc, yc)
  sol = solver.solve()

  print("approximation ratio:", c_sol['min_cost'] / sol['min_cost'])

vrp_solver(n, m)
