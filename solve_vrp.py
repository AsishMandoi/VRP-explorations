import numpy as np
from VRP.classical.ras import RAS as ExactSolver
from VRP.quantum.CQM_based.fqs_v2 import FQS

n=7     # number of clients
m=3     # number of vehicles

def vrp_solver(n, m, **params):
  np.random.seed(0)
  xc = (np.random.rand(n + 1) - 0.5) * 20
  yc = (np.random.rand(n + 1) - 0.5) * 20
  xc[0], yc[0] = 0, 0
  cost = np.zeros((n + 1, n + 1))
  for i in range(n + 1):
      for j in range(i + 1, n + 1):
          cost[i, j] = np.sqrt((xc[i] - xc[j]) ** 2 + (yc[i] - yc[j]) ** 2) * 100
          cost[j, i] = cost[i, j]

  cost = cost.astype(int)

  c_solver = ExactSolver(n, m, cost, xc, yc)
  c_sol = c_solver.formulate_and_solve()

  solver = FQS(n, m, cost, xc, yc)
  sol = solver.solve()

  print("\napproximation ratio:", c_sol['min_cost'] / sol['min_cost'])

vrp_solver(n, m)
