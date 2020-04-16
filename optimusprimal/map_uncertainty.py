import numpy as np

def find_credible_upper_limit(x_sol, region, objective_function, gamma, iters, tol):
    eta1 = 0
    eta2 = 1e20
    x = x_sol
    for i in range(iters):
        x[region] = 0
        x[region] += eta1
        obj1 = objective_function(x)
        x[region] = 0
        x[region] += eta2
        obj2 = objective_function(x)
        eta3 = (eta2 - eta1) * 0.5
        x[region] = 0
        x[region] += eta3
        obj3 = objective_function(x)
        if(np.abs(obj3 - gamma) < tol):
            return eta3
