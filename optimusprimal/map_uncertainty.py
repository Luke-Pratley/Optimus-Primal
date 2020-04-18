import numpy as np


def bisection_method_credible_interval(x_sol, region, objective_function, bound, iters, tol):
    """
    Bisection method for finding credible interval
    """

    eta1 = 0
    eta2 = 10
    x = x_sol
    obj3 = objective_function(x_sol) - bound
    for i in range(int(iters)):
        print(eta1, eta2, obj3)
        x[region] = 0
        x[region] += eta1
        obj1 = objective_function(x) - bound
        eta3 = (eta2 + eta1) * 0.5
        x[region] = 0
        x[region] += eta3
        obj3 = objective_function(x) - bound
        if(np.abs(obj3) < tol):
            return eta3
        if(np.sign(obj1) == np.sign(obj3)):
            eta1 = eta3
        else:
            eta2 = eta3
