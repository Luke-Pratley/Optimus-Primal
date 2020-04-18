import numpy as np


def bisection_method_credible_interval(x_sol, region, objective_function, start_interval, bound, iters, tol):
    """
    Bisection method for finding credible interval
    """

    eta1 = start_interval[0]
    eta2 = start_interval[1]
    x = np.copy(x_sol)
    obj3 = objective_function(x_sol) - bound
    for i in range(int(iters)):
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
    return np.nan


def create_credible_region(x_sol, region_size, objective_function, bound, iters, tol, top, bottom):
    region = np.zeros(x_sol.shape, dtype=bool)
    region[:,:] = False
    region[:region_size, :region_size] = True
    dsizey, dsizex = int(x_sol.shape[0]/region_size), int(x_sol.shape[1]/region_size)
    error_p = np.zeros((dsizey, dsizex))
    error_m = np.zeros((dsizey, dsizex))
    for i in range(dsizey):
        for j in range(dsizex):
            mask = np.roll(region, shift=(i * region_size, j * region_size))
            x_mean = np.mean(np.ravel(x_sol[(mask)]))
            error_p[i, j] = bisection_method_credible_interval(x_sol, (mask), objective_function, [x_mean, top], bound, iters, tol)
            error_m[i, j] = bisection_method_credible_interval(x_sol, (mask), objective_function, [bottom, x_mean], bound, iters, tol)
    return error_p, error_m
