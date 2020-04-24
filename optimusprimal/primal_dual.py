import optimusprimal.Empty as Empty
import logging
import numpy as np
import time
import sys

logger = logging.getLogger('Optimus Primal')

def FBPD(x_init, options=None, f=None, h=None, p=None, g=None):
    """Takes in an input signal with proximal operators and a gradient operator
    and returns a solution with diagnostics."""
    # default inputs
    if f is None:
        f = Empty.EmptyProx()
    if g is None:
        g = Empty.EmptyGrad()
    if h is None:
        h = Empty.EmptyProx()
    if p is None:
        p = Empty.EmptyProx()
    if options is None:
        options = {'tol': 1e-4, 'iter': 500,
                   'update_iter': 100, 'record_iters': False}

    # checking minimum requrements for inputs
    assert hasattr(f, 'prox')
    assert hasattr(h, 'prox')
    assert hasattr(h, 'dir_op')
    assert hasattr(h, 'adj_op')
    assert hasattr(p, 'prox')
    assert hasattr(p, 'dir_op')
    assert hasattr(p, 'adj_op')
    assert hasattr(g, 'grad')

    # algorithmic parameters
    tol = options['tol']
    max_iter = options['iter']
    update_iter = options['update_iter']
    record_iters = options['record_iters']
    # step-sizes
    tau = 0.5 / 3.
    sigmah = 1.
    sigmap = 1.
    # initialization
    x = x_init
    y = h.dir_op(x)
    z = p.dir_op(x)

    logger.info('Running Forward Backward Primal Dual')
    timing = np.zeros(max_iter)
    criter = np.zeros(max_iter)

    # algorithm loop
    for it in range(0, max_iter):

        t = time.time()
        # primal forward-backward step
        x_old = x
        x = x - tau * (g.grad(x) / g.beta / 2. + h.adj_op(y) /
                       h.beta + p.adj_op(z) / p.beta)
        x = f.prox(x, tau)
        # dual forward-backward step
        y = y + sigmah * h.dir_op(2 * x - x_old)
        y = y - sigmah * h.prox(y / sigmah, 1. / sigmah)

        z = z + sigmap * p.dir_op(2 * x - x_old)
        z = z - sigmap * p.prox(z / sigmap, 1. / sigmap)
        # time and criterion
        if(record_iters):
            timing[it] = time.time() - t
            criter[it] = f.fun(x) + g.fun(x) + \
                h.fun(h.dir_op(x)) + p.fun(p.dir_op(x))

        # stopping rule
        if np.linalg.norm(x - x_old) < tol * np.linalg.norm(x_old) and it > 10:
            logger.info('[Primal Dual] converged in %d iterations', it)
            break
        if(update_iter >= 0):
            if(it % update_iter == 0):
                logger.info('[Primal Dual] %d out of %d iterations, tol = %f', it, max_iter, np.linalg.norm(x - x_old) / np.linalg.norm(x_old))
        logger.debug('[Primal Dual] %d out of %d iterations, tol = %f', it, max_iter, np.linalg.norm(x - x_old) / np.linalg.norm(x_old))

    criter = criter[0:it + 1]
    timing = np.cumsum(timing[0:it + 1])
    solution = x
    diagnostics = {'max_iter': it, 'times': timing, 'Obj_vals': criter}
    return solution, diagnostics
