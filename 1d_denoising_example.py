import prox_operators
import linear_operators
import primal_dual
import numpy as np

options = {'tol': 1e-4, 'iter': 5000, 'update_iter': 50, 'record_iters': False}
sigma = 0.01
size = 128
epsilon = np.sqrt(size + 2. * np.sqrt(size))
x = np.sin(np.linspace(0, 1 * np.pi, size))

W = np.ones((size,))

y = W * x + np.random.normal(0, sigma, size)

p = prox_operators.l2_ball(epsilon, y, linear_operators.diag_matrix_operator(W))
p.beta = 2.

wav = ["db1", "db4", "db8"]
levels = 3
shape = (size,)
psi = linear_operators.dictionary(wav, levels, shape)

h = prox_operators.l1_norm(np.max(np.abs(psi.dir_op(y))) * 1e-3, psi)
h.beta = len(wav) * 2.

x, diag = primal_dual.FBPD(y, options, None, h, p, None)
