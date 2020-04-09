import sys
sys.path.insert(0,'..')
import optimusprimal.prox_operators as prox_operators
import optimusprimal.linear_operators as linear_operators
import optimusprimal.primal_dual as primal_dual
import numpy as np
import matplotlib.pyplot as plt

output_dir = "output/"

options = {'tol': 1e-5, 'iter': 5000, 'update_iter': 50, 'record_iters': False}
ISNR = 20.
sigma = 10**(-ISNR/20.)
size = 1024
epsilon = np.sqrt(size + 2. * np.sqrt(size)) * sigma
x = np.linspace(0, 1 * np.pi, size)

W = np.ones((size,))

y = W * x + np.random.normal(0, sigma, size)

p = prox_operators.l2_ball(epsilon, y, linear_operators.diag_matrix_operator(W))
p.beta = 1.

wav = ["db1", "db4"]
levels = 6
shape = (size,)
psi = linear_operators.dictionary(wav, levels, shape)

h = prox_operators.l1_norm(np.max(np.abs(psi.dir_op(y))) * 1e-3, psi)
h.beta = 1.
f = prox_operators.real_prox()
z, diag = primal_dual.FBPD(y, options, f, h, p, None)

plt.plot(np.real(y))
plt.plot(np.real(x))
plt.plot(np.real(z))
plt.legend(["data", "true", "fit"])
SNR = np.log10(np.sqrt(np.sum(np.abs(x)**2))/np.sqrt(np.sum(np.abs(x - z)**2))) * 20.
plt.title("SNR = " + str(SNR))
plt.savefig(output_dir+"1d_constrained_example.png")
