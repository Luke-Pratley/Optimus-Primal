import operators
import pytest
import FBPD

import numpy as np

def forward_operator(op, inp, result):
    assert np.all(op.dir_op(inp) == result)
def adjoint_operator(op, inp, result):
    assert np.all(op.adj_op(inp) == result)

def test_id_op():
    id_op = operators.identity()
    inp = np.random.normal(0, 10., (10, 10))
    out = inp
    forward_operator(id_op, inp, out)
    adjoint_operator(id_op, inp, out)
    inp = np.random.normal(0, 10., (10))
    out = inp
    forward_operator(id_op, inp, out)
    adjoint_operator(id_op, inp, out)

def test_matrix_op():
    A = np.random.normal(0, 10., (10, 5)) * 1j
    op = operators.matrix_operator(A)
    inp = np.random.normal(0, 10., (5))
    out = A @ inp
    forward_operator(op, inp, out)
    inp = np.random.normal(0, 10., (10))
    out = np.conj(A.T) @ inp
    adjoint_operator(op, inp, out)
