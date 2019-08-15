import numpy as np
import pytest

from ddtruss import Truss, DataDrivenSolver

points = np.array([[0, 0], [1, 0], [0.5, 0.5], [2, 1]])
lines = np.array([[0, 2], [1, 2], [1, 3], [2, 3]], dtype=int)

truss = Truss(points, lines)

E = 1.962e11
A = [2e-4, 2e-4, 1e-4, 1e-4]
U_dict = {0: [0, 0], 1: [0, 0]}
F_dict = {3: [0, -9.81e3]}

u_ref = np.array(
    [0, 0, 0, 0, 2.65165043e-4, 8.83883476e-5, 3.47902545e-3, -5.60034579e-3]
)


def test_truss():
    u, *_ = truss.solve(A=A, E=E, U_dict=U_dict, F_dict=F_dict)
    assert np.allclose(u, u_ref)


@pytest.mark.parametrize(
    "n_data", [5000, 10000]
)
def test_data_driven_solver(n_data):
    ddsolver = DataDrivenSolver(truss)

    eps_max = 1.1e-3
    eps = np.linspace(-eps_max, eps_max, n_data)
    sig = E * eps
    material_data = np.hstack([eps.reshape((-1, 1)), sig.reshape((-1, 1))])
    ddsolver.load_material_data(material_data)

    u, *_ = ddsolver.solve(A=A, U_dict=U_dict, F_dict=F_dict)
    assert np.allclose(u, u_ref, rtol=1e-2)
