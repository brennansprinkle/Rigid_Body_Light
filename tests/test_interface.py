import numpy as np
import pytest
from Rigid import RigidBody
from scipy.spatial.transform import Rotation
import utils


def test_create():
    a = 1.0
    eta = 1.0
    _, config = utils.load_config("structures/shell_N_12_Rg_0_7921_Rh_1.vertex")

    N = 10
    X = np.random.randn(N, 3)
    Q = np.random.randn(N, 4)

    cb = RigidBody(config, X, Q, a, eta, dt=0.01)
    cb = RigidBody(config, X, Q, a, eta, dt=0.01, wall_PC=True)
    cb = RigidBody(config, X, Q, a, eta, dt=0.01, block_PC=True)


def test_config():
    n = 10
    X_0 = np.random.rand(n, 3)
    Q_0 = np.random.rand(n, 4)

    cb = utils.create_solver(rigid_config=np.array([[0.0, 0.0, 0.0]]), X=X_0, Q=Q_0)
    cb.set_config(X_0, Q_0)

    Q_0 = Rotation.from_quat(Q_0).as_quat()

    X, Q = cb.get_config()
    print(Q - Q_0)
    assert np.allclose(X, X_0)
    assert np.allclose(Q, Q_0)


def test_bad_config():
    n = 10
    X_0 = np.random.rand(n, 3)
    Q_0 = np.random.rand(n, 4)

    cb = utils.create_solver(rigid_config=np.array([[0.0, 0.0, 0.0]]), X=X_0, Q=Q_0)

    with pytest.raises(RuntimeError):
        cb.set_config(X_0, Q_0[: n - 1])

    with pytest.raises(RuntimeError):
        cb.set_config(X_0[: n - 1], Q_0)
