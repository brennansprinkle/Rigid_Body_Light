import numpy as np
import pytest
from Rigid import RigidBody
from scipy.spatial.transform import Rotation
import utils

struct_shell_12 = "../structures/shell_N_12_Rg_0_7921_Rh_1.vertex"


def test_create():
    a = 1.0
    eta = 1.0
    _, config = utils.load_config(struct_shell_12)

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


def test_blob_positions():
    N = 5
    X, Q = utils.create_random_positions(N)
    _, config = utils.load_config(struct_shell_12)
    blobs_per_body = config.shape[0]
    cb = utils.create_solver(rigid_config=config, X=X, Q=Q)

    N_blobs = N * blobs_per_body
    pos = cb.get_blob_positions()
    assert pos.shape == (N_blobs, 3)

    ref_pos = np.zeros((N_blobs, 3))
    for i in range(N):
        x_i = X[i, :]
        r_i = Rotation.from_quat(Q[i, :], scalar_first=True)
        pos_i = r_i.apply(config.copy()) + x_i
        ref_pos[i * blobs_per_body : (i + 1) * blobs_per_body, :] = pos_i

    assert np.allclose(pos, ref_pos)
