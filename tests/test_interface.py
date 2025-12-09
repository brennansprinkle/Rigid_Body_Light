import numpy as np
import pytest
from Rigid import RigidBody
from scipy.spatial.transform import Rotation
import utils


def test_create():
    a = 1.0
    eta = 1.0
    _, config = utils.load_config(utils.struct_shell_12)

    N = 10
    X = np.random.randn(N, 3)
    Q = np.random.randn(N, 4)

    cb = RigidBody(config, X, Q, a, eta, dt=0.01)
    cb = RigidBody(config, X, Q, a, eta, dt=0.01, wall_PC=True)
    cb = RigidBody(config, X, Q, a, eta, dt=0.01, block_PC=True)

    with pytest.raises(RuntimeError):
        config = config.flatten()[:-1]
        cb = RigidBody(config, X, Q, a, eta, dt=0.01)


def test_config():
    n = 10
    X_0 = np.random.rand(n, 3)
    Q_0 = np.random.rand(n, 4)

    cb = utils.create_solver(X=X_0, Q=Q_0)
    cb.set_config(X_0, Q_0)

    Q_0 = Rotation.from_quat(Q_0).as_quat()

    X, Q = cb.get_config()
    assert np.allclose(X, X_0)
    assert np.allclose(Q, Q_0)


def test_bad_config():
    n = 10
    X_0 = np.random.rand(n, 3)
    Q_0 = np.random.rand(n, 4)

    cb = utils.create_solver(X=X_0, Q=Q_0)

    with pytest.raises(RuntimeError):
        cb.set_config(X_0, Q_0[: n - 1])

    with pytest.raises(RuntimeError):
        cb.set_config(X_0[: n - 1], Q_0)


def test_blob_positions():
    N = 5
    X, Q = utils.create_random_positions(N)
    _, config = utils.load_config(utils.struct_shell_12)
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

    assert np.allclose(pos, ref_pos, atol=1e-5)


def test_K_dot():
    N_rigid = 3
    X, Q = utils.create_random_positions(N_rigid)
    _, config = utils.load_config(utils.struct_shell_12)
    cb = utils.create_solver(rigid_config=config, X=X, Q=Q)
    blobs_per_body = config.shape[0]

    U_bad_size = np.random.randn(6 * N_rigid - 3)
    with pytest.raises(RuntimeError):
        cb.K_dot(U_bad_size)

    U_vec = np.random.randn(6 * N_rigid)
    result = cb.K_dot(U_vec)
    shape = (N_rigid * blobs_per_body, 3)
    assert result.shape == shape
    assert np.linalg.norm(result) > 0.0


def test_KT_dot():
    N_rigid = 3
    X, Q = utils.create_random_positions(N_rigid)
    _, config = utils.load_config(utils.struct_shell_12)
    cb = utils.create_solver(rigid_config=config, X=X, Q=Q)
    blobs_per_body = config.shape[0]

    lambda_bad_size = np.random.randn(3 * blobs_per_body * N_rigid - 5)
    with pytest.raises(RuntimeError):
        cb.KT_dot(lambda_bad_size)

    lambda_vec = np.random.randn(3 * blobs_per_body * N_rigid)
    result = cb.KT_dot(lambda_vec)
    shape = (2 * N_rigid, 3)
    assert result.shape == shape
    assert np.linalg.norm(result) > 0.0


def test_get_K_Kinv():
    N_rigid = 3
    X, Q = utils.create_random_positions(N_rigid)
    _, config = utils.load_config(utils.struct_shell_12)
    cb = utils.create_solver(rigid_config=config, X=X, Q=Q)

    K = cb.get_K()
    K_inv = cb.get_Kinv()

    assert np.sum(np.abs(K)) > 0.0
    assert np.sum(np.abs(K_inv)) > 0.0


@pytest.mark.parametrize(
    ("block_PC", "wall_PC"),
    ((False, False), (True, False), (False, True), (True, True)),
)
def test_apply_PC(block_PC, wall_PC):
    N_rigid = 3
    X, Q = utils.create_random_positions(N_rigid, wall_PC=wall_PC)
    _, config = utils.load_config(utils.struct_shell_12)
    cb = utils.create_solver(
        rigid_config=config, X=X, Q=Q, block_PC=block_PC, wall_PC=wall_PC
    )
    blobs_per_body = config.shape[0]

    size = 3 * blobs_per_body * N_rigid + 6 * N_rigid
    b = np.random.randn(size)
    PC = cb.apply_PC(b)

    assert PC.shape == (size,)
    assert np.linalg.norm(PC) > 0.0

    with pytest.raises(RuntimeError):
        b_bad_size = np.random.randn(size - 4)
        cb.apply_PC(b_bad_size)

def test_apply_M():
    N_rigid = 2
    X, Q = utils.create_random_positions(N_rigid)
    _, config = utils.load_config(utils.struct_shell_12)
    cb = utils.create_solver(rigid_config=config, X=X, Q=Q)
    blobs_per_body = config.shape[0]

    F_bad_size = np.random.randn(3 * blobs_per_body * N_rigid - 4)
    with pytest.raises(RuntimeError):
        cb.apply_M(F_bad_size)

    F = np.random.randn(3 * blobs_per_body * N_rigid)
    result = cb.apply_M(F)
    shape = (3 * blobs_per_body * N_rigid,)
    assert result.shape == shape
    assert np.linalg.norm(result) > 0.0


def test_apply_saddle():
    N_rigid = 2
    X, Q = utils.create_random_positions(N_rigid)
    _, config = utils.load_config(utils.struct_shell_12)
    cb = utils.create_solver(rigid_config=config, X=X, Q=Q)
    blobs_per_body = config.shape[0]

    size = 3 * blobs_per_body * N_rigid + 6 * N_rigid
    x = np.random.randn(size)

    out = cb.apply_saddle(x)
    assert out.shape == (size,)
    assert np.linalg.norm(out) > 0.0

    x_bad_size = np.random.randn(size - 2)
    with pytest.raises(RuntimeError):
        cb.apply_saddle(x_bad_size)


def test_evolve_rigid_bodies():
    N_rigid = 3
    X, Q = utils.create_random_positions(N_rigid)
    _, config = utils.load_config(utils.struct_shell_12)
    cb = utils.create_solver(rigid_config=config, X=X, Q=Q)

    U = np.random.randn(6 * N_rigid)
    cb.evolve_rigid_bodies(U)

    X_new, Q_new = cb.get_config()

    assert np.linalg.norm(X_new - X) > 0.0
    assert np.linalg.norm(Q_new - Q) > 0.0
