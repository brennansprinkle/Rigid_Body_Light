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


@pytest.mark.parametrize(
    ("X_shape", "Q_shape"), (((10, 3), (10, 4)), ((15, 1), (20, 1)))
)
@pytest.mark.parametrize("vector_type", (list, np.array))
def test_config(X_shape, Q_shape, vector_type):
    X_0 = np.random.rand(*X_shape)
    Q_0 = np.random.rand(*Q_shape)
    X_0 = vector_type(X_0)
    Q_0 = vector_type(Q_0)

    cb = utils.create_solver(X=X_0, Q=Q_0)
    cb.set_config(X_0, Q_0)

    Q_0 = Rotation.from_quat(np.reshape(Q_0, (-1, 4))).as_quat().reshape(Q_shape)

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


@pytest.mark.parametrize("vector_type", (list, np.array))
@pytest.mark.parametrize("flat_inputs", [False, True])
def test_K_dot(vector_type, flat_inputs):
    N_rigid = 3
    X, Q = utils.create_random_positions(N_rigid)
    _, config = utils.load_config(utils.struct_shell_12)
    cb = utils.create_solver(rigid_config=config, X=X, Q=Q)
    blobs_per_body = config.shape[0]

    U_bad_size = vector_type(np.random.randn(6 * N_rigid - 3))
    with pytest.raises(RuntimeError):
        cb.K_dot(U_bad_size)

    U_vec = np.random.randn(6 * N_rigid)
    if flat_inputs:
        out_shape = (3 * blobs_per_body * N_rigid,)
    else:
        U_vec = np.reshape(U_vec, (-1, 6))
        out_shape = (blobs_per_body * N_rigid, 3)

    result = cb.K_dot(vector_type(U_vec))
    assert result.shape == out_shape
    assert np.linalg.norm(result) > 0.0


@pytest.mark.parametrize("vector_type", (list, np.array))
@pytest.mark.parametrize("flat_inputs", [False, True])
def test_KT_dot(vector_type, flat_inputs):
    N_rigid = 3
    X, Q = utils.create_random_positions(N_rigid)
    _, config = utils.load_config(utils.struct_shell_12)
    cb = utils.create_solver(rigid_config=config, X=X, Q=Q)
    blobs_per_body = config.shape[0]

    lambda_bad_size = vector_type(np.random.randn(3 * blobs_per_body * N_rigid - 5))
    with pytest.raises(RuntimeError):
        cb.KT_dot(lambda_bad_size)

    lambda_vec = np.random.randn(3 * blobs_per_body * N_rigid)
    if flat_inputs:
        out_shape = (6 * N_rigid,)
    else:
        lambda_vec = np.reshape(lambda_vec, (-1, 3))
        out_shape = (2 * N_rigid, 3)
    result = cb.KT_dot(vector_type(lambda_vec))
    print("result shape:", (result.shape), "lambda_vec shape:", np.shape(lambda_vec))
    assert result.shape == out_shape
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
@pytest.mark.parametrize("vector_type", (list, np.array))
@pytest.mark.parametrize("flat_inputs", [False, True])
def test_apply_PC(block_PC, wall_PC, vector_type, flat_inputs):
    N_rigid = 3
    X, Q = utils.create_random_positions(N_rigid)
    _, config = utils.load_config(utils.struct_shell_12)
    cb = utils.create_solver(
        rigid_config=config, X=X, Q=Q, block_PC=block_PC, wall_PC=wall_PC
    )
    blobs_per_body = config.shape[0]

    lambda_vec = np.random.randn(3 * blobs_per_body * N_rigid)
    U = np.random.randn(6 * N_rigid)
    if not flat_inputs:
        lambda_vec = np.reshape(lambda_vec, (-1, 3))
        U = np.reshape(U, (-1, 6))
    PC = cb.apply_PC(vector_type(lambda_vec), vector_type(U))

    assert np.linalg.norm(PC) > 0.0

    with pytest.raises(RuntimeError):
        lambda_bad_size = np.random.randn(3 * blobs_per_body * N_rigid - 5)
        cb.apply_PC(lambda_bad_size, U)
    with pytest.raises(RuntimeError):
        U_bad_size = np.random.randn(6 * N_rigid - 3)
        cb.apply_PC(lambda_vec, U_bad_size)


@pytest.mark.parametrize("vector_type", (list, np.array))
@pytest.mark.parametrize("flat_inputs", [False, True])
def test_apply_M(vector_type, flat_inputs):
    N_rigid = 2
    X, Q = utils.create_random_positions(N_rigid)
    _, config = utils.load_config(utils.struct_shell_12)
    cb = utils.create_solver(rigid_config=config, X=X, Q=Q)
    blobs_per_body = config.shape[0]

    F_bad_size = np.random.randn(3 * blobs_per_body * N_rigid - 4)
    with pytest.raises(RuntimeError):
        cb.apply_M(F_bad_size)

    F = vector_type(np.random.randn(3 * blobs_per_body * N_rigid))
    if not flat_inputs:
        F = np.reshape(F, (-1, 3))
    print(type(F))
    result = cb.apply_M(vector_type(F))
    shape = (3 * blobs_per_body * N_rigid,)
    assert result.shape == shape
    assert np.linalg.norm(result) > 0.0


@pytest.mark.parametrize("vector_type", (list, np.array))
@pytest.mark.parametrize("flat_inputs", [False, True])
def test_evolve_rigid_bodies(vector_type, flat_inputs):
    N_rigid = 3
    X, Q = utils.create_random_positions(N_rigid)
    _, config = utils.load_config(utils.struct_shell_12)
    cb = utils.create_solver(rigid_config=config, X=X, Q=Q)

    U = np.random.randn(6 * N_rigid)
    if not flat_inputs:
        U = np.reshape(U, (-1, 6))
    cb.evolve_rigid_bodies(vector_type(U))

    X_new, Q_new = cb.get_config()

    assert np.linalg.norm(X_new - X) > 0.0
    assert np.linalg.norm(Q_new - Q) > 0.0
