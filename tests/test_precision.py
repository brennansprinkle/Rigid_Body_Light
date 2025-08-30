from Rigid import RigidBody
import numpy as np
import utils
import pytest


@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_precision(precision):
    N_rigid = 5
    X, Q = utils.create_random_positions(N_rigid)
    X = np.array(X, dtype=precision)
    Q = np.array(Q, dtype=precision)
    cb = utils.create_solver(X, Q, precision=precision)
    cb.set_config(X, Q)

    N_per = cb.blobs_per_body
    N_blobs = N_rigid * N_per

    U = np.random.randn(6 * N_rigid).astype(precision)
    L = np.random.randn(3 * N_blobs).astype(precision)
    ku = cb.K_dot(U)
    ktl = cb.KT_dot(L)

    assert ku.dtype == precision
    assert ktl.dtype == precision
    assert np.linalg.norm(ku) > 0.0
    assert np.linalg.norm(ktl) > 0.0


@pytest.mark.parametrize(
    ("block_PC", "wall_PC"),
    ((False, False), (True, False), (False, True), (True, True)),
)
@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_pc_precision(precision, block_PC, wall_PC):
    N_rigid = 3
    X, Q = utils.create_random_positions(N_rigid)
    X = np.array(X, dtype=precision)
    Q = np.array(Q, dtype=precision)
    cb = utils.create_solver(
        X, Q, block_PC=block_PC, wall_PC=wall_PC, precision=precision
    )

    lambda_vec = np.random.randn(3 * cb.blobs_per_body * N_rigid).astype(precision)
    U = np.random.randn(6 * N_rigid).astype(precision)
    PC = cb.apply_PC(lambda_vec, U)

    assert PC.dtype == precision
    assert np.linalg.norm(PC) > 0.0
