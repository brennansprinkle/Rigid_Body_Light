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
    cb = utils.create_solver(X, Q)
    cb.set_config(X, Q)

    N_per = cb.blobs_per_body
    N_blobs = N_rigid * N_per

    U = np.random.randn(6 * N_rigid).astype(precision)
    lambda_vec = np.random.randn(3 * N_blobs).astype(precision)
    ku = cb.K_dot(U)
    ktl = cb.KT_dot(lambda_vec)

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
    cb = utils.create_solver(X, Q, block_PC=block_PC, wall_PC=wall_PC)

    u_slip = np.random.randn(3 * cb.blobs_per_body * N_rigid).astype(precision)
    F = np.random.randn(6 * N_rigid).astype(precision)
    lambda_vec, U = cb.apply_PC(u_slip, F)

    assert np.linalg.norm(lambda_vec) > 0.0
    assert np.linalg.norm(U) > 0.0
