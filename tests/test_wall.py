from Rigid import RigidBody
import numpy as np
import utils
import pytest


def test_above_wall():
    N = 1
    X = np.array([[0.0, 0.0, 1.0]])
    Q = np.array([[1.0, 0.0, 0.0, 0.0]])
    _, config = utils.load_config(utils.struct_shell_12)
    cb = utils.create_solver(rigid_config=config, X=X, Q=Q, wall_PC=True)

    size = 3 * cb.blobs_per_body * N + 6 * N
    vec = np.random.randn(size)
    PC = cb.apply_PC(vec)
    saddle = cb.apply_saddle(vec)
    M_applied = cb.apply_M(vec[: 3 * cb.blobs_per_body * N])
    assert np.linalg.norm(PC) > 0.0
    assert np.linalg.norm(saddle) > 0.0
    assert np.linalg.norm(M_applied) > 0.0


def test_under_wall():
    N = 1
    X = np.array([[0.0, 0.0, 0.0]])
    Q = np.array([[1.0, 0.0, 0.0, 0.0]])
    _, config = utils.load_config(utils.struct_shell_12)
    cb = utils.create_solver(rigid_config=config, X=X, Q=Q, wall_PC=True)

    size = 3 * cb.blobs_per_body * N + 6 * N
    vec = np.random.randn(size)
    with pytest.raises(RuntimeError):
        cb.apply_saddle(vec)
    with pytest.raises(RuntimeError):
        cb.apply_PC(vec)
    with pytest.raises(RuntimeError):
        cb.apply_M(vec[: 3 * cb.blobs_per_body * N])
