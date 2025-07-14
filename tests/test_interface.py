import numpy as np
import pytest
from Rigid import RigidBody
from scipy.spatial.transform import Rotation


def test_config():
    n = 10
    X_0 = np.random.rand(n, 3)
    Q_0 = np.random.rand(n, 4)

    cb = RigidBody()
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

    cb = RigidBody()

    with pytest.raises(RuntimeError):
        cb.set_config(X_0, Q_0[: n - 1])

    with pytest.raises(RuntimeError):
        cb.set_config(X_0[: n - 1], Q_0)
