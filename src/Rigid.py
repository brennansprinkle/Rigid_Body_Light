from Rigid import c_rigid as crigid
import numpy as np


class RigidBody:
    X_shape = None
    Q_shape = None

    def __init__(self):
        self.cb = crigid.CManyBodies()

    def get_config(self):
        X, Q = self.cb.getConfig()

        X = X.reshape(self.X_shape)
        Q = Q.reshape(self.Q_shape)
        return X, Q

    def set_config(self, X, Q):
        self.__check_and_set_shapes(X, Q)
        X = X.flatten()
        Q = Q.flatten()
        self.cb.setConfig(X, Q)

    def __check_and_set_shapes(self, X, Q):
        x_size = np.prod(np.shape(X))
        q_size = np.prod(np.shape(Q))

        if x_size % 3 != 0:
            raise RuntimeError("X must have total length 3N")
        if q_size % 4 != 0:
            raise RuntimeError("Q must have total length 4N")

        nx = x_size // 3
        nq = q_size // 4

        if nx != nq:
            raise RuntimeError("X and Q must have the same number of bodies")

        self.N_bodies = nx
        self.X_shape = X.shape
        self.Q_shape = Q.shape
