from Rigid import c_rigid_obj as crigid
import numpy as np


class RigidBody:
    X_shape = None
    Q_shape = None

    def __init__(self, rigid_config, X, Q, a, eta, dt, wall_PC=False, block_PC=False):
        self.cb = crigid.CManyBodies()
        kbt = 1.0  # TODO temp, do we need kbt in c_rigid at all?

        if rigid_config.size % 3 != 0:
            raise RuntimeError(
                "Rigid config must have length 3N. Rigid config shape: "
                + str(rigid_config.shape)
            )
        self.blobs_per_body = rigid_config.size // 3

        self.cb.setParameters(a, dt, kbt, eta, rigid_config)
        self.cb.setBlkPC(block_PC)
        self.cb.setWallPC(wall_PC)

        self.set_config(X, Q)

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
        self.cb.set_K_mats()

        self.total_blobs = self.N_bodies * self.blobs_per_body

    def get_blob_positions(self):
        # TODO can we avoid casting here by changing the C++ code?
        shape = (-1, 3) if len(self.X_shape) == 2 else (-1)
        return np.array(self.cb.multi_body_pos()).reshape(shape)

    def KT_dot(self, lambda_vec):
        if lambda_vec.size != 3 * self.total_blobs:
            raise RuntimeError(
                "lambda must have total size 3*N_blobs = "
                + str(3 * self.total_blobs)
                + ". lambda_vec shape: "
                + str(lambda_vec.shape)
            )
        result = self.cb.KT_x_Lam(lambda_vec)
        shape = (-1, 3) if len(self.X_shape) == 2 else (-1)
        return np.array(result).reshape(shape)

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
