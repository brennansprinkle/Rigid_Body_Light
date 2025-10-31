from Rigid import c_rigid as crigid
import numpy as np


class RigidBody:
    X_shape = None
    Q_shape = None

    def __init__(
        self,
        rigid_config,
        X,
        Q,
        a,
        eta,
        dt,
        wall_PC=False,
        block_PC=False,
    ):
        self.cb = crigid.CManyBodies()
        self.precision = self.cb.precision

        kbt = 1.0  # TODO temp, do we need kbt in c_rigid at all?

        if rigid_config.size % 3 != 0:
            raise RuntimeError(
                f"Rigid config must have length 3N. Rigid config shape: {rigid_config.shape}"
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
        self.__check_and_set_configs(X, Q)
        X = X.flatten()
        Q = Q.flatten()
        self.cb.setConfig(X, Q)
        self.cb.set_K_mats()

        self.total_blobs = self.N_bodies * self.blobs_per_body

    def get_blob_positions(self):
        shape = (-1, 3) if len(self.X_shape) == 2 else (-1)
        return np.array(self.cb.multi_body_pos()).reshape(shape)

    def KT_dot(self, lambda_vec):
        self.__check_input_size(lambda_vec=lambda_vec)
        result = self.cb.KT_x_Lam(lambda_vec)
        shape = (-1, 3) if len(self.X_shape) == 2 else (-1)
        return np.array(result).reshape(shape)

    def K_dot(self, U):
        self.__check_input_size(U_vec=U)
        result = self.cb.K_x_U(U)
        shape = (-1, 3) if len(self.X_shape) == 2 else (-1)
        return np.array(result).reshape(shape)

    def apply_PC(self, lambda_vec, U_vec):
        self.__check_input_size(lambda_vec=lambda_vec, U_vec=U_vec)
        in_vec = np.concatenate((lambda_vec, U_vec))
        return self.cb.apply_PC(in_vec)

    def get_K(self):
        return self.cb.get_K()

    def get_Kinv(self):
        return self.cb.get_Kinv()

    def evolve_rigid_bodies(self, U):
        self.__check_input_size(U_vec=U)
        self.cb.evolve_X_Q(U)

    def __check_and_set_configs(self, X, Q):
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

    def __check_input_size(self, lambda_vec=None, U_vec=None):
        if lambda_vec is not None:
            if lambda_vec.size != 3 * self.total_blobs:
                raise RuntimeError(
                    f"lambda must have total size 3*N_blobs = {3 * self.total_blobs}. lambda_vec shape: {lambda_vec.shape}"
                )
        if U_vec is not None:
            if U_vec.size != 6 * self.N_bodies:
                raise RuntimeError(
                    f"U must have total size 6*N_bodies = {6*self.N_bodies}. U shape: {U_vec.shape}"
                )
