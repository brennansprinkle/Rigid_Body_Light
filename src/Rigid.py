from Rigid import c_rigid as crigid
import numpy as np
from typing import TypeAlias

vector: TypeAlias = list | np.ndarray
"""Rigid body interface for Python.

Dev notes:
- The C++ code expects inputs as flattened numpy arrays. There's a decent bit of wrangling done in this Python interface to make it work seamlessly with lists & multi-dimensional arrays.
"""


class RigidBody:
    X_shape: tuple[int, ...]
    Q_shape: tuple[int, ...]

    def __init__(
        self,
        rigid_config,
        X: vector,
        Q: vector,
        a: float,
        eta: float,
        dt: float,
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

    def get_config(self) -> tuple[np.ndarray, np.ndarray]:
        X, Q = self.cb.getConfig()

        X = X.reshape(self.X_shape)
        Q = Q.reshape(self.Q_shape)
        return X, Q

    def set_config(self, X: vector, Q: vector) -> None:
        self.__check_and_set_configs(X, Q)
        X = np.array(X).ravel()
        Q = np.array(Q).ravel()
        self.cb.setConfig(X, Q)
        self.cb.set_K_mats()

        self.total_blobs = self.N_bodies * self.blobs_per_body

    def get_blob_positions(self) -> np.ndarray:
        shape = (-1, 3) if len(self.X_shape) == 2 else (-1)
        return np.array(self.cb.multi_body_pos()).reshape(shape)

    def K_dot(self, U: vector) -> np.ndarray:
        self.__check_input_size(U_vec=U)
        result = self.cb.K_x_U(np.array(U).ravel())
        shape = (-1, 3) if np.ndim(U) == 2 else (-1)
        return result.reshape(shape)

    def KT_dot(self, lambda_vec: vector) -> np.ndarray:
        self.__check_input_size(lambda_vec=lambda_vec)
        result = self.cb.KT_x_Lam(np.array(lambda_vec).ravel())
        shape = (-1, 3) if np.ndim(lambda_vec) == 2 else (-1)
        return result.reshape(shape)

    def apply_PC(self, lambda_vec: vector, U_vec: vector) -> np.ndarray:
        self.__check_input_size(lambda_vec=lambda_vec, U_vec=U_vec)
        in_vec = np.concatenate((np.array(lambda_vec).ravel(), np.array(U_vec).ravel()))
        return self.cb.apply_PC(in_vec)

    def apply_M(self, F: vector) -> np.ndarray:
        self.__check_input_size(lambda_vec=F)
        r_vecs = self.get_blob_positions()
        return self.cb.apply_M(np.array(F).ravel(), r_vecs.ravel())

    def get_K(self) -> np.ndarray:
        return self.cb.get_K()

    def get_Kinv(self) -> np.ndarray:
        return self.cb.get_Kinv()

    def evolve_rigid_bodies(self, U: vector) -> None:
        self.__check_input_size(U_vec=U)
        self.cb.evolve_X_Q(np.array(U).ravel())

    def __check_and_set_configs(self, X: vector, Q: vector) -> None:
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
        self.X_shape = np.shape(X)
        self.Q_shape = np.shape(Q)

    def __check_input_size(
        self, lambda_vec: vector | None = None, U_vec: vector | None = None
    ):
        if lambda_vec is not None:
            if np.size(lambda_vec) != 3 * self.total_blobs:
                raise RuntimeError(
                    f"lambda must have total size 3*N_blobs = {3 * self.total_blobs}. lambda_vec shape: {np.shape(lambda_vec)}"
                )
        if U_vec is not None:
            if np.size(U_vec) != 6 * self.N_bodies:
                raise RuntimeError(
                    f"U must have total size 6*N_bodies = {6*self.N_bodies}. U shape: {np.shape(U_vec)}"
                )
