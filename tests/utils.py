import numpy as np
from Rigid import RigidBody

struct_shell_12 = "../structures/shell_N_12_Rg_0_7921_Rh_1.vertex"


def load_config(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
        s = float(lines[0].split()[0])
        config = np.array([[float(j) for j in i.split()] for i in lines[1:]])
    return s, config


def create_solver(X, Q, rigid_config=None, wall_PC=False, block_PC=False):
    if rigid_config is None:
        _, rigid_config = load_config(struct_shell_12)

    return RigidBody(
        rigid_config, X, Q, a=1.0, eta=1.0, dt=1.0, wall_PC=wall_PC, block_PC=block_PC
    )


def create_random_positions(N):
    n_placed = 0

    X = np.zeros((N, 3))
    while n_placed < N:
        x_i = np.random.uniform(-10.0, 10.0, (N, 3))
        dists = np.linalg.norm(X[:n_placed, :] - x_i[n_placed, :], axis=1)
        if np.all(dists > 2.0):
            X[n_placed, :] = x_i[n_placed, :]
            n_placed += 1

    Q = np.random.randn(N, 4)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    return X, Q
