import numpy as np
from Rigid import RigidBody
import os

struct_dir = os.path.dirname(os.path.abspath(__file__)) + "/../structures/"
struct_shell_12 = struct_dir + "shell_N_12.csv"


def load_config(file_name):
    with open(file_name, "r") as f:
        _ = f.readline()
        params = f.readline().strip().split(",")
        sep = float(params[0].split(" ")[1])
        N = int(params[1])
        rg = float(params[2])
        rh = int(params[3])
        cfg = np.loadtxt(f, delimiter=" ")
        params = {"sep": sep, "N": N, "Rg": rg, "Rh": rh}
    return params, cfg


def create_solver(X, Q, rigid_config=None, wall_PC=False, block_PC=False):
    if rigid_config is None:
        _, rigid_config = load_config(struct_shell_12)

    return RigidBody(
        rigid_config,
        X,
        Q,
        a=1.0,
        eta=1.0,
        dt=1.0,
        wall_PC=wall_PC,
        block_PC=block_PC,
    )


def create_random_positions(N, wall_PC=False):
    n_placed = 0

    X = np.zeros((N, 3))
    while n_placed < N:
        lower_limit = 1.0 if wall_PC else -10.0
        x_i = np.random.uniform(lower_limit, 10.0, (N, 3))
        dists = np.linalg.norm(X[:n_placed, :] - x_i[n_placed, :], axis=1)
        if np.all(dists > 2.0):
            X[n_placed, :] = x_i[n_placed, :]
            n_placed += 1

    Q = np.random.randn(N, 4)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    return X, Q
