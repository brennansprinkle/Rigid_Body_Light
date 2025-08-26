import numpy as np
from Rigid import RigidBody


def load_config(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
        s = float(lines[0].split()[0])
        config = np.array([[float(j) for j in i.split()] for i in lines[1:]])
    return s, config


def create_solver(rigid_config, X, Q):

    return RigidBody(
        rigid_config, X, Q, a=1.0, eta=1.0, dt=1.0, wall_PC=False, block_PC=False
    )
