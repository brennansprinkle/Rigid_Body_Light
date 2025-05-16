import argparse
import numpy as np
import scipy.linalg as la
import scipy.spatial as spatial
import scipy.sparse.linalg as spla

from functools import partial
import sys
import time
import copy

import scipy as sp
import pyamg
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import utils


from libMobility import NBody, DPStokes

# Find project functions
found_functions = False
path_to_append = "./"
sys.path.append("../")
sys.path.append("./build/")

for i in range(10):
    path_to_append += "../"
    sys.path.append(path_to_append)

import c_rigid_obj as cbodies


def main():

    # load a numeric data file into Cfg and skip the first line
    # but keep the second number in the first line and save as a variable called s

    struct_file = "./Structures/Cylinder_N_86_Lg_1_9384_Rg_0_1484.vertex"
    # struct_file = "./Structures/Cylinder_N_324_Lg_2_0299_Rg_0_1554.vertex"
    Cfg = load_data(struct_file)
    L_rod = 1.9384
    R_rod = 0.1484

    sep = np.min(sp.spatial.distance.pdist(Cfg))

    # Set some variables for the simulation
    a = 0.5 * sep
    L = 5 * L_rod
    H = L_rod

    # Create rigid bodies
    N_rigid_bodies = 10
    X_0 = []
    quat = []

    diameter = 1.1 * L_rod
    radius = diameter / 2

    attempts = 0
    max_attempts = 1000
    while len(X_0) < N_rigid_bodies and attempts < max_attempts:
        x = np.random.uniform(radius, L - radius)
        y = np.random.uniform(radius, L - radius)
        new_pos = np.array([x, y, 0.25 * H])  # TODO could change z from constant

        overlap = False
        for pos in X_0:
            if np.linalg.norm(new_pos - pos) < diameter:
                overlap = True
                break

        if not overlap:
            X_0.append(new_pos)
            theta = np.random.uniform(0, 2 * np.pi)
            q = R.from_euler("z", theta, degrees=False)
            q = q.as_quat(scalar_first=True)
            quat.append(q)
        attempts += 1

    if len(X_0) < N_rigid_bodies:
        raise RuntimeError(
            f"Could only place {len(X_0)} discs after {max_attempts} attempts. Try increasing L or reducing N."
        )

    X_0 = np.array(X_0).flatten()
    quat = np.array(quat).flatten()

    # read in misc. parameters
    n_steps = 10000
    n_plot = 50
    eta = 1.4e-3  # viscosity (Pa*s)
    dt = 2e-3
    kBT = 0.004142  # aJ
    g = 14.2926 * kBT
    theta = 0.0  # floor tilt angle
    debye_length = 0.1 * a
    repulsion_strength = 4.0 * kBT
    Tol = 1.0e-3
    periodic_length = np.array([L, L, 0.0])

    # kBT = 0.0

    solver = DPStokes("periodic", "periodic", "single_wall")
    solver.setParameters(Lx=L, Ly=L, zmin=0.0, zmax=H, allowChangingBoxSize=False)

    # solver = NBody("open", "open", "single_wall")
    # solver.setParameters(wallHeight=0.0)

    solver.initialize(
        temperature=kBT, viscosity=eta, hydrodynamicRadius=a, needsTorque=False
    )

    print("a is: " + str(a))
    print("diffusive blob timestep is: " + str(kBT * dt / (6 * np.pi * eta * a**3)))

    # Make solver object
    cb = cbodies.CManyBodies()

    # Sets the PC type
    # If true will use the block diag 'Dilute suspension approximation' to M
    # For dense suspensions this is a bad approximation (try setting false)
    # for rigid bodies with lots of blobs this is expensive (try setting flase)
    cb.setBlkPC(False)

    # Set the domain to have a wall
    cb.setWallPC(True)

    numParts = N_rigid_bodies * len(Cfg)
    cb.setParameters(numParts, a, dt, kBT, eta, periodic_length, Cfg)
    cb.setConfig(X_0, quat)
    print("set config")
    cb.set_K_mats()
    print("set K mats")

    ######################################################################################################
    ########################################## Solver params ###############################################
    ######################################################################################################

    sz = 3 * N_rigid_bodies * len(Cfg)
    Nsize = sz + 6 * N_rigid_bodies

    num_rejects = 0
    Sol = np.zeros(Nsize)

    ############################################
    #### Solve the system

    Qs, Xs = cb.getConfig()
    r_vectors = np.array(cb.multi_body_pos())

    num_rejects = 0

    U_Guess = None

    # h_coord = [[],[],[]]
    # make h_coord a list of emppty lists
    h_coords = []
    for k in range(N_rigid_bodies):
        h_coords.append([])

    fig_index = 0
    for n in range(n_steps):
        print("Step: ", n)
        if n % 1000 == 0:
            print("Progress: ", 100 * ((1.0 * n) / n_steps))
        Qs, Xs = cb.getConfig()
        Xs_start = copy.deepcopy(Xs)
        Qs_start = copy.deepcopy(Qs)
        r_vectors = np.array(cb.multi_body_pos())

        if n % n_plot == 0:
            plot_positions(
                r_vectors, fig_index, L, H, Xs, Qs, cyl_length=L_rod, cyl_radius=R_rod
            )
            fig_index += 1

        step_start = time.time()
        ########################################################
        ################## Step 1 ##############################
        ########################################################
        # get Random rigid velocity for RFD
        # print("Step 1")
        Slip = np.random.randn(sz)
        Force = np.zeros(6 * N_rigid_bodies)

        start = time.time()

        RHS = np.concatenate((-Slip, -Force))

        RHS_norm = np.linalg.norm(RHS)
        end = time.time()
        # print("Time RHS: "+str(end - start)+" s")

        solver.setPositions(r_vectors)

        def apply_Saddle_mdot(x):
            out = 0 * x
            Lam = x[0:sz]
            U = x[sz::]
            vels, _ = solver.Mdot(Lam)
            out[0:sz] = vels - cb.K_x_U(U)
            out[sz::] = cb.KT_x_Lam(Lam)
            out[sz::] *= -1.0
            return out

        A = spla.LinearOperator(
            (Nsize, Nsize), matvec=apply_Saddle_mdot, dtype="float64"
        )
        PC = spla.LinearOperator((Nsize, Nsize), matvec=cb.apply_PC, dtype="float64")

        res_list = []
        start = time.time()

        (Sol_RFD, info_RFD) = pyamg.krylov.gmres(
            A,
            (RHS / RHS_norm),
            x0=None,
            tol=Tol,
            M=PC,  #
            maxiter=min(300, Nsize),
            restrt=None,
            residuals=res_list,
        )

        end = time.time()
        # print("Time GMRES: " + str(end - start) + " s")
        # # print(res_list)
        # print("GMRES its: " + str(len(res_list)))

        # Extract the velocities from the GMRES solution
        Sol_RFD *= RHS_norm
        Lambda_RFD = Sol_RFD[0:sz]  # Don't care
        U_RFD = Sol_RFD[sz::]  # U =  N*K^T*M^{-1}*W

        ########## RFD ##########
        delta_rfd = 1.0e-3
        # r_vec_p,r_vec_m = cb.M_RFD_cfgs(U_RFD,delta_rfd)
        # r_vec_p = np.array(r_vec_p)
        # r_vec_m = np.array(r_vec_m)
        # solver.setPositions(r_vec_p)
        # MpW, _ = solver.Mdot(Slip)
        # solver.setPositions(r_vec_m)
        # MmW, _ = solver.Mdot(Slip)
        # solver.setPositions(r_vectors)
        # M_RFD = (1.0/delta_rfd)*(MpW - MmW)
        # KT_RFD = cb.KT_RFD_from_U(U_RFD,Slip)

        # print('KT_RFD: ',np.linalg.norm(KT_RFD))
        # print('M_RFD: ',np.linalg.norm(M_RFD))
        # print('U_RFD: ',U_RFD)

        ########################################################
        ################## Step 2 ##############################
        ########################################################
        # Predictor step
        # print("Step 2")
        ### Slip to be used in both steps
        Slip, _ = solver.sqrtMdotW()
        Slip *= np.sqrt(2 * kBT / dt)
        ### Forces at Q^n
        v_prescribed = 1.0
        rod_mob_fact = 6 * np.pi * eta * a * v_prescribed
        Force = Calc_Foces_Bodies(
            cb,
            theta,
            g,
            r_vectors,
            a,
            debye_length,
            repulsion_strength,
            N_rigid_bodies,
            rod_mob_fact,
            periodic_length,
        )

        RHS = np.concatenate((-Slip, -Force))
        RHS_norm = np.linalg.norm(RHS)
        res_list = []
        start = time.time()

        (Sol_Pred, info_Pred) = pyamg.krylov.gmres(
            A,
            (RHS / RHS_norm),
            x0=None,
            tol=Tol,
            M=PC,  #
            maxiter=min(300, Nsize),
            restrt=None,
            residuals=res_list,
        )

        end = time.time()
        # print("Time GMRES: " + str(end - start) + " s")
        # # print(res_list)
        # print("GMRES its: " + str(len(res_list)))

        # Extract the velocities from the GMRES solution
        Sol_Pred *= RHS_norm
        Lambda_Pred = Sol_Pred[0:sz]  # Don't care
        U_Pred = Sol_Pred[sz::]  # N*F + sqrt(2*kBT/dt)*N^{1/2}*W

        cb.evolve_X_Q(U_Pred)

        ########################################################
        ################## Step 3 ##############################
        ########################################################
        # Corrector step
        ### Update slip from predictor step
        # print("Step 3")
        # Slip += 2.0 * kBT * M_RFD
        ### Forces at Q^n+1/2
        r_vectors = np.array(cb.multi_body_pos())
        Force = Calc_Foces_Bodies(
            cb,
            theta,
            g,
            r_vectors,
            a,
            debye_length,
            repulsion_strength,
            N_rigid_bodies,
            rod_mob_fact,
            periodic_length,
        )
        # Force += -2.0 * kBT * KT_RFD

        RHS = np.concatenate((-Slip, -Force))
        RHS_norm = np.linalg.norm(RHS)

        solver.setPositions(r_vectors)
        A = spla.LinearOperator(
            (Nsize, Nsize), matvec=apply_Saddle_mdot, dtype="float64"
        )

        res_list = []
        start = time.time()

        (Sol_Corr, info_Corr) = pyamg.krylov.gmres(
            A,
            (RHS / RHS_norm),
            x0=(Sol_Pred / RHS_norm),
            tol=Tol,
            M=PC,  #
            maxiter=min(300, Nsize),
            restrt=None,
            residuals=res_list,
        )

        end = time.time()
        # print("Time GMRES: " + str(end - start) + " s")
        # # print(res_list)
        # print("GMRES its: " + str(len(res_list)))

        # Extract the velocities from the GMRES solution
        Sol_Corr *= RHS_norm
        Lambda_Corr = Sol_Corr[0:sz]  # Don't care
        U_Corr = Sol_Corr[sz::]  # Corrector veloctity

        cb.setConfig(Xs_start, Qs_start)
        cb.set_K_mats()

        # full velocity
        U_full = 0.5 * (U_Pred + U_Corr)
        # sys.exit()

        if np.linalg.norm(dt * U_full[0:3]) > 0.25 * L_rod:
            num_rejects += 1
            continue
            # sys.exit()

        for k in range(N_rigid_bodies):
            h_coords[k].append(float(Xs[3 * k + 2]))
        # evolve rigid bodies
        cb.evolve_X_Q(U_full)
        step_end = time.time()
        print("Step time: ", step_end - step_start)

    print("Number of rejects: ", num_rejects)


def create_cylinder(radius=0.1, length=1.0, resolution=16):
    """
    Create a cylinder aligned along the x-axis, centered at the origin.
    """
    theta = np.linspace(0, 2 * np.pi, resolution)
    x = np.linspace(-length / 2, length / 2, 2)  # <-- shift center
    theta, x = np.meshgrid(theta, x)

    y = radius * np.cos(theta)
    z = radius * np.sin(theta)

    return x, y, z


def plot_positions(r_vectors, fig_index, L, H, Xs, Qs, cyl_length, cyl_radius):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    Xs = Xs.reshape(-1, 3)
    Qs = Qs.reshape(-1, 4)

    box = np.array([L, L, 0])
    temp = copy.deepcopy(Xs)
    Xs = utils.periodize_r_vecs(Xs, box, len(Xs))

    # ax.scatter(r_vectors[0::3], r_vectors[1::3], r_vectors[2::3], c="r", marker="o")

    cyl_x, cyl_y, cyl_z = create_cylinder(radius=cyl_radius, length=cyl_length)

    for i in range(len(Xs)):
        pos = Xs[i]
        quat = Qs[i]

        # Convert quaternion to rotation matrix
        rot = R.from_quat(quat, scalar_first=True).as_matrix()

        # Flatten cylinder grid for transformation
        points = np.vstack([cyl_x.ravel(), cyl_y.ravel(), cyl_z.ravel()])  # (3, N)
        rotated = rot @ points  # Rotate (3, N)
        rotated[0, :] += pos[0]
        rotated[1, :] += pos[1]
        rotated[2, :] += pos[2]

        # Reshape back to meshgrid shape
        X = rotated[0].reshape(cyl_x.shape)
        Y = rotated[1].reshape(cyl_y.shape)
        Z = rotated[2].reshape(cyl_z.shape)

        ax.plot_surface(X, Y, Z, color="b", alpha=0.7, linewidth=0)

    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_zlim(0, H)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([L, L, H])  # Ensure aspect ratio
    plt.tight_layout()
    plt.savefig(f"img/fig_{fig_index}.png")
    plt.close()


def load_data(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
        # s should be a float
        Cfg = np.array([[float(j) for j in i.split()] for i in lines[1:]])
    return Cfg


def wall_force_blobs(r_vectors, a, debye_length, repulsion_strength):
    """
    Calculate the wall force using the Debye-Hückel theory.

    Parameters
    ----------
    r_vectors : ndarray
        blob positions
    a : float
        Radius of the blob.
    debye_length : float
        Debye length.
    repulsion_strength : float
        Repulsion strength.

    Returns
    -------
    ndarray
        Wall force.
    """
    # reshape r_vecrors to be (N,3)
    r_vectors = r_vectors.reshape(-1, 3)

    forces = np.zeros_like(r_vectors)
    h = r_vectors[:, 2]
    for k in range(len(h)):
        if h[k] > a:
            forces[k, 2] = (repulsion_strength / debye_length) * np.exp(
                -(h[k] - a) / debye_length
            )
        else:
            forces[k, 2] = repulsion_strength / debye_length

    return forces.flatten()


def calc_grav_force(theta, g):
    f = np.zeros(3)
    f[2] = -g * np.cos(theta)
    f[0] = -g * np.sin(theta)
    return f


def forward_orientation_force(cb, mob_fact, N):
    F_mag = 6 * np.pi * mob_fact
    quats, _ = cb.getConfig()
    quats = quats.reshape(N, 4)
    forces = np.zeros((2 * N, 3))

    for i in range(N):
        q = quats[i]
        Ration = R.from_quat(q, scalar_first=True)
        Ration = Ration.as_matrix()
        forward_vector = np.array([1, 0, 0])
        forward_vector = np.dot(Ration, forward_vector)
        forward_vector = forward_vector / np.linalg.norm(forward_vector)
        forward_force = F_mag * forward_vector

        forces[2 * i] = forward_force

    return forces


def calc_sterics(r_vectors, L, a, repulsion_strength, debye_length, Nbods):

    N_per_body = len(r_vectors) // 3 // Nbods

    r_cut = 2 * a + 4 * debye_length
    offsets, list_of_neighbors = utils.build_neighbor_list(r_vectors, L, r_cut)

    blob_sterics = utils.blob_blob_sterics(
        r_vectors, L, a, repulsion_strength, debye_length, list_of_neighbors, offsets
    )

    return blob_sterics.flatten()


def Calc_Foces_Bodies(
    cb, theta, g, r_vectors, a, debye_length, repulsion_strength, Nbods, rod_mob_fact, L
):
    FT = np.zeros((2 * Nbods, 3))
    blob_force = wall_force_blobs(r_vectors, a, debye_length, repulsion_strength)
    FT += np.reshape(cb.KT_x_Lam(blob_force), (2 * Nbods, 3))

    grav_force = calc_grav_force(theta, g)
    FT[0::2, :] += grav_force

    propelling_force = forward_orientation_force(cb, rod_mob_fact, Nbods)
    FT += propelling_force

    blob_sterics = calc_sterics(
        r_vectors, L, a, repulsion_strength, debye_length, Nbods
    )
    FT += np.reshape(cb.KT_x_Lam(blob_sterics), (2 * Nbods, 3))

    return FT.flatten()


if __name__ == "__main__":
    main()
