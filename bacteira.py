import argparse
import numpy as np
import scipy.linalg as la
import scipy.spatial as spatial
import scipy.sparse.linalg as spla

from functools import partial
import sys
import time
import copy

import scipy.sparse as sp
import pyamg
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as pyrot

from libMobility import NBody, DPStokes

# Find project functions
found_functions = False
path_to_append = './'
sys.path.append('../')
sys.path.append('./build/')

for i in range(10):
    path_to_append += '../'
    sys.path.append(path_to_append)

import c_rigid_obj as cbodies


def main():

    # load a numeric data file into Cfg and skip the first line
    # but keep the second number in the first line and save as a variable called s

    struct_file = "./Structures/shell_N_42_Rg_0_8913_Rh_1.vertex"
    # struct_file = './Structures/shell_N_162_Rg_0_9497_Rh_1.vertex'
    # struct_file = './Structures/shell_N_642_Rg_0_9767_Rh_1.vertex'
    # struct_file = './Structures/shell_N_2562_Rg_0_9888_Rh_1.vertex'
    s, Cfg = load_data(struct_file)
    Radius = 1.0  # 1.486

    s, Cfg = load_data(struct_file)

    s *= Radius
    Cfg *= Radius

    Nblobs_Per = len(Cfg)

    # Set some variables for the simulation
    a = 0.5 * s
    output_name = "./data/test"

    # Create rigid bodies
    X_0 = []
    Quat = []

    struct_location = np.array([0, 0, 1.2]) * Radius
    struct_orientation = np.array([1.0, 0.0, 0.0, 0.0])
    for k in range(10):
        # random x y  location in the range -1e6 to 1e6
        xy_loc = np.random.rand(2) * 1.0e3
        xy_loc = 2.0 * xy_loc - 1e3
        zshift = np.random.rand() * 0.4 * Radius
        zshift -= 0.2 * Radius
        loc = struct_location + np.array([xy_loc[0], xy_loc[1], zshift])
        X_0.append(loc)
        Quat.append(struct_orientation)

    Nbods = len(X_0)
    X_0 = np.array(X_0).flatten()
    Quat = np.array(Quat).flatten()

    # read in misc. parameters
    n_steps = 10000
    n_save = 1
    eta = 1.4e-3  # viscosity (Pa*s)
    dt = 1e-2
    kBT = 0.004142  # aJ # #0.0
    g = 14.2926 * kBT
    theta = 0.0  # np.pi/6.0
    debye_length = 0.1 * a
    repulsion_strength = 4.0 * kBT
    # kBT = 0.0
    Tol = 1.0e-3
    periodic_length = np.array([0.0, 0.0, 0.0])

    solver = NBody("open", "open", "single_wall")
    solver.setParameters(wallHeight=0.0, Nbatch=Nbods, NperBatch=Nblobs_Per)

    # solver = DPStokes("periodic", "periodic", "two_walls")
    # solver.setParameters(Lx=L, Ly=L,zmin=0.0, zmax=5*a, allowChangingBoxSize=True)

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

    numParts = Nbods * len(Cfg)
    cb.setParameters(numParts, a, dt, kBT, eta, periodic_length, Cfg)
    cb.setConfig(X_0, Quat)
    print("set config")
    cb.set_K_mats()
    print("set K mats")

    ######################################################################################################
    ########################################## Solver params ###############################################
    ######################################################################################################

    sz = 3 * Nbods * len(Cfg)
    Nsize = sz + 6 * Nbods

    num_rejects = 0
    Sol = np.zeros(Nsize)

    ############################################
    #### Solve the system

    Qs, Xs = cb.getConfig()
    r_vectors = np.array(cb.multi_body_pos())
    #

    num_rejects = 0

    U_Guess = None

    # h_coord = [[],[],[]]
    # make h_coord a list of emppty lists
    h_coords = []
    for k in range(Nbods):
        h_coords.append([])

    for n in range(n_steps):
        if n % 1000 == 0:
            print("Progress: ", 100 * ((1.0 * n) / n_steps))
        Qs, Xs = cb.getConfig()
        Xs_start = copy.deepcopy(Xs)
        Qs_start = copy.deepcopy(Qs)
        r_vectors = np.array(cb.multi_body_pos())

        # Calc_Foces_Bodies(cb, theta, g, r_vectors, a, debye_length, repulsion_strength)

        ########################################################
        ################## Step 1 ##############################
        ########################################################
        # get Random rigid velocity for RFD
        print("Step 1")
        Slip = np.random.randn(sz)
        Force = np.zeros(6 * Nbods)

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
        print("Time GMRES: " + str(end - start) + " s")
        # print(res_list)
        print("GMRES its: " + str(len(res_list)))

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
        print("Step 2")
        ### Slip to be used in both steps
        Slip, _ = solver.sqrtMdotW()
        Slip *= np.sqrt(2 * kBT / dt)
        ### Forces at Q^n
        Force = Calc_Foces_Bodies(
            cb, theta, g, r_vectors, a, debye_length, repulsion_strength, Nbods
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
        print("Time GMRES: " + str(end - start) + " s")
        # print(res_list)
        print("GMRES its: " + str(len(res_list)))

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
        print("Step 3")
        # Slip += 2.0 * kBT * M_RFD
        ### Forces at Q^n+1/2
        r_vectors = np.array(cb.multi_body_pos())
        Force = Calc_Foces_Bodies(
            cb, theta, g, r_vectors, a, debye_length, repulsion_strength, Nbods
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
        print("Time GMRES: " + str(end - start) + " s")
        # print(res_list)
        print("GMRES its: " + str(len(res_list)))

        # Extract the velocities from the GMRES solution
        Sol_Corr *= RHS_norm
        Lambda_Corr = Sol_Corr[0:sz]  # Don't care
        U_Corr = Sol_Corr[sz::]  # Corrector veloctity

        cb.setConfig(Xs_start, Qs_start)
        cb.set_K_mats()

        # full velocity
        U_full = 0.5 * (U_Pred + U_Corr)
        # sys.exit()

        if np.linalg.norm(dt * U_full[0:3]) > 0.25 * Radius:
            num_rejects += 1
            continue
            # sys.exit()

        for k in range(Nbods):
            h_coords[k].append(float(Xs[3 * k + 2]))
        # evolve rigid bodies
        cb.evolve_X_Q(U_full)

    print("Number of rejects: ", num_rejects)

def load_data(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
        # s should be a float
        s = float(lines[0].split()[1])
        Cfg = np.array([[float(j) for j in i.split()] for i in lines[1:]])
    return s, Cfg


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
    #
    fb = 0 * r_vectors
    h = r_vectors[:, 2]
    # h -= 4*debye_length
    for k in range(len(h)):
        if h[k] > a:
            fb[k, 2] = (repulsion_strength / debye_length) * np.exp(
                -(h[k] - a) / debye_length
            )
        else:
            fb[k, 2] = repulsion_strength / debye_length
    # lr_mask = h > a
    # sr_mask = h <= a
    # fb[lr_mask,2] += (repulsion_strength / debye_length) * np.exp(-(h[lr_mask]-a)/debye_length)
    # fb[sr_mask,2] += (repulsion_strength / debye_length)

    return fb.flatten()

def wall_force_particle(theta, g):
    """
    Calculate the wall force using the Debye-Hückel theory.

    Parameters
    ----------
    theta : float
        Angle of the wall.
    g : float
        boyancy force.
    Returns
    -------
    float
        Wall force.
    """
    f = np.zeros(3)
    f[2] = -g * np.cos(theta)
    f[0] = -g * np.sin(theta)
    return f


def Calc_Foces_Bodies(
    cb, theta, g, r_vectors, a, debye_length, repulsion_strength, Nbods
):
    FT = np.zeros((2 * Nbods, 3))
    blob_force = wall_force_blobs(r_vectors, a, debye_length, repulsion_strength)
    # sys.exit()
    FT += np.reshape(cb.KT_x_Lam(blob_force), (2 * Nbods, 3))
    # FT[0,:] += np.sum(blob_force.reshape(-1, 3), axis=0)
    grav_force = wall_force_particle(theta, g)
    FT[0::2, :] += grav_force
    return FT.flatten()


if __name__ == "__main__":
    main()
