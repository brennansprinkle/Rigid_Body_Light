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
from sksparse.cholmod import cholesky
import pyamg
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as pyrot

# Find project functions
found_functions = False
path_to_append = './'
sys.path.append('../')
sys.path.append('./build/')

for i in range(10):
    path_to_append += '../'
    sys.path.append(path_to_append)

import c_rigid_obj as cbodies

def load_data(file_name):
            with open(file_name, 'r') as f:
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
    fb = 0*r_vectors
    h = r_vectors[:,2]
    lr_mask = h > a
    sr_mask = h <= a
    fb[lr_mask,2] += (repulsion_strength / debye_length) * np.exp(-(h[lr_mask]-a)/debye_length)
    fb[sr_mask,2] += (repulsion_strength / debye_length)
    
    return fb.flatten()

def wall_energy_blobs(r_vectors, a, debye_length, repulsion_strength):
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
    h = r_vectors[:,2]
    # this seems to fix the energy for debeye = 0.1*a but not for 0.5*a
    #h += 0.75*debye_length
    E = 0
    for k in range(len(h)):
        if h[k] > a:
            E += (repulsion_strength) * np.exp(-(h[k]-a)/debye_length)
        else:
            E += (repulsion_strength) * (1.0 - (h[k]-a)/debye_length)
    # E = 0*h
    # lr_mask = h > a
    # sr_mask = h <= a
    # E[lr_mask] += (repulsion_strength) * np.exp(-(h[lr_mask]-a)/debye_length)
    # E[sr_mask] += (repulsion_strength) * (1.0 - (h[sr_mask]-a)/debye_length)
    
    return E #np.sum(E)

def particle_wall_energy(Cfg, Radius, a, debye_length, repulsion_strength, g, kBT, heights=None):
    if heights is None:
        heights = np.linspace(0.0, 2*Radius, 100)
    P_h = np.zeros(heights.shape)
    for k in range(len(heights)):
        r_vectors = Cfg + np.array([0, 0, heights[k]])
        E = wall_energy_blobs(r_vectors, a, debye_length, repulsion_strength)
        E += (heights[k])*g
        P_h[k] = np.exp(-E/(kBT))
    # normalize P_h using the trapezoidal rule
    dh = heights[1] - heights[0]
    P_h /= np.trapz(P_h, dx=dh)
    return P_h, heights

def particle_wall_energy_cfg_avg(Cfg, Radius, a, debye_length, repulsion_strength, g, kBT, heights):
    Ph_avg = np.zeros(heights.shape)
    Nsamp = 1000
    for junk in range(Nsamp):
        Cfg_rot = np.zeros(Cfg.shape)
        q_k = np.random.randn(4)
        q_k /= np.linalg.norm(q_k)
        for k in range(len(Cfg)):
            Cfg_rot[k] = pyrot.from_quat(q_k).apply(Cfg[k])
        Ph,hs = particle_wall_energy(Cfg_rot, Radius, a, debye_length, repulsion_strength, g, kBT, heights=(Radius*bin_locs))
        Ph_avg += Ph
    Ph_avg /= (1.0*Nsamp)
    return Ph_avg, hs


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
    f[2] = -g*np.cos(theta)
    f[0] = -g*np.sin(theta)
    return f

if __name__ == '__main__':

        # load a numeric data file into Cfg and skip the first line
        # but keep the second number in the first line and save as a variable called s
        
        struct_file = './Structures/shell_N_42_Rg_0_8913_Rh_1.vertex'
        #struct_file = './Structures/shell_N_162_Rg_0_9497_Rh_1.vertex'
        #struct_file = './Structures/shell_N_642_Rg_0_9767_Rh_1.vertex'
        #struct_file = './Structures/shell_N_2562_Rg_0_9888_Rh_1.vertex'
        s, Cfg = load_data(struct_file)
        Radius = 1.0 #1.486

        s, Cfg = load_data(struct_file)

        s *= Radius
        Cfg *= Radius

        # Set some variables for the simulation
        a = 0.5*s
        output_name = './data/test'
        

        # Create rigid bodies
        X_0 = []
        Quat = []

        struct_location = np.array([0,0,1.2])*Radius
        struct_orientation = np.array([1.0,0.0,0.0,0.0])
        for k in range(1):
            X_0.append(struct_location)
            Quat.append(struct_orientation)


        Nbods = len(X_0)
        X_0 = np.array(X_0).flatten()
        Quat = np.array(Quat).flatten()
        
           
        # read in misc. parameters
        n_steps = 400000
        n_save = 1
        eta = 1.4e-3 # viscosity (Pa*s)
        dt = 5e-3
        kBT = 0.004142 #aJ # #0.0
        g = 14.2926*kBT
        theta = 0.0 #np.pi/6.0
        debye_length = 0.1*a
        repulsion_strength = 4.0*kBT
        #kBT = 0.0
        Tol = 1.0e-3
        periodic_length = np.array([0.0,0.0,0.0])
        
        
        
        print('a is: '+str(a))
        print('diffusive blob timestep is: '+str(kBT*dt/(6*np.pi*eta*a**3)))
    
        
        # Make solver object
        cb = cbodies.CManyBodies()
        
        # Sets the PC type
        # If true will use the block diag 'Dilute suspension approximation' to M
        # For dense suspensions this is a bad approximation (try setting false)
        # for rigid bodies with lots of blobs this is expensive (try setting flase)
        cb.setBlkPC(False)

        # Set the domain to have a wall
        cb.setWallPC(True)
        
       
        numParts = Nbods*len(Cfg)
        cb.setParameters(numParts, a, dt, kBT, eta, periodic_length, Cfg)
        cb.setConfig(X_0,Quat)
        print('set config')
        cb.set_K_mats()
        print('set K mats')
        
        
        ######################################################################################################
        ########################################## Solver params ###############################################
        ######################################################################################################
        
        sz = 3*Nbods*len(Cfg)
        Nsize = sz + 6*Nbods
        
        num_rejects = 0
        Sol = np.zeros(Nsize)
        
        ############################################
        #### Solve the system
            
        Qs, Xs = cb.getConfig()
        r_vectors = np.array(cb.multi_body_pos())
        FT = np.zeros((2, 3))
        

        Force = FT.flatten()
        Slip = np.zeros(sz)
        gamma = 1.0
        #
        # print every 3rd element of r_vectors
            
        num_rejects = 0

        X_Guess = None
        h_coord = []
        for n in range(n_steps):
            Qs, Xs = cb.getConfig()
            Xs_start = copy.deepcopy(Xs)
            Qs_start = copy.deepcopy(Qs)
            r_vectors = np.array(cb.multi_body_pos())
            
            FT = np.zeros((2, 3))
            blob_force = wall_force_blobs(r_vectors, a, debye_length, repulsion_strength)
            FT += np.reshape(cb.KT_x_Lam(blob_force), (2*Nbods, 3))
            # print(cb.KT_x_Lam(blob_force))
            # sys.exit()
            FT[0,:] += wall_force_particle(theta, g)
            Force = FT.flatten()
            Slip = np.zeros(sz)
            #gamma = 1.0
            #Slip[0::3] = -1.0*r_vectors[2::3] * gamma # - gamma


            start = time.time()
            RHS = cb.RHS_and_Midpoint(Slip, Force)

            r_vectors_mid = np.array(cb.multi_body_pos())
            r_vectors_mid = np.reshape(r_vectors_mid, (Nbods*len(Cfg), 3))
            
            RHS_norm = np.linalg.norm(RHS)
            end = time.time()
            print("Time RHS: "+str(end - start)+" s")
            
            A = spla.LinearOperator((Nsize, Nsize), matvec=cb.apply_Saddle, dtype='float64')
            PC = spla.LinearOperator((Nsize, Nsize), matvec=cb.apply_PC, dtype='float64')
            
            res_list = []
            start = time.time()

            if X_Guess is not None:
                X_0 = X_Guess/RHS_norm
            else:
                X_0 = None
            
            (Sol, info_precond) = pyamg.krylov.gmres(A, (RHS/RHS_norm), x0=X_0, tol=Tol, M=PC, #
                                                        maxiter=min(300, Nsize), restrt=None, residuals=res_list)
            
            end = time.time()
            print("Time GMRES: "+str(end - start)+" s")
            #print(res_list)
            print('GMRES its: '+str(len(res_list)))
            
            # Extract the velocities from the GMRES solution
            Sol *= RHS_norm
            X_Guess = Sol
            Lambda_s = Sol[0:sz]
            U_s = Sol[sz::]

            print('U_s: ',U_s)
            if np.linalg.norm(dt*U_s[0:3]) > 0.25*Radius:
                num_rejects += 1
                ########################
                print('Bad Timestep U_s!!')
                cb.setConfig(Xs_start,Qs_start)
                continue

            h_coord.append(Xs[2])
            # evolve rigid bodies
            cb.evolve_X_Q(U_s)

            

        print('Number of rejects: ',num_rejects)
        # make a histogram of the h_coord
        plt.figure()
        h_coord = np.array(h_coord)
        # remove the first 1000 elements
        cut = int(1.0/dt)
        h_coord = h_coord[cut::]
        #normalize by Raidus
        h_coord = h_coord
        # use 100 bins from a to 2*a
        bin_locs = np.linspace(1, 2, 200)
        #plt.hist(h_coord, bins=bin_locs)
        # get the counts from the histogram without plotting
        counts, bin_edges = np.histogram(h_coord, bins=bin_locs)
        counts = counts.astype(float)
        # normalize counts using trapezoidal rule
        bin_dx = bin_edges[1] - bin_edges[0]
        counts = counts / np.trapz(counts, dx=bin_dx)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0
        #plt.plot(bin_centers, counts, 'b--', label='Histogram')
        # normalize the histogram
        #exact = np.load('./Hist_Data/hist_theory_42_blob.npz')
        #plt.plot(exact['z'], exact['P'], 'k-', label='Theory')
        Ph,hs = particle_wall_energy(Cfg, Radius, a, debye_length, repulsion_strength, g, kBT, heights=(Radius*bin_locs))
        # bin_h = (hs[1] - hs[0])/Radius
        # Ph = Ph/np.trapz(Ph, dx=bin_h)
        plt.plot(hs, Ph, 'r-', label='Theory')

        Ph_avg, hs_avg = particle_wall_energy_cfg_avg(Cfg, Radius, a, debye_length, repulsion_strength, g, kBT, heights=(Radius*bin_locs))
        plt.plot(bin_centers, counts, 'b--', label='Histogram', linewidth=4)
        plt.plot(hs_avg, Ph_avg, 'k-', label='Histogram', linewidth=4)

        plt.xlabel('h (um)')
        plt.ylabel('Count')
        plt.show()
        # save the data to a file
        np.savetxt('./Hist_Data/mid_dt_1em2_h_coord.txt', h_coord)

        sys.exit()




        Qs, Xs = cb.getConfig()
        print(Xs)
        r_vectors = np.array(cb.multi_body_pos())
        # look at flow field
        x = np.linspace(Xs[0]-2, Xs[0]+2, 100)
        z = np.linspace(max(0.0,Xs[2]-2.0), Xs[2]+2.0, 100)
        X, Z = np.meshgrid(x, z)
        # make those points into a flattened array of [x,y,z] points with y = 0
        points = np.vstack([X.ravel(), np.zeros(X.ravel().shape)+Xs[1], Z.ravel()]).T
        point_forces = 0*points
        # make a new array of points with r_vectors stacked ontop points.flatten()
        # but don't use vstack
        all_points = np.concatenate((r_vectors, points.flatten()))
        all_forces = np.concatenate((Lambda_s, point_forces.flatten()))

        all_velocities = cb.apply_M(all_forces,all_points)
        v_blobs = all_velocities[0:sz]
        v_grid = all_velocities[sz::]

        u = v_grid[0::3]
        w = v_grid[2::3]
        u = u.reshape(X.shape)  #+ Z*gamma
        w = w.reshape(X.shape)
        # set all values of u and w to zero where X**2 + (Z-0.6)**2 < 1

        mask = (X-Xs[0])**2 + (Z-Xs[2])**2 < Radius**2
        u[mask] = 0
        w[mask] = 0


        # plot the flow field magnitude and streamlines
        plt.figure()
        #plt.streamplot(X, Z, u, w)
        # change the streamlines to balck and thinken the linewidth
        plt.streamplot(X, Z, u, w, color='k', linewidth=2,density=1)
        # on the smae plot show a solid black circls at [0,0.6] with radius 1
        #circle = plt.Circle((0, 0.6), 1, color='k', fill=False)
        # and the velocity magnitude as a contour plot
        u_mag = np.sqrt(u**2 + w**2)
        # set the colormap to be cmocean ice
        import cmocean
        plt.contourf(X, Z, u_mag, levels=50, cmap=cmocean.cm.dense)
        #plt.contourf(X, Z, u_mag, levels=50)
        # add a colorbar    
        cbar = plt.colorbar()
        cbar.set_label(r'$|U| (\mu m/s)$', rotation=90)
        plt.show()
