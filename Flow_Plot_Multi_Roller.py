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

if __name__ == '__main__':

        # load a numeric data file into Cfg and skip the first line
        # but keep the second number in the first line and save as a variable called s
        
        #struct_file = './Structures/shell_N_42_Rg_0_8913_Rh_1.vertex'
        struct_file = './Structures/shell_N_642_Rg_0_9767_Rh_1.vertex'
        Radius = 1.0

        s, Cfg = load_data(struct_file)

        s *= Radius
        Cfg *= Radius

        # Set some variables for the simulation
        a = 0.5*s
        output_name = './data/test'
        

        # Create rigid bodies
        X_0 = []
        Quat = []

        struct_location = [[0,0,2.2],[2.5,0,2.2]]
        struct_orientation = [[1.0,0.0,0.0,0.0],[1.0,0.0,0.0,0.0]]
        # struct_location = [[0,0,2.2]]
        # struct_orientation = [[1.0,0.0,0.0,0.0]]
        for k in range(len(struct_location)):
            X_0.append(np.array(struct_location[k]))
            Quat.append(np.array(struct_orientation[k]))


        Nbods = len(X_0)
        X_0 = np.array(X_0).flatten()
        Quat = np.array(Quat).flatten()
        
           
        # read in misc. parameters
        n_steps = 1 
        n_save = 1
        eta = 1e-3 # viscosity (Pa*s)
        dt = 1.0
        g = 0.0
        kBT = 0.0
        Tol = 1.0e-4
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
        print(r_vectors)
        FT = np.zeros((2*Nbods, 3))
        flip = 1.0
        ind = 0
        for i in range(Nbods):
            flip *= -1.0
            FT[2*i+1, ind] = flip*8.0*np.pi*eta*(2.0*np.pi*9.0)*Radius**3
            ind += 1
        

        print('Force:')
        print(FT)

        Force = FT.flatten()
        Slip = np.zeros(sz)
            
        
        start = time.time()
        RHS = cb.RHS_and_Midpoint(Slip, Force)
        
        RHS_norm = np.linalg.norm(RHS)
        end = time.time()
        print("Time RHS: "+str(end - start)+" s")
        
        A = spla.LinearOperator((Nsize, Nsize), matvec=cb.apply_Saddle, dtype='float64')
        PC = spla.LinearOperator((Nsize, Nsize), matvec=cb.apply_PC, dtype='float64')
        
        res_list = []
        start = time.time()
        
        (Sol, info_precond) = pyamg.krylov.gmres(A, (RHS/RHS_norm), x0=None, tol=Tol, M=PC, #
                                                    maxiter=min(300, Nsize), restrt=None, residuals=res_list)
        
        end = time.time()
        print("Time GMRES: "+str(end - start)+" s")
        #print(res_list)
        print('GMRES its: '+str(len(res_list)))
        
        # Extract the velocities from the GMRES solution
        Sol *= RHS_norm
        Lambda_s = Sol[0:sz]
        U_s = Sol[sz::]

        # make a grid of points in the x-z plance saparated by a from 0 to 4 in the z direction
        # and -2 to 2 in the x direction
        print('Vel: ')
        print(U_s.reshape((2*Nbods,3)))

        x = np.linspace(-2, 5, 100)
        z = np.linspace(0.01, 4, 100)
        dx  = x[2] - x[1]
        X, Z = np.meshgrid(x, z)
        # make those points into a flattened array of [x,y,z] points with y = 0
        points = np.vstack([X.ravel(), np.zeros(X.ravel().shape), Z.ravel()]).T
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
        u = u.reshape(X.shape)
        w = w.reshape(X.shape)
        # set all values of u and w to zero where X**2 + (Z-0.6)**2 < 1
 

        for i in range(Nbods):
            mask = (X-X_0[3*i])**2 + (Z-X_0[3*i+2])**2 < (Radius+0.1)**2
            u[mask] = 0
            w[mask] = 0


        # plot the flow field magnitude and streamlines
        plt.figure()
        #plt.streamplot(X, Z, u, w)
        # change the streamlines to balck and thinken the linewidth
        plt.streamplot(X, Z, u, w, color='k', linewidth=2,density=2)
        # on the smae plot show a solid black circls at [0,0.6] with radius 1
        #circle = plt.Circle((0, 0.6), 1, color='k', fill=False)
        # and the velocity magnitude as a contour plot
        u_mag = np.sqrt(u**2 + w**2)
        plt.contourf(X, Z, u_mag, levels=50)
        # set the aspect ratio to be equal
        plt.gca().set_aspect('equal', adjustable='box')
        # add a colorbar    
        cbar = plt.colorbar()
        cbar.set_label(r'$|U| (\mu m/s)$', rotation=90)
        plt.show()
        
