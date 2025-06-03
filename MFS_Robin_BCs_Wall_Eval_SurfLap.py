#!/usr/bin/env python3
"""
MFS Solver for 2 spheres
Converted from MATLAB to Python
Last edited: 6/2/2025
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse as sp
from scipy.spatial.distance import pdist
from scipy.spatial import Delaunay
from scipy.special import j0, j1
from scipy.linalg import qr
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib parameters for LaTeX-like appearance
plt.rcParams.update({
    'font.size': 18,
    'lines.linewidth': 5,
    'axes.linewidth': 2,
    'figure.figsize': (10, 8)
})

def main():
    """Main function to run the MFS solver"""
    
    # Parameters
    r_src = 0.3  # radially shrink source points
    
    # Green's function params
    n = 100  # number of quad points in Hankel transform
    alpha = 1.0
    E_0 = 4.0 * 2.0
    
    # Eval grid params
    N = 64
    L = 3.5
    
    # Biot numbers (you'll need to define these)
    Bi_1 = 2.0  # Define this value
    Bi_2 = 0.1  # Define this value
    
    print("Loading grid data...")
    
    # Load grid data (you'll need to replace these paths with your actual data files)
    try:
        x_bdry_1 = np.loadtxt('Spherical_T_data/ss029.00438')
        x_source_1 = np.loadtxt('Spherical_T_data/ss027.00380')
        
        x_bdry_2 = np.loadtxt('Spherical_T_data/ss019.00192')
        x_source_2 = np.loadtxt('Spherical_T_data/ss017.00156')
    except FileNotFoundError:
        print("Warning: Grid data files not found. Creating dummy data for demonstration.")
        # Create dummy spherical data
        theta = np.linspace(0, np.pi, 20)
        phi = np.linspace(0, 2*np.pi, 40)
        theta_mesh, phi_mesh = np.meshgrid(theta, phi)
        
        x = np.sin(theta_mesh) * np.cos(phi_mesh)
        y = np.sin(theta_mesh) * np.sin(phi_mesh)
        z = np.cos(theta_mesh)
        
        x_bdry_1 = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
        x_source_1 = 0.8 * x_bdry_1
        x_bdry_2 = x_bdry_1.copy()
        x_source_2 = x_source_1.copy()
    
    # Process first sphere
    x_source_1 = r_src * x_source_1
    dxbd_1 = np.min(pdist(x_bdry_1))
    z_p = 1.0 + 0.2
    x_bdry_1[:, 2] += z_p
    x_source_1[:, 2] += z_p
    
    # Process second sphere
    dxbd_2 = np.min(pdist(x_bdry_2))
    scale_factor = dxbd_1 / dxbd_2
    x_bdry_2 *= scale_factor
    x_source_2 = r_src * scale_factor * x_source_2
    
    x_bdry_2[:, 2] += 0.7 * z_p
    x_source_2[:, 2] += 0.7 * z_p
    x_p = 1.7
    x_bdry_2[:, 0] += x_p
    x_source_2[:, 0] += x_p
    
    # Combine sources
    x_source = np.vstack([x_source_1, x_source_2])
    
    print("Setting up triangulations and FEM matrices...")
    
    # Create Delaunay triangulations and get FEM matrices
    tri_1 = Delaunay(x_bdry_1)
    L_1, M_1 = get_FEM_Lap(x_bdry_1, tri_1.simplices)
    
    tri_2 = Delaunay(x_bdry_2)
    L_2, M_2 = get_FEM_Lap(x_bdry_2, tri_2.simplices)
    
    # Set up block diagonal matrices
    m_plus = 0.5
    Du_1 = (1.0 + 2.0 * m_plus) * Bi_1
    Du_2 = (1.0 + 2.0 * m_plus) * Bi_2
    
    L_full = sp.block_diag([Du_1 * L_1, Du_2 * L_2])
    M_full = sp.block_diag([M_1, M_2])
    
    # Combine boundary points and compute normals
    x_bdry = np.vstack([x_bdry_1, x_bdry_2])
    
    n_hat_1 = x_bdry_1 - np.mean(x_bdry_1, axis=0)
    n_hat_1 = n_hat_1 / np.linalg.norm(n_hat_1, axis=1, keepdims=True)
    
    n_hat_2 = x_bdry_2 - np.mean(x_bdry_2, axis=0)
    n_hat_2 = n_hat_2 / np.linalg.norm(n_hat_2, axis=1, keepdims=True)
    
    n_hat = np.vstack([n_hat_1, n_hat_2])
    
    print("Setting up evaluation grid...")
    
    # Set up evaluation grid
    x = np.linspace(-4, 7, 2*N+1)
    y = np.linspace(-5, 5, 2*N+1)
    x = 0.5 * (x[:-1] + x[1:])
    y = 0.5 * (y[:-1] + y[1:])
    
    X, Y = np.meshgrid(x, y)
    X_eval = X
    Y_eval = Y
    Z_eval = np.zeros_like(X)
    
    print("Calling MFS solver...")
    
    # Call solver
    A, b, Phi, Phi_x, Phi_y, Phi_z = get_MFS_solve(
        E_0, alpha, n, n_hat, x_bdry, x_source, 
        X_eval, Y_eval, Z_eval, L_full, M_full
    )
    
    print("Plotting results...")
    
    # Plotting
    fig = plt.figure(figsize=(15, 5))
    
    # 3D plot of spheres
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(x_bdry[:, 0], x_bdry[:, 1], x_bdry[:, 2], c='k', s=20)
    ax1.scatter(x_source[:, 0], x_source[:, 1], x_source[:, 2], c='r', s=20, marker='^')
    ax1.quiver(x_bdry[:, 0], x_bdry[:, 1], x_bdry[:, 2], 
              n_hat[:, 0], n_hat[:, 1], n_hat[:, 2], 
              color=[0.0, 0.65, 0.95], length=0.1)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Sphere Configuration')
    
    # Unit conversions for velocity field
    eps_p_eta = 699240  # (um/V)^2 * (1/s)
    phi_star_T = 2.5e-4
    
    U_x = -eps_p_eta * np.real((1j*(phi_star_T*E_0/alpha) + phi_star_T*Phi) * 
                               np.conj(phi_star_T*Phi_x))
    U_y = -eps_p_eta * np.real((1j*(phi_star_T*E_0/alpha) + phi_star_T*Phi) * 
                               np.conj(phi_star_T*Phi_y))
    
    U_mag = np.sqrt(U_x**2 + U_y**2)
    
    # Velocity magnitude plot
    ax2 = fig.add_subplot(132)
    c = ax2.pcolormesh(X, Y, U_mag, cmap='turbo_r', shading='auto')
    ax2.streamplot(X, Y, U_x, U_y, color='k', linewidth=1, density=2)
    plt.colorbar(c, ax=ax2)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Velocity Magnitude')
    ax2.set_aspect('equal')
    
    # Solution check
    ax3 = fig.add_subplot(133)
    solution_check = np.imag(Phi_z - 1j*alpha*Phi)
    c = ax3.pcolormesh(X, Y, solution_check, shading='auto')
    plt.colorbar(c, ax=ax3)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('Solution Check')
    ax3.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    # Save results
    results = {
        'U_x': U_x,
        'U_y': U_y,
        'U_mag': U_mag,
        'Phi': Phi,
        'Phi_x': Phi_x,
        'Phi_y': Phi_y,
        'Phi_z': Phi_z,
        'Du_1': Du_1,
        'Du_2': Du_2,
        'alpha': alpha,
        'E_0': E_0,
        'X_eval': X_eval,
        'Y_eval': Y_eval,
        'Z_eval': Z_eval,
        'N': N
    }
    
    np.savez(f'MFS_solution_n_{n}_alpha_{alpha}_Bi_0p1_2.npz', **results)
    print("Results saved to MFS_solution_n_{}_alpha_{}_Bi_0p1_2.npz".format(n, alpha))

def green(x, x_0, y, y_0, z, z_0, alpha, n):
    """Green's function"""
    r = np.sqrt((x - x_0)**2 + (y - y_0)**2)
    R = np.sqrt(r**2 + (z - z_0)**2)
    R_prime = np.sqrt(r**2 + (z + z_0)**2)
    
    G = (1/(4*np.pi)) * (1/R + 1/R_prime - robin_integral(r, z, z_0, alpha, n))
    return G

def green_x(x, x_0, y, y_0, z, z_0, alpha, n):
    """x-derivative of Green's function"""
    r = np.sqrt((x - x_0)**2 + (y - y_0)**2)
    R = np.sqrt(r**2 + (z - z_0)**2)
    R_prime = np.sqrt(r**2 + (z + z_0)**2)
    
    G_x = -(1/(4*np.pi)) * (x - x_0) * (1/R**3 + 1/R_prime**3)
    
    if r >= 1e-10:
        G_x = G_x - (1/(4*np.pi)) * ((x - x_0)/r) * robin_integral_r(r, z, z_0, alpha, n)
    
    return G_x

def green_y(x, x_0, y, y_0, z, z_0, alpha, n):
    """y-derivative of Green's function"""
    r = np.sqrt((x - x_0)**2 + (y - y_0)**2)
    R = np.sqrt(r**2 + (z - z_0)**2)
    R_prime = np.sqrt(r**2 + (z + z_0)**2)
    
    G_y = -(1/(4*np.pi)) * (y - y_0) * (1/R**3 + 1/R_prime**3)
    
    if r >= 1e-10:
        G_y = G_y - (1/(4*np.pi)) * ((y - y_0)/r) * robin_integral_r(r, z, z_0, alpha, n)
    
    return G_y

def green_z(x, x_0, y, y_0, z, z_0, alpha, n):
    """z-derivative of Green's function"""
    r = np.sqrt((x - x_0)**2 + (y - y_0)**2)
    R = np.sqrt(r**2 + (z - z_0)**2)
    R_prime = np.sqrt(r**2 + (z + z_0)**2)
    
    G_z = (1/(4*np.pi)) * (-(z - z_0)/R**3 - (z + z_0)/R_prime**3 - 
                           robin_integral_z(r, z, z_0, alpha, n))
    return G_z

def robin_integral(r, z, z_0, alpha, n):
    """Robin integral for Green's function"""
    z_tilde = z + z_0
    
    def imaginary_func(k):
        return 2*alpha*np.exp(-k*np.abs(z_tilde))*j0(k*r)*k / (k**2 + alpha**2)
    
    def real_func(k):
        return 2*alpha**2*np.exp(-k*np.abs(z_tilde))*j0(k*r) / (k**2 + alpha**2)
    
    i1 = clenshaw_curtis_half_space(imaginary_func, n)
    i2 = clenshaw_curtis_half_space(real_func, n)
    
    return 1j*i1 + i2

def robin_integral_r(r, z, z_0, alpha, n):
    """r-derivative of Robin integral"""
    z_tilde = z + z_0
    
    def imaginary_func(k):
        return -k*np.exp(-k*np.abs(z_tilde))*j1(k*r)*k / (k**2 + alpha**2)
    
    def real_func(k):
        return -k*np.exp(-k*np.abs(z_tilde))*j1(k*r) / (k**2 + alpha**2)
    
    i1 = clenshaw_curtis_half_space(imaginary_func, n)
    i2 = clenshaw_curtis_half_space(real_func, n)
    
    return 2j*alpha*i1 + 2*alpha**2*i2

def robin_integral_z(r, z, z_0, alpha, n):
    """z-derivative of Robin integral"""
    z_tilde = z + z_0
    
    def imaginary_func(k):
        return -2*alpha*k*np.sign(z_tilde)*np.exp(-k*np.abs(z_tilde))*j0(k*r)*k / (k**2 + alpha**2)
    
    def real_func(k):
        return -2*alpha**2*k*np.sign(z_tilde)*np.exp(-k*np.abs(z_tilde))*j0(k*r) / (k**2 + alpha**2)
    
    i1 = clenshaw_curtis_half_space(imaginary_func, n)
    i2 = clenshaw_curtis_half_space(real_func, n)
    
    return 1j*i1 + i2

def clenshaw_curtis_half_space(f, n):
    """Clenshaw-Curtis quadrature for half-space integrals"""
    x, w = ccfi_1(n, 1.0)
    fx = np.array([f(xi) for xi in x])
    return np.dot(w, fx)

def ccfi_1(n, ell):
    """Boyd quadrature rule for Laguerre integral"""
    t = np.pi * np.arange(1, n+1) / (n + 1)
    x = ell * (1.0 / np.tan(0.5 * t))**2
    
    w = np.zeros(n)
    for i in range(n):
        for j in range(1, n+1):
            w[i] += np.sin(j * t[i]) * (1.0 - np.cos(j * np.pi)) / j
    
    w = w * 2.0 * ell * np.sin(t) / (1.0 - np.cos(t))**2 * 2.0 / (n + 1)
    
    return x, w

def get_MFS_solve(E_0, alpha, n, n_hat, x_bdry, x_src, X, Y, Z, L_full, M_full):
    """Main MFS solver function"""
    print("Setting up MFS system...")
    
    M = len(x_bdry)
    N = len(x_src)
    
    Gn_mat = np.zeros((M, N))
    G_mat = np.zeros((M, N))
    bn = np.zeros(M)
    b = np.zeros(M)
    
    print("Building system matrices...")
    for j in range(M):
        if j % 10 == 0:
            print(f'A loop is: {100*j/M:.1f}% done')
        
        x = x_bdry[j, :]
        normal = n_hat[j, :]
        
        bn[j] = E_0 * normal[2]  # d\phi/dn = -d \phi_E / dn
        b[j] = E_0 * x[2]
        
        for i in range(N):
            y = x_src[i, :]
            
            # Compute gradient components
            G_x_val = green_x(x[0], y[0], x[1], y[1], x[2], y[2], alpha, n)
            G_y_val = green_y(x[0], y[0], x[1], y[1], x[2], y[2], alpha, n)
            G_z_val = green_z(x[0], y[0], x[1], y[1], x[2], y[2], alpha, n)
            
            Gn_mat[j, i] = np.dot(normal, [G_x_val, G_y_val, G_z_val])
            G_mat[j, i] = green(x[0], y[0], x[1], y[1], x[2], y[2], alpha, n)
    
    RHS = M_full @ bn - L_full @ b
    A = M_full @ Gn_mat - L_full @ G_mat
    
    print('System matrix built')
    
    # Solve system using QR decomposition
    Q, R = qr(A, mode='economic')
    coefs = np.linalg.solve(R, Q.T @ RHS)
    
    print('System solved')
    
    # Evaluate solution on grid
    Phi = np.zeros_like(X)
    Phi_x = np.zeros_like(X)
    Phi_y = np.zeros_like(X)
    Phi_z = np.zeros_like(X)
    
    K = X.size
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()
    
    print("Evaluating solution...")
    for k in range(K):
        if k % (K//10) == 0:
            print(f'Eval loop is: {100*k/K:.1f}% done')
        
        x_eval = [X_flat[k], Y_flat[k], Z_flat[k]]
        
        # Background field contribution
        Phi.flat[k] = -E_0 * x_eval[2]
        Phi_z.flat[k] = -E_0
        
        # MFS contribution
        for i in range(N):
            y = x_src[i, :]
            
            G_val = green(x_eval[0], y[0], x_eval[1], y[1], x_eval[2], y[2], alpha, n)
            G_x_val = green_x(x_eval[0], y[0], x_eval[1], y[1], x_eval[2], y[2], alpha, n)
            G_y_val = green_y(x_eval[0], y[0], x_eval[1], y[1], x_eval[2], y[2], alpha, n)
            G_z_val = green_z(x_eval[0], y[0], x_eval[1], y[1], x_eval[2], y[2], alpha, n)
            
            Phi.flat[k] += coefs[i] * G_val
            Phi_x.flat[k] += coefs[i] * G_x_val
            Phi_y.flat[k] += coefs[i] * G_y_val
            Phi_z.flat[k] += coefs[i] * G_z_val
    
    return A, RHS, Phi, Phi_x, Phi_y, Phi_z

def get_FEM_Lap(vert, tri):
    """Get FEM Laplacian and mass matrices"""
    ntris = tri.shape[0]
    nvert = vert.shape[0]
    
    row = []
    col = []
    L_val = []
    M_val = []
    
    for k in range(ntris):
        t = tri[k, :]
        
        r1 = vert[t[0], :]  # vertex i
        r2 = vert[t[1], :]  # vertex j
        r3 = vert[t[2], :]  # vertex l
        
        E3 = r1 - r2  # edge 1
        E2 = r3 - r1  # edge 2
        E1 = r2 - r3  # edge 3
        
        A_tri = 0.5 * np.linalg.norm(np.cross(E1, E2))  # area of triangle
        
        # Local stiffness matrix
        A_k = (1.0 / (4.0 * A_tri)) * np.array([
            [np.dot(E1, E1), np.dot(E1, E2), np.dot(E1, E3)],
            [np.dot(E2, E1), np.dot(E2, E2), np.dot(E2, E3)],
            [np.dot(E3, E1), np.dot(E3, E2), np.dot(E3, E3)]
        ])
        
        # Local mass matrix
        M_k = (1.0 / 12.0) * A_tri * np.array([
            [2, 1, 1],
            [1, 2, 1],
            [1, 1, 2]
        ])
        
        # Assemble global matrices
        for rr in range(3):
            for cc in range(3):
                row.append(t[rr])
                col.append(t[cc])
                L_val.append(A_k[rr, cc])
                M_val.append(M_k[rr, cc])
    
    L = sp.csr_matrix((L_val, (row, col)), shape=(nvert, nvert))
    M = sp.csr_matrix((M_val, (row, col)), shape=(nvert, nvert))
    
    return L, M

if __name__ == "__main__":
    main()