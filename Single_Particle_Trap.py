import numpy as np
import scipy
import single_particle
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as pyrot


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


if __name__ == '__main__':
        
    dt = 1e-2

    data, metadata = single_particle.run(
        t_max  = 50, # was 2000
        dt     = dt,
        t_save = dt,
        method = 'Trap',
        struct_file_to_load = 162,
    )

    h_coord = data[:, 2]

    # make a histogram of the h_coord
    plt.figure()
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
    Radius = metadata['particle_diameter'] / 2
    a      = metadata['blob_diameter'] / 2
    atto = 1e-18
    kB = scipy.constants.k / atto
    kBT = kB * metadata['T']

    Ph,hs = particle_wall_energy(metadata['initial_blob_coords'], Radius, a, metadata['debye_length'], metadata['repulsion_strength'],
                                 metadata['f_gravity_mag'], kBT, heights=(Radius*bin_locs))
    # bin_h = (hs[1] - hs[0])/Radius
    # Ph = Ph/np.trapz(Ph, dx=bin_h)
    plt.plot(hs, Ph, 'r-', label='Theory')

    Ph_avg, hs_avg = particle_wall_energy_cfg_avg(metadata['initial_blob_coords'], Radius, a, metadata['debye_length'], metadata['repulsion_strength'],
                                                  metadata['f_gravity_mag'], kBT, heights=(Radius*bin_locs))
    plt.plot(bin_centers, counts, 'b--', label='Histogram', linewidth=4)
    plt.plot(hs_avg, Ph_avg, 'k-', label='Histogram', linewidth=4)

    plt.xlabel('h (um)')
    plt.ylabel('Count')
    plt.show()

    # save the data to a file
    np.savetxt('./Hist_Data/42_blob_dt_1em2_h_coord.txt', h_coord)

    print('normalization of Ph:', np.trapz(Ph, dx=bin_dx))