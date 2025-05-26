import time, copy, sys
import numpy as np
import solvers
import scipy

# find c++ functions
sys.path.append('../')
sys.path.append('./build/')
import c_rigid_obj

# we use tqdm for nice progress bars if it is available
try:
    import tqdm
    import functools
    progressbar = functools.partial(tqdm.tqdm)
except ImportError:
    class progressbar:
        def __init__(*args, **kwargs):
            pass
        def update(*args, **kwargs):
            pass
        def close(*args, **kwargs):
            pass


struct_files = {
    42: './Structures/shell_N_42_Rg_0_8913_Rh_1.vertex',
    162: './Structures/shell_N_162_Rg_0_9497_Rh_1.vertex',
    642: './Structures/shell_N_642_Rg_0_9767_Rh_1.vertex',
    2562: './Structures/shell_N_2562_Rg_0_9888_Rh_1.vertex',
}

def load_struct_data(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        # s should be a float
        s = float(lines[0].split()[1]) # what is s
        blob_coords = np.array([[float(j) for j in i.split()] for i in lines[1:]])
    return s, blob_coords

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

def calc_force_torque_body(gravity, wall, f_gravity_mag, theta, blob_radius, debye_length, repulsion_strength, cb, Nbodies):
    force_torque_body = np.zeros((Nbodies, 2, 3)) # axis=0: body, axis=1: force/torque, axis=2: x/y/z

    blob_coords = np.array(cb.multi_body_pos())

    ### add gravity
    if gravity:
        f_gravity_x = - f_gravity_mag * np.sin(theta * np.pi / 180)
        f_gravity_z = - f_gravity_mag * np.cos(theta * np.pi / 180)
        assert f_gravity_z <= 0
        force_torque_body[:, 0, 0] += f_gravity_x # [:, 0, 0] is all bodies, force, x
        force_torque_body[:, 0, 2] += f_gravity_z # [:, 0, 2] is all bodies, force, z

    ### add sterics with wall
    if wall:
        wall_force_temp = wall_force_blobs(blob_coords, blob_radius, debye_length, repulsion_strength)
        wall_force = np.reshape(cb.KT_x_Lam(wall_force_temp), (Nbodies, 2, 3))
        force_torque_body += wall_force

    return force_torque_body.flatten()

def run(t_max, t_save, dt=1e-2, shear_size=0.0, gravity=True, theta=0, wall=True, T=298,
        method='EMmid', struct_file_to_load=42, skip_seconds=2, verbose=False, Nbodies=1,
        libmobility_solver=None, body_coords=None):
    
    # parameter checking
    if Nbodies > 1:
        assert method in ['Trap']
    
    if body_coords is not None:
        assert body_coords.shape[0] == Nbodies
        assert body_coords.shape[1] == 3

    hydrodynamic_diameter = 2.972 # um
    bare_diameter = 2.82 # um
    hydrodynamic_radius = hydrodynamic_diameter / 2
    bare_radius = bare_diameter / 2

    metadata = dict(
        dt = dt,
        shear = shear_size,
        gravity = gravity,
        theta = theta,
        wall = wall,
        T = T,
        method = method,
        particle_diameter = hydrodynamic_diameter,
        bare_particle_diameter = bare_diameter,
        time_step = t_save
    )

    print('body_radius', hydrodynamic_radius)
    print(f'theta = {theta} = {theta * np.pi / 180}')

    struct_file = struct_files[struct_file_to_load]
    metadata['num_blobs'] = struct_file_to_load
    
    blob_diameter, initial_blob_coords = load_struct_data(struct_file)
    Nblobs_per_body = len(initial_blob_coords)

    blob_diameter       *= hydrodynamic_radius
    initial_blob_coords *= hydrodynamic_radius
    metadata['initial_blob_coords'] = initial_blob_coords
    metadata['blob_diameter'] = blob_diameter

    # Set some variables for the simulation
    blob_radius = 0.5*blob_diameter

    # Create rigid bodies
    X_0 = []
    Quat = []

    default_struct_location = np.array([0, 0, 1.2]) * hydrodynamic_radius
    default_struct_orientation = np.array([1.0, 0.0, 0.0, 0.0])

    if Nbodies == 1:
        X_0 = [default_struct_location]
        Quat = [default_struct_orientation]

    elif body_coords is not None:
        X_0 = body_coords
        Quat = [default_struct_orientation] * Nbodies
    
    else:
        for k in range(Nbodies):
            # random x y  location in the range -1e6 to 1e6
            xy_loc = np.random.rand(2)*1.0e3
            xy_loc = 2.0*xy_loc - 1e3
            zshift = np.random.rand()*0.4*hydrodynamic_radius
            zshift -= 0.2*hydrodynamic_radius # why are we doing this?
            loc = default_struct_location + np.array([xy_loc[0],xy_loc[1],zshift])
            X_0.append(loc)
            Quat.append(default_struct_orientation)

    X_0 = np.array(X_0).flatten()
    Quat = np.array(Quat).flatten()
    
        
    # misc. parameters
    eta = 1.75e-03 # viscosity (Pa*s), from Eleanor in Slack
    atto = 1e-18
    k_B = scipy.constants.k / atto
    kT = k_B * T # in aJ
    metadata['eta'] = eta

    # calculate magnitude of gravity force
    body_volume = 4/3*np.pi*bare_radius**3 # um^3
    rho_particles = 1510 # kg/m^3 (collective diffusion paper SI)
    rho_water = 970 # kg/m^3 (Eleanor in slack)
    delta_rho = rho_particles - rho_water # kg/m^3
    delta_rho *= (1e-6)**3 # kg/um^3
    effective_mass = delta_rho * body_volume # kg
    g = scipy.constants.g # ms^-2
    f_gravity_mag = effective_mass * g # N
    f_gravity_mag *= 1e12 # pN. approx 16kT
    metadata['delta_rho'] = delta_rho
    metadata['f_gravity_mag'] = f_gravity_mag

    # other parameters
    # debye_length = 0.1*blob_radius
    debye_length = 0.0122 * 120/170 # 0.1*blob_radius for 42 blobs, diameter=2.92. 120/170 b/c that gave d=170nm, but Eleanor says it's 120nm
    print('debye length', debye_length)
    print('debye ratio', 0.1395/debye_length)
    repulsion_strength = 4.0 * 296 * k_B # 2D monolayer sims used 0.0163 which is 0.98*4*kT
    # print('repulsion ratio', 0.0163/repulsion_strength)
    Tol = 1.0e-3
    periodic_length = np.array([0.0, 0.0, 0.0])

    metadata['debye_length'] = debye_length
    metadata['repulsion_strength'] = repulsion_strength
    metadata['gmres_tol'] = Tol
    
    
    print('a is: '+str(blob_radius))
    print('diffusive blob timestep is: '+str(kT*dt/(6*np.pi*eta*blob_radius**3)))

    
    # Make solver object
    cb = c_rigid_obj.CManyBodies()
    
    # Sets the PC type
    # If true will use the block diag 'Dilute suspension approximation' to M
    # For dense suspensions this is a bad approximation (try setting false)
    # for rigid bodies with lots of blobs this is expensive (try setting flase)
    cb.setBlkPC(False)

    # Set the domain to have a wall
    cb.setWallPC(True)
    
    
    numParts = Nbodies*len(initial_blob_coords)
    cb.setParameters(numParts, blob_radius, dt, kT, eta, periodic_length, initial_blob_coords)
    cb.setConfig(X_0, Quat)
    print('set config')
    cb.set_K_mats()
    print('set K mats')
    
    
    num_blob_coords = 3*Nbodies*len(initial_blob_coords)
    Nsize = num_blob_coords + 6*Nbodies
    
    num_rejects = 0
    Sol = np.zeros(Nsize)
        
    # Qs, Xs = cb.getConfig()
    # r_vectors = np.array(cb.multi_body_pos())
    # FT = np.zeros((2, 3))
    
    # Force = FT.flatten()
    # Slip = np.zeros(sz)
    # gamma = 1.0
        
    num_rejects = 0

    print(f'shear: {shear_size}, gravity: {gravity}, wall: {wall}, kT: {kT}')

    n_steps = int(t_max  / dt)
    n_save  = int(t_save / dt)
    skip_timesteps = skip_seconds / dt
    skip_saves = skip_timesteps / n_save
    num_output_rows = Nbodies * int(n_steps / n_save - 1 - skip_saves)
    particles = np.full((num_output_rows, 9), np.nan)
    print(f'particles array size, {particles.nbytes/1e9:.1f}GB')
    print('particles shape', particles.shape)

    gmres_guess = None

    if Nbodies > 1:
        if method == 'Trap':
            assert libmobility_solver is not None

            libmobility_solver.initialize(
                temperature        = kT,
                viscosity          = eta,
                hydrodynamicRadius = hydrodynamic_radius,
                needsTorque        = False
            )

    
    # we need to let the solver calculate the forces b/c the Trap solver calcluates forces for different configurations
    # so we don't calculate it here, but make it easy for the solver by giving it a functions with the parameters mostly already bound
    force_torque_body_function = functools.partial(
        calc_force_torque_body,
        gravity            = gravity,
        wall               = wall,
        f_gravity_mag      = f_gravity_mag,
        theta              = theta,
        blob_radius        = blob_radius,
        debye_length       = debye_length,
        repulsion_strength = repulsion_strength,
        Nbodies            = Nbodies,
    )

    progress = progressbar(total=n_steps, mininterval=10)
    n = 0
    while n < n_steps:
        # we use a while not a for, because if the step gets retried we don't want n to increment
        
        Slip = np.zeros(num_blob_coords)

        blob_coords = np.array(cb.multi_body_pos())
        Qs, Xs = cb.getConfig()
        
        Xs_start = copy.deepcopy(Xs) # we copy these now so we can use them to
        Qs_start = copy.deepcopy(Qs) # reset the configuration after the RFD, or if we get a bad timestep

        if not gravity and not wall and shear_size == 0 and kT == 0:
            assert np.all(Xs == X_0)

        ### add shear
        Slip[0::3] = -1.0*blob_coords[2::3] * shear_size

        ### use the solver to get the velocities from the forces
        if method == 'EMmid':
            Lambda, U_s, gmres_guess = solvers.solver_EMmid(
                Nbodies                    = Nbodies,
                Nblobs                     = Nblobs_per_body,
                force_torque_body_function = force_torque_body_function,
                Slip                       = Slip,
                cb                         = cb,
                Tol                        = Tol,
                X_Guess                    = gmres_guess,
                verbose                    = verbose
            )

        elif method.startswith('EMRFD'):
            Lambda, U_s, gmres_guess = solvers.solver_EMRFD(
                Nbodies                    = Nbodies,
                Nblobs                     = Nblobs_per_body,
                force_torque_body_function = force_torque_body_function,
                Slip                       = Slip,
                cb                         = cb,
                dt                         = dt,
                kBT                        = kT,
                Tol                        = Tol,
                U_guess                    = gmres_guess,
                Xs_start                   = Xs_start,
                Qs_start                   = Qs_start,
                verbose                    = verbose
            )

        elif method == 'Trap':
            if Nbodies == 1:
                Lambda, U_s = solvers.solver_trap(
                    Nbodies                    = Nbodies,
                    Nblobs                     = Nblobs_per_body,
                    force_torque_body_function = force_torque_body_function,
                    Slip                       = Slip,
                    cb                         = cb,
                    dt                         = dt,
                    kBT                        = kT,
                    Tol                        = Tol,
                    Xs_start                   = Xs_start,
                    Qs_start                   = Qs_start,
                    verbose                    = verbose,
                )

            else:
                Lambda, U_s = solvers.solver_trap_libmobility(
                    Nbodies                    = Nbodies,
                    Nblobs                     = Nblobs_per_body,
                    force_torque_body_function = force_torque_body_function,
                    Slip                       = Slip,
                    cb                         = cb,
                    dt                         = dt,
                    kBT                        = kT,
                    Tol                        = Tol,
                    Xs_start                   = Xs_start,
                    Qs_start                   = Qs_start,
                    libmobility_solver         = libmobility_solver,
                    verbose                    = verbose,
                )
        
        else:
            raise Exception(f'unknown method {method}')
            
        if np.any(np.abs(U_s).max() > 1e5):
            print('U_s has elements greater than 1e5')

        # evolve rigid bodies (use the velocities to update the positions and orientations)
        cb.evolve_X_Q(U_s)
        blob_coords_new = np.array(cb.multi_body_pos())
        Qs_new, Xs_new = cb.getConfig()

        if num_rejects > 100 and num_rejects > n:
            print(f'{num_rejects} rejects, {n} successes. Stopping.')
            break
        
        if dist := np.linalg.norm(blob_coords - blob_coords_new, ord=np.inf) > 4*blob_radius: # does this work when blob_coords is across multiple bodies? do we not need to specify an axis or something?
            num_rejects += 1
            print(f'Bad timestep! ({n})')
            cb.setConfig(Xs_start, Qs_start) # reset the positions and orientations
            continue

        if fraction_unchanged := np.sum(Xs_start == Xs_new) / Xs_start.size > 0.2:
            num_rejects += 1
            print(f'{fraction_unchanged*100}% of blob coords did not change')
            cb.setConfig(Xs_start, Qs_start) # reset the positions and orientations
            continue

        # save data
        if n % n_save == 0 and n > skip_timesteps:
            frame = int((n - skip_timesteps) / n_save) - 1
            row_i = frame * Nbodies
            assert row_i >= 0
            
            particles[row_i:row_i+Nbodies, [0, 1, 2]] = Xs_new.reshape((Nbodies, 3)) # Xs is [x0, y0, z0, x1, y1, z1, ...]
            particles[row_i:row_i+Nbodies, 3] = frame
            particles[row_i:row_i+Nbodies, 4] = np.arange(0, Nbodies) # particle ID
            particles[row_i:row_i+Nbodies, [5, 6, 7, 8]] = Qs_new.reshape((Nbodies, 4))

        n += 1
        progress.update()

    print(f'num_rejects: {num_rejects}')
    
    return particles, metadata