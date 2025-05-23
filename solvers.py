import time
import numpy as np
import scipy.sparse.linalg as spla
import pyamg


def solver_EMmid(Nbodies, Nblobs, force_torque_body_function, Slip, cb, Tol, X_Guess, verbose=False):
    #gamma = 1.0
    #Slip[0::3] = -1.0*r_vectors[2::3] * gamma # - gamma

    Nsize = 3 * Nbodies * Nblobs + 6 * Nbodies

    start = time.time()

    force_torque_body = force_torque_body_function(cb=cb)

    assert np.isfinite(Slip).all() # passes
    assert np.isfinite(force_torque_body).all()
    RHS = cb.RHS_and_Midpoint(Slip, force_torque_body)
    assert np.isfinite(Slip).all() # fails
    assert np.isfinite(force_torque_body).all()

    r_vectors_mid = np.array(cb.multi_body_pos())
    r_vectors_mid = np.reshape(r_vectors_mid, (Nbodies*Nblobs, 3))
            
    RHS_norm = np.linalg.norm(RHS)
    end = time.time()
    if verbose:
        print("Time RHS: "+str(end - start)+" s")
            
    A  = spla.LinearOperator((Nsize, Nsize), matvec=cb.apply_Saddle, dtype='float64')
    PC = spla.LinearOperator((Nsize, Nsize), matvec=cb.apply_PC,     dtype='float64')
            
    res_list = []
    start = time.time()

    if X_Guess is not None:
        X_0 = X_Guess/RHS_norm
    else:
        X_0 = None
            
    (Sol, info_precond) = pyamg.krylov.gmres(A, (RHS/RHS_norm), x0=X_0, tol=Tol, M=PC,
                                                        maxiter=min(300, Nsize), restrt=None, residuals=res_list)
            
    end = time.time()
    if verbose:
        print("Time GMRES: "+str(end - start)+" s")
                #print(res_list)
        print('GMRES its: '+str(len(res_list)))
            
            # Extract the velocities from the GMRES solution
    Sol *= RHS_norm
    X_Guess = Sol
    Lambda_s = Sol[0:3*Nbodies*Nblobs]
    U_s = Sol[3*Nbodies*Nblobs::]
    return Lambda_s, U_s, X_Guess


def solver_EMRFD(Nbodies, Nblobs, force_torque_body_function, Slip, cb, dt, kBT, Tol, U_guess, Xs_start, Qs_start, verbose=False):
    start = time.time()

    Nsize = 3 * Nbodies * Nblobs + 6 * Nbodies

    # stack Slip and Force into RHS
    if kBT > 0.0:
        Slip += -1.0*np.sqrt(2*kBT/dt)*cb.M_half_W() #np.random.normal(size=sz)
        ################
    
    force_torque_body = force_torque_body_function(cb=cb)
    RHS = np.concatenate((Slip, -force_torque_body))
            
    RHS_norm = np.linalg.norm(RHS)
    end = time.time()
            #print("Time RHS: "+str(end - start)+" s")
            
    A  = spla.LinearOperator((Nsize, Nsize), matvec=cb.apply_Saddle, dtype='float64')
    PC = spla.LinearOperator((Nsize, Nsize), matvec=cb.apply_PC,     dtype='float64')
            
    res_list = []
    start = time.time()

    X_guess = None
    if U_guess is not None:
         X_guess = U_guess/RHS_norm
            
    (Sol_Brownian, info_Brownian) = pyamg.krylov.gmres(A, (RHS/RHS_norm), x0=X_guess, tol=Tol, M=PC,
                                                        maxiter=min(300, Nsize), restrt=None, residuals=res_list)
            
    end = time.time()
            #print("Time GMRES: "+str(end - start)+" s")
            #print(res_list)
    if verbose:
        print('GMRES its: '+str(len(res_list)))
            
            # Extract the velocities from the GMRES solution
    Sol_Brownian *= RHS_norm
    Lambda_Brownian = Sol_Brownian[0:3*Nbodies*Nblobs] # Don't care - what does that mean?!
    U_Brownian = Sol_Brownian[3*Nbodies*Nblobs::] # N*F - N*K^T*M^{-1}*Slip + sqrt(2*kBT/dt)*N^{1/2}*W

    U_guess = Sol_Brownian
            #print('U_Brownian: ',U_Brownian)
    U_RFD = np.zeros(U_Brownian.shape)

    if kBT > 0.0: #False: #
        W_rfd = np.random.normal(size=6*Nbodies)
        delta_rfd = 1e-4
        U_plus = 0.5*delta_rfd*W_rfd
        U_minus = -0.5*delta_rfd*W_rfd
        RHS_RFD = np.concatenate((np.zeros(3*Nbodies*Nblobs), -1.0*W_rfd))

        # positivly perturb the system for the RFD solve
        cb.evolve_X_Q_RFD(U_plus)
        # RFD solve #2
        A = spla.LinearOperator((Nsize, Nsize), matvec=cb.apply_Saddle, dtype='float64')
        #PC = spla.LinearOperator((Nsize, Nsize), matvec=cb.apply_PC, dtype='float64')
        res_list = []
        start = time.time()
        (Sol_Plus, info_Plus) = pyamg.krylov.gmres(A, RHS_RFD, x0=None, tol=Tol, M=PC,
                                                            maxiter=min(300, Nsize), restrt=None, residuals=res_list)
        end = time.time()
        # Extract the velocities from the GMRES solution
        U_plus = Sol_Plus[3*Nbodies*Nblobs:] # N_{+}*W_RFD
        #print("Time GMRES plus: "+str(end - start)+" s")
        if verbose:
            print('GMRES its plus: '+str(len(res_list)))

        # set configuration back to original
        cb.setConfig(Xs_start, Qs_start)
        cb.set_K_mats()

        # negativly perturb the system for the RFD solve
        cb.evolve_X_Q_RFD(U_minus)
        # RFD solve #3
        A = spla.LinearOperator((Nsize, Nsize), matvec=cb.apply_Saddle, dtype='float64')
        #PC = spla.LinearOperator((Nsize, Nsize), matvec=cb.apply_PC, dtype='float64')
        res_list = []
        start = time.time()
        (Sol_Minus, info_Minus) = pyamg.krylov.gmres(A, RHS_RFD, x0=Sol_Plus, tol=Tol, M=PC,
                                                            maxiter=min(300, Nsize), restrt=None, residuals=res_list)
        end = time.time()
        # Extract the velocities from the GMRES solution
        U_Minus = Sol_Minus[3*Nbodies*Nblobs:] # N_{-}*W_RFD
        #print("Time GMRES minus: "+str(end - start)+" s")
        if verbose:
            print('GMRES its minus: '+str(len(res_list)))

        # set configuration back to original
        cb.setConfig(Xs_start,Qs_start)
        cb.set_K_mats()

        # RFD velocity (make sure to multiply by kBT)
        U_RFD = (kBT/delta_rfd)*(U_plus - U_Minus)  

    # full velocity
    U_full = U_Brownian + U_RFD
    return Lambda_Brownian, U_full, U_guess

def solver_trap(Nbodies, Nblobs, force_torque_body_function, Slip, cb, dt, kBT, Tol, Xs_start, Qs_start, verbose=False):

    Nsize = 3 * Nbodies * Nblobs + 6 * Nbodies
    
    ########################################################
    ################## Step 1 ##############################
    ########################################################
    # get Random rigid velocity for RFD
    if verbose:
        print('Step 1')

    Slip_noforces = np.random.randn(3*Nbodies*Nblobs)
    Force_noforces = np.zeros(6*Nbodies)

    start = time.time()
    
    RHS = np.concatenate((-Slip_noforces, -Force_noforces))
    
    RHS_norm = np.linalg.norm(RHS)
    end = time.time()
    #print("Time RHS: "+str(end - start)+" s")
    
    A = spla.LinearOperator((Nsize, Nsize), matvec=cb.apply_Saddle, dtype='float64')
    PC = spla.LinearOperator((Nsize, Nsize), matvec=cb.apply_PC, dtype='float64')
    
    res_list = []
    start = time.time()
    
    (Sol_RFD, info_RFD) = pyamg.krylov.gmres(A, (RHS/RHS_norm), x0=None, tol=Tol, M=PC, #
                                                maxiter=min(300, Nsize), restrt=None, residuals=res_list)
    
    end = time.time()
    #print("Time GMRES: "+str(end - start)+" s")
    #print(res_list)
    if verbose:
        print('GMRES its: '+str(len(res_list)))
    
    # Extract the velocities from the GMRES solution
    Sol_RFD *= RHS_norm
    Lambda_RFD = Sol_RFD[0:3*Nbodies*Nblobs] # Don't care
    U_RFD = Sol_RFD[3*Nbodies*Nblobs::] # U =  N*K^T*M^{-1}*W 

    KT_RFD = cb.KT_RFD_from_U(U_RFD, Slip)
    M_RFD = cb.M_RFD_from_U(U_RFD, Slip)

    # print('KT_RFD: ',np.linalg.norm(KT_RFD))
    # print('M_RFD: ',np.linalg.norm(M_RFD))
    # print('U_RFD: ',U_RFD)
    
    ########################################################
    ################## Step 2 ##############################
    ########################################################
    # Predictor step
    if verbose:
        print('Step 2')
    ### Slip to be used in both steps
    Slip = np.sqrt(2*kBT/dt)*cb.M_half_W()
    ### Forces at Q^n
    force_torque_body = force_torque_body_function(cb=cb)

    RHS = np.concatenate((-Slip, -force_torque_body))
    RHS_norm = np.linalg.norm(RHS)
    res_list = []
    start = time.time()
    
    
    (Sol_Pred, info_Pred) = pyamg.krylov.gmres(A, (RHS/RHS_norm), x0=None, tol=Tol, M=PC, #
                                                maxiter=min(300, Nsize), restrt=None, residuals=res_list)
    
    end = time.time()
    #print("Time GMRES: "+str(end - start)+" s")
    #print(res_list)
    if verbose:
        print('GMRES its: '+str(len(res_list)))
    
    # Extract the velocities from the GMRES solution
    Sol_Pred *= RHS_norm
    Lambda_Pred = Sol_Pred[0:3*Nbodies*Nblobs] # Don't care
    U_Pred = Sol_Pred[3*Nbodies*Nblobs::] # N*F + sqrt(2*kBT/dt)*N^{1/2}*W


    cb.evolve_X_Q(U_Pred)


    ########################################################
    ################## Step 3 ##############################
    ########################################################
    # Corrector step
    ### Update slip from predictor step
    if verbose:
        print('Step 3')
    Slip += 2.0*kBT*M_RFD
    ### Forces at Q^n+1/2
    force_torque_body = force_torque_body_function(cb=cb)
    force_torque_body += -2.0*kBT*KT_RFD # can somebody explain this?


    RHS = np.concatenate((-Slip, -force_torque_body))
    RHS_norm = np.linalg.norm(RHS)

    res_list = []
    start = time.time()
    
    (Sol_Corr, info_Corr) = pyamg.krylov.gmres(A, (RHS/RHS_norm), x0=(Sol_Pred/RHS_norm), tol=Tol, M=PC, #
                                                maxiter=min(300, Nsize), restrt=None, residuals=res_list)
    
    end = time.time()
    #print("Time GMRES: "+str(end - start)+" s")
    #print(res_list)
    if verbose:
        print('GMRES its: '+str(len(res_list)))
    
    # Extract the velocities from the GMRES solution
    Sol_Corr *= RHS_norm
    Lambda_Corr = Sol_Corr[0:3*Nbodies*Nblobs] # Don't care
    U_Corr = Sol_Corr[3*Nbodies*Nblobs::] # Corrector veloctity



    cb.setConfig(Xs_start,Qs_start)
    cb.set_K_mats() 

    # full velocity
    U_full = 0.5*(U_Pred + U_Corr)

    return Lambda_Corr, U_full


def solver_trap_libmobility(Nbodies, Nblobs, force_torque_body_function, Slip, cb, dt, kBT, Tol, Xs_start, Qs_start, libmobility_solver, verbose=False):
    Nsize = 3 * Nbodies * Nblobs + 6 * Nbodies
    sz = 3 * Nbodies * Nblobs # can we please come up with a better name for this?

    ########################################################
    ################## Step 1 ##############################
    ########################################################

    # get Random rigid velocity for RFD
    if verbose:
        print('Step 1')
    Slip = np.random.randn(sz)
    Force = np.zeros(6*Nbodies)

    start = time.time()
    
    RHS = np.concatenate((-Slip, -Force))
    
    RHS_norm = np.linalg.norm(RHS)
    end = time.time()
    #print("Time RHS: "+str(end - start)+" s")
    
    libmobility_solver.setPositions(r_vectors)
    def apply_Saddle_mdot(x):
        out = 0*x
        Lam = x[0:sz]
        U = x[sz::]
        vels, _ = libmobility_solver.Mdot(Lam) 
        out[0:sz] = vels - cb.K_x_U(U)
        out[sz::] = cb.KT_x_Lam(Lam)
        out[sz::] *= -1.0
        return out
    
    A = spla.LinearOperator((Nsize, Nsize), matvec=apply_Saddle_mdot, dtype='float64')
    PC = spla.LinearOperator((Nsize, Nsize), matvec=cb.apply_PC, dtype='float64')
    
    res_list = []
    start = time.time()
    
    (Sol_RFD, info_RFD) = pyamg.krylov.gmres(A, (RHS/RHS_norm), x0=None, tol=Tol, M=PC, #
                                                maxiter=min(300, Nsize), restrt=None, residuals=res_list)
    
    end = time.time()
    if verbose:
        print("Time GMRES: "+str(end - start)+" s")
        #print(res_list)
        print('GMRES its: '+str(len(res_list)))
    
    # Extract the velocities from the GMRES solution
    Sol_RFD *= RHS_norm
    Lambda_RFD = Sol_RFD[0:sz] # Don't care
    U_RFD = Sol_RFD[sz::] # U =  N*K^T*M^{-1}*W 

    ########## RFD ##########
    delta_rfd = 1.0e-3
    r_vec_p,r_vec_m = cb.M_RFD_cfgs(U_RFD,delta_rfd)
    r_vec_p = np.array(r_vec_p)
    r_vec_m = np.array(r_vec_m)
    libmobility_solver.setPositions(r_vec_p)
    MpW, _ = libmobility_solver.Mdot(Slip)
    libmobility_solver.setPositions(r_vec_m)
    MmW, _ = libmobility_solver.Mdot(Slip)
    libmobility_solver.setPositions(r_vectors)
    M_RFD = (1.0/delta_rfd)*(MpW - MmW)
    KT_RFD = cb.KT_RFD_from_U(U_RFD,Slip)

    # print('KT_RFD: ',np.linalg.norm(KT_RFD))
    # print('M_RFD: ',np.linalg.norm(M_RFD))
    # print('U_RFD: ',U_RFD)
    
    ########################################################
    ################## Step 2 ##############################
    ########################################################
    # Predictor step
    if verbose:
        print('Step 2')
    ### Slip to be used in both steps
    Slip, _ = libmobility_solver.sqrtMdotW()
    Slip *= np.sqrt(2*kBT/dt)
    ### Forces at Q^n
    Force = force_torque_body_function(cb=cb)

    RHS = np.concatenate((-Slip, -Force))
    RHS_norm = np.linalg.norm(RHS)
    res_list = []
    start = time.time()
    
    
    (Sol_Pred, info_Pred) = pyamg.krylov.gmres(A, (RHS/RHS_norm), x0=None, tol=Tol, M=PC, #
                                                maxiter=min(300, Nsize), restrt=None, residuals=res_list)
    
    end = time.time()
    if verbose:
        print("Time GMRES: "+str(end - start)+" s")
        #print(res_list)
        print('GMRES its: '+str(len(res_list)))
    
    # Extract the velocities from the GMRES solution
    Sol_Pred *= RHS_norm
    Lambda_Pred = Sol_Pred[0:sz] # Don't care
    U_Pred = Sol_Pred[sz::] # N*F + sqrt(2*kBT/dt)*N^{1/2}*W


    cb.evolve_X_Q(U_Pred)


    ########################################################
    ################## Step 3 ##############################
    ########################################################
    # Corrector step
    ### Update slip from predictor step
    if verbose:
        print('Step 3')
    Slip += 2.0*kBT*M_RFD
    ### Forces at Q^n+1/2
    r_vectors = np.array(cb.multi_body_pos())
    Force = force_torque_body_function(cb=cb)
    Force += -2.0*kBT*KT_RFD


    RHS = np.concatenate((-Slip, -Force))
    RHS_norm = np.linalg.norm(RHS)

    libmobility_solver.setPositions(r_vectors)
    A = spla.LinearOperator((Nsize, Nsize), matvec=apply_Saddle_mdot, dtype='float64')

    res_list = []
    start = time.time()
    
    (Sol_Corr, info_Corr) = pyamg.krylov.gmres(A, (RHS/RHS_norm), x0=(Sol_Pred/RHS_norm), tol=Tol, M=PC, #
                                                maxiter=min(300, Nsize), restrt=None, residuals=res_list)
    
    end = time.time()
    if verbose:
        print("Time GMRES: "+str(end - start)+" s")
        #print(res_list)
        print('GMRES its: '+str(len(res_list)))
    
    # Extract the velocities from the GMRES solution
    Sol_Corr *= RHS_norm
    Lambda_Corr = Sol_Corr[0:sz] # Don't care
    U_Corr = Sol_Corr[sz::] # Corrector veloctity

    cb.setConfig(Xs_start,Qs_start)
    cb.set_K_mats() 

    # full velocity
    U_full = 0.5*(U_Pred + U_Corr)

    return Lambda_Corr, U_full