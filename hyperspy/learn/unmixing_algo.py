"""Code Description: MVSA estimates the vertices M = {m_1, ...m_p} of the (p-1)-dimensional simplex of minimum volume
containing the vectors [y_1, y_2,...y_N], under the assumption that y_i belongs to a (p-1) dimensional affine set.
                                                y_i = M*x_i
                                            where x_i belongs to the probability (p-1) - simplex
                                            """

"""Steps to obtain the matrix M:
    1) Project y onto a p-dimensional subspace containing the data set y
        y_p = U_p(transpose) * y    (where Up is an orthonormal matrix)
    2) solve the optimization problem
        Q^* = arg_min Q  -\log abs(det(Q))
        subject to: Q*yp >= 0 and ones(1,p)*Q=mq,
        where mq = ones(1,N)*yp'inv(yp*yp)
    3) Compute
        M = U_p*inv(Q^*)"""

"""This Python script attempts to replicate the MVSA code developed via Matlab. Certain segments of the code would be
removed due to the difference between Python and Matlab."""

#-------------------------------MVSA ALgorithm ------------------------------------------------------------#
def mvsa(X, p, MMilters = 10, Spherize = 'yes', verbose = 1, Lambda = 1e-10, mu = 1e-6, M = 0, tol_f = 1e-2):

    """Input Y: matrix with L(channels) X N(pixels). Each pixel is a linear mixture of p endmembers,
    Input p: number of independent columns of M."""
    from scipy.sparse import linalg
    import math
    import numpy as np
    """ Defaults parameters for optional variables
    Maximum of number of OPs
        MMilters = 10
        spherize = 'yes'
    Display only MVSA warnings
        verbose = 1
    Spherization regularization parameter
        Lambda = 1e-10
    Quadractic regularization parameter for the Hesssian
        Hreg = = mu * I + H
        mu = 1e-6
    No initial simplex
        M = 0
    Tolerance for the termination test
        tol_f = 1e-2
    """
    """Local Variables"""
    slack = 1e-3  # maximum violation of inequalities
    any_energy_decreasing = 0  # flag energy decreasing
    f_val_back = float("inf")  # equivalence to the double scalar infinity in Matlab

    """Set Display Mode"""
    # Algorithm unclear. Return to this later.
    Y = np.array(X, copy = True)
    Y = Y.T
    #data set size
    L, N = np.shape(Y)
    if (L < p):
        raise ValueError("Insufficient number of columns in y")
    if p == 0:
        p = L - 1
    # Identifying the affine set that best represents the data set Y
    my = np.mean(Y, axis=1)
    Y = Y - np.tile(my[:, np.newaxis], N)
    Y_transposed = np.transpose(Y)
    a_temp = np.dot(Y, Y_transposed)
    a_temp = a_temp/N
    p_temp = p-1
    Up, D, scrap_value = linalg.svds(a_temp,k=p_temp)
    Up = np.fliplr(Up)*(-1)
    D = D[::-1]
    # represent Y  in subspace R^(p-1)
    Y = np.dot(np.dot(Up, Up.T),Y)
    # lift Y
    Y = Y + np.tile(my[:, np.newaxis], N)
    # compute the orthogonal component of my
    my_ortho = my - np.dot(np.dot(Up, Up.T), my)
    # define another orthonormal direction
    Up = np.append(Up, np.expand_dims(my_ortho/math.sqrt(sum(my_ortho**2)), axis = 1), axis =1)
    sing_values = np.array(D, copy = True)
    D = np.diag(D)
    # gets coordinate in R^p
    Y = np.dot(Up.T, Y)

    if Spherize == 'yes':
        Y = np.dot(Up, Y)
        Y = Y - np.tile(my[:, np.newaxis], N)

        temp_1 = D + Lambda*np.identity(p-1)
        temp_1 = np.sqrt(temp_1.diagonal())
        temp_1 = 1/temp_1
        C = np.diag(temp_1)
        Y = np.dot(np.dot(C, Up[:,:p-1].T),Y)
        # lift
        temp_array_1 = np.ones((1,np.shape(Y)[1]))
        Y = np.append(Y, temp_array_1, axis = 0)
        # normalize to unit norm
        Y = Y/math.sqrt(p)

    # Initialization
    if M == 0:
            M, indice, Rp, loadings = VCA(Y,p)  # Initialize with VCA
            # expand Q
            Ym = np.mean(M, axis = 1)
            Ym  = np.tile(Ym[:, np.newaxis],p)
            dQ = M - Ym
            # fraction: multiply by p is to make sure Q0 starts with
            # a feasible initial value
            M = M + p*dQ
    else:
            M = M - np.tile(my[:, np.newaxis], p)
            M = np.dot(np.dot(Up[:,:p-1],Up[:,:p-1].T), M)
            M = M + np.tile(my[:, np.newaxis], p)
            M = np.dot(Up.T, M) # represent the data in subspace
            # If spherization is set
            if Spherize == 'yes':
                M = np.dot(Up, M) - np.tile(my[:, np.newaxis], p)
                M = np.dot(np.dot(C, Up[:,:p-1].T), M)
                # lift
                M[p-1, :] = 1
                # normalize to unit norm
                M = M/math.sqrt(p)
    Q0 = np.linalg.inv(M)
    Q = Q0

    # Build Constraint Matrices
    A = np.kron(Y.T, np.identity(p)) # inequality matrix size np * p^2
    E = np.kron(np.identity(p), np.ones((1,p)))
    # Equality Independent Vector
    qm = np.sum(np.dot(np.linalg.inv(np.dot(Y,   Y.T)), Y), axis = 1)[:, np.newaxis]
    print("qm value is: ", qm.shape)

    """Sequence of QPs  - main body """
    throtle = 0
    for k in range(1, MMilters+1):
        M = np.linalg.inv(Q) # make initial point feasible
        Ym = np.mean(M, axis = 1)
        Ym = np.tile(Ym[:, np.newaxis],p)
        dW = M - Ym
        count = 0
        print("Shape of M is: ", M.shape)
        if not throtle:
            while np.sum(np.sum(np.dot(np.linalg.inv(M), Y) < 0, axis = 0), axis = 0)> 0:
                print('\n making M feasible...\n')
                M = M + 0.01*dW
                count = count + 1
                if count > 100:
                    if verbose:
                        print('\n could not make M feasible after 100 expansions \n')
                    break
        Q = np.linalg.inv(M)
        # gradient of -log(abs(det(Q)))
        g = -M.T
        g = g.ravel(order='F')[:, np.newaxis]

        # quadratic term mu*I + diag(H)

        H = mu*np.identity(p**2) + np.diagonal(g**2)
        q0 = Q.flatten(order='F')[:, np.newaxis]
        print("Shape of q0: ", q0.shape)
        Q0 = Q
        f = g - np.dot(H, q0)

        # initial function value
        f0_val = -math.log(abs(np.linalg.det(Q0)))
        f0_quad = f0_val # (q-q0)'*g+1/2*(q-q0)'*H*(q-q0);

        # energy decreasing in this quadratic problem
        energy_decreasing = 0

        # QP

        q = quadProgl(H, f, Y, np.zeros((p*N, 1)), E, qm, q0, throtle)

        exitflag = 1

        if exitflag < 0: #This segment is not needed from the Matlab Source code
            pass

        if energy_decreasing or (exitflag ==1):
            Q = np.reshape(q, (p,p), order = 'F')
            # f bound
            f_val = -math.log(abs(np.linalg.det(Q)))
            if verbose:
                print('\n iter = %d, f0 = %2.4f, f = %2.4f, exitflag = %d \n'%(k, f0_val, f_val, exitflag))

            # line search
            if f0_val < f_val:
                Q = Q0
                if throtle != 1:
                    throtle = 1
            else:
                throtle = 0

            energy_decreasing = 1
        if energy_decreasing:
            if abs((f_val_back - f_val)/f_val) < tol_f and not throtle:
                if verbose:
                    print('\n iter: = %d termination test PASSED\n'%k)
            f_val_back = f_val

    if Spherize == 'yes':
        M = np.linalg.inv(Q)
        M = M * math.sqrt(p)
        M = M[:p-1, :]
        temp_1 = D + Lambda*np.identity(p-1)
        temp_1 = temp_1.diagonal()
        temp_1 = np.sqrt(temp_1)
        M = np.dot(np.dot(Up[:, :p-1], np.diag(temp_1)), M)
        M = M + np.tile(my[:, np.newaxis], p)
        Sest = np.dot(Q, Y)
    else:
        M = np.linalg.inv(Q)
        M = np.dot(Up, M)
        Sest = np.dot(Q, Y)

    return M, Sest, Up, my, sing_values

def quadProgl(G, c, Y, b, Aeq, beq, x0, throtle):
    import math
    import numpy as np
    p, N = np.shape(Y)
    neq = np.shape(Aeq)[0]
    m = p*p
    mu = np.ones((neq, 1))
    y0 = np.ones((N*p, 1))
    l0 = np.ones((N*p, 1))
    n = N*p
    mu0 = mu
    xnew = x0
    ynew = y0
    lnew = l0
    munew = mu0
    Q0 = np.reshape(x0, (p,p), order = 'F')
    previous = -math.log(abs(np.linalg.det(Q0)))
    below_threshold = 0

    nniter = 150
    if throtle == 1:
        nniter = 300
    for k in range(1, nniter+1):
        x, y, mu, l = [xnew, ynew, munew, lnew]
        # DEBUG
        Q = np.reshape(xnew, (p,p), order = 'F')
        true_value = -math.log(abs(np.linalg.det(Q)))
        quadratic_value = np.dot(c.T, x) + 0.5*np.dot(np.dot(x.T, G), x)
        threshold = previous

        yinvl = ((y**(-1))*l)

        temp = np.dot(np.reshape(l, (p,N), order = 'F'), Y.T)
        rd = np.dot(G, x) + c - temp.flatten(order='F')[:, np.newaxis] + np.dot(Aeq.T, mu)
        temp = np.dot(np.reshape(x, (p,p), order = 'F'), Y)
        rb = temp.flatten(order = 'F')[:, np.newaxis] - y - b
        rbeq = np.dot(Aeq, x) - beq
        print(np.dot(Aeq,x).shape)
        print(beq.shape)
        AtransYinvLA = computeAtransYinvLA(Y, yinvl)
        K = G + AtransYinvLA
        K_copy = np.array(K, copy = True)
        K_copy = np.append(K_copy, Aeq.T, axis = 1)
        Aeq_copy = np.array(Aeq, copy = True)
        Aeq_copy = np.append(Aeq_copy, np.zeros((neq, neq)), axis = 1)
        Aug = np.append(K_copy, Aeq_copy, axis = 0)
        rh = yinvl * (rb + y)
        temp = np.dot(np.reshape(rh, (p,N), order = 'F'), Y.T)
        rhaugl = -rd - temp.flatten(order = 'F')[:, np.newaxis]
        rhaug = np.array(rhaugl, copy = True)
        print(rhaug.shape)
        print(rbeq.shape)
        rhaug = np.append(rhaug, -rbeq, axis = 0)
        DxDm = np.linalg.lstsq(Aug, rhaug)[0]
        DxAff = DxDm[:m]
        temp = np.dot(np.reshape(DxAff, (p,p)), Y)
        DyAff = temp.flatten(order = 'F')[:, np.newaxis] + rb
        DlAff = -yinvl*(y+DyAff)
        mi = np.dot(y.T, l/n)

        aDymin = np.amin(-y[DyAff<0]/DyAff[DyAff<0])
        if not isinstance(aDymin, np.ndarray) or aDymin > 1:
            aDymin = 1

        aDlmin = np.amin(-l[DlAff<0]/DlAff[DlAff<0])
        if not isinstance(aDlmin, np.ndarray) or aDlmin > 1:
            aDlmin = 1

        if isinstance(aDymin, np.ndarray) or isinstance(aDlmin, np.ndarray):
            aAff = np.minimum(aDymin, aDlmin)
        else:
            aAff = min(aDymin, aDlmin)

        miAff = np.dot((y+np.dot(aAff, DyAff)).T,((l+np.dot(aAff, DlAff))/n))

        sigma = (miAff/mi)**3

        ycorrected = y + (l*-1)*DlAff*DyAff - sigma*mi*(l*-1)
        rh = yinvl*(rb + ycorrected)
        temp = np.dot(np.reshape(rh, (p,N)), Y.T)
        rhaugl = -rd - temp.flatten(order = 'F')[:, np.newaxis]
        rhaug = np.array(rhaugl, copy = True)
        rhaug = np.append(rhaug, -rbeq, axis = 0)
        DxDm = np.linalg.lstsq(Aug, rhaug)[0]
        Dx = DxDm.flatten(order = 'F')[0:m][:, np.newaxis]
        print("Shape of Dx: ", Dx.shape)
        Dmu = DxDm.flatten(order = 'F')[m:][:, np.newaxis]
        temp = np.dot(np.reshape(Dx, (p,p)), Y)
        Dy = temp.flatten(order = 'F')[:, np.newaxis] + rb
        Dl  = -yinvl*(Dy + ycorrected)
        tk = 1 - 1/(k+1)
        print("shape of amin calculation: ", -tk*y[Dy<0]/Dy[Dy<0])
        apri= np.amin(-tk*y[Dy<0]/Dy[Dy<0])
        if not isinstance(apri, np.ndarray) or apri > 1:
            apri = 1

        adual = np.amin(-tk*l[Dl<0]/Dl[Dl<0])
        if not isinstance(adual, np.ndarray) or adual > 1:
            adual = 1

        if isinstance(apri, np.ndarray) or isinstance(adual, np.ndarray):
            a = np.minimum(apri,adual)
        else:
            a = min(apri, adual)


        if throtle == 1:
            a = a/10

        xnew = x + a*Dx
        print("shape of x: ", x.shape)
        print("shape of a*Dx: ", (a*Dx).shape)
        print("shape of xnew: ",xnew.shape)
        Qnew = np.reshape(xnew, (p,p))
        true_value_new = -math.log(abs(np.linalg.det(Qnew)))
        ynew = y + a*Dy
        lnew = l + a*Dl
        munew = mu + a*Dmu

        if sigma < 1e-8 and mi < 1e-8:
            break

        if true_value_new < previous:
            previous = true_value_new
            below_threshold = 1
        else:
            if below_threshold == 1:
                xnew = x
                break
    x = xnew
    return x

def computeAtransYinvLA(Y,sinvl):
    import numpy as np
    p, N = np.shape(Y)
    Yaug = np.zeros((p*p, N))
    Sinvl = np.reshape(sinvl, (p,N))
    SinvLaug = np.tile(Sinvl, (p,1))
    for i in range(1, p+1):
        Yaug[(i-1)*p:i*p, :] = np.tile(Y[i-1,:][np.newaxis,:], (p,1))

    AtransYinvLcompact = Yaug * SinvLaug
    AtransYinvLAcompact = np.dot(AtransYinvLcompact, Y.T)
    AtransYinvLA = np.zeros((p*p, p*p))

    for i in range(1, p+1):
        for k in range(1, p+1):
            for j in range(1, p+1):
                AtransYinvLA[(i-1)*p+k-1, k+(j-1)*p - 1] = AtransYinvLAcompact[(i-1)*p+k-1, j-1]

    return AtransYinvLA


##--------------------------------------------------------VCA------------------------------------------##
def VCA(R, p, verbose = 'on', snr_input = 0, compress = 'svd', **kwargs):
    import math
    import time
    import numpy as np
    from scipy.linalg import svd
    from hyperspy.learn.rpca import orpca
    start_time = time.clock()
    L, N = R.shape
    print('Shape of R is: ', R.shape)
    if p<0 or p>L or (p%1) != 0:
        raise ValueError('ENDMEMBER parameter must be integer between 1 and L')

    if snr_input == 0:
        r_m = np.mean(R, 1)[:, np.newaxis]
        R_m = np.tile(r_m, N)
        R_o = R - R_m
        if compress == 'svd':
            Ud, Sd, Vd = svd(np.dot(R_o, R_o.T)/N)
            Ud = Ud[:, :p]
            del Sd, Vd
        elif compress == 'orpca':
            X, E, Ud, Sd, Vd = orpca(np.dot(R_o, R_o.T)/N, rank = p, fast = True, **kwargs)
            del X, E, Sd, Vd
        x_p = np.dot(Ud.T, R_o)
        #print('shape of x_p: ', x_p.shape)
        SNR = estimate_snr(R, r_m, x_p)
    else:
        SNR = snr_input

    SNR_th = 15 + 10*math.log10(p)
    #print('SNR_th value: ', SNR_th)
    #print('SNR is: ', SNR)
    #print('SNR_th is: ', SNR_th)
    if SNR < SNR_th:
        #print('Selected the projective proj.', SNR)
        d = p-1
        if snr_input == 0:
            Ud = Ud[:, :d]
        else:
            r_m = np.mean(R, 1)[:, np.newaxis]
            R_m = np.tile(r_m, N)
            R_o = R - R_m
            Ud, Sd, Vd = svd(np.dot(R_o, R_o.T)/N)
            Ud = Ud[:, :d]
            x_p = np.dot(Ud.T, R_o)
        Rp = np.dot(Ud, x_p[0:d, :]) + np.tile(r_m, N)
        x = x_p[0:d, :]
        #print('shape of x: ', x.shape)
        c = np.max(np.sum(x**2, axis = 0))**0.5
        c_arr_temp = c*np.ones((1,N))
        y = np.array(x, copy = True)
        y = np.append(y, c_arr_temp, axis = 0)
    else:
        d = p
        Ud, Sd, Vd = svd(np.dot(R, R.T)/N)
        #print('Shape of Ud: ', Ud.shape)
        Ud = Ud[:, :d]
        x_p = np.dot(Ud.T, R)
        Rp = np.dot(Ud, x_p[:d, :])
        x = np.dot(Ud.T, R)
        u = np.mean(x, 1)[:, np.newaxis]
        y = x/np.tile(np.sum(x*np.tile(u, N), axis=0)[np.newaxis, :], (d, 1))
    indice = np.zeros((1,p), dtype = int)
    A = np.zeros((p,p))
    A[p-1,0] = 1

    for i in range(p):
        Ip = np.eye(p)
        mean = np.zeros(p)
        w = np.random.rand(p, 1)
        f = w - np.dot(np.dot(A, np.linalg.pinv(A)),w)
        f = f/math.sqrt(np.sum(f**2))
        v = np.dot(f.T, y)
        v = np.absolute(v)
        indice[0, i] = v.argmax()
        A[:, i] = y[:, indice[0, i]]
    #print('indice: ', indice)
    if SNR < SNR_th:
        Ae = np.dot(Ud, x[:, indice[0]]) + np.tile(r_m, p)
    else:
        Ae = np.dot(Ud, x[:, indice[0]])
    #print('shape of Ae is: ', Ae.shape)
    #loadings = np.dot(np.linalg.pinv(Ae), R)
    loadings = np.dot(np.dot(np.linalg.pinv(np.dot(Ae.T,Ae)),Ae.T),R)
    print("Execution Time---%s seconds ---" %(time.clock() - start_time))
    return Ae, loadings, SNR

def estimate_snr(R, r_m, x_p):
    import math
    import numpy as np

    #print("ESTIMATE SNR")
    #print('shape of R ', np.shape(R))
    #print('shape of flatten R ', np.shape(R.flatten(order = 'F')))
    #print('-----------')
    L, N = np.shape(R)
    p, N = np.shape(x_p)

    P_x = np.sum(x_p.flatten(order = 'F')[:, np.newaxis]**2)/N + np.dot(r_m.T, r_m)
    P_y = np.sum(R.flatten(order = 'F')[:, np.newaxis]**2)/N
    
    snr_est = 10*math.log10((P_x - p/L*P_y)/(P_y-P_x))#Change to absolute value of snr_est
    return snr_est


##---------------------------------------BLU Algorithm --------------------------------------------------

def blind_unmix(Y, R, Nmc = 100, geo_method = 'vca', bool_plot = 0):
    import math
    import numpy as np

    P = Y.shape[0]
    Y = Y.T
    #initial values
    Tsigma2r = np.ones((R,1))*5e1
    #print(P)
    # dimension of space of interest
    L_red = R - 1

    # (crude) geometrical estimation
    print("GEOMETRICAL ENDMEMBER EXTRACTION")
    endm, matP, matU, Y_bar, endm_proj, y_proj = find_endm(Y, R, geo_method)
    endm = abs(endm)
    # projected endmember
    endm_proj = np.dot(matP, endm - np.dot(Y_bar, np.ones((1,R))))

    # MC initialization

    M_est = endm
    T_est = np.dot(matP, M_est - np.dot(Y_bar, np.ones((1,R))))

    A_est = abs(np.dot(np.linalg.pinv(M_est), Y))
    A_est = np.nan_to_num(A_est/np.dot(np.sum(A_est, axis = 0)[:, np.newaxis], np.ones((1,R))).T)
    

    # MC Chains
    Tab_A = np.zeros((Nmc,R,P));
    Tab_T = np.zeros((Nmc,L_red,R));
    Tab_sigma2 = np.zeros((Nmc,1));

    # MCMC Algorithm
    m_compt = 1
    run_simulation = 1
    j = 1
    while m_compt < Nmc and run_simulation:
        print('---MCMC iteration: ', j)
        #sampling noise variance
        sigma2_est = sample_sigma2(A_est, M_est, Y)
        #print('Check point 1')
        #sampling abundance vectors
        A_est = sample_A(Y.T, M_est.T, A_est.T, R, P, sigma2_est).T
        #print('A values: ', A_est.shape)
        #print(A_est[0, :])
        #print('Check point 2')
        #print('A_est nan status: ', np.isnan(A_est).any())
        #print('Check point 1')
        #sampling endmember projections
        T_est, M_est = sample_T_const(A_est,M_est,T_est,sigma2_est,Tsigma2r,matU,Y_bar,Y,endm_proj,bool_plot,y_proj)
        #print('Check point 2')
        #print('Check point 3')
        #saving the MC chains
        Tab_A[m_compt-1, :, :] = A_est
        Tab_T[m_compt-1,:,:] = T_est
        Tab_sigma2[m_compt-1,:] = sigma2_est
        #print('Check point 3')
        m_compt = m_compt + 1
        j += 1

    Nbi = math.floor(Nmc/3)
    A_MMSE = np.reshape(np.mean(Tab_A[Nbi:Nmc, :, :], axis = 0), (R, P), order = 'F')
    T_MMSE = np.reshape(np.mean(Tab_T[Nbi:Nmc, :, :], axis = 0), (R-1, R), order = 'F')
    M_est = np.dot(matU, T_MMSE) + np.dot(Y_bar, np.ones((1,R)))
    print('A_MMSE shape: ', A_MMSE.shape)
    print('T_MMSE shape: ', T_MMSE.shape)
    print('M_est shape: ', M_est.shape)
    return A_MMSE, M_est, Tab_A, Tab_T, Tab_sigma2, matU, Y_bar, Nmc

    
def sample_sigma2(A_est, M_est, Y):
    import numpy as np
    L, P = np.shape(Y)
    
    #coeff1 = ones(P,1)*L/2
    tmp = Y - np.dot(M_est, A_est)
    coeff1 = P * L/2
    coeff2 = np.sum((np.sum(tmp**2, 0)/2))
    
    Tsigma2p = 1/np.random.gamma(shape = coeff1, scale = coeff2**(-1))
    
    return Tsigma2p

def sample_A(X,S,A,R,P,sigma2e):
    import numpy as np
    """------------------------------------------------------------------
This function allows to sample the abundances 
   according to its posterior f(A|...)
USAGE
   A = sample_abundances(X,S,A,R,P,L,sigma2e,mu0,rhoa,psia)

INPUT
   X,S,A : matrices of mixture, sources and mixing coefficients 
   R,P,L  : number of sources, observations and samples
   sigma2e : the current state of the sigma2 parameter
   rho,psi  : hyperprior parameters 

OUTPUT
   A     : the new state of the A arameter

---------------------------------------------------------------------------"""
    y = X
    ord = np.random.permutation(R)
    jr = ord[R-1]
    comp_jr = ord[:(R-1)]
    #print('shape of A: ', A.shape)
    alpha = A[:, comp_jr]
    #print('shape of alpha: ', alpha.shape)
    #useful quantities
    u = np.ones((R-1, 1))
    M_R = S[jr, :][:, np.newaxis]
    M_Ru = np.dot(M_R, u.T)
    M = S[comp_jr, :].T
    T = np.dot((M-M_Ru).T, (M-M_Ru))

    for p in range(P):
            Sigma = np.linalg.inv(T/sigma2e)
            
            Mu = np.dot(Sigma, np.dot((M-M_Ru).T/sigma2e, y[p, :][:, np.newaxis] - M_R))

            #print('shape of alpha bef func: ', alpha.shape)
            alpha[p, :] = dtrandnmult(alpha[p, :], Mu, Sigma)
            
    A[:, ord[:(R-1)]] = alpha
    A[:, ord[R-1]] = np.maximum(1-np.sum(alpha, 1),0)
    
    return A

def dtrandnmult(S,Mu,Re):
    
    import numpy as np
    import math 
    

    S = S.flatten(order = 'F')[:, np.newaxis]
    #print('S shape: ', S.shape)
    #print(S)

    Mu = Mu.flatten(order = 'F')[:, np.newaxis]
    #print('Mu values: ')
    #print(Mu)

    R = S.shape[0]
    
    if R == 1:
            S = trandn(Mu, np.sqrt(Re))
            #print(' S condition 1 shape: ', S.shape)
            #print(S)
            return S.flatten()
    else:
            
            Sigma_mat = [0]*R # 634-636: create empty list of size R for loop iteration and value storage
            
            Sigma_vect = [0]*R
            
            for r in range(R):
                    Rm = Re
                    Rm = np.delete(Rm, r, 0)
                    Rv = Rm[:, r][:, np.newaxis]
                    Rm = np.delete(Rm, r, 1)
                    Sigma_mat[r] = np.linalg.inv(Rm)
                    Sigma_vect[r] = Rv
            for iter in range(5):
                    randperm = np.random.permutation(R)
                    
                    Moy_Sv = np.zeros(len(randperm))
                    
                    Var_Sv = np.zeros(len(randperm))
                    
                    Std_Sv = np.zeros(len(randperm))
                    
                    for k in randperm:
                            Sk = S
                            Sk = np.delete(Sk, k, 0) #potential bug
                            Muk = Mu
                            Muk = np.delete(Muk, k, 0) #potential bug

                            """print('Sk shape: ', Sk.shape)
                            print('Muk shape: ', Muk.shape)
                            print('Sigma_vect shape: ', Sigma_vect[k].shape)
                            print('Sigma_mat shape: ', Sigma_mat[k].shape)"""

                            test = np.dot(Sigma_vect[k].T, np.dot(Sigma_mat[k], Sk-Muk))

                            Moy_Sv[k] = Mu[k] + np.dot(np.dot(Sigma_vect[k].T,Sigma_mat[k]), Sk-Muk)
                            Var_Sv[k] = Re[k,k] - np.dot(Sigma_vect[k].T, np.dot(Sigma_mat[k], Sigma_vect[k]))
                            Std_Sv[k] = math.sqrt(abs(Var_Sv[k]))
                            S[k] = dtrandn_MH(S[k], Moy_Sv[k], Std_Sv[k], 0, (1 - sum(S) + S[k]))
            
    #if np.isnan(S[:, 0]).any():
    #print('S return nan status: ', np.isnan(S[:, 0]).any())
    #raise ValueError('NaN detected, stopping function')
    return S[:, 0]

def trandn(Mu, Sigma):
    
    import numpy as np
    from scipy.special import erf, erfinv
    import math
    
    Mu = Mu.flatten(order = 'F')[:, np.newaxis]

    Sigma = Sigma.flatten(order = 'F')[:, np.newaxis]

    #print('Mu values: ', Mu)
    #print('Re sqrt value: ', Sigma)

    U = np.random.rand(len(Mu),1)
    #print('Sigma multi: ', np.maximum(Sigma, np.spacing(1)))
    #print('math sqrt multi: ', (math.sqrt(2)*np.maximum(Sigma, np.spacing(1))))
    V = erf(-Mu/(math.sqrt(2)*np.maximum(Sigma, np.spacing(1))))
    #print('V values: ', V)

    #print('sqrt values: ', 2*(Sigma**2))
    #print('erfinv values: ', -(((1-V)*U + V)==1))
    
    X = Mu + np.sqrt(2*(Sigma**2))*erfinv(-(((1-V)*U + V) == 1)*np.spacing(1) + (1-V)*U + V)
    #print(' X value: ', X)
    
    X = np.maximum(X, np.spacing(1))

    #if np.isnan(X).any():
    #print(X)
    #raise ValueError('NaN detected, stopping function, recheck X value')
    return X

def dtrandn_MH(X, Mu, Sigma, Mum, Mup, sample_T = False):
    
    import numpy as np
    
    #Mu_new, Sigma are float64 variables
    Mu_new = Mu - Mum
    #if sample_T == True:
        #print('Mu_new: ', Mu_new)
    
    Mup_new = Mup - Mum
    #if sample_T == True:
        #print('Mup_new: ', Mup_new)
    #if sample_T:
        #print('Mu is: ', Mu_new)
        #print('Mup is : ', Mup)
    if Mu < Mup:
            Z = randnt(Mu_new, Sigma, 1, sample_T)
    else:
            delta = Mu_new - Mup_new
            Mu_new = -delta
            Z = randnt(Mu_new, Sigma, 1, sample_T)
            Z = -(Z-Mup_new)
    Z = Z + Mum
    cond = (Z<=Mup) and (Z>=Mum)
    #if sample_T == True:
        #print(cond)
    X = (Z*cond + X*np.logical_not(cond))

    #if sample_T:
        #print('output X: ', X)
    
    return np.asscalar(X)


def sample_T_const(A,M,T,sigma2p,Tsigma2r,matU,Y_bar,Y,E_prior,bool_plot,y_proj):
    
    import math
    
    import numpy as np
    
    T_out = T

    K = T_out.shape[0]
 
    M_out = M
    
    R = A.shape[0]
    
    L, P = np.shape(Y)

    #print('A shape: ', A.shape)

    randperm = np.random.permutation(R)
    for r in randperm:
            comp_r = np.setdiff1d(np.arange(R), r)

            alpha_r = A[comp_r, :]
            #print('alpha_r shape: ', alpha_r.shape)
            
            alphar = A[r, :][np.newaxis, :]

            #first value before multiplication with matU.T should be a scalar)
            #In case of nan values, check for inf in A[r, :]
            invSigma_r = np.dot(np.dot(np.sum((A[r, :]**2)[:, np.newaxis]/sigma2p), matU.T), matU) + 1/Tsigma2r[r]*np.eye(R-1)
            if np.isnan(np.dot(np.sum((A[r, :]**2)[:, np.newaxis]/sigma2p), matU.T)).any():
                print('r value: ', r)
                print('matU.T values: ', matU.T)
                print('A r: ', A[r, :])
            """print('matU nan status: ', np.isnan(matU.T).any())
            print('sigma2p: ', sigma2p)
            print('Check point T1')
            print('invsigmaR nan status: ', np.isnan(invSigma_r).any())"""
            Sigma_r = np.linalg.inv(invSigma_r)
            
            er = E_prior[:, r][:, np.newaxis]
            
            randperm1 = np.random.permutation(K)

            for k in randperm1:
                    #print('randperm iteration:', k)
                    tr = T_out[:, r][:, np.newaxis]

                    comp_k = np.setdiff1d(np.arange(K), k)

                    M_r = M_out[:, comp_r]

                    #print('M_r shape: ', M_r.shape)

                    
                    Delta_r = Y - np.dot(M_r, alpha_r) - np.dot(Y_bar, alphar)

                    mu = np.dot(Sigma_r, np.dot(matU.T, np.sum(Delta_r*np.dot(np.ones((L,1)),alphar)/sigma2p, axis = 1))) + er/Tsigma2r[r]
                                        
                    """print('Sigma_r nan status: ', np.isnan(Sigma_r).any())
                    print('matU nan status: ', np.isnan(matU).any())
                    print('Delta_r nan status: ', np.isnan(Delta_r).any())
                    print('alphar nan status: ', np.isnan(alphar).any())
                    print('sigma2p: ', sigma2p)
                    print('second calc nan status: ', np.isnan(er/Tsigma2r[r]).any())
                    print('mu nan status: ', np.isnan(mu).any())"""
                    
                    skr = Sigma_r[comp_k, k][:, np.newaxis]

                    comp_id = comp_k
                    Sigma_r_k = np.empty((len(comp_k), len(comp_k)))
                    """Point of interest in case there's a value error"""
                    for i in range(Sigma_r_k.shape[0]):
                        for j in range(Sigma_r_k.shape[1]):
                            Sigma_r_k[i, j] = Sigma_r[comp_id[i], comp_id[j]]
                    inv_Sigma_r_k = np.linalg.inv(Sigma_r_k)

                    #flatten version of mu
                    mu_flattened = mu.flatten(order = 'F')
                    #print('mu flattened value: ', mu_flattened[k])

                    muk = mu_flattened[k] + np.dot(skr.T, np.dot(inv_Sigma_r_k, tr[comp_k,0][:, np.newaxis] - er[comp_k,0][:, np.newaxis]))
                    
                    s2k = Sigma_r[k,k] - np.dot(skr.T, np.dot(inv_Sigma_r_k, skr))

                    
                    #troncature
                    #print('matU pure shape: ', matU.shape)
                    #print('matU k shape: ', matU[:, k][:, np.newaxis].shape)


                    vect_e = ((-Y_bar - np.dot(matU[:, comp_k], tr[comp_k, 0][:, np.newaxis]))/(matU[:, k][:, np.newaxis])).flatten()

                    #print('check point T2')

                    """print('Y_bar nan status: ', np.isnan(Y_bar).any())
                    print('matU nan status: ', np.isnan(matU[:, comp_k]).any())
                    print('matU nan status with k ', np.isnan(matU[:, k][:, np.newaxis]).any())
                    print('tr nan status: ', np.isnan(tr[comp_k, 0]).any())
                    print('vect_e nan status: ', np.isnan(vect_e).any())"""

                    setUp = (matU[:, k]>0)
                    setUm = (matU[:, k]<0)
                    
                    mup = min(1/np.spacing(1), np.min(vect_e[setUm]))
                    mum = max(-1/np.spacing(1), np.max(vect_e[setUp]))
                    
                    T_out[k,r] = dtrandn_MH(T_out[k,r], muk[0,0], math.sqrt(s2k), mum, mup, sample_T = True)

                    #print('check point T3')

                    M_out[:, r] = (np.dot(matU, T_out[:, r][:, np.newaxis]) + Y_bar).flatten()
                    #print('Check point T3')
                    
    return T_out, M_out
    

    
def randnt(m, s, N, sample_T = False):
    import numpy as np
    import math
    
    """
    RPNORM    Random numbers from the positive normal distribution.
    RPNORM(N,M,S) is an N-by-1 vector with random entries, generated
    from a positive normal distribution with mean M and standard
    deviation S.x

    (c) Vincent Mazet, 06/2005
    Centre de Recherche en Automatique de Nancy, France
    vincent.mazet@cran.uhp-nancy.fr

    Reference:
    V. Mazet, D. Brie, J. Idier, "Simulation of Positive Normal Variables
    using several Proposal Distributions", IEEE Workshop Statistical
    Signal Processing 2005, july 17-20 2005, Bordeaux, France. 
    """
    if s<0:
        raise ValueError('Standard Deviation must be positive')
        pass
    if N<=0:
        raise ValueError('N is wrong')
        pass
    
    
    NN = N
    A  = 1.136717791056118
    mA = (1 - A**2)/A*s
    mC = s*math.sqrt(math.pi/2)
    x = [] #create a list to mimic the behavior in MatLab, which can be truncated later.
    k = 0
    while (len(x) < NN):
        if (m < mA): #4. Exponential Distribution
            a = (-m + math.sqrt(m**2+4*s**2))*0.5/(s**2)
            z = -np.log(1-np.random.random((N,1)))/a
            rho = np.exp(-((z-m)**2)/(2*s**2) - a*(m-z+a*s**2/2))
        elif m <= 0: #3. Normal distribution truncated at the mean equality because 3 is faster to compute than 2
            z = np.nan_to_num(abs(np.random.random((N,1)))*s + m)
            rho = (z>=0)
        elif m < mC: #2. Normal distribution coupled with the uniform ones
            distr_array = np.random.random((N,1))
            s_m_array = m/(m+math.sqrt(math.pi/2)*s)
            r = (distr_array < s_m_array)
            u = np.random.random((N,1))*m
            g = abs(np.random.random((N,1))*s) + m
            z = r*u + (1-r)*g
            if sample_T:
                print(' m<mC z is: ', z)
            rho = r*np.exp(-(z-m)**2/(2*s**2)) + (1-r)*np.ones((N,1))
        else: #1. Normal distribution
            z = np.random.random((N,1))*s + m
            #if sample_T:
                #print('else z is: ', z)
            rho =(z>=0)
                    
        #Accept or reject the propositions
        reject = (np.random.random((N,1)) > rho)
        z = z[-reject]
        if len(z) > 0:
            for i in z:
                x.append(i)
        N = N - len(z)
    x = np.array(x)[:, np.newaxis]
    return x


def find_endm(Y, R, method): 
    
    import numpy as np
    
    L_red = R - 1
    P = np.size(Y, 1)
    
    #PCA
    Rmat = Y - np.dot(np.mean(Y,1)[:, np.newaxis], np.ones((1,P)))
    Rmat = np.dot(Rmat, Rmat.T)
    
    vect_prop, D, S = np.linalg.svd(Rmat)
    del S

    #D = np.diag(D[:L_red])
    D = np.ones(L_red)

    vect_prop = vect_prop[:, :L_red].T
    
    #First L_red eigenvectors
    V = vect_prop[:L_red,:]

    #First L_red eigenvalues
    V_inv = np.linalg.pinv(V)
    Y_bar = np.mean(Y, axis = 1)[:, np.newaxis]
    #print('Y_bar shape: ', Y_bar.shape)


    #D fractional square
    #D_square = np.eye(L_red) #avoids inf values
    #for i in range(D.shape[0]):
     #   D_square[i, i] = D[i, i]**(-1/2)
    #print('D_square: ', D_square)
    #projector
    matP = np.dot(np.diag(D[:L_red]**(-1/2)), V)
    #inverse projecter
    matU = np.dot(V_inv, np.diag(D[:L_red]**(1/2)))

    #projecting
    
    y_proj = np.dot(matP,(Y - np.dot(Y_bar, np.ones((1,P)))))
    
    if method == "nfindr":
        Nb_iter = 5000
        endm_proj = nfindr(y_proj.T, Nb_iter)
        endm = np.dot(matU, endm_proj) + np.dot(Y_bar, np.ones((1,R)))
    elif method == "vca":
        endm = VCA(Y, R)[0]
        endm_proj = np.dot(matP, endm - np.dot(Y_bar, np.ones((1,R))))

    return endm, matP, matU, Y_bar, endm_proj, y_proj




























































