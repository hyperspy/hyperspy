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
            M, indice, Rp = VCA(Y,p)  # Initialize with VCA
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


def VCA(R, p, verbose = 'on', snr_input = 0):
    import math
    import numpy as np
    L, N = np.shape(R)
    if p<0 or p>L or (p%1) != 0:
        raise ValueError('ENDMEMBER parameter must be integer between 1 and L')

    if snr_input == 0:
        r_m = np.expand_dims(np.mean(R, 1), axis = 1)
        R_m = np.tile(r_m, N)
        R_o = R - R_m
        Ud, Sd, Vd = np.linalg.svd(np.dot(R_o, R_o.T)/N, full_matrices = False)
        x_p = np.dot(Ud.T, R_o)
        SNR = estimate_snr(R, r_m, x_p)

    SNR_th = 15 + 10*math.log10(p)

    if SNR < SNR_th:
        if verbose == 'on':
            print('Select the projective proj.', SNR)
        d = p-1
        if snr_input == 0:
            Ud = Ud[:,0:d]
        else:
            r_m = np.expand_dims(np.mean(R, 1), axis = 1)
            R_m = np.tile(r_m, N)
            R_o = R - R_m
            Ud, Sd, Vd = np.linalg.svd(np.dot(R_o, R_o.T)/N, full_matrices = False)
            Ud = Ud[:, 0:p-1]
            x_p = np.dot(Ud.T, R_o)
        Rp = np.dot(Ud, x_p[0:d, :]) + np.tile(r_m, N)
        x = x_p[0:d, :]
        c = np.max(np.sum(x**2, axis = 0))**0.5
        c_arr_temp = c*np.ones((1,N))
        y = np.array(x, copy = True)
        y = np.append(y, c_arr_temp, axis = 0)
    else:
        if verbose == 'on':
            print()
        d = p
        Ud, Sd, Vd = np.linalg.svd(np.dot(R, R.T)/N, full_matrices = False)
        x_p = np.dot(Ud.T, R)
        Rp = np.dot(Ud, x_p[0:d, :])
        x = np.dot(Ud.T, R)
        u = np.expand_dims(np.mean(x, 1), axis = 1)
        temp = x * np.tile(u, N)
        y = x/np.tile(np.expand_dims(np.sum(temp, axis=0), axis=0), (d, 1))
    indice = np.zeros((1,p), dtype = int)
    A = np.zeros((p,p))
    A[p-1,0] = 1

    for i in range(1, p+1):
        w = np.random.rand(p, 1)
        f = w - np.dot(np.dot(A, np.linalg.pinv(A)),w)
        f = f/math.sqrt(np.sum(f**2))
        v = np.dot(f.T, y)
        v = np.absolute(v)
        indice[0][i-1] = v.argmax()
        A[:, i-1] = y[:, indice[0][i-1]]

    Ae = Rp[:, indice[0]]

    return Ae, indice[0], Rp

def estimate_snr(R, r_m, x_p):
    import math
    import numpy as np
    L, N = np.shape(R)
    p, N = np.shape(x_p)
    P_y = float(np.sum(R.flatten(order = 'F')[:, np.newaxis]**2)/N)
    P_x = float(np.sum(x_p.flatten(order = 'F')[:, np.newaxis]**2)/N + np.dot(r_m.T, r_m))
    # the signal given by the example picture causes a negative value, so an absolute
    # transformation is added to avoid this issue.
    if (P_y - P_x) < 0 or (P_y - P_x) < 1e-10 or (P_x - p/L*P_y) < 0 or (P_x - p/L*P_y) < 1e-10:
        snr_est = 0
    else:
        snr_est = 10*math.log10((P_x - p/L*P_y)/abs(P_y-P_x))
    return snr_est





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



