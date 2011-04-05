# -*- coding: utf-8 -*-
# Copyright © 2007 Francisco Javier de la Peña
#
# This file is part of EELSLab.
#
# EELSLab is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# EELSLab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EELSLab; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  
# USA

import numpy as np

def mlpca(X,varX,p, convlim = 1E-10, maxiter = 50000):
    """
    This function performs MLPCA with missing
    data.
    
    Arguments:
    X       is the mxn matrix of observations.
    stdX    is the mxn matrix of standard deviations
            associated with X (zeros for missing
            measurements).
    p       is the model dimensionality.
    
    Returns:
    U,S,V   are the pseudo-svd parameters.
    Sobj    is the value of the objective function.
    ErrFlag indicates exit conditions:
            0 = nkmal termination
            1 = max iterations exceeded.
    """

    XX = X
#    varX = stdX**2
    n = XX.shape[1]
    print "\nPerforming maximum likelihood principal components analysis"
    # Generate initial estimates
    print "Generating initial estimates"
#    CV = np.zeros((X.shape[0], X.shape[0]))
#    for i in range(X.shape[0]):
#        for j in range(X.shape[0]):
#            denom = np.min((len(np.where(X[i,:] != 0)), 
#            len(np.where(X[j,:] != 0))))
#            CV[i,j] = np.dot(X[i,:], (X[j,:]).T) / denom
    CV = np.cov(X)
    U, S, Vh = np.linalg.svd(CV, 0)
    U0 = U[:,:p]

    # Loop for alternating least squares
    print "Optimization iteration loop"
    count = 0
    Sold = 0
    ErrFlag = -1
    while ErrFlag < 0:
        count += 1
        Sobj = 0
        MLX = np.zeros(XX.shape)
        for i in range(n):
#            Q = sp.sparse.lil_matrix((varX.shape[0] ,varX.shape[0]))
#            Q.setdiag((1/(varX[:,i])).squeeze())
#            Q.tocsc()

            Q = np.diag((1/(varX[:,i])).squeeze())
            U0m = np.matrix(U0)
            F = np.linalg.inv((U0m.T * Q * U0m))
            MLX[:,i] = np.array(U0m * F * U0m.T * Q * ((np.matrix(XX[:,i])).T)).squeeze()
            dx = np.matrix((XX[:,i] - MLX[:,i]).squeeze())
            Sobj = Sobj + float(dx *Q * dx.T)
        if (count % 2) == 1:
            print "Iteration : %s" % (count / 2)
            if (abs(Sold - Sobj) / Sobj) < convlim:
                ErrFlag = 1
            print "(abs(Sold - Sobj) / Sobj) = %s" % (abs(Sold - Sobj) / Sobj)
            if count > maxiter:
                ErrFlag = 1
        
        if ErrFlag < 0:
            Sold = Sobj
            U,S,Vh = np.linalg.svd(MLX, 0)
            V = Vh.T
            XX = XX.T
            varX = varX.T
            n = XX.shape[1]
            U0 = V[:,:p]
    # Finished
    
    U, S, Vh = np.linalg.svd(MLX, 0)
    V = Vh.T
    U = U[:,:p]
#    S = S[:p]
    V = V[:,:p]
    return U,S,V,Sobj, ErrFlag
