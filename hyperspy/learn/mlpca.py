# -*- coding: utf-8 -*-
# This file is a transcription of a MATLAB code obtained from the  
# following research paper: Darren T. Andrews and Peter D. Wentzell, 
# “Applications of maximum likelihood principal component analysis: 
# incomplete data sets and calibration transfer,” 
# Analytica Chimica Acta 350, no. 3 (September 19, 1997): 341-352.
# 
# Copyright 1997 Darren T. Andrews and Peter D. Wentzell
# Copyright 2007-2011 The Hyperspy developers
#
# This file is part of  Hyperspy.
#
#  Hyperspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  Hyperspy.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import scipy.linalg
from hyperspy.misc.import_sklearn import *


def mlpca(X,varX,p, convlim = 1E-10, maxiter = 50000, fast=False):
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
    if fast is True and sklearn_installed is True:
        def svd(X):
            return fast_svd(X, p)
    else:
        def svd(X):
            return scipy.linalg.svd(X, full_matrices = False)
    XX = X
#    varX = stdX**2
    n = XX.shape[1]
    print "\nPerforming maximum likelihood principal components analysis"
    # Generate initial estimates
    print "Generating initial estimates"
#    CV = np.zeros((X.shape[0], X.shape[0]))
#    for i in xrange(X.shape[0]):
#        for j in xrange(X.shape[0]):
#            denom = np.min((len(np.where(X[i,:] != 0)), 
#            len(np.where(X[j,:] != 0))))
#            CV[i,j] = np.dot(X[i,:], (X[j,:]).T) / denom
    CV = np.cov(X)
    U, S, Vh = svd(CV)
    U0 = U

    # Loop for alternating least squares
    print "Optimization iteration loop"
    count = 0
    Sold = 0
    ErrFlag = -1
    while ErrFlag < 0:
        count += 1
        Sobj = 0
        MLX = np.zeros(XX.shape)
        for i in xrange(n):
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
            U,S,Vh = svd(MLX)
            V = Vh.T
            XX = XX.T
            varX = varX.T
            n = XX.shape[1]
            U0 = V[:]
    # Finished
    
    U, S, Vh = svd(MLX)
    V = Vh.T
#    S = S[:p]
    return U,S,V,Sobj, ErrFlag
