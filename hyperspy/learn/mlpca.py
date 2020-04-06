# -*- coding: utf-8 -*-
# This file is a transcription of a MATLAB code obtained from the
# following research paper: Darren T. Andrews and Peter D. Wentzell,
# “Applications of maximum likelihood principal component analysis:
# incomplete data sets and calibration transfer,”
# Analytica Chimica Acta 350, no. 3 (September 19, 1997): 341-352.
#
# Copyright 1997 Darren T. Andrews and Peter D. Wentzell
# Copyright 2007-2020 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

import logging

import numpy as np

from hyperspy.learn.svd_pca import svd_solve

_logger = logging.getLogger(__name__)


def mlpca(
    X, varX, output_dimension, svd_solver="auto", tol=1e-10, max_iter=50000, **kwargs
):
    """Performs maximum likelihood PCA with missing data.

    This function is a transcription of a MATLAB code obtained
    from [Andrews1997]_.

    Parameters
    ----------
    X : numpy array, shape (m, n)
        Matrix of observations.
    varX : numpy array
        Matrix of variances associated with X
        (zeros for missing measurements).
    output_dimension : int
        The model dimensionality.
    svd_solver : {'auto', 'full', 'arpack', 'randomized'}
        If auto :
            The solver is selected by a default policy based on `data.shape` and
            `output_dimension`: if the input data is larger than 500x500 and the
            number of components to extract is lower than 80% of the smallest
            dimension of the data, then the more efficient 'randomized'
            method is enabled. Otherwise the exact full SVD is computed and
            optionally truncated afterwards.
        If full :
            use an exact LAPACK SVD solver to estimate the full number
            of components
        If arpack :
            use a truncated ARPACK SVD solver to estimate a limited
            number of components, given by output_dimension. Requires
            0 < output_dimension < min(data.shape).
        If randomized :
            use a truncated, randomized SVD from sklearn to estimate
            a limited number of components, given by output_dimension.
    tol : float
        Tolerance of the stopping condition.
    max_iter : int
        Maximum number of iterations before exiting without convergence.

    Returns
    -------
    U, S, V: numpy array
        The pseudo-SVD parameters.
    s_obj : float
        Value of the objective function.

    References
    ----------
    .. [Andrews1997] Darren T. Andrews and Peter D. Wentzell, "Applications
           of maximum likelihood principal component analysis: incomplete
           data sets and calibration transfer", Analytica Chimica Acta 350,
           no. 3 (September 19, 1997): 341-352.

    """
    m, n = X.shape

    with np.errstate(divide="ignore"):
        # Shouldn't really have zero variance anywhere,
        # but handle it here.
        inv_v = 1.0 / varX
        inv_v[~np.isfinite(inv_v)] = 1.0

    _logger.info("Performing maximum likelihood principal components analysis")

    # Generate initial estimates
    _logger.info("Generating initial estimates")
    U, _, _ = svd_solve(np.cov(X), svd_solver=svd_solver, **kwargs)
    U = U[:, :output_dimension]
    s_old = 0.0

    # Placeholders
    F = np.empty((m, n))
    M = np.zeros((m, n))
    Uq = np.zeros((output_dimension, m))

    # Loop for alternating least squares
    _logger.info("Optimization iteration loop")
    for itr in range(max_iter):
        s_obj = 0.0

        for i in range(n):
            Uq = U.T * inv_v[:, i]
            F = np.linalg.inv(Uq @ U)
            M[:, i] = np.linalg.multi_dot([U, F, Uq, X[:, i].T])
            dx = X[:, i] - M[:, i]
            s_obj += (dx * inv_v[:, i]) @ dx.T

        # Every second iteration, check the stop criterion
        if itr > 0 and itr % 2 == 0:
            stop_criterion = np.abs(s_old - s_obj) / s_obj
            _logger.info(
                "Iteration: {}, convergence: {}".format(itr // 2, stop_criterion)
            )

            if stop_criterion < tol:
                break

        # Transpose for next iteration
        s_old = s_obj
        _, _, V = svd_solve(M, svd_solver=svd_solver, **kwargs)

        X = X.T
        inv_v = inv_v.T
        F = F.T
        M = M.T

        m, n = X.shape
        U = V[:output_dimension].T

    U, S, V = svd_solve(M, svd_solver=svd_solver, **kwargs)
    V = V.T

    return U, S, V, s_obj
