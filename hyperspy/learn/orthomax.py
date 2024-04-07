# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import numpy as np
from numpy.linalg import svd


def orthomax(A, gamma=1.0, tol=1.4901e-07, max_iter=256):
    """Calculate orthogonal rotations for a matrix of factors or loadings from PCA.

    When gamma=1.0, this is known as varimax rotation, which finds a
    rotation matrix W that maximizes the variance of the squared
    components of A @ W. The rotation matrix preserves orthogonality of
    the components.

    Taken from metpy.

    Parameters
    ----------
    A : numpy array
        Input data to unmix
    gamma : float
        If gamma in range [0, 1], use SVD approach, otherwise
        solve with a sequence of bivariate rotations.
    tol : float
        Tolerance of the stopping condition.
    max_iter : int
        Maximum number of iterations before exiting without convergence.

    Returns
    -------
    B : numpy array
        Rotated data matrix
    W : numpy array
        The unmixing matrix

    """

    d, m = A.shape
    oo_d = 1.0 / d

    B = np.copy(A)
    Bsq = np.empty((d, m))
    W = np.eye(m)
    R = np.empty((2, 2))

    if 0.0 <= gamma and gamma <= 1.0:
        # Use Lawley and Maxwell's fast version
        converged = False

        while not converged:
            S = 0.0
            for _ in range(max_iter):  # pragma: no branch
                Sold = S
                Bsq = B**2
                U, S, V = svd(
                    A.T @ (d * B * Bsq - gamma * B * np.sum(Bsq, axis=0)),
                    full_matrices=False,
                )
                W = U @ V
                S = np.sum(S)
                B = A @ W

                if abs(S - Sold) < tol * S:
                    converged = True
                    break

    else:
        # TODO: this doesn't seem to work...perhaps
        # someone with more knowledge can either fix
        # or remove this?
        # Use a sequence of bivariate rotations
        for _ in range(max_iter):  # pragma: no branch
            maxTheta = 0.0

            for i in range(m - 1):
                for j in range(i, m):
                    Bi = B[:, i]
                    Bj = B[:, j]
                    u = Bi * Bi - Bj * Bj
                    v = 2.0 * Bi * Bj

                    usum = u.sum()
                    vsum = v.sum()

                    numer = 2.0 * u.T @ v - 2.0 * gamma * usum * vsum * oo_d
                    denom = u.T @ u - v.T @ v - gamma * (usum**2 - vsum**2) * oo_d

                    theta = 0.25 * np.arctan2(numer, denom)
                    maxTheta = max(maxTheta, abs(theta))

                    R = np.array(
                        [
                            [np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)],
                        ]
                    )

                    B[:, [i, j]] = B[:, [i, j]] @ R
                    W[:, [i, j]] = W[:, [i, j]] @ R

            if maxTheta < tol:
                break

    return B, W.T
