# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANdata WARRANTdata; without even the implied warranty of
# MERCHANTABILITdata or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# dataou should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

import logging

import numpy as np
import scipy.linalg

from hyperspy.misc.machine_learning.import_sklearn import (
    fast_svd, sklearn_installed)

_logger = logging.getLogger(__name__)


def svd_pca(data, fast=False, output_dimension=None, centre=None,
            auto_transpose=True):
    """Perform PCA using SVD.

    Parameters
    ----------
    data : numpy array
        MxN array of input data (M variables, N trials)
    fast : bool
        Wheter to use randomized svd estimation to estimate a limited number of
        componentes given by output_dimension
    output_dimension : int
        Number of components to estimate when fast is True
    centre : None | 'variables' | 'trials'
        If None no centring is applied. If 'variable' the centring will be
        performed in the variable axis. If 'trials', the centring will be
        performed in the 'trials' axis.
    auto_transpose : bool
        If True, automatically transposes the data to boost performance

    Returns
    -------

    factors : numpy array
    loadings : numpy array
    explained_variance : numpy array
    mean : numpy array or None (if center is None)
    """
    N, M = data.shape
    if centre is not None:
        if centre == 'variables':
            mean = data.mean(1)[:, np.newaxis]
        elif centre == 'trials':
            mean = data.mean(0)[np.newaxis, :]
        else:
            raise AttributeError(
                'centre must be one of: None, variables, trials')
        data -= mean
    else:
        mean = None
    if auto_transpose is True:
        if N < M:
            _logger.info("Auto transposing the data")
            data = data.T
        else:
            auto_transpose = False
    if fast is True and sklearn_installed is True:
        if output_dimension is None:
            raise ValueError('When using fast_svd it is necessary to '
                             'define the output_dimension')
        U, S, V = fast_svd(data, output_dimension)
    else:
        U, S, V = scipy.linalg.svd(data, full_matrices=False)
    if auto_transpose is False:
        factors = V.T
        explained_variance = S ** 2 / N
        loadings = U * S
    else:
        loadings = V.T
        explained_variance = S ** 2 / N
        factors = U * S
    return factors, loadings, explained_variance, mean
