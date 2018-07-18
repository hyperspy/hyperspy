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
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import dask.array as da


def linear_regression(y, comp_data):
    '''
    Performs linear regression on single pixels as well
    as multidimensional arrays

    Parameters
    ----------
    y : array_like, shape: (signal_axis) or (nav_shape, signal_axis)
        The data to be fit to
    comp_data : array_like, shape: (number_of_comp, signal_axis) or (nav_shape,
                                    number_of_comp, signal_axis)
        The components to fit to the data

    Returns:
    ----------
    fit_coefficients : array_like, 
                        shape: (number_of_comp) or (nav_shape, number_of_comp)

    '''
    if isinstance(comp_data, da.Array):
        lazy = True
        matmul = da.matmul
        inv = da.linalg.inv
        dot = da.dot
    else:
        lazy = False
        matmul = np.matmul
        inv = np.linalg.inv
        dot = np.dot

    square = matmul(comp_data, comp_data.T)
    square_inv = inv(square)
    comp_data2 = matmul(square_inv, comp_data)
    fit_coefficients = dot(y, comp_data2.T)
    if lazy:
        fit_coefficients = fit_coefficients.compute()
    return fit_coefficients


def standard_error_from_covariance(covariance):
    'Get standard error coefficients from the diagonal of the covariance'
    # dask diag only supports 2D arrays, so we cannot use diag (for now)
    # if isinstance(data, da.Array):
    #     sqrt = da.sqrt
    #     diag = da.diag
    # else:
    #     sqrt = np.sqrt
    #     diag = np.diag
    standard_error = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
    return standard_error


def get_top_parent_twin(parameter):
    'Get the top parent twin, if there is one'
    if parameter.twin:
        return get_top_parent_twin(parameter.twin)
    else:
        return parameter


def get_full_twin_function(parameter):
    'If there is chaining of twins, get the full twin_function'
    func = twin_function
    if parameter.twin:
        return get_top_parent_twin(parameter.twin)
    else:
        return parameter
