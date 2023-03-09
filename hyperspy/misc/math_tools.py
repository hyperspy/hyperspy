# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
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

from functools import reduce
import math
import numbers
from packaging.version import Version
import warnings

import numpy as np
import dask
import dask.array as da


def symmetrize(a):
    return a + a.swapaxes(0, 1) - np.diag(a.diagonal())


def antisymmetrize(a):
    return a - a.swapaxes(0, 1) + np.diag(a.diagonal())


def closest_nice_number(number):
    oom = 10 ** math.floor(math.log10(number))
    return oom * (number // oom)


def get_linear_interpolation(p1, p2, x):
    """Given two points in 2D returns y for a given x for y = ax + b

    Parameters
    ----------
    p1,p2 : (x, y)
    x : float

    Returns
    -------
    y : float

    """
    x1, y1 = p1
    x2, y2 = p2
    a = (y2 - y1) / (x2 - x1)
    b = (x2 * y1 - x1 * y2) / (x2 - x1)
    y = a * x + b
    return y


def order_of_magnitude(number):
    """Order of magnitude of the given number

    Parameters
    ----------
    number : float

    Returns
    -------
    Float
    """
    return math.floor(math.log10(number))


def isfloat(number):
    """Check if a number or array is of float type.

    This is necessary because e.g. isinstance(np.float32(2), float) is False.

    """
    if hasattr(number, "dtype"):
        return np.issubdtype(number, np.floating)
    else:
        return isinstance(number, float)


def anyfloatin(things):
    """Check if iterable contains any non integer."""
    for n in things:
        if isfloat(n) and not n.is_integer():
            return True
    return False


def outer_nd(*vec):
    """
    Calculates outer product of n vectors

    Parameters
    ----------
    vec : vector

    Return
    ------
    out : ndarray
    """
    return reduce(np.multiply.outer, vec)


def hann_window_nth_order(m, order):
    """
    Calculates 1D Hann window of nth order

    Parameters
    ----------
    m : int
        number of points in window (typically the length of a signal)
    order : int
        Filter order

    Return
    ------
    window : array
        window
    """
    if not isinstance(m, int) or m <= 0:
        raise ValueError('Parameter m has to be positive integer greater than 0.')
    if not isinstance(order, int) or order <= 0:
        raise ValueError('Filter order has to be positive integer greater than 0.')
    sin_arg = np.pi * (m - 1.) / m
    cos_arg = 2. * np.pi / (m - 1.) * (np.arange(m))

    return m / (order * 2 * np.pi) * sum([(-1) ** i / i *
                                          np.sin(i * sin_arg) * (np.cos(i * cos_arg) - 1)
                                          for i in range(1, order + 1)])


def optimal_fft_size(target, real=False):
    """Wrapper around scipy function next_fast_len() for calculating optimal FFT padding.

    scipy.fft was only added in 1.4.0, so we fall back to scipy.fftpack
    if it is not available. The main difference is that next_fast_len()
    does not take a second argument in the older implementation.

    Parameters
    ----------
    target : int
        Length to start searching from. Must be a positive integer.
    real : bool, optional
        True if the FFT involves real input or output, only available
        for scipy > 1.4.0

    Returns
    -------
    int
        Optimal FFT size.

    """

    try:
        from scipy.fft import next_fast_len

        support_real = True

    except ImportError:  # pragma: no cover
        from scipy.fftpack import next_fast_len

        support_real = False

    if support_real:
        return next_fast_len(target, real)
    else:  # pragma: no cover
        return next_fast_len(target)


def check_random_state(seed, lazy=False):
    """Turn a random seed into a RandomState or Generator instance.

    Parameters
    ----------
    seed : None or int or np.random.RandomState or np.random.Generator or
        dask.array.random.RandomState or da.random.Generator
        If None:
            Return the random state singleton used by
            np.random or dask.array.random
        If int:
            Return a new random state instance seeded with ``seed``.
        If np.random.RandomState, np.random.Generator,
        dask.array.random.RandomState:
            Return seed.
    lazy : bool, default False
        If True, and seed is ``None`` or ``int``, return
        a dask.array.random.RandomState instance instead for dask < 2023.2.1,
        otherwise returns a dask.array.random.Generator instance

    Returns
    -------
    np.random.Generator instance or

    """
    # Derived from `sklearn.utils.check_random_state`.
    # Copyright (c) 2007-2020 The scikit-learn developers.
    # All rights reserved.
    dask_version = Version(dask.__version__)
    if seed is None:
        if lazy:
            if dask_version < Version('2022.10.0'):
                return da.random._state
            elif dask_version < Version('2023.2.1'):
                backend = da.backends.array_creation_dispatch.backend
                if backend not in da.random._cached_random_states.keys():
                    # Need to initialise the backend
                    da.random.seed()
                return da.random._cached_random_states[backend]
            else:
                return da.random.default_rng()
        else:
            return np.random.default_rng()

    if isinstance(seed, numbers.Integral):
        if lazy:
            try:
                return da.random.default_rng(seed)
            except AttributeError:
                return da.random.RandomState(seed)
        else:
            return np.random.default_rng(seed)

    if isinstance(seed, (np.random.RandomState, da.random.RandomState)):
        warnings.warn(
            "Support for RandomState generators have been deprecated and will be removed "
            " in HyperSpy 2.0, use `default_rng` instead.",
            DeprecationWarning
            )
        return seed

    if isinstance(seed, np.random.Generator):
        return seed

    if (dask_version >= Version('2023.2.1')
            and isinstance(seed, da.random.Generator)):
        return seed

    raise ValueError(
        f"{seed} cannot be used to seed a RandomState or a Generator instance"
        )
