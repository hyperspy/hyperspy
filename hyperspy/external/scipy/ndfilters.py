# Copyright (c) 2001-2002 Enthought, Inc. 2003-2026, SciPy Developers.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from collections.abc import Iterable
import operator
import warnings

import numpy as np


def _check_axes(axes, ndim):
    if axes is None:
        return tuple(range(ndim))
    elif np.isscalar(axes):
        axes = (operator.index(axes),)
    elif isinstance(axes, Iterable):
        for ax in axes:
            axes = tuple(operator.index(ax) for ax in axes)
            if ax < -ndim or ax > ndim - 1:
                raise ValueError(f"specified axis: {ax} is out of range")
        axes = tuple(ax % ndim if ax < 0 else ax for ax in axes)
    else:
        message = "axes must be an integer, iterable of integers, or None"
        raise ValueError(message)
    if len(tuple(set(axes))) != len(axes):
        raise ValueError("axes must be unique")
    return axes


def _normalize_sequence(input, rank):
    """If input is a scalar, create a sequence of length equal to the
    rank by duplicating the input. If input is a sequence,
    check if its length is equal to the length of array.
    """
    is_str = isinstance(input, str)
    if not is_str and isinstance(input, Iterable):
        normalized = list(input)
        if len(normalized) != rank:
            err = "sequence argument must have length equal to input rank"
            raise RuntimeError(err)
    else:
        normalized = [input] * rank
    return normalized



def _get_footprint(
        input, size=None, footprint=None, mode="reflect", origin=0, axes=None
        ):

    if (size is not None) and (footprint is not None):
        warnings.warn("ignoring size because footprint is set",
                      UserWarning, stacklevel=3)
    # input = np.asarray(input)
    if np.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    axes = _check_axes(axes, input.ndim)
    num_axes = len(axes)
    origins = _normalize_sequence(origin, num_axes)
    if footprint is None:
        if size is None:
            raise RuntimeError("no footprint or filter size provided")
        sizes = _normalize_sequence(size, num_axes)
        footprint = np.ones(sizes, dtype=bool)
    else:
        footprint = np.asarray(footprint, dtype=bool)
    if num_axes < input.ndim:
        # set origin = 0 for any axes not being filtered
        origins_temp = [0,] * input.ndim
        for o, ax in zip(origins, axes):
            origins_temp[ax] = o
        origins = origins_temp

        if not isinstance(mode, str) and isinstance(mode, Iterable):
            # set mode = 'constant' for any axes not being filtered
            modes = _normalize_sequence(mode, num_axes)
            modes_temp = ['constant'] * input.ndim
            for m, ax in zip(modes, axes):
                modes_temp[ax] = m
            mode = modes_temp

        # insert singleton dimension along any non-filtered axes
        if footprint.ndim != num_axes:
            raise RuntimeError("footprint array has incorrect shape")
        footprint = np.expand_dims(
            footprint,
            tuple(ax for ax in range(input.ndim) if ax not in axes)
        )

    return footprint
