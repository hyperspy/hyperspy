# Copyright 2007-2021 The HyperSpy developers
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
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>


import numpy as np


def check_axes_calibration(ax1, ax2):
    """Check if the calibration of two Axis objects matches.

    Raises a logger warning if there is a mismatch.
    ``scale`` and ``offset`` are compared as floats
    using np.allclose, while ``units`` is compared
    with a simple inequality (!=).

    Parameters
    ----------
    ax1, ax2 : Axis objects

    Returns
    -------
    bool
        If the two axes have identical calibrations.

    """
    if ax1.size == ax2.size:
        if ax1.is_uniform == ax2.is_uniform == True:
            if np.allclose(ax1.scale, ax2.scale, atol=0, rtol=1e-7) and \
               np.allclose(ax1.offset, ax2.offset, atol=0, rtol=1e-7) and\
               ax1.units == ax2.units:
                return True
        elif hasattr(ax1,'axis') and hasattr(ax2,'axis'):
            if np.allclose(ax1.axis, ax2.axis, atol=0, rtol=1e-7):
                return True
        elif hasattr(ax1,'expression') and hasattr(ax2,'expression'):
            if ax1.expression == ax2.expression and \
               check_axes_calibration(ax1.x, ax2.x):
                return True
    return False
