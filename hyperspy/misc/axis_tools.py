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
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>


import numpy as np

from hyperspy.api import _ureg


def check_axes_calibration(ax1, ax2, rtol=1e-7):
    """Check if the calibration of two Axis objects matches.

    Raises a logger warning if there is a mismatch.
    ``scale`` and ``offset`` are compared as floats
    using np.allclose, while ``units`` is compared
    with a simple inequality (!=).

    Parameters
    ----------
    ax1, ax2 : Axis objects
        Axes objects that should be compared.
    rtol : float
        Tolerance passed to `np.allclose` for comparison. Default 1e-7.

    Returns
    -------
    bool
        If the two axes have identical calibrations.

    """
    if ax1.size == ax2.size:
        try:
            unit1 = _ureg.Unit(ax1.units)
        except Exception:
            unit1 = ax1.units
        try:
            unit2 = ax2.units
            unit2 = _ureg.Unit(ax2.units)
        except Exception:
            pass
        if np.allclose(ax1.axis, ax2.axis, atol=0, rtol=rtol) and unit1 == unit2:
            return True
    return False
