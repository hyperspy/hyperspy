# Copyright 2007-2026 The HyperSpy developers
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
    from hyperspy.api import _ureg

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


def calculate_convolution1D_axis(f_axis, g_axis):
    """
    Creates an axis that includes padding for convolution. Typically used
    in model fitting implementing the convolution of components with a given functions.

    Parameters
    ----------
    f_axis, g_axis : hyperspy.axes.UniformDataAxis
        The axes of the signals to be convolved.

    Returns
    -------
    numpy.ndarray

    Examples
    --------
    Create a signal with a Lorentzian peak
    >>> f = hs.model.components1D.Lorentzian(centre=220)
    >>> f_signal = hs.signals.Signal1D(f.function(np.arange(200, 300)))
    >>> f_signal.axes_manager.signal_axes.set(offset=200)
    >>> f_signal.plot()

    Create a second signal, for example a detector response
    >>> g = hs.model.components1D.Gaussian(sigma=3)
    >>> g_signal = hs.signals.Signal1D(g.function(np.arange(-20, 20)))
    >>> g_signal.axes_manager.signal_axes.set(offset=-20)
    >>> g_signal.plot()

    Calculate the convolution axis and pad the signal
    >>> convolution_axis = calculate_convolution1D_axis(
    ...     f_signal.axes_manager.signal_axes[0], g_signal.axes_manager.signal_axes[0]
    ...     )
    >>> f_padded_data = f.function(convolution_axis)

    Convolve the data:
    >>> result = np.convolve(f_padded_data, g_signal.data, mode="valid")
    >>> comparison = hs.signals.ComplexSignal1D(f_signal.data + 1j * result)
    >>> comparison.plot()
    """

    for axis_ in [f_axis, g_axis]:
        if not axis_.is_uniform:  # pragma: no cover
            raise ValueError("Only uniform axes are supported.")

    offset = f_axis.offset
    scale = f_axis.scale
    size = f_axis.size + g_axis.size - 1
    offset_index = g_axis.size - g_axis.value2index(0) - 1

    return np.linspace(
        offset - offset_index * scale, offset + scale * (size - 1 - offset_index), size
    )
