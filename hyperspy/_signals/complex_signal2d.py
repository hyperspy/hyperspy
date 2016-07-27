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


from hyperspy._signals.common_signal2d import CommonSignal2D
from hyperspy._signals.complex_signal import (ComplexSignal, LazyComplexSignal)
from hyperspy.docstrings.plot import (
    BASE_PLOT_DOCSTRING, PLOT2D_DOCSTRING, COMPLEX_DOCSTRING, KWARGS_DOCSTRING)


class Complex2Dmixin:

    """BaseSignal subclass for complex 2-dimensional data."""

    _signal_dimension = 2

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        if self.axes_manager.signal_dimension != 2:
            self.axes_manager.set_signal_dimension(2)

    def add_phase_ramp(self, ramp_x, ramp_y, offset=0):
        """Add a linear phase ramp to the wave.

        Parameters
        ----------
        ramp_x: float
            Slope of the ramp in x-direction.
        ramp_y: float
            Slope of the ramp in y-direction.
        offset: float, optional
            Offset of the ramp at the fulcrum.
        Notes
        -----
            The fulcrum of the linear ramp is at the origin and the slopes are given in units of
            the axis with the according scale taken into account. Both are available via the
            `axes_manager` of the signal.

        """
        phase = self.phase
        phase.add_ramp(ramp_x, ramp_y, offset)
        self.phase = phase

    def plot(self,
             colorbar=True,
             scalebar=True,
             scalebar_color="white",
             axes_ticks=None,
             saturated_pixels=0,
             vmin=None,
             vmax=None,
             no_nans=False,
             centre_colormap="auto",
             **kwargs
             ):
        """%s
        %s
        %s
        %s

        """
        super().plot(
            colorbar=colorbar,
            scalebar=scalebar,
            scalebar_color=scalebar_color,
            axes_ticks=axes_ticks,
            saturated_pixels=saturated_pixels,
            vmin=vmin,
            vmax=vmax,
            no_nans=no_nans,
            centre_colormap=centre_colormap,
            **kwargs
        )
    plot.__doc__ %= (BASE_PLOT_DOCSTRING, PLOT2D_DOCSTRING,
                     COMPLEX_DOCSTRING, KWARGS_DOCSTRING)


class ComplexSignal2D(Complex2Dmixin, ComplexSignal, CommonSignal2D):

    """BaseSignal subclass for complex 2-dimensional data."""
    pass


class LazyComplexSignal2D(Complex2Dmixin, LazyComplexSignal, CommonSignal2D):

    """BaseSignal subclass for lazy complex 2-dimensional data."""
    pass
