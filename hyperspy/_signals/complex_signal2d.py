# -*- coding: utf-8 -*-
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
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.


from hyperspy._signals.common_signal2d import CommonSignal2D
from hyperspy._signals.complex_signal import (ComplexSignal, LazyComplexSignal)
from hyperspy.docstrings.plot import (
    BASE_PLOT_DOCSTRING, BASE_PLOT_DOCSTRING_PARAMETERS, PLOT2D_DOCSTRING,
    COMPLEX_DOCSTRING, PLOT2D_KWARGS_DOCSTRING)


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
            The fulcrum of the linear ramp is at the origin and the slopes are
            given in units of the axis with the according scale taken into
            account. Both are available via the `axes_manager` of the signal.

        """
        phase = self.phase
        phase.add_ramp(ramp_x, ramp_y, offset)
        self.phase = phase

    def plot(self,
             power_spectrum=False,
             fft_shift=False,
             navigator="auto",
             plot_markers=True,
             autoscale='v',
             saturated_pixels=None,
             norm="auto",
             vmin=None,
             vmax=None,
             gamma=1.0,
             linthresh=0.01,
             linscale=0.1,
             scalebar=True,
             scalebar_color="white",
             axes_ticks=None,
             axes_off=False,
             axes_manager=None,
             no_nans=False,
             colorbar=True,
             centre_colormap="auto",
             min_aspect=0.1,
             **kwargs
             ):
        """%s
        %s
        %s
        %s
        %s

        """
        super().plot(
            power_spectrum=power_spectrum,
            fft_shift=fft_shift,
            navigator=navigator,
            plot_markers=plot_markers,
            autoscale=autoscale,
            saturated_pixels=saturated_pixels,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            gamma=gamma,
            linthresh=linthresh,
            linscale=linscale,
            scalebar=scalebar,
            scalebar_color=scalebar_color,
            axes_ticks=axes_ticks,
            axes_off=axes_off,
            axes_manager=axes_manager,
            no_nans=no_nans,
            colorbar=colorbar,
            centre_colormap=centre_colormap,
            min_aspect=min_aspect,
            **kwargs
        )
    plot.__doc__ %= (BASE_PLOT_DOCSTRING, COMPLEX_DOCSTRING,
                     BASE_PLOT_DOCSTRING_PARAMETERS,
                     PLOT2D_DOCSTRING, PLOT2D_KWARGS_DOCSTRING)


class ComplexSignal2D(Complex2Dmixin, ComplexSignal, CommonSignal2D):

    """BaseSignal subclass for complex 2-dimensional data."""
    pass


class LazyComplexSignal2D(ComplexSignal2D, LazyComplexSignal):

    """BaseSignal subclass for lazy complex 2-dimensional data."""
    pass
