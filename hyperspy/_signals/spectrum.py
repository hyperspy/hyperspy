# -*- coding: utf-8 -*-
# Copyright 2007-2015 The HyperSpy developers
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

import warnings

import matplotlib.pyplot as plt
import numpy as np

from hyperspy.exceptions import DataDimensionError
from hyperspy.signal import Signal
from hyperspy.gui.egerton_quantification import SpikesRemoval


class Spectrum(Signal):

    """
    """
    _record_by = 'spectrum'

    def __init__(self, *args, **kwargs):
        Signal.__init__(self, *args, **kwargs)
        self.axes_manager.set_signal_dimension(1)

    def to_image(self):
        """Returns the spectrum as an image.

        See Also
        --------
        as_image : a method for the same purpose with more options.
        signals.Spectrum.to_image : performs the inverse operation on images.

        Raises
        ------
        DataDimensionError: when data.ndim < 2

        """
        if self.data.ndim < 2:
            raise DataDimensionError(
                "A Signal dimension must be >= 2 to be converted to an Image")
        im = self.rollaxis(-1 + 3j, 0 + 3j)
        im.metadata.Signal.record_by = "image"
        im._assign_subclass()
        return im

    def _spikes_diagnosis(self, signal_mask=None,
                          navigation_mask=None):
        """Plots a histogram to help in choosing the threshold for
        spikes removal.

        Parameters
        ----------
        signal_mask: boolean array
            Restricts the operation to the signal locations not marked
            as True (masked)
        navigation_mask: boolean array
            Restricts the operation to the navigation locations not
            marked as True (masked).

        See also
        --------
        spikes_removal_tool

        """
        self._check_signal_dimension_equals_one()
        dc = self.data
        if signal_mask is not None:
            dc = dc[..., ~signal_mask]
        if navigation_mask is not None:
            dc = dc[~navigation_mask, :]
        der = np.abs(np.diff(dc, 1, -1))
        n = ((~navigation_mask).sum() if navigation_mask else
             self.axes_manager.navigation_size)

        # arbitrary cutoff for number of spectra necessary before histogram
        # data is compressed by finding maxima of each spectrum
        tmp = Signal(der) if n < 2000 else Signal(np.ravel(der.max(-1)))

        # get histogram signal using smart binning and plot
        tmph = tmp.get_histogram()
        tmph.plot()

        # Customize plot appearance
        plt.gca().set_title('')
        plt.gca().fill_between(tmph.axes_manager[0].axis,
                               tmph.data,
                               facecolor='#fddbc7',
                               interpolate=True,
                               color='none')
        ax = tmph._plot.signal_plot.ax
        axl = tmph._plot.signal_plot.ax_lines[0]
        axl.set_line_properties(color='#b2182b')
        plt.xlabel('Derivative magnitude')
        plt.ylabel('Log(Counts)')
        ax.set_yscale('log')
        ax.set_ylim(10 ** -1, plt.ylim()[1])
        ax.set_xlim(plt.xlim()[0], 1.1 * plt.xlim()[1])
        plt.draw()

    def spikes_removal_tool(self, signal_mask=None,
                            navigation_mask=None):
        """Graphical interface to remove spikes from EELS spectra.

        Parameters
        ----------
        signal_mask: boolean array
            Restricts the operation to the signal locations not marked
            as True (masked)
        navigation_mask: boolean array
            Restricts the operation to the navigation locations not
            marked as True (masked)

        See also
        --------
        _spikes_diagnosis,

        """
        self._check_signal_dimension_equals_one()
        sr = SpikesRemoval(self,
                           navigation_mask=navigation_mask,
                           signal_mask=signal_mask)
        sr.configure_traits()
        return sr

    def create_model(self):
        """Create a model for the current data.

        Returns
        -------
        model : `Model` instance.

        """

        from hyperspy.model import Model
        model = Model(self)
        return model
