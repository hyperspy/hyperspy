# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
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

"""Plotting funtions.

Functions:

plot_spectra, plot_images
    Plot multiple spectra/images in the same figure.
plot_signals
    Plot multiple signals at the same time.
plot_histograms
    Compute and plot the histograms of multiple signals in the same figure.

The :mod:`~hyperspy.api.plot` module contains the following submodules:

:mod:`~hyperspy.api.markers`
        Markers that can be added to `Signal` plots.

"""

from hyperspy.drawing.utils import (plot_histograms, plot_images, plot_signals,
                                    plot_spectra)
from hyperspy.utils import markers
