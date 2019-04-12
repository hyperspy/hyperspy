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
import pytest

from hyperspy import signals, components1d
from hyperspy._signals.signal1d import BackgroundRemoval


BASELINE_DIR = "plot_signal_tools"
DEFAULT_TOL = 2.0
STYLE_PYTEST_MPL = 'default'


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR,
                               tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
def test_plot_BackgroundRemoval():
    pl = components1d.PowerLaw()
    pl.A.value = 1e10
    pl.r.value = 3
    s = signals.Signal1D(pl.function(np.arange(100, 200)))
    s.axes_manager[0].offset = 100

    br = BackgroundRemoval(s,
                           background_type='Power Law',
                           polynomial_order=2,
                           fast=True,
                           plot_remainder=True,
                           show_progressbar=None)

    br.span_selector.set_initial((105, 115))
    br.span_selector.onmove_callback()

    return br.signal._plot.signal_plot.figure
