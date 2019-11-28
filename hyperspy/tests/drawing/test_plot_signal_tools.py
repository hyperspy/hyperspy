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
import matplotlib.pyplot as plt

from hyperspy import signals, components1d
from hyperspy._signals.signal1d import BackgroundRemoval
from hyperspy.signal_tools import ImageContrastEditor


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


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR,
                               tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
@pytest.mark.parametrize("gamma", (0.7, 1.2))
@pytest.mark.parametrize("saturated_pixels", (0.3, 0.5))
def test_plot_contrast_editor(gamma, saturated_pixels):
    np.random.seed(1)
    data = np.random.random(size=(10, 10, 100, 100))*1000
    data += np.arange(10*10*100*100).reshape((10, 10, 100, 100))
    s = signals.Signal2D(data)
    s.plot(gamma=gamma, saturated_pixels=saturated_pixels)
    ceditor = ImageContrastEditor(s._plot.signal_plot)
    assert ceditor.gamma == gamma
    assert ceditor.saturated_pixels == saturated_pixels
    return plt.gcf()


@pytest.mark.parametrize("norm", ("linear", "log", "power", "symlog"))
def test_plot_contrast_editor_norm(norm):
    np.random.seed(1)
    data = np.random.random(size=(100, 100))*1000
    data += np.arange(100*100).reshape((100, 100))
    s = signals.Signal2D(data)
    s.plot(norm=norm)
    ceditor = ImageContrastEditor(s._plot.signal_plot)
    if norm == "log":
        # test log with negative numbers
        s2 = s - 5E3
        s2.plot(norm=norm)
        ceditor2 = ImageContrastEditor(s._plot.signal_plot)
    assert ceditor.norm == norm.capitalize()

