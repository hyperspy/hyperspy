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

import matplotlib.pyplot as plt
import numpy as np
import pytest

from hyperspy import signals, components1d, datasets
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
    s.add_poissonian_noise(random_state=1)

    br = BackgroundRemoval(s,
                           background_type='Power Law',
                           polynomial_order=2,
                           fast=False,
                           plot_remainder=True)

    br.span_selector.set_initial((105, 150))
    br.span_selector.onmove_callback()
    br.span_selector_changed()

    return br.signal._plot.signal_plot.figure


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR,
                               tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
@pytest.mark.parametrize("gamma", (0.7, 1.2))
@pytest.mark.parametrize("percentile", (["0.15th", "99.85th"], ["0.25th", "99.75th"]))
def test_plot_contrast_editor(gamma, percentile):
    np.random.seed(1)
    data = np.random.random(size=(10, 10, 100, 100))*1000
    data += np.arange(10*10*100*100).reshape((10, 10, 100, 100))
    s = signals.Signal2D(data)
    s.plot(gamma=gamma, vmin=percentile[0], vmax=percentile[1])
    ceditor = ImageContrastEditor(s._plot.signal_plot)
    assert ceditor.gamma == gamma
    assert ceditor.vmin_percentile == float(percentile[0].split("th")[0])
    assert ceditor.vmax_percentile == float(percentile[1].split("th")[0])
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
        _ = ImageContrastEditor(s._plot.signal_plot)
    assert ceditor.norm == norm.capitalize()


def test_plot_contrast_editor_complex():
    s = datasets.example_signals.object_hologram()
    fft = s.fft(True)
    fft.plot(True, vmin=None, vmax=None)
    ceditor = ImageContrastEditor(fft._plot.signal_plot)
    assert ceditor.bins == 250
    np.testing.assert_allclose(ceditor._vmin, fft._plot.signal_plot._vmin)
    np.testing.assert_allclose(ceditor._vmax, fft._plot.signal_plot._vmax)
    np.testing.assert_allclose(ceditor._vmin, 1.495977361e+3)
    np.testing.assert_allclose(ceditor._vmax, 3.568838458887e+17)
