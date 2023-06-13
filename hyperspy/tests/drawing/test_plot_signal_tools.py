# Copyright 2007-2022 The HyperSpy developers
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
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import matplotlib.pyplot as plt
import numpy as np
import pytest

from hyperspy.datasets.artificial_data import get_core_loss_eels_model
from hyperspy import signals, components1d, datasets
from hyperspy.signal_tools import (
    ImageContrastEditor,
    BackgroundRemoval,
    SpanSelectorInSignal1D,
    Signal1DCalibration,
    )


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

    br.span_selector.extents = (105, 150)
    # will draw the line
    br.span_selector_changed()
    # will update the right axis
    br.span_selector_changed()

    return br.signal._plot.signal_plot.figure


def test_plot_BackgroundRemoval_change_background():
    pl = components1d.PowerLaw()
    pl.A.value = 1e10
    pl.r.value = 3
    s = signals.Signal1D(pl.function(np.arange(100, 200)))
    s.axes_manager[0].offset = 100
    s.add_gaussian_noise(100)

    br = BackgroundRemoval(s,
                           background_type='Power Law',
                           polynomial_order=2,
                           fast=False,
                           plot_remainder=True)

    br.span_selector.extents = (105, 150)
    # will draw the line
    br.span_selector_changed()
    # will update the right axis
    br.span_selector_changed()
    assert isinstance(br.background_estimator, components1d.PowerLaw)
    br.background_type = 'Polynomial'
    assert isinstance(
        br.background_estimator,
        type(components1d.Polynomial(legacy=False))
        )


def test_plot_BackgroundRemoval_close_figure():
    s = signals.Signal1D(np.arange(1000).reshape(10, 100))
    br = BackgroundRemoval(s, background_type='Gaussian')
    signal_plot = s._plot.signal_plot

    assert len(signal_plot.events.closed.connected) == 5
    assert len(s.axes_manager.events.indices_changed.connected) == 4
    s._plot.close()
    assert not br._fit in s.axes_manager.events.indices_changed.connected
    assert not br.disconnect in signal_plot.events.closed.connected


def test_plot_BackgroundRemoval_close_tool():
    s = signals.Signal1D(np.arange(1000).reshape(10, 100))
    br = BackgroundRemoval(s, background_type='Gaussian')
    br.span_selector.extents = (20, 40)
    br.span_selector_changed()
    signal_plot = s._plot.signal_plot

    assert len(signal_plot.events.closed.connected) == 5
    assert len(s.axes_manager.events.indices_changed.connected) == 4
    br.on_disabling_span_selector()
    assert not br._fit in s.axes_manager.events.indices_changed.connected
    s._plot.close()
    assert not br.disconnect in signal_plot.events.closed.connected


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR,
                               tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL)
@pytest.mark.parametrize("gamma", (0.7, 1.2))
@pytest.mark.parametrize("percentile", (["0.15th", "99.85th"], ["0.25th", "99.75th"]))
def test_plot_contrast_editor(gamma, percentile):
    rng = np.random.default_rng(1)
    data = rng.random(size=(10, 10, 100, 100))*1000
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
    rng = np.random.default_rng(1)
    data = rng.random(size=(100, 100))*1000
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


def test_plot_constrast_editor_setting_changed():
    # Test that changing setting works
    rng = np.random.default_rng(1)
    data = rng.random(size=(100, 100))*1000
    data += np.arange(100*100).reshape((100, 100))
    s = signals.Signal2D(data)
    s.plot()
    ceditor = ImageContrastEditor(s._plot.signal_plot)
    ceditor.span_selector.extents = (3E3, 5E3)
    ceditor.update_span_selector_traits()
    np.testing.assert_allclose(ceditor.ss_left_value, 3E3)
    np.testing.assert_allclose(ceditor.ss_right_value, 5E3)
    assert ceditor.auto
    # Do a cycle to trigger traits changed
    ceditor.auto = False
    assert not ceditor.auto
    ceditor.auto = True # reset and clear span selector
    assert ceditor.auto
    assert not ceditor.span_selector.visible
    assert not ceditor._is_selector_visible
    assert not ceditor.line.line.get_visible()
    ceditor.span_selector.extents = (3E3, 5E3)
    ceditor.span_selector.set_visible(True)
    ceditor.update_line()
    assert ceditor._is_selector_visible
    assert ceditor.line.line.get_visible()

    assert ceditor.bins == 24
    assert ceditor.line.axis.shape == (ceditor.bins, )
    ceditor.bins = 50
    assert ceditor.bins == 50
    assert ceditor.line.axis.shape == (ceditor.bins, )

    # test other parameters
    ceditor.linthresh = 0.1
    assert ceditor.image.linthresh == 0.1

    ceditor.linscale = 0.5
    assert ceditor.image.linscale == 0.5


def test_plot_constrast_editor_auto_indices_changed():
    rng = np.random.default_rng(1)
    data = rng.random(size=(10, 10, 100, 100))*1000
    data += np.arange(10*10*100*100).reshape((10, 10, 100, 100))
    s = signals.Signal2D(data)
    s.plot()
    ceditor = ImageContrastEditor(s._plot.signal_plot)
    ceditor.span_selector.extents = (3E3, 5E3)
    ceditor.update_span_selector_traits()
    s.axes_manager.indices = (0, 1)
    # auto is None by default, the span selector need to be removed:
    assert not ceditor.span_selector.visible
    assert not ceditor._is_selector_visible
    ref_value = (100020.046452, 110953.450532)
    np.testing.assert_allclose(ceditor._get_current_range(), ref_value)

    # Change auto to False
    ceditor.auto = False
    s.axes_manager.indices = (0, 2)
    # vmin, vmax shouldn't have changed
    np.testing.assert_allclose(ceditor._get_current_range(), ref_value)


def test_plot_constrast_editor_reset():
    rng = np.random.default_rng(1)
    data = rng.random(size=(100, 100))*1000
    data += np.arange(100*100).reshape((100, 100))
    s = signals.Signal2D(data)
    s.plot()
    ceditor = ImageContrastEditor(s._plot.signal_plot)
    ceditor.span_selector.extents = (3E3, 5E3)
    ceditor._update_image_contrast()
    vmin, vmax = 36.559113, 10960.787649
    np.testing.assert_allclose(ceditor._vmin, vmin)
    np.testing.assert_allclose(ceditor._vmax, vmax)
    np.testing.assert_allclose(ceditor._get_current_range(), (3E3, 5E3))

    ceditor.reset()
    assert not ceditor.span_selector.visible
    assert not ceditor._is_selector_visible
    np.testing.assert_allclose(ceditor._get_current_range(), (vmin, vmax))
    np.testing.assert_allclose(ceditor.image._vmin, vmin)
    np.testing.assert_allclose(ceditor.image._vmax, vmax)


def test_plot_constrast_editor_apply():
    rng = np.random.default_rng(1)
    data = rng.random(size=(100, 100))*1000
    data += np.arange(100*100).reshape((100, 100))
    s = signals.Signal2D(data)
    s.plot()
    ceditor = ImageContrastEditor(s._plot.signal_plot)
    ceditor.span_selector.extents = (3E3, 5E3)
    ceditor._update_image_contrast()
    image_vmin_vmax = ceditor.image._vmin, ceditor.image._vmax
    ceditor.apply()
    assert not ceditor.span_selector.visible
    assert not ceditor._is_selector_visible
    np.testing.assert_allclose(
        (ceditor.image._vmin, ceditor.image._vmax),
        image_vmin_vmax,
        )


def test_span_selector_in_signal1d():
    s = signals.Signal1D(np.arange(1000).reshape(10, 100))
    calibration_tool = SpanSelectorInSignal1D(s)
    calibration_tool.span_selector.extents = (20, 40)
    calibration_tool.span_selector_changed()
    calibration_tool.span_selector.extents = (10.1, 10.2)
    calibration_tool.span_selector_changed()


def test_span_selector_in_signal1d_model():
    m = get_core_loss_eels_model()
    calibration_tool = SpanSelectorInSignal1D(m)
    assert len(m.signal._plot.signal_plot.ax_lines) == 2
    assert m.signal is calibration_tool.signal
    calibration_tool.span_selector.extents = (420, 460)
    calibration_tool.span_selector_changed()
    calibration_tool.span_selector_switch(False)
    assert calibration_tool.span_selector is None


def test_signal1d_calibration():
    s = signals.Signal1D(np.arange(1000).reshape(10, 100))
    s.axes_manager[-1].scale = 0.1
    calibration_tool = Signal1DCalibration(s)
    np.testing.assert_allclose(
        calibration_tool.span_selector.snap_values,
        s.axes_manager.signal_axes[0].axis
        )
    calibration_tool.span_selector.extents = (2.0, 4.0)
    calibration_tool.span_selector_changed()
    assert calibration_tool.ss_left_value == 2.0
    assert calibration_tool.ss_right_value == 4.0
    calibration_tool.span_selector.extents = (3.02, 5.09)
    np.testing.assert_allclose(
        calibration_tool.span_selector.extents,
        (3.0, 5.1)
        )
    calibration_tool.span_selector_changed()
    np.testing.assert_allclose(calibration_tool.ss_left_value, 3.0)
    np.testing.assert_allclose(calibration_tool.ss_right_value, 5.1)
