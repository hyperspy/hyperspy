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
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import matplotlib.pyplot as plt
import numpy as np
import pytest

import hyperspy.api as hs
from hyperspy import components1d, signals
from hyperspy.exceptions import SignalDimensionError
from hyperspy.signal_tools import (
    BackgroundRemoval,
    ImageContrastEditor,
    LineInSignal1D,
    LineInSignal2D,
    Signal1DCalibration,
    Signal2DCalibration,
    SpanSelectorInSignal1D,
)

BASELINE_DIR = "plot_signal_tools"
DEFAULT_TOL = 2.0
STYLE_PYTEST_MPL = "default"


@pytest.mark.mpl_image_compare(
    baseline_dir=BASELINE_DIR, tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL
)
def test_plot_BackgroundRemoval():
    pl = components1d.PowerLaw()
    pl.A.value = 1e10
    pl.r.value = 3
    s = signals.Signal1D(pl.function(np.arange(100, 200)))
    s.axes_manager[0].offset = 100
    s.add_poissonian_noise(random_state=1)

    br = BackgroundRemoval(
        s,
        background_type="Power Law",
        polynomial_order=2,
        fast=False,
        plot_remainder=True,
    )

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

    br = BackgroundRemoval(
        s,
        background_type="Power Law",
        polynomial_order=2,
        fast=False,
        plot_remainder=True,
    )

    br.span_selector.extents = (105, 150)
    # will draw the line
    br.span_selector_changed()
    # will update the right axis
    br.span_selector_changed()
    assert isinstance(br.background_estimator, components1d.PowerLaw)
    br.background_type = "Polynomial"
    assert isinstance(br.background_estimator, type(components1d.Polynomial()))


def test_plot_BackgroundRemoval_close_figure():
    s = signals.Signal1D(np.arange(1000).reshape(10, 100))
    br = BackgroundRemoval(s, background_type="Gaussian")
    signal_plot = s._plot.signal_plot

    assert len(signal_plot.events.closed.connected) == 3
    assert len(s._plot.events.closed.connected) == 1
    assert len(s.axes_manager.events.indices_changed.connected) == 4
    s._plot.close()
    assert br._fit not in s.axes_manager.events.indices_changed.connected
    assert len(s._plot.events.closed.connected) == 0
    assert len(signal_plot.events.closed.connected) == 0


def test_plot_BackgroundRemoval_close_tool():
    s = signals.Signal1D(np.arange(1000).reshape(10, 100))
    br = BackgroundRemoval(s, background_type="Gaussian")
    br.span_selector.extents = (20, 40)
    br.span_selector_changed()
    signal_plot = s._plot.signal_plot

    assert len(signal_plot.events.closed.connected) == 3
    assert len(s._plot.events.closed.connected) == 1
    assert len(s.axes_manager.events.indices_changed.connected) == 4
    br.on_disabling_span_selector()
    assert br._fit not in s.axes_manager.events.indices_changed.connected
    s._plot.close()
    assert len(s._plot.events.closed.connected) == 0
    assert len(signal_plot.events.closed.connected) == 0


@pytest.mark.mpl_image_compare(
    baseline_dir=BASELINE_DIR, tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL
)
@pytest.mark.parametrize("gamma", (0.7, 1.2))
@pytest.mark.parametrize("percentile", (["0.15th", "99.85th"], ["0.25th", "99.75th"]))
def test_plot_contrast_editor(gamma, percentile):
    rng = np.random.default_rng(1)
    data = rng.random(size=(10, 10, 100, 100)) * 1000
    data += np.arange(10 * 10 * 100 * 100).reshape((10, 10, 100, 100))
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
    data = rng.random(size=(100, 100)) * 1000
    data += np.arange(100 * 100).reshape((100, 100))
    s = signals.Signal2D(data)
    s.plot(norm=norm)
    ceditor = ImageContrastEditor(s._plot.signal_plot)
    if norm == "log":
        # test log with negative numbers
        s2 = s - 5e3
        s2.plot(norm=norm)
        _ = ImageContrastEditor(s._plot.signal_plot)
    assert ceditor.norm == norm.capitalize()


def test_plot_contrast_editor_complex():
    s = hs.data.wave_image(random_state=0)

    fft = s.fft(True)
    fft.plot(True, vmin=None, vmax=None)
    ceditor = ImageContrastEditor(fft._plot.signal_plot)
    assert ceditor.bins == 250
    np.testing.assert_allclose(ceditor._vmin, fft._plot.signal_plot._vmin)
    np.testing.assert_allclose(ceditor._vmax, fft._plot.signal_plot._vmax)
    np.testing.assert_allclose(ceditor._vmin, 0.2002909426101699)
    np.testing.assert_allclose(ceditor._vmax, 1074314272.3907123)


def test_plot_constrast_editor_setting_changed():
    # Test that changing setting works
    rng = np.random.default_rng(1)
    data = rng.random(size=(100, 100)) * 1000
    data += np.arange(100 * 100).reshape((100, 100))
    s = signals.Signal2D(data)
    s.plot()
    ceditor = ImageContrastEditor(s._plot.signal_plot)
    ceditor.span_selector.extents = (3e3, 5e3)
    ceditor.update_span_selector_traits()
    np.testing.assert_allclose(ceditor.ss_left_value, 3e3)
    np.testing.assert_allclose(ceditor.ss_right_value, 5e3)
    assert ceditor.auto
    # Do a cycle to trigger traits changed
    ceditor.auto = False
    assert not ceditor.auto
    ceditor.auto = True  # reset and clear span selector
    assert ceditor.auto
    assert not ceditor.span_selector.get_visible()
    assert not ceditor._is_selector_visible
    assert not ceditor.line.line.get_visible()
    ceditor.span_selector.extents = (3e3, 5e3)
    ceditor.span_selector.set_visible(True)
    ceditor.update_line()
    assert ceditor._is_selector_visible
    assert ceditor.line.line.get_visible()

    assert ceditor.bins == 24
    assert ceditor.line.axis.shape == (ceditor.bins,)
    ceditor.bins = 50
    assert ceditor.bins == 50
    assert ceditor.line.axis.shape == (ceditor.bins,)

    # test other parameters
    ceditor.linthresh = 0.1
    assert ceditor.image.linthresh == 0.1

    ceditor.linscale = 0.5
    assert ceditor.image.linscale == 0.5


def test_plot_constrast_editor_auto_indices_changed():
    rng = np.random.default_rng(1)
    data = rng.random(size=(10, 10, 100, 100)) * 1000
    data += np.arange(10 * 10 * 100 * 100).reshape((10, 10, 100, 100))
    s = signals.Signal2D(data)
    s.plot()
    ceditor = ImageContrastEditor(s._plot.signal_plot)
    ceditor.span_selector.extents = (3e3, 5e3)
    ceditor.update_span_selector_traits()
    s.axes_manager.indices = (0, 1)
    # auto is None by default, the span selector need to be removed:
    assert not ceditor.span_selector.get_visible()
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
    data = rng.random(size=(100, 100)) * 1000
    data += np.arange(100 * 100).reshape((100, 100))
    s = signals.Signal2D(data)
    s.plot()
    ceditor = ImageContrastEditor(s._plot.signal_plot)
    ceditor.span_selector.extents = (3e3, 5e3)
    ceditor._update_image_contrast()
    vmin, vmax = 36.559113, 10960.787649
    np.testing.assert_allclose(ceditor._vmin, vmin)
    np.testing.assert_allclose(ceditor._vmax, vmax)
    np.testing.assert_allclose(ceditor._get_current_range(), (3e3, 5e3))

    ceditor.reset()
    assert not ceditor.span_selector.get_visible()
    assert not ceditor._is_selector_visible
    np.testing.assert_allclose(ceditor._get_current_range(), (vmin, vmax))
    np.testing.assert_allclose(ceditor.image._vmin, vmin)
    np.testing.assert_allclose(ceditor.image._vmax, vmax)


def test_plot_constrast_editor_apply():
    rng = np.random.default_rng(1)
    data = rng.random(size=(100, 100)) * 1000
    data += np.arange(100 * 100).reshape((100, 100))
    s = signals.Signal2D(data)
    s.plot()
    ceditor = ImageContrastEditor(s._plot.signal_plot)
    ceditor.span_selector.extents = (3e3, 5e3)
    ceditor._update_image_contrast()
    image_vmin_vmax = ceditor.image._vmin, ceditor.image._vmax
    ceditor.apply()
    assert not ceditor.span_selector.get_visible()
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
    m = hs.data.two_gaussians().create_model()
    calibration_tool = SpanSelectorInSignal1D(m)
    assert len(m.signal._plot.signal_plot.ax_lines) == 2
    assert m.signal is calibration_tool.signal
    calibration_tool.span_selector.extents = (40, 60)
    calibration_tool.span_selector_changed()
    calibration_tool.span_selector_switch(False)
    assert calibration_tool.span_selector is None


def test_signal1d_calibration():
    s = signals.Signal1D(np.arange(1000).reshape(10, 100))
    s.axes_manager[-1].scale = 0.1
    calibration_tool = Signal1DCalibration(s)
    np.testing.assert_allclose(
        calibration_tool.span_selector.snap_values, s.axes_manager.signal_axes[0].axis
    )
    calibration_tool.span_selector.extents = (2.0, 4.0)
    calibration_tool.span_selector_changed()
    assert calibration_tool.ss_left_value == 2.0
    assert calibration_tool.ss_right_value == 4.0
    calibration_tool.span_selector.extents = (3.02, 5.09)
    np.testing.assert_allclose(calibration_tool.span_selector.extents, (3.0, 5.1))
    calibration_tool.span_selector_changed()
    np.testing.assert_allclose(calibration_tool.ss_left_value, 3.0)
    np.testing.assert_allclose(calibration_tool.ss_right_value, 5.1)


def test_line_in_signal1d():
    s = signals.Signal1D(np.arange(1000).reshape(10, 100))
    axis = s.axes_manager.signal_axes[0]
    line = LineInSignal1D(s)
    # default position is in the middle of the signal axis
    assert line.position == (axis.high_value - axis.low_value) / 2
    line.position = 30
    assert line.position == line._line.position[0] == 30
    assert len(s._plot.signal_plot.ax.get_lines()) == 2

    # Remove the line
    line.on = False
    assert line._line is None
    assert len(s._plot.signal_plot.ax.get_lines()) == 1

    # Add the line; default position is used
    line.on = True
    assert line._line is not None
    assert line.position == (axis.high_value - axis.low_value) / 2
    assert len(s._plot.signal_plot.ax.get_lines()) == 2

    # Check disconnection on figure close
    s._plot.close()
    assert line.on is False
    assert line._line is None

    # this does nothing because the figure is closed
    line.on = True
    assert line.on is False
    assert line._line is None

    # Re-open the signal plot and add the line back
    s.plot()
    line.on = True
    assert line._line is not None


def test_line_in_signal1d_wrong_dimension():
    # Test that LineInSignal1D raises error for non-1D signals
    s2d = signals.Signal2D(np.arange(100).reshape(10, 10))
    with pytest.raises(SignalDimensionError):
        LineInSignal1D(s2d)


def test_line_in_signal1d_with_navigation():
    # Check the correct axis are used for the line
    s = signals.Signal1D(np.arange(50).reshape(5, 10))
    line = LineInSignal1D(s)
    assert line._axis is s.axes_manager.signal_axes[0]
    assert line._line.axes[0] is s.axes_manager.signal_axes[0]


def test_line_in_signal2d():
    s = signals.Signal2D(np.arange(10000).reshape(100, 100))
    xaxis = s.axes_manager.signal_axes[0]
    yaxis = s.axes_manager.signal_axes[1]
    line = LineInSignal2D(s)

    # Check initial position is set to default values
    expected_x0 = xaxis.low_value + (xaxis.high_value - xaxis.low_value) / 4
    expected_y0 = yaxis.low_value + (yaxis.high_value - yaxis.low_value) / 4
    expected_x1 = xaxis.high_value - (xaxis.high_value - xaxis.low_value) / 4
    expected_y1 = yaxis.high_value - (yaxis.high_value - yaxis.low_value) / 4

    # Check that line coordinates are properly initialized
    np.testing.assert_allclose(line.x0, expected_x0)
    np.testing.assert_allclose(line.y0, expected_y0)
    np.testing.assert_allclose(line.x1, expected_x1)
    np.testing.assert_allclose(line.y1, expected_y1)

    # Check that the line widget was created
    assert line._line is not None
    assert line.on is True

    # Test setting custom line position
    line.x0 = 10
    line.y0 = 20
    line.x1 = 30
    line.y1 = 40
    assert line.x0 == line._line.position[0][0] == 10
    assert line.y0 == line._line.position[0][1] == 20
    assert line.x1 == line._line.position[1][0] == 30
    assert line.y1 == line._line.position[1][1] == 40

    # Check that length is calculated correctly
    expected_length = np.sqrt((30 - 10) ** 2 + (40 - 20) ** 2)
    np.testing.assert_allclose(line.length, expected_length)

    # Remove the line
    line.on = False
    assert line._line is None

    # Add the line back - should restore to default position
    line.on = True
    assert line._line is not None
    np.testing.assert_allclose(line.x0, expected_x0)
    np.testing.assert_allclose(line.y0, expected_y0)
    np.testing.assert_allclose(line.x1, expected_x1)
    np.testing.assert_allclose(line.y1, expected_y1)

    # Check disconnection on figure close
    s._plot.close()
    assert line.on is False
    assert line._line is None

    # this does nothing because the figure is closed
    line.on = True
    assert line._line is None

    # Re-open the signal plot and add the line back
    s.plot()
    line.on = True
    assert line._line is not None


def test_line_in_signal2d_wrong_dimension():
    # Test that LineInSignal2D raises error for non-2D signals
    s1d = signals.Signal1D(np.arange(100))
    with pytest.raises(SignalDimensionError):
        LineInSignal2D(s1d)


def test_line_in_signal2d_with_navigation():
    # Check the correct axes are used for the line
    s = signals.Signal2D(np.arange(1000).reshape(2, 5, 10, 10))
    line = LineInSignal2D(s)
    assert line._xaxis is s.axes_manager.signal_axes[0]
    assert line._yaxis is s.axes_manager.signal_axes[1]
    assert line._line.axes[0] is s.axes_manager.signal_axes[0]
    assert line._line.axes[1] is s.axes_manager.signal_axes[1]


def test_signal2d_calibration():
    s = signals.Signal2D(np.arange(10000).reshape(100, 100))
    s.axes_manager[0].scale = 0.5
    s.axes_manager[1].scale = 0.5
    s.axes_manager[0].units = "nm"
    s.axes_manager[1].units = "nm"

    calibration_tool = Signal2DCalibration(s)

    # Check initialization
    assert calibration_tool.units == "nm"
    assert calibration_tool.scale == 0.5
    assert calibration_tool.on is True
    assert calibration_tool._line is not None

    # Set line position
    position = (10.0, 10.0), (30.0, 30.0)
    calibration_tool.x0 = position[0][0]
    calibration_tool.y0 = position[0][1]
    calibration_tool.x1 = position[1][0]
    calibration_tool.y1 = position[1][1]

    # Check that length is calculated
    expected_length = np.sqrt((30.0 - 10.0) ** 2 + (30.0 - 10.0) ** 2)
    np.testing.assert_allclose(calibration_tool.length, expected_length)

    # Set new length and check scale calculation is called
    new_length = 50.0
    calibration_tool.new_length = new_length
    expected_scale = s._get_signal2d_scale(*position[0], *position[1], new_length)
    np.testing.assert_allclose(calibration_tool.scale, expected_scale)

    # Test that changing line position updates scale
    new_position = (40.0, 40.0)
    calibration_tool.x1 = new_position[0]
    calibration_tool.y1 = new_position[1]
    expected_scale = s._get_signal2d_scale(*position[0], *new_position, new_length)
    np.testing.assert_allclose(calibration_tool.scale, expected_scale)

    # Test units change
    calibration_tool.units = "um"
    assert calibration_tool.units == "um"


def test_signal2d_calibration_wrong_dimension():
    # Test that Signal2DCalibration raises error for non-2D signals
    s1d = signals.Signal1D(np.arange(100))
    with pytest.raises(SignalDimensionError):
        Signal2DCalibration(s1d)
