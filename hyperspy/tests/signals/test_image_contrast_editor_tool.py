# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
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
from hyperspy.signal_tools import ImageContrastEditor


class TestContrastEditorTool:
    def setup_method(self, method):
        s = hs.signals.Signal2D(np.arange(2 * 3 * 10 * 10).reshape(2, 3, 10, 10))
        self.s = s

    def test_reset_vmin_vmax(self):
        s = self.s
        s.plot(vmin="10th", vmax="99th")

        ceditor = ImageContrastEditor(s._plot.signal_plot)
        np.testing.assert_allclose(ceditor._vmin, 9.9)
        np.testing.assert_allclose(ceditor._vmax, 98.01)

        ceditor._vmin = 20
        ceditor._vmax = 90
        ceditor._reset_original_settings()
        np.testing.assert_allclose(ceditor._vmin, 9.9)
        np.testing.assert_allclose(ceditor._vmax, 98.01)

    def test_reset_span_selector(self):
        s = self.s
        s.plot(vmin="10th", vmax="99th")
        ceditor = ImageContrastEditor(s._plot.signal_plot)

        ceditor.span_selector.extents = (20, 90)
        ceditor._update_image_contrast()
        ax_image = s._plot.signal_plot.ax.images[0]
        np.testing.assert_allclose(ax_image.norm.vmin, 20)
        np.testing.assert_allclose(ax_image.norm.vmax, 90)

        ceditor._clear_span_selector()
        assert not ceditor.span_selector.get_visible()
        np.testing.assert_allclose(ax_image.norm.vmin, 20)
        np.testing.assert_allclose(ax_image.norm.vmax, 90)

        ceditor._update_image_contrast()
        np.testing.assert_allclose(ax_image.norm.vmin, 9.9)
        np.testing.assert_allclose(ax_image.norm.vmax, 98.01)

    def test_change_navigation_coordinate(self):
        s = self.s
        s.plot(vmin="10th", vmax="99th")
        ceditor = ImageContrastEditor(s._plot.signal_plot)

        np.testing.assert_allclose(ceditor._vmin, 9.9)
        np.testing.assert_allclose(ceditor._vmax, 98.01)

        s.axes_manager.indices = (1, 1)
        np.testing.assert_allclose(ceditor._vmin, 409.9)
        np.testing.assert_allclose(ceditor._vmax, 498.01)

    def test_vmin_vmax_changed(self):
        s = self.s
        s.plot(vmin="0th", vmax="100th")

        ceditor = ImageContrastEditor(s._plot.signal_plot)
        np.testing.assert_allclose(ceditor._vmin, 0.0)
        np.testing.assert_allclose(ceditor._vmax, 99.0)

        ceditor._vmin_percentile_changed(0, 10)
        ceditor._vmax_percentile_changed(100, 99)
        np.testing.assert_allclose(ceditor._vmin, 9.9)
        np.testing.assert_allclose(ceditor._vmax, 98.01)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_close_vmin_vmax():
    data = np.random.random(10 * 10 * 10).reshape([10] * 3)
    s = hs.signals.Signal2D(data)
    s.plot()

    image_plot = s._plot.signal_plot
    display_range = (0.6, 0.9)

    ceditor = ImageContrastEditor(image_plot)

    # Simulate selecting a range on the histogram
    ceditor.span_selector.extents = display_range
    plt.pause(0.001)  # in case, interactive backend is used
    ceditor._update_image_contrast()

    # Need to use auto=False to pick up the current display when closing
    ceditor.auto = False
    ceditor.close()

    assert (image_plot.vmin, image_plot.vmax) == display_range
