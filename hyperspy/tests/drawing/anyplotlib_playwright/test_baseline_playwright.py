# -*- coding: utf-8 -*-
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

"""PNG baseline tests for the anyplotlib backend.

The browser-rendered counterpart of the matplotlib
``@pytest.mark.mpl_image_compare`` suites (test_plot_signal1d /
test_plot_signal2d / plot_markers): each test plots a deterministic signal,
screenshots the standalone HTML in headless Chromium, and compares the
result bitwise against a golden PNG using the pixel comparator vendored
from anyplotlib (``_png_utils.compare_arrays``).

Baselines live in ``baselines/<platform>/`` next to this file — a browser
renders text with the host's fonts, so a PNG only matches on the platform
that produced it.  A missing baseline is written on first run (the test
skips so the run that created it does not silently pass); set
``HSPY_UPDATE_APL_BASELINES=1`` to regenerate all.
"""

from __future__ import annotations

import numpy as np

import hyperspy.api as hs

from .conftest import assert_matches_baseline, real_figure


def _sine_stack(n_nav=10, n_pts=100):
    x = np.linspace(0.0, 2.0 * np.pi, n_pts)
    return np.array([np.sin(x + 0.35 * i) + 0.02 * i for i in range(n_nav)])


def _gradient_image(h=32, w=32):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.sin(xx / 4.0) * np.cos(yy / 5.0) + (xx + yy) / (h + w)


def _shot(take_screenshot, s):
    return take_screenshot(real_figure(s._plot.signal_plot.ax.figure))


class TestSignal1DBaselines:
    def test_signal1d_simple(self, apl_backend, take_screenshot):
        s = hs.signals.Signal1D(np.sin(np.linspace(0, 2 * np.pi, 100)))
        s.plot()
        try:
            assert_matches_baseline(_shot(take_screenshot, s), "signal1d_simple")
        finally:
            s._plot.close()

    def test_signal1d_with_navigator(self, apl_backend, take_screenshot):
        s = hs.signals.Signal1D(_sine_stack())
        s.axes_manager.indices = (4,)
        s.plot()
        try:
            assert_matches_baseline(_shot(take_screenshot, s), "signal1d_navigator")
        finally:
            s._plot.close()

    def test_signal1d_vlines_markers(self, apl_backend, take_screenshot):
        s = hs.signals.Signal1D(np.sin(np.linspace(0, 2 * np.pi, 100)))
        s.plot()
        try:
            m = hs.plot.markers.VerticalLines(offsets=np.array([20.0, 50.0, 80.0]))
            s.add_marker(m)
            assert_matches_baseline(
                _shot(take_screenshot, s), "signal1d_vlines_markers"
            )
        finally:
            s._plot.close()


class TestSignal2DBaselines:
    def test_signal2d_simple(self, apl_backend, take_screenshot):
        s = hs.signals.Signal2D(_gradient_image())
        s.plot()
        try:
            assert_matches_baseline(_shot(take_screenshot, s), "signal2d_simple")
        finally:
            s._plot.close()

    def test_signal2d_cmap_no_colorbar(self, apl_backend, take_screenshot):
        s = hs.signals.Signal2D(_gradient_image())
        s.plot(cmap="viridis", colorbar=False)
        try:
            assert_matches_baseline(
                _shot(take_screenshot, s), "signal2d_viridis_nocbar"
            )
        finally:
            s._plot.close()

    def test_signal2d_log_norm(self, apl_backend, take_screenshot):
        s = hs.signals.Signal2D(np.exp(_gradient_image() * 3.0))
        s.plot(norm="log")
        try:
            assert_matches_baseline(_shot(take_screenshot, s), "signal2d_lognorm")
        finally:
            s._plot.close()

    def test_signal2d_with_navigator(self, apl_backend, take_screenshot):
        data = np.array([_gradient_image() * (i + 1) for i in range(6)])
        s = hs.signals.Signal2D(data)
        s.axes_manager.indices = (2,)
        s.plot()
        try:
            assert_matches_baseline(_shot(take_screenshot, s), "signal2d_navigator")
        finally:
            s._plot.close()


class TestMarkerBaselines:
    def test_signal2d_point_markers(self, apl_backend, take_screenshot):
        s = hs.signals.Signal2D(np.zeros((32, 32)))
        s.plot(colorbar=False)
        try:
            m = hs.plot.markers.Points(
                offsets=np.array([[8.0, 8.0], [16.0, 16.0], [24.0, 24.0]]), sizes=20
            )
            s.add_marker(m)
            assert_matches_baseline(_shot(take_screenshot, s), "signal2d_point_markers")
        finally:
            s._plot.close()

    def test_signal2d_mixed_markers(self, apl_backend, take_screenshot):
        s = hs.signals.Signal2D(np.zeros((32, 32)))
        s.plot(colorbar=False)
        try:
            s.add_marker(
                hs.plot.markers.Rectangles(
                    offsets=np.array([[10.0, 10.0]]), widths=[8.0], heights=[6.0]
                )
            )
            s.add_marker(
                hs.plot.markers.Texts(
                    offsets=np.array([[16.0, 26.0]]), texts=["label"], color="yellow"
                )
            )
            s.add_marker(
                hs.plot.markers.Circles(
                    offsets=np.array([[24.0, 8.0]]), sizes=np.array([4.0])
                )
            )
            assert_matches_baseline(_shot(take_screenshot, s), "signal2d_mixed_markers")
        finally:
            s._plot.close()

    def test_navigation_changes_signal_panel(self, apl_backend, take_screenshot):
        """Same signal, two navigation positions -> different golden files."""
        s = hs.signals.Signal1D(_sine_stack())
        s.axes_manager.indices = (0,)
        s.plot()
        try:
            first = _shot(take_screenshot, s)
            s.axes_manager.indices = (9,)
            second = _shot(take_screenshot, s)
            assert_matches_baseline(first, "signal1d_nav_index0")
            assert_matches_baseline(second, "signal1d_nav_index9")
        finally:
            s._plot.close()
