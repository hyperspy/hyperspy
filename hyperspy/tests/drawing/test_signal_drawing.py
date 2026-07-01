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

"""Unit tests for the low-level component/loading plot helpers in
hyperspy.drawing.signal (_plot_1D_component, _plot_2D_component,
_plot_loading). These are only reachable through BaseSignal._plot_factors /
_plot_loadings with an explicit ax already created, so some branches
(ax=None, calibrate=False, comp_label combinations) are exercised here via
direct calls instead.
"""

import numpy as np
import pytest

from hyperspy.drawing.signal import (
    _plot_1D_component,
    _plot_2D_component,
    _plot_loading,
)
from hyperspy.signals import Signal1D, Signal2D


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def test_plot_1d_component_creates_axes_when_none():
    s = Signal1D(np.random.random((10, 100)))
    factors = np.random.random((100, 5))
    ax = _plot_1D_component(factors, 0, s.axes_manager, ax=None)
    assert ax is not None


def test_plot_1d_component_uncalibrated_uses_channel_index_label():
    s = Signal1D(np.random.random((10, 100)))
    factors = np.random.random((100, 5))
    ax = _plot_1D_component(factors, 0, s.axes_manager, calibrate=False)
    assert ax.get_xlabel() == "Channel index"


def test_plot_1d_component_sets_title_when_not_same_window():
    s = Signal1D(np.random.random((10, 100)))
    factors = np.random.random((100, 5))
    ax = _plot_1D_component(
        factors, 0, s.axes_manager, comp_label="my component", same_window=False
    )
    assert ax.get_title() == "my component"


def test_plot_2d_component_creates_axes_when_none():
    s = Signal2D(np.random.random((5, 10, 10)))
    factors = np.random.random((100, 5))
    ax = _plot_2D_component(factors, 0, s.axes_manager, ax=None)
    assert ax is not None


def test_plot_loading_nav2_creates_axes_when_none():
    s = Signal1D(np.random.random((6, 7, 100)))
    loadings = np.random.random((5, 6 * 7))
    _plot_loading(loadings, 0, s.axes_manager, ax=None)


def test_plot_loading_nav2_uncalibrated_uses_pixel_labels():
    s = Signal1D(np.random.random((6, 7, 100)))
    loadings = np.random.random((5, 6 * 7))
    import matplotlib.pyplot as plt

    ax = plt.gca()
    _plot_loading(loadings, 0, s.axes_manager, ax=ax, calibrate=False)
    assert ax.get_xlabel() == "pixels"
    assert ax.get_ylabel() == "pixels"


def test_plot_loading_nav2_title_without_same_window():
    s = Signal1D(np.random.random((6, 7, 100)))
    loadings = np.random.random((5, 6 * 7))
    import matplotlib.pyplot as plt

    ax = plt.gca()
    _plot_loading(loadings, 0, s.axes_manager, ax=ax, comp_label="c", same_window=False)
    assert ax.get_title() == "0 #0"


def test_plot_loading_nav1_creates_axes_when_none():
    s = Signal1D(np.random.random((6, 100)))
    loadings = np.random.random((5, 6))
    _plot_loading(loadings, 0, s.axes_manager, ax=None)


def test_plot_loading_nav1_title_without_same_window():
    s = Signal1D(np.random.random((6, 100)))
    loadings = np.random.random((5, 6))
    import matplotlib.pyplot as plt

    ax = plt.gca()
    _plot_loading(loadings, 0, s.axes_manager, ax=ax, comp_label="c", same_window=False)
    assert ax.get_title() == "c #0"


def test_plot_loading_nav1_calibrated_with_units():
    s = Signal1D(np.random.random((6, 100)))
    s.axes_manager.navigation_axes[0].units = "nm"
    loadings = np.random.random((5, 6))
    import matplotlib.pyplot as plt

    ax = plt.gca()
    _plot_loading(loadings, 0, s.axes_manager, ax=ax, calibrate=True)
    assert ax.get_xlabel() == "nm"


def test_plot_loading_nav1_uncalibrated_uses_depth_label():
    s = Signal1D(np.random.random((6, 100)))
    loadings = np.random.random((5, 6))
    import matplotlib.pyplot as plt

    ax = plt.gca()
    _plot_loading(loadings, 0, s.axes_manager, ax=ax, calibrate=False)
    assert ax.get_xlabel() == "depth"
