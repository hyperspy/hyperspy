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

"""Unit tests for hyperspy.drawing.tiles.HistogramTilePlot (SAMFire debug plot)."""

import importlib
from unittest import mock

import numpy as np
import pytest

from hyperspy.drawing.backends._protocol import BackendCapabilityError
from hyperspy.drawing.tiles import HistogramTilePlot


@pytest.fixture
def db():
    return {
        "component1": {
            "param1": (np.array([1, 2, 3, 0, 5]), np.array([0, 1, 2, 3, 4, 5])),
        },
    }


@pytest.fixture(autouse=True)
def _close_htp():
    plots = []
    yield plots
    for htp in plots:
        if htp.figure is not None:
            htp.close()


def test_require_mpl_raises_when_matplotlib_missing():
    htp = HistogramTilePlot()
    with mock.patch.object(importlib.util, "find_spec", return_value=None):
        with pytest.raises(BackendCapabilityError, match="requires matplotlib"):
            htp._require_mpl()


def test_require_mpl_raises_for_non_mpl_backend():
    class _FakeBackend:
        pass

    htp = HistogramTilePlot()
    with mock.patch("hyperspy.drawing.tiles.get_backend", return_value=_FakeBackend()):
        with pytest.raises(BackendCapabilityError, match="FakeBackend"):
            htp._require_mpl()


def test_require_mpl_passes_for_mpl_backend():
    htp = HistogramTilePlot()
    htp._require_mpl()  # must not raise: the default backend is MplBackend


def test_plot_creates_figure_and_axis(db, _close_htp):
    htp = HistogramTilePlot()
    _close_htp.append(htp)
    htp.plot(db, color="#4C72B0")
    assert htp.figure is not None
    assert htp.ax is not None


def test_plot_empty_db_creates_figure_but_no_axis(_close_htp):
    htp = HistogramTilePlot()
    _close_htp.append(htp)
    htp.plot({})
    assert htp.figure is not None
    assert htp.ax is None


def test_update_replots_histogram_bars(db, _close_htp):
    htp = HistogramTilePlot()
    _close_htp.append(htp)
    htp.plot(db, color="#4C72B0")
    n_patches_before = len(htp.ax.patches)
    htp.update(db, color="red")
    assert len(htp.ax.patches) == n_patches_before
