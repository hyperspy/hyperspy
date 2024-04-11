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

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pytest

from hyperspy._signals.signal2d import Signal2D
from hyperspy.drawing._markers.texts import Texts
from hyperspy.misc.test_utils import update_close_figure

BASELINE_DIR = "markers"
DEFAULT_TOL = 2.0
STYLE_PYTEST_MPL = "default"
plt.style.use(STYLE_PYTEST_MPL)


class TestTextCollection:
    @pytest.fixture
    def data(self):
        d = np.empty((3,), dtype=object)
        for i in np.ndindex(d.shape):
            d[i] = np.stack([np.arange(3), np.ones(3) * i], axis=1)
        return d

    @pytest.fixture
    def lazy_data(self):
        d = np.empty((3,), dtype=object)
        for i in np.ndindex(d.shape):
            d[i] = np.stack([np.arange(3), np.ones(3) * i], axis=1)
        d = da.from_array(d, chunks=(1, 1, 1))

        return d

    @pytest.mark.parametrize(
        "texts",
        (
            ("test",),
            "test",
            ("test", "test"),
            "ragged_text",
        ),
    )
    @pytest.mark.parametrize("iter_data", ("lazy_data", "data"))
    def test_iterating_marker(self, texts, request, iter_data):
        data = request.getfixturevalue(iter_data)
        s = Signal2D(np.ones((3, 5, 6)))
        s.plot()
        ragged_texts = texts == "ragged_text"
        if ragged_texts:
            t = np.empty((3,), dtype=object)
            for i in np.ndindex(t.shape):
                t[i] = ("test" + str(i),)
            texts = t
        markers = Texts(offsets=data, texts=texts)
        children_before = s._plot.signal_plot.ax.get_children()
        s.add_marker(markers)
        s.axes_manager.navigation_axes[0].index = 1
        children_after = s._plot.signal_plot.ax.get_children()
        assert len(children_after) - len(children_before) == 1

    @pytest.mark.mpl_image_compare(
        baseline_dir=BASELINE_DIR, tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL
    )
    def test_text_marker_plot(self):
        s = Signal2D(np.ones((3, 5, 6)))
        s.data[:, :, ::2] = np.nan
        markers = Texts(offsets=[[2.0, 3.0]], texts=("test",), sizes=(20,))
        s.add_marker(markers, render_figure=True)
        return s._plot.signal_plot.figure


def _test_text_collection_close():
    signal = Signal2D(np.ones((10, 10)))
    markers = Texts(offsets=[[1, 1], [4, 4]], texts=("test",))
    signal.add_marker(markers)
    return signal


@update_close_figure()
def test_text_collection_close():
    return _test_text_collection_close()


def test_text_collection_close_render():
    s = Signal2D(np.ones((2, 10, 10)))
    markers = Texts(
        offsets=[[1, 1], [4, 4]], texts=("test",), sizes=(10,), color=("black",)
    )
    s.plot()
    children_before = s._plot.signal_plot.ax.get_children()
    s.add_marker(markers, render_figure=True)
    children_after = s._plot.signal_plot.ax.get_children()
    assert len(children_after) - len(children_before) == 1

    markers.close(render_figure=True)
    assert len(s._plot.signal_plot.ax.get_children()) == len(children_before)
