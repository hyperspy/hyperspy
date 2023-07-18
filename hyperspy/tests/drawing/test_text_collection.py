# Copyright 2007-2023 The HyperSpy developers
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
import pytest


import numpy as np


from matplotlib.patches import RegularPolygon
import matplotlib.pyplot as plt
import dask.array as da

from hyperspy.drawing._markers.text_collection import TextCollection, RelativeTextCollection
from hyperspy._signals.signal2d import Signal2D, BaseSignal, Signal1D
from hyperspy.misc.test_utils import update_close_figure

from copy import deepcopy

BASELINE_DIR = "marker_collection"
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

    @pytest.mark.parametrize("text", (("test",),
                                      "test",
                                      ("test", "test"),
                                      "ragged_text",))
    @pytest.mark.parametrize("iter_data", ("lazy_data", "data"))
    def test_iterating_marker(self, text, request,  iter_data):
        data = request.getfixturevalue(iter_data)
        s = Signal2D(np.ones((3, 5, 6)))
        ragged_text = text is "ragged_text"
        if ragged_text:
            t = np.empty((3,), dtype=object)
            for i in np.ndindex(t.shape):
                t[i] = ("test"+str(i),)
            text = t
        markers = TextCollection(offsets=data, s=text)
        s.add_marker(markers)
        s.axes_manager.navigation_axes[0].index = 1
        children = s._plot.signal_plot.ax.get_children()
        children = [str(c) for c in children]
        if ragged_text:
            assert "Text(2.0, 1.0, 'test(1,)')" in children
        else:
            assert "Text(2.0, 1.0, 'test')" in children


class TestRelativeTextCollection:
    def test_relative_text_collection(self):
        s = Signal1D(np.arange(10).reshape(5, 2))
        markers = RelativeTextCollection(offsets=[[0, 1], [1, 2]],
                                         s=("test",))
        s.add_marker(markers)
        s.axes_manager.navigation_axes[0].index = 1
        children = s._plot.signal_plot.ax.get_children()
        assert "Text(0, 2, 'test')" in str(children)
        assert "Text(1, 6, 'test')" in str(children)
        s.axes_manager.navigation_axes[0].index = 2
        children = s._plot.signal_plot.ax.get_children()
        assert "Text(0, 4, 'test')" in str(children)
        assert "Text(1, 10, 'test')" in str(children)

    def test_relative_text_collection_with_reference(self):
        s = Signal1D(np.arange(10).reshape(5, 2))
        markers = RelativeTextCollection(offsets=[[0, 1], [1, 2]],
                                         s=("test",), reference="data_index",
                                         indexes=[0, 0])
        s.add_marker(markers)
        s.axes_manager.navigation_axes[0].index = 1
        children = s._plot.signal_plot.ax.get_children()
        assert "Text(0, 2, 'test')" in str(children)
        assert "Text(1, 4, 'test')" in str(children)
        s.axes_manager.navigation_axes[0].index = 2
        children = s._plot.signal_plot.ax.get_children()
        assert "Text(0, 4, 'test')" in str(children)
        assert "Text(1, 8, 'test')" in str(children)

    def test_plot_fail(self):
        markers = TextCollection(offsets=[[1, 1],
                                          [4, 4]], s=("test",))
        with pytest.raises(AttributeError):
            markers.plot()
    def test_plot_and_render(self):
        markers = TextCollection(offsets=[[1, 1],
                                          [4, 4]], s=("test",))
        s = Signal1D(np.arange(100).reshape((10,10)))
        s.add_marker(markers)
        markers.plot(render_figure=True)

    def test_static_update(self):
        markers = TextCollection(offsets=[[1, 1],
                                          [4, 4]], s=("test",))
        s = Signal1D(np.arange(100).reshape((10, 10)))
        s.plot()
        s.add_marker(markers)
        s.axes_manager.navigation_axes[0].index=2



def _test_text_collection_close():
    signal = Signal2D(np.ones((10, 10)))
    markers = TextCollection(offsets=[[1, 1],
                                      [4, 4]], s=("test",))
    signal.add_marker(markers)
    return signal
@update_close_figure()
def test_text_collection_close():
    return _test_text_collection_close()