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

from hyperspy.drawing._markers.text_collection import TextCollection
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

    @pytest.mark.parametrize("text", ("test",
                                     ("test1", "test2"),
                                      "ragged_text",))
    @pytest.mark.parametrize("iter_data", ("lazy_data", "data"))
    def test_iterating_marker(self, text, request,  iter_data):
        data = request.getfixturevalue(iter_data)
        s = Signal2D(np.ones((3, 5, 6)))
        if text is "ragged_text":
            t = np.empty((3,), dtype=object)
            for i in np.ndindex(t.shape):
                t[i] = "test"+str(i)
            text = t
        markers = TextCollection(offsets=data, s=text)
        s.add_marker(markers)
        s.axes_manager.navigation_axes[0].index = 2
        assert s._plot.signal_plot.ax.te


def _test_text_collection_close():
    signal = Signal2D(np.ones((10,10)))
    markers = TextCollection(offsets =[[1,1],[4,4]], s=("test",))
    signal.add_marker(markers)
    return signal
@update_close_figure()
def test_text_collection_close():
    return _test_text_collection_close()