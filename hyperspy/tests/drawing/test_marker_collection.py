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

from hyperspy.drawing._markers.marker_collection import MarkerCollection
from matplotlib.collections import LineCollection, CircleCollection
from hyperspy._signals.signal2d import Signal2D
import numpy as np

default_tol = 2.0
baseline_dir = 'plot_markers'
style_pytest_mpl = 'default'
import matplotlib


class TestLineMarkerCollection:

    @pytest.fixture
    def static_line_collection(self):
        s = Signal2D(np.random.random((3, 4, 5, 6)))
        markers = MarkerCollection(LineCollection,
                                   segments=np.random.random((10,10, 2)))
        markers.axes_manager = s.axes_manager
        return markers

    @pytest.fixture
    def iterating_line_collection(self):
        data = np.empty((3,4), dtype=object)
        for i in np.ndindex(data.shape):
            data[i] = np.random.random((10, 10, 2))
        s = Signal2D(np.random.random((3, 4, 5, 6)))
        markers = MarkerCollection(LineCollection,
                                   segments=data)
        markers.axes_manager = s.axes_manager
        return markers

    def test_init(self,static_line_collection):
        assert isinstance(static_line_collection, MarkerCollection)

    def test_get_data(self,static_line_collection):
        kwds = static_line_collection.get_data_position()
        assert isinstance(kwds, dict)
        assert kwds["segments"].shape == (10,10,2)

    def test_get_data2(self, iterating_line_collection):
        kwds = iterating_line_collection.get_data_position()
        assert isinstance(kwds, dict)
        assert kwds["segments"].shape == (10,10,2)

    def test_initialize_collection(self, iterating_line_collection):
        iterating_line_collection.initialize_collection()
        assert isinstance(iterating_line_collection.collection, LineCollection)

    def test_update_collection(self, iterating_line_collection):
        iterating_line_collection.initialize_collection()
        iterating_line_collection.axes_manager.navigation_axes[0].index=1

    def test_add_marker(self):
        s = Signal2D(np.random.random((3, 4, 5, 6)))
        markers = MarkerCollection(LineCollection,
                                   segments=np.random.random((10, 2, 2))*3, lw=2)

        s.plot()
        s.add_marker(markers,plot_on_signal=True)
        s._plot.signal_plot.figure.savefig("test.png")
        s.axes_manager.navigation_axes[0].index=1
        s._plot.signal_plot.figure.savefig("test1.png")

    def test_add_marker(self):
        data = np.empty((3, 4), dtype=object)
        for i in np.ndindex(data.shape):
            data[i] = np.random.random((10, 2, 2))*3
        s = Signal2D(np.random.random((3, 4, 5, 6)))
        markers = MarkerCollection(LineCollection,
                                   segments=data, lw=2)

        s.plot()
        s.add_marker(markers,plot_on_signal=True)
        s._plot.signal_plot.figure.savefig("test.png")
        s.axes_manager.navigation_axes[0].index=1
        s._plot.signal_plot.figure.savefig("test1.png")

class TestCircleMarkerCollection:

    @pytest.fixture
    def static_collection(self):
        s = Signal2D(np.random.random((3, 4, 5, 6)))
        markers = MarkerCollection(offsets=np.random.random((10,2)),
                                   sizes=2)
        markers.axes_manager = s.axes_manager
        return markers

    @pytest.fixture
    def iterating_collection(self):
        data = np.empty((3,4), dtype=object)
        for i in np.ndindex(data.shape):
            data[i] = np.random.random((10, 2,))
        s = Signal2D(np.random.random((3, 4, 5, 6)))
        markers = MarkerCollection(offsets=data,
                                   sizes=np.random.random(10))
        markers.axes_manager = s.axes_manager
        return markers

    def test_init(self,static_collection):
        assert isinstance(static_collection, MarkerCollection)

    def test_get_data(self,static_collection):
        kwds = static_collection.get_data_position()
        assert isinstance(kwds, dict)
        assert kwds["offsets"].shape == (10,2)

    def test_get_data2(self, iterating_collection):
        kwds = iterating_collection.get_data_position()
        assert isinstance(kwds, dict)
        assert kwds["offsets"].shape == (10,2)

    def test_initialize_collection(self, iterating_collection):
        iterating_collection.initialize_collection()
        assert isinstance(iterating_collection.collection, CircleCollection)

    def test_update_collection(self, iterating_collection):
        iterating_collection.initialize_collection()
        iterating_collection.axes_manager.navigation_axes[0].index=1

    def test_add_marker(self):
        s = Signal2D(np.random.random((3, 4, 5, 6)))
        markers = MarkerCollection(offsets=np.random.rand(10, 2)*5,
                                   sizes=(5,),
                                   linewidths=(2,),
                                   facecolors="black",)
        s.axes_manager.signal_axes[0].scale=1
        s.axes_manager.signal_axes[1].scale=1
        s.plot()
        s.add_marker(markers, plot_on_signal=True)
        s._plot.signal_plot.figure.savefig("test.png")
        s.axes_manager.navigation_axes[0].index=1
        s._plot.signal_plot.figure.savefig("test1.png")

    def test_add_marker2(self):
        data = np.empty((3, 4), dtype=object)
        for i in np.ndindex(data.shape):
            data[i] = np.random.random((10, 2, 2))*3
        s = Signal2D(np.random.random((3, 4, 5, 6)))
        s.axes_manager.signal_axes[0].scale=1/6
        s.axes_manager.signal_axes[1].scale=1/5

        markers = MarkerCollection(LineCollection,
                                   segments=data, lw=2)

        s.plot()
        s.add_marker(markers,plot_on_signal=True)
        s._plot.signal_plot.figure.savefig("test.png")
        s.axes_manager.navigation_axes[0].index=1
        s._plot.signal_plot.figure.savefig("test1.png")


