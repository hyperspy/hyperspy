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
from hyperspy._signals.signal2d import Signal2D, BaseSignal
import numpy as np
from matplotlib.collections import (LineCollection,
                                    CircleCollection,
                                    EllipseCollection,
                                    StarPolygonCollection,
                                    PolyCollection, PatchCollection)
from matplotlib.patches import RegularPolygon

BASELINE_DIR = 'marker_collection'
DEFAULT_TOL = 2.0
STYLE_PYTEST_MPL = 'default'
import matplotlib.pyplot as plt
plt.style.use(STYLE_PYTEST_MPL)

class TestCollections:
    @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR,
                                   tolerance=DEFAULT_TOL,
                                   style=STYLE_PYTEST_MPL)
    def test_multi_collections_signal(self):
        collections = [LineCollection, CircleCollection,
                       EllipseCollection,StarPolygonCollection,
                       PolyCollection,PatchCollection, None]
        num_col = len(collections)
        offsets = [np.stack([np.ones(num_col)*i, np.arange(num_col)], axis=1)
                   for i in range(len(collections))]
        kwargs = [{"segments":np.array([[[0, 0],
                                        [0,-.5]]]),
                   "lw":4},
                  {"sizes":(.4,)},
                  {"widths":(.2,), "heights":(.7,), "angles":(60,), "units":"xy"},
                  {"numsides": 7, "sizes":(.4,)},
                  {"verts": np.array([[[0, 0], [.3, .3], [.3, .6], [.6, .3]]])},
                  {"patches":[RegularPolygon(xy=(0,0), numVertices=7, radius=.5,),],},
                  {"sizes":(0.5,)},
                  ]
        for k, o, c in zip(kwargs, offsets, collections):
            k["offsets"] = o
            k["collection_class"]= c

        collections = [MarkerCollection(**k) for k in kwargs]
        s = Signal2D(np.zeros((2, num_col, num_col)))
        s.axes_manager.signal_axes[0].offset= 0
        s.axes_manager.signal_axes[1].offset = 0
        s.plot(interpolation=None)
        [s.add_marker(c) for c in collections]
        return s._plot.signal_plot.figure


    @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR,
                                   tolerance=DEFAULT_TOL,
                                   style=STYLE_PYTEST_MPL)
    def test_multi_collections_navigator(self):
        collections = [LineCollection, CircleCollection,
                       EllipseCollection,StarPolygonCollection,
                       PolyCollection,PatchCollection, None]
        num_col = len(collections)
        offsets = [np.stack([np.ones(num_col)*i, np.arange(num_col)], axis=1)
                   for i in range(len(collections))]
        kwargs = [{"segments":np.array([[[0, 0],
                                        [0,-.5]]]),
                   "lw":4},
                  {"sizes":(.4,)},
                  {"widths":(.2,), "heights":(.7,), "angles":(60,), "units":"xy"},
                  {"numsides": 7, "sizes":(.4,)},
                  {"verts": np.array([[[0, 0], [.3, .3], [.3, .6], [.6, .3]]])},
                  {"patches":[RegularPolygon(xy=(0,0), numVertices=7, radius=.5,),],},
                  {"sizes":(0.5,)},
                  ]
        for k, o, c in zip(kwargs, offsets, collections):
            k["offsets"] = o
            k["collection_class"]= c

        collections = [MarkerCollection(**k) for k in kwargs]
        s = Signal2D(np.zeros((num_col, num_col, 1, 1)))
        s.axes_manager.signal_axes[0].offset= 0
        s.axes_manager.signal_axes[1].offset = 0
        s.plot(interpolation=None)
        [s.add_marker(c, plot_on_signal=False) for c in collections]
        return s._plot.navigator_plot.figure

    @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR,
                                   tolerance=DEFAULT_TOL,
                                   style=STYLE_PYTEST_MPL)
    def test_iterating_marker(self):
        data = np.empty((3,), dtype=object)
        for i in np.ndindex(data.shape):
            data[i] = np.stack([np.arange(3), np.ones(3)*i], axis=1)
        s = Signal2D(np.ones((3, 5, 6)))
        markers = MarkerCollection(None,
                                   offsets=data,
                                   sizes=(.2,))
        s.add_marker(markers)
        s.axes_manager.navigation_axes[0].index=2
        return s._plot.signal_plot.figure

    def test_from_signal(self):
        data = np.empty((3,), dtype=object)
        for i in np.ndindex(data.shape):
            data[i] = np.stack([np.arange(3), np.ones(3) * i], axis=1)

        col = MarkerCollection.from_signal(BaseSignal(data,
                                                      ragged=True), sizes=(.3,))
        s = Signal2D(np.ones((3, 5, 6)))
        s.add_marker(col)
        s.axes_manager.navigation_axes[0].index = 2


class TestLineMarkerCollection:

    @pytest.fixture
    def static_line_collection(self):
        s = Signal2D(np.random.random((3, 4, 5, 6)))
        markers = MarkerCollection(LineCollection,
                                   segments=np.random.random((10,2, 2)))
        markers.axes_manager = s.axes_manager
        return markers

    @pytest.fixture
    def iterating_line_collection(self):
        data = np.empty((3,4), dtype=object)
        for i in np.ndindex(data.shape):
            data[i] = np.random.random((10, 2, 2))
        s = Signal2D(np.random.random((3, 4, 5, 6)))
        markers = MarkerCollection(LineCollection,
                                   segments=data)
        markers.axes_manager = s.axes_manager
        return markers

    @pytest.mark.parametrize("collection", ("iterating_line_collection",
                                            "static_line_collection"))
    def test_init(self,collection,request):
        col = request.getfixturevalue(collection)
        assert isinstance(col, MarkerCollection)

    @pytest.mark.parametrize("collection", ("iterating_line_collection",
                                            "static_line_collection"))
    def test_get_data(self,collection,request):
        col = request.getfixturevalue(collection)
        kwds = col.get_data_position()
        assert isinstance(kwds, dict)
        assert kwds["segments"].shape == (10,2,2)