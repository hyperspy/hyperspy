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
from matplotlib.collections import (
    LineCollection,
    CircleCollection,
    EllipseCollection,
    StarPolygonCollection,
    PolyCollection,
    PatchCollection,
)
from matplotlib.patches import RegularPolygon
import matplotlib.pyplot as plt
import dask.array as da

from hyperspy.drawing.marker_collection import MarkerCollection
from hyperspy.drawing._markers.relative_collection import RelativeCollection
from hyperspy.drawing._markers.line_collection import HorizontalLineCollection, VerticalLineCollection
from hyperspy.drawing.marker_collection import markers2collection
from hyperspy._signals.signal2d import Signal2D, BaseSignal, Signal1D
from hyperspy.axes import UniformDataAxis
from hyperspy.misc.test_utils import update_close_figure

from copy import deepcopy

BASELINE_DIR = "marker_collection"
DEFAULT_TOL = 2.0
STYLE_PYTEST_MPL = "default"
plt.style.use(STYLE_PYTEST_MPL)


class TestCollections:
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

    @pytest.fixture
    def signal(self, data):
        sig = BaseSignal(data, ragged=True)
        sig.metadata.set_item(
            "Peaks.signal_axes",
            (
                UniformDataAxis(scale=0.5, offset=-1),
                UniformDataAxis(scale=2.0, offset=-2),
            ),
        )
        return sig

    @pytest.fixture
    def collections(self):
        collections = [
            LineCollection,
            CircleCollection,
            EllipseCollection,
            StarPolygonCollection,
            PolyCollection,
            PatchCollection,
            None,
        ]
        num_col = len(collections)
        offsets = [
            np.stack([np.ones(num_col) * i, np.arange(num_col)], axis=1)
            for i in range(len(collections))
        ]
        kwargs = [
            {"segments": np.array([[[0, 0], [0, -0.5]]]), "lw": 4},
            {"sizes": (0.4,)},
            {"widths": (0.2,), "heights": (0.7,), "angles": (60,), "units": "xy"},
            {"numsides": 7, "sizes": (0.4,)},
            {"verts": np.array([[[0, 0], [0.3, 0.3], [0.3, 0.6], [0.6, 0.3]]])},
            {
                "patches": [
                    RegularPolygon(
                        xy=(0, 0),
                        numVertices=7,
                        radius=0.5,
                    ),
                ],
            },
            {"sizes": (0.5,)},
        ]
        for k, o, c in zip(kwargs, offsets, collections):
            k["offsets"] = o
            k["collection_class"] = c

        collections = [MarkerCollection(**k) for k in kwargs]
        return collections

    @pytest.mark.mpl_image_compare(
        baseline_dir=BASELINE_DIR, tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL
    )
    def test_multi_collections_signal(self, collections):
        num_col = len(collections)
        s = Signal2D(np.zeros((2, num_col, num_col)))
        s.axes_manager.signal_axes[0].offset = 0
        s.axes_manager.signal_axes[1].offset = 0
        s.plot(interpolation=None)
        [s.add_marker(c) for c in collections]
        return s._plot.signal_plot.figure

    @pytest.mark.mpl_image_compare(
        baseline_dir=BASELINE_DIR, tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL
    )
    def test_multi_collections_navigator(self, collections):
        num_col = len(collections)
        s = Signal2D(np.zeros((num_col, num_col, 1, 1)))
        s.axes_manager.signal_axes[0].offset = 0
        s.axes_manager.signal_axes[1].offset = 0
        s.plot(interpolation=None)
        [s.add_marker(c, plot_on_signal=False) for c in collections]
        return s._plot.navigator_plot.figure

    @pytest.mark.mpl_image_compare(
        baseline_dir=BASELINE_DIR, tolerance=DEFAULT_TOL, style=STYLE_PYTEST_MPL
    )
    @pytest.mark.parametrize("iter_data", ("lazy_data", "data"))
    def test_iterating_marker(self, request, iter_data):
        data = request.getfixturevalue(iter_data)
        s = Signal2D(np.ones((3, 5, 6)))
        markers = MarkerCollection(None, offsets=data, sizes=(0.2,))
        s.add_marker(markers)
        s.axes_manager.navigation_axes[0].index = 2
        return s._plot.signal_plot.figure

    def test_parameters_2_scatter(self, data):
        m = MarkerCollection(None,
                             offsets=np.array([[100, 70],
                                               [70, 100]]),
                             color=("b", "g"),
                             sizes=(3,),)
        s = Signal2D(np.zeros((100, 100)))
        s.add_marker(m)

    def test_parameters_singletons(self, signal, data):
        m = MarkerCollection(None,
                             offsets=np.array([[100, 70],
                                               [70, 100]]),
                             color="b",
                             sizes=3)
        s = Signal2D(np.zeros((2, 100, 100)))
        s.add_marker(m)

    def test_parameters_singletons_iterating(self):
        data = np.empty(2, dtype=object)
        data[0] = np.array([[100, 70], [70, 100]])
        data[1] = np.array([[100, 70], [70, 100]])
        sizes = np.empty(2, dtype=object)
        sizes[0] = 3
        sizes[1] = 4
        m = MarkerCollection(None,
                             offsets=np.array([[100, 70],
                                               [70, 100]]),
                             color="b",
                             sizes=sizes)
        s = Signal2D(np.zeros((2, 100, 100)))
        s.add_marker(m)

    @pytest.mark.parametrize(
        "signal_axes",
        (
            "metadata",
            (
                UniformDataAxis(scale=0.5, offset=-1),
                UniformDataAxis(scale=2.0, offset=-2),
            ),
            None,
        ),
    )
    def test_from_signal(self, signal, data, signal_axes):
        col = MarkerCollection.from_signal(
            signal, sizes=(0.3,), signal_axes=signal_axes
        )

        s = Signal2D(np.ones((3, 5, 6)))
        s.add_marker(col)
        s.axes_manager.navigation_axes[0].index = 1
        if isinstance(signal_axes, (tuple, str)):
            ans = np.zeros_like(data[1])
            ans[:, 1] = data[1][:, 0] * 2 - 2
            ans[:, 0] = data[1][:, 1] * 0.5 - 1
            np.testing.assert_array_equal(col.get_data_position()["offsets"], ans)
        else:
            np.testing.assert_array_equal(col.get_data_position()["offsets"], data[1])

    def test_from_signal_fail(self, signal):
        with pytest.raises(ValueError):
            col = MarkerCollection.from_signal(signal, sizes=(0.3,), signal_axes="test")

    def test_find_peaks(self):
        from skimage.draw import disk
        from skimage.morphology import disk as disk2

        rr, cc = disk(
            center=(10, 8),
            radius=4,
        )
        img = np.zeros((2, 20, 20))
        img[:, rr, cc] = 1
        s = Signal2D(img)
        s.axes_manager.signal_axes[0].scale = 1.5
        s.axes_manager.signal_axes[1].scale = 2
        s.axes_manager.signal_axes[0].offset = -1
        s.axes_manager.signal_axes[1].offset = -1
        pks = s.find_peaks(
            interactive=False,
            method="template_matching",
            template=disk2(4),
        )
        col = MarkerCollection.from_signal(
            pks, sizes=(0.3,), signal_axes=s.axes_manager.signal_axes
        )
        s.add_marker(col)
        np.testing.assert_array_equal(col.get_data_position()["offsets"], [[11, 19]])

    def test_find_peaks0d(self):
        from skimage.draw import disk
        from skimage.morphology import disk as disk2

        rr, cc = disk(
            center=(10, 8),
            radius=4,
        )
        img = np.zeros((1, 20, 20))
        img[:, rr, cc] = 1
        s = Signal2D(img)
        s.axes_manager.signal_axes[0].scale = 1.5
        s.axes_manager.signal_axes[1].scale = 2
        s.axes_manager.signal_axes[0].offset = -1
        s.axes_manager.signal_axes[1].offset = -1
        pks = s.find_peaks(
            interactive=False,
            method="template_matching",
            template=disk2(4),
        )
        col = MarkerCollection.from_signal(
            pks, sizes=(0.3,), signal_axes=s.axes_manager.signal_axes
        )
        s.add_marker(col)
        np.testing.assert_array_equal(col.get_data_position()["offsets"], [[11, 19]])


class TestInitMarkerCollection:
    @pytest.fixture
    def signal(self):
        signal = Signal2D(np.zeros((3, 10, 10)))
        return signal

    @pytest.fixture
    def static_line_collection(self, signal):
        segments = np.ones((10, 2, 2))
        markers = MarkerCollection(LineCollection, segments=segments)
        markers.axes_manager = signal.axes_manager
        return markers

    @pytest.fixture
    def iterating_line_collection(self, signal):
        data = np.empty((3,), dtype=object)
        for i in np.ndindex(data.shape):
            data[i] = np.ones((10, 2, 2)) * i
        markers = MarkerCollection(LineCollection, segments=data)
        markers.axes_manager = signal.axes_manager
        return markers

    def test_multiple_collections(self, static_line_collection,
                                  iterating_line_collection,
                                  signal):
        signal.add_marker(static_line_collection, permanent=True)
        signal.add_marker(iterating_line_collection, permanent=True)
        assert(len(signal.metadata.Markers) == 2)

    @pytest.mark.parametrize(
        "collection", ("iterating_line_collection", "static_line_collection")
    )
    def test_init(self, collection, request):
        col = request.getfixturevalue(collection)
        assert isinstance(col, MarkerCollection)

    @pytest.mark.parametrize(
        "collection", ("iterating_line_collection", "static_line_collection")
    )
    def test_get_data(self, collection, request):
        col = request.getfixturevalue(collection)
        kwds = col.get_data_position()
        assert isinstance(kwds, dict)
        assert kwds["segments"].shape == (10, 2, 2)

    @pytest.mark.parametrize(
        "collection", ("iterating_line_collection", "static_line_collection")
    )
    def test_to_dictionary(self, collection, request):
        col = request.getfixturevalue(collection)
        dict = col._to_dictionary()
        assert dict["collection_class"] == LineCollection
        assert dict["plot_on_signal"] is True

    @pytest.mark.parametrize(
        "collection", ("iterating_line_collection", "static_line_collection")
    )
    def test_update(self, collection, request, signal):
        col = request.getfixturevalue(collection)
        signal.plot()
        signal.add_marker(col)
        signal.axes_manager.navigation_axes[0].index = 2
        if collection is "iterating_line_collection":
            col.get_data_position()["segments"]
            np.testing.assert_array_equal(
                col.get_data_position()["segments"], np.ones((10, 2, 2)) * 2
            )
        else:
            np.testing.assert_array_equal(
                col.get_data_position()["segments"], np.ones((10, 2, 2))
            )

    @pytest.mark.parametrize(
        "collection", ("iterating_line_collection", "static_line_collection")
    )
    def test_fail_plot(self, collection, request):
        col = request.getfixturevalue(collection)
        with pytest.raises(AttributeError):
            col.plot()
    def test_deepcopy(self,iterating_line_collection):
        it_2 = deepcopy(iterating_line_collection)
        assert it_2 is not iterating_line_collection



class TestMarkers2Collection:
    @pytest.fixture
    def iter_data(self):
        data = {"x1":np.arange(10),
                "y1": np.arange(10),
                "x2": np.arange(10),
                "y2": np.arange(10),
                "size": np.arange(10),
                }
        return data
    @pytest.fixture
    def static_data(self):
        data = {"x1": 1,
                "y1": 2,
                "x2": 3,
                "y2": 4,
                "text": None,
                "size": 5,
                }
        return data

    @pytest.fixture
    def static_and_iter_data(self):
        data = {"x1": 1,
                "y1": np.arange(10),
                "x2": np.arange(10),
                "y2": 4,
                "text": None,
                "size": np.arange(10),
                }
        return data

    @pytest.fixture
    def signal(self):
        return Signal1D(np.ones((10, 20)))

    @pytest.mark.parametrize(
        "data", ("iter_data", "static_data", "static_and_iter_data")
    )
    @pytest.mark.parametrize(
        "marker_type", ("Point",
                        "HorizontalLineSegment",
                        "LineSegment",
                        "Ellipse",
                        "HorizontalLine",
                        "VerticalLine",
                        "Arrow",
                        "Rectangle",
                        "VerticalLineSegment",

    ))
    def test_marker2collection(self, request, marker_type, data, signal):
        d = request.getfixturevalue(data)
        test_dict = {}
        test_dict["data"] = d
        test_dict["marker_type"] = marker_type
        test_dict['marker_properties'] = {"color": "black"}
        test_dict['plot_on_signal'] = True
        markers = markers2collection(test_dict)

        signal.add_marker(markers,)
        signal.plot()

    @pytest.mark.parametrize(
        "data", ("iter_data", "static_data", "static_and_iter_data")
    )
    @pytest.mark.parametrize(
        "marker_type", "Text")
    def test_marker2collectionfail(self, request, marker_type, data):
        d = request.getfixturevalue(data)
        test_dict = {}
        test_dict["data"] = d
        test_dict["marker_type"] = marker_type
        test_dict['marker_properties'] = {"color": "black"}
        test_dict['plot_on_signal'] = True
        with pytest.raises(ValueError):
            markers2collection(test_dict)


def _test_marker_collection_close():
    signal = Signal2D(np.ones((10,10)))
    segments = np.ones((10, 2, 2))
    markers = MarkerCollection(LineCollection, segments=segments)
    signal.add_marker(markers)
    return signal

@update_close_figure()
def test_marker_collection_close():
    return _test_marker_collection_close()


class TestRelativeMarkerCollection:
    def test_relative_marker_collection(self):
        signal = Signal1D((np.arange(100)+1).reshape(10, 10))
        segments = np.zeros((10, 2, 2))
        segments[:, 1, 1] = 1  # set y values end
        segments[:, 0, 0] = np.arange(10).reshape(10)  # set x values
        segments[:, 1, 0] = np.arange(10).reshape(10)  # set x values

        markers = RelativeCollection(collection_class=LineCollection,
                                     segments=segments)
        signal.plot()
        signal.add_marker(markers)
        signal.axes_manager.navigation_axes[0].index = 1
        segs = markers.collection.get_segments()
        assert markers.collection.get_segments()[0][0][0] == 0
        assert markers.collection.get_segments()[0][1][1] == 11

    def test_relative_marker_collection_fail(self):
        with pytest.raises(ValueError):
            segments = np.zeros((10, 2, 2))
            segments[:, 1, 1] = 1  # set y values end
            segments[:, 0, 0] = np.arange(10).reshape(10)  # set x values
            segments[:, 1, 0] = np.arange(10).reshape(10)  # set x values
            markers = RelativeCollection(collection_class=LineCollection,
                                         segments=segments,
                                         reference="data_index")

    def test_relative_marker_collection_fail_ref(self):
        with pytest.raises(ValueError):
            segments = np.zeros((10, 2, 2))
            segments[:, 1, 1] = 1  # set y values end
            segments[:, 0, 0] = np.arange(10).reshape(10)  # set x values
            segments[:, 1, 0] = np.arange(10).reshape(10)  # set x values
            markers = RelativeCollection(collection_class=LineCollection,
                                         segments=segments,
                                         reference="data_in")


class TestLineCollections:
    @pytest.fixture
    def segments(self):
        d = np.empty((3,), dtype=object)
        for i in np.ndindex(d.shape):
            d[i] = np.arange(i[0]+1)
        return d

    def test_vertical_line_collection(self, segments):
        vert = VerticalLineCollection(segments=segments)
        s = Signal2D(np.zeros((3, 3, 3)))
        s.axes_manager.signal_axes[0].offset = 0
        s.axes_manager.signal_axes[1].offset = 0
        s.plot(interpolation=None)
        s.add_marker(vert)
        kwargs = vert.get_data_position()
        np.testing.assert_array_equal(kwargs["segments"], [[[0., -5], [0., 1]]])

    def test_horizontal_line_collection(self, segments):
        hor = HorizontalLineCollection(segments=segments)
        s = Signal2D(np.zeros((3, 3, 3)))
        s.axes_manager.signal_axes[0].offset = 0
        s.axes_manager.signal_axes[1].offset = 0
        s.plot(interpolation=None)
        s.add_marker(hor)
        kwargs = hor.get_data_position()
        np.testing.assert_array_equal(kwargs["segments"], [[[-1, 0], [5, 0]]])
