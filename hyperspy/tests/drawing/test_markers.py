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
from copy import deepcopy
from pathlib import Path

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import (
    LineCollection,
    PolyCollection,
    StarPolygonCollection,
)
from matplotlib.transforms import (
    CompositeGenericTransform,
    IdentityTransform,
)

import hyperspy.api as hs
from hyperspy._signals.signal2d import BaseSignal, Signal1D, Signal2D
from hyperspy.axes import UniformDataAxis
from hyperspy.drawing.markers import markers_dict_to_markers
from hyperspy.external.matplotlib.collections import (
    CircleCollection,
    EllipseCollection,
    RectangleCollection,
    SquareCollection,
    TextCollection,
)
from hyperspy.external.matplotlib.quiver import Quiver
from hyperspy.misc.test_utils import update_close_figure
from hyperspy.utils.markers import (
    Arrows,
    Circles,
    Ellipses,
    HorizontalLines,
    Lines,
    Markers,
    Points,
    Polygons,
    Rectangles,
    Squares,
    Texts,
    VerticalLines,
)

BASELINE_DIR = "markers"
DEFAULT_TOL = 2.0
STYLE_PYTEST_MPL = "default"
FILE_PATH = Path(__file__).resolve().parent


plt.style.use(STYLE_PYTEST_MPL)


class TestMarkers:
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
            Circles,
            Ellipses,
            Rectangles,
            Squares,
            Points,
        ]
        num_col = len(collections)
        offsets = [
            np.stack([np.ones(num_col) * i, np.arange(num_col)], axis=1)
            for i in range(len(collections))
        ]
        kwargs = [
            {"sizes": (0.4,)},
            {"widths": (0.2,), "heights": (0.7,), "angles": (60,), "units": "xy"},
            {"widths": (0.2,), "heights": (0.7,), "angles": (60,), "units": "xy"},
            {"widths": (0.4,)},
            {"sizes": (20,)},
        ]
        for k, o in zip(kwargs, offsets):
            k["offsets"] = o
        collections = [c(**k) for k, c in zip(kwargs, collections)]
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
        baseline_dir=BASELINE_DIR,
        tolerance=DEFAULT_TOL,
        style=STYLE_PYTEST_MPL,
        filename="test_iterating_markers.png",
    )
    @pytest.mark.parametrize("iter_data", ("lazy_data", "data"))
    def test_iterating_markers(self, request, iter_data):
        data = request.getfixturevalue(iter_data)
        s = Signal2D(np.ones((3, 5, 6)))
        markers = Points(offsets=data, sizes=(50,))
        s.add_marker(markers)
        s.axes_manager.navigation_axes[0].index = 2
        return s._plot.signal_plot.figure

    def test_parameters_2_scatter(self, data):
        m = Points(
            offsets=np.array([[100, 70], [70, 100]]),
            color="g",
            sizes=(3,),
        )
        s = Signal2D(np.zeros((100, 100)))
        s.add_marker(m)

    def test_parameters_singletons(self, signal, data):
        m = Points(offsets=np.array([[100, 70], [70, 100]]), color="b", sizes=3)
        s = Signal2D(np.zeros((2, 100, 100)))
        s.add_marker(m)

    def test_parameters_singletons_iterating(self):
        data = np.empty(2, dtype=object)
        data[0] = np.array([[100, 70], [70, 100]])
        data[1] = np.array([[100, 70], [70, 100]])
        sizes = np.empty(2, dtype=object)
        sizes[0] = 3
        sizes[1] = 4
        m = Points(offsets=np.array([[100, 70], [70, 100]]), color="b", sizes=sizes)
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
        col = Points.from_signal(signal, sizes=(10,), signal_axes=signal_axes)

        s = Signal2D(np.ones((3, 5, 6)))
        s.add_marker(col)
        s.axes_manager.navigation_axes[0].index = 1
        if isinstance(signal_axes, (tuple, str)):
            ans = np.zeros_like(data[1])
            ans[:, 1] = data[1][:, 0] * 2 - 2
            ans[:, 0] = data[1][:, 1] * 0.5 - 1
            np.testing.assert_array_equal(col.get_current_kwargs()["offsets"], ans)
        else:
            np.testing.assert_array_equal(col.get_current_kwargs()["offsets"], data[1])

    def test_from_signal_lines(self, signal, data):
        data = np.empty((3,), dtype=object)
        for i in np.ndindex(data.shape):
            data[i] = np.ones((10, 2, 2)) * i

        signal = BaseSignal(data, ragged=True)
        lines = Lines.from_signal(signal, key="segments", signal_axes=None)
        s = Signal2D(np.ones((3, 5, 6)))
        s.add_marker(lines)

    def test_from_signal_not_ragged(self):
        s = hs.signals.Signal2D(np.ones((2, 3, 5, 6, 7)))

        # not a ragged signal, navigation_dim must be zero
        pos = hs.signals.BaseSignal(np.arange(10).reshape((5, 2)))
        col = hs.plot.markers.Points.from_signal(pos)

        s.add_marker(col)

    def test_from_signal_fail(self, signal):
        with pytest.raises(ValueError):
            _ = Points.from_signal(signal, sizes=(10,), signal_axes="test")

    def test_find_peaks(self):
        from skimage.draw import disk
        from skimage.morphology import disk as disk2

        import hyperspy.api as hs

        rr, cc = disk(
            center=(10, 8),
            radius=4,
        )
        img = np.zeros((2, 3, 4, 20, 20))
        img[:, :, :, rr, cc] = 1
        img[:, 1, 0] = 0
        img[:, 0, 1] = 0
        s = hs.signals.Signal2D(img)
        s.axes_manager.signal_axes[0].scale = 1.5
        s.axes_manager.signal_axes[1].scale = 2
        s.axes_manager.signal_axes[0].offset = -1
        s.axes_manager.signal_axes[1].offset = -1
        pks = s.find_peaks(
            interactive=False,
            method="template_matching",
            template=disk2(4),
        )
        col = hs.plot.markers.Points.from_signal(
            pks, sizes=(10,), signal_axes=s.axes_manager.signal_axes
        )
        s.add_marker(col)

        marker_pos = [11, 19]
        np.testing.assert_array_equal(col.get_current_kwargs()["offsets"], [marker_pos])
        s.axes_manager.indices = (0, 1, 0)
        np.testing.assert_array_equal(
            col.get_current_kwargs()["offsets"], [[np.nan, np.nan]]
        )

        s.axes_manager.indices = (1, 0, 0)
        np.testing.assert_array_equal(
            col.get_current_kwargs()["offsets"], [[np.nan, np.nan]]
        )

        # Go to end with last navigation axis at 0
        s.axes_manager.indices = (3, 2, 0)
        np.testing.assert_array_equal(col.get_current_kwargs()["offsets"], [marker_pos])

        # Go to end with last navigation axis at 1
        s.axes_manager.indices = (3, 2, 1)
        np.testing.assert_array_equal(col.get_current_kwargs()["offsets"], [marker_pos])

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
        col = Points.from_signal(
            pks, sizes=(0.3,), signal_axes=s.axes_manager.signal_axes
        )
        s.add_marker(col)
        np.testing.assert_array_equal(col.get_current_kwargs()["offsets"], [[11, 19]])

    def test_deepcopy_signal_with_markers(self, collections):
        num_col = len(collections)
        s = Signal2D(np.zeros((2, num_col, num_col)))
        s.plot()
        [s.add_marker(c, permanent=True) for c in collections]
        new_s = deepcopy(s)
        assert len(new_s.metadata["Markers"]) == num_col

    def test_deepcopy_signal_with_muultiple_markers_same_class(self):
        markers_list = [Points(offsets=np.array([1, 2]) * i) for i in range(3)]
        num_markers = len(markers_list)
        s = Signal2D(np.zeros((2, 10, 10)))
        s.plot()
        [s.add_marker(m, permanent=True) for m in markers_list]
        s2 = deepcopy(s)
        assert len(s2.metadata["Markers"]) == num_markers
        ref_name = ["Points", "Points1", "Points2"]
        assert s.metadata.Markers.keys() == s2.metadata.Markers.keys() == ref_name

    def test_get_current_signal(self, collections):
        num_col = len(collections)
        s = Signal2D(np.zeros((2, num_col, num_col)))
        s.plot()
        [s.add_marker(c, permanent=True) for c in collections]
        cs = s.get_current_signal()
        assert len(cs.metadata["Markers"]) == num_col
        assert isinstance(cs.metadata.Markers.Circles, Circles)

    def test_plot_and_render(self):
        markers = Points(offsets=[[1, 1], [4, 4]])
        s = Signal1D(np.arange(100).reshape((10, 10)))
        s.add_marker(markers)
        markers.plot(render_figure=True)


class TestInitMarkers:
    @pytest.fixture
    def signal(self):
        signal = Signal2D(np.zeros((3, 10, 10)))
        return signal

    @pytest.fixture
    def static_line_collection(self, signal):
        segments = np.ones((10, 2, 2))
        markers = Lines(segments=segments)
        markers._signal = signal
        return markers

    @pytest.fixture
    def iterating_line_collection(self, signal):
        data = np.empty((3,), dtype=object)
        for i in np.ndindex(data.shape):
            data[i] = np.ones((10, 2, 2)) * i
        markers = Lines(segments=data)
        markers._signal = signal
        return markers

    def test_multiple_collections(
        self, static_line_collection, iterating_line_collection, signal
    ):
        signal.add_marker(static_line_collection, permanent=True)
        signal.add_marker(iterating_line_collection, permanent=True)
        assert len(signal.metadata.Markers) == 2

    @pytest.mark.parametrize(
        "collection", ("iterating_line_collection", "static_line_collection")
    )
    def test_init(self, collection, request):
        col = request.getfixturevalue(collection)
        assert isinstance(col, Markers)

    @pytest.mark.parametrize(
        "collection", ("iterating_line_collection", "static_line_collection")
    )
    def test_get_data(self, collection, request):
        col = request.getfixturevalue(collection)
        kwds = col.get_current_kwargs()
        assert isinstance(kwds, dict)
        assert kwds["segments"].shape == (10, 2, 2)

    @pytest.mark.parametrize(
        "collection", ("iterating_line_collection", "static_line_collection")
    )
    def test_to_dictionary(self, collection, request):
        col = request.getfixturevalue(collection)
        dict = col._to_dictionary()
        assert dict["class"] == "Lines"
        assert dict["name"] == ""
        assert dict["plot_on_signal"] is True

    @pytest.mark.parametrize(
        "collection", ("iterating_line_collection", "static_line_collection")
    )
    def test_update(self, collection, request, signal):
        col = request.getfixturevalue(collection)
        signal.plot()
        signal.add_marker(col)
        signal.axes_manager.navigation_axes[0].index = 2
        if collection == "iterating_line_collection":
            col.get_current_kwargs()["segments"]
            np.testing.assert_array_equal(
                col.get_current_kwargs()["segments"], np.ones((10, 2, 2)) * 2
            )
        else:
            np.testing.assert_array_equal(
                col.get_current_kwargs()["segments"], np.ones((10, 2, 2))
            )

    @pytest.mark.parametrize(
        "collection", ("iterating_line_collection", "static_line_collection")
    )
    def test_fail_plot(self, collection, request):
        col = request.getfixturevalue(collection)
        with pytest.raises(AttributeError):
            col.plot()

    def test_deepcopy_iterating_line_collection(self, iterating_line_collection):
        it_2 = deepcopy(iterating_line_collection)
        assert it_2 is not iterating_line_collection

    def test_wrong_navigation_size(self):
        s = Signal2D(np.zeros((2, 3, 3)))
        offsets = np.empty((3, 2), dtype=object)
        for i in np.ndindex(offsets.shape):
            offsets[i] = np.ones((3, 2))
        m = Points(offsets=offsets)
        with pytest.raises(ValueError):
            s.add_marker(m)

    def test_add_markers_to_multiple_signals(self):
        s = Signal2D(np.zeros((2, 3, 3)))
        s2 = Signal2D(np.zeros((2, 3, 3)))
        m = Points(offsets=[[1, 1], [2, 2]])
        s.add_marker(m, permanent=True)
        with pytest.raises(ValueError):
            s2.add_marker(m, permanent=True)

    def test_add_markers_to_same_signal(self):
        s = Signal2D(np.zeros((2, 3, 3)))
        m = Points(offsets=[[1, 1], [2, 2]])
        s.add_marker(m, permanent=True)
        with pytest.raises(ValueError):
            s.add_marker(m, permanent=True)

    def test_add_markers_to_navigator_without_nav(self):
        s = Signal2D(np.zeros((3, 3)))
        m = Points(offsets=[[1, 1], [2, 2]])
        with pytest.raises(ValueError):
            s.add_marker(m, plot_on_signal=False)

    def test_marker_collection_lazy_nonragged(self):
        m = Points(offsets=da.array([[1, 1], [2, 2]]))
        assert not isinstance(m.kwargs["offsets"], da.Array)

    def test_rep(self):
        offsets = np.array([[1, 1], [2, 2]])
        m = Markers(offsets=offsets, verts=3, sizes=3, collection=PolyCollection)
        assert m.__repr__() == "<Markers, length: 2>"

        m = Points(offsets=offsets)
        assert m.__repr__() == "<Points, length: 2>"

    def test_rep_iterating(self, signal):
        offsets = np.empty(3, dtype=object)
        for i in range(3):
            offsets[i] = np.array([[1, 1], [2, 2]])
        m = Points(offsets=offsets)
        assert m.__repr__() == "<Points, length: variable (current: not plotted)>"

        signal.plot()
        signal.add_marker(m)
        assert m.__repr__() == "<Points, length: variable (current: 2)>"

    def test_update_static(self):
        m = Points(offsets=([[1, 1], [2, 2]]))
        s = Signal1D(np.ones((10, 10)))
        s.plot()
        s.add_marker(m)
        s.axes_manager.navigation_axes[0].index = 2

    @pytest.mark.parametrize(
        "subclass",
        (
            (Arrows, Quiver, {"offsets": [[1, 1]], "U": [1], "V": [1]}),
            (Circles, CircleCollection, {"offsets": [[1, 1]], "sizes": [1]}),
            (
                Ellipses,
                EllipseCollection,
                {"offsets": [1, 2], "widths": [1], "heights": [1]},
            ),
            (HorizontalLines, LineCollection, {"offsets": [1, 2]}),
            (Points, CircleCollection, {"offsets": [[1, 1]], "sizes": [1]}),
            (VerticalLines, LineCollection, {"offsets": [1, 2]}),
            (
                Rectangles,
                RectangleCollection,
                {"offsets": [[1, 1]], "widths": [1], "heights": [1]},
            ),
            (Squares, SquareCollection, {"offsets": [[1, 1]], "widths": [1]}),
            (Texts, TextCollection, {"offsets": [[1, 1]], "texts": ["a"]}),
            (Lines, LineCollection, {"segments": [[0, 0], [1, 1]]}),
            (
                Markers,
                StarPolygonCollection,
                {"collection": "StarPolygonCollection", "offsets": [[1, 1]]},
            ),
        ),
    )
    def test_initialize_subclasses(self, subclass):
        m = subclass[0](**subclass[2])
        assert m._collection_class is subclass[1]

    @pytest.mark.parametrize(
        "subclass",
        (
            (Arrows, {"offsets": [[1, 1]], "U": [1], "V": [1]}),
            (Circles, {"offsets": [[1, 1]], "sizes": [1]}),
            (Ellipses, {"offsets": [1, 2], "widths": [1], "heights": [1]}),
            (HorizontalLines, {"offsets": [1, 2]}),
            (Points, {"offsets": [[1, 1]], "sizes": [1]}),
            (VerticalLines, {"offsets": [1, 2]}),
            (Rectangles, {"offsets": [[1, 1]], "widths": [1], "heights": [1]}),
            (Squares, {"offsets": [[1, 1]], "widths": [1]}),
            (Texts, {"offsets": [[1, 1]], "texts": ["a"]}),
            (Lines, {"segments": [[0, 0], [1, 1]]}),
            (Markers, {"collection": "StarPolygonCollection", "offsets": [[1, 1]]}),
        ),
    )
    def test_deepcopy(self, subclass):
        transforms_kwargs = {}
        # only add transform for compatible mMrkers
        if subclass[0] not in [HorizontalLines, VerticalLines]:
            transforms_kwargs["transform"] = "display"
            transforms_kwargs["offset_transform"] = "display"

        m = subclass[0](**subclass[1], **transforms_kwargs)
        m2 = deepcopy(m)

        assert m2 is not m
        print(m.kwargs.keys())
        for key, value in m.kwargs.items():
            print(key, value, m2.kwargs[key])
            assert np.all(m2.kwargs[key] == value)

        assert m2.offset_transform == m.offset_transform == "display"
        if subclass[0] not in [HorizontalLines, VerticalLines]:
            assert m2.transform == "display"
            assert m2.transform == m.transform == "display"
        assert "transform" not in m.kwargs
        assert "transform" not in m2.kwargs

    def test_markers_errors_incorrect_transform(self):
        with pytest.raises(ValueError):
            Circles(offsets=[[1, 1]], sizes=1, transform="data")
        with pytest.raises(ValueError):
            Lines(segments=[[0, 0], [1, 1]], offset_transform="data")
        with pytest.raises(ValueError):
            Polygons(verts=[[0, 0], [1, 1]], offset_transform="data")
        with pytest.raises(ValueError):
            Rectangles(offsets=[[1, 1]], widths=1, heights=1, transform="data")
        with pytest.raises(ValueError):
            Squares(offsets=[[1, 1]], widths=1, transform="data")
        with pytest.raises(ValueError):
            Ellipses(offsets=[[1, 1]], widths=1, heights=1, transform="data")


class TestsMarkersAddRemove:
    def test_add_items_variable(self):
        offsets = np.array([[1, 1], [2, 2]])
        m = Points(offsets=offsets)
        assert len(m) == 2
        m.add_items(offsets=np.array([[0, 1]]))
        assert len(m) == 3
        assert not m._is_iterating

    def test_add_items_variable_length(self):
        offsets = np.empty(2, dtype=object)
        for i in range(2):
            offsets[i] = np.array([[1, 1], [2, 2]])
        m = Points(offsets=offsets)
        assert m._is_iterating
        for offset in m.kwargs["offsets"].flat:
            assert offset.shape == (2, 2)

        m.add_items(offsets=np.array([[0, 1]]))
        for offset in m.kwargs["offsets"].flat:
            assert offset.shape == (3, 2)
            np.testing.assert_allclose(offset[-1], [0, 1])
        assert len(m.kwargs["offsets"][0]) == 3
        assert m._is_iterating

        m.add_items(offsets=np.array([[0, 2]]))
        for offset in m.kwargs["offsets"].flat:
            assert offset.shape == (4, 2)
            np.testing.assert_allclose(offset[-1], [0, 2])

    def test_remove_items_iterable_navigation_indices(self):
        offsets = np.empty(4, dtype=object)
        texts = np.empty(4, dtype=object)
        for i in range(len(offsets)):
            offsets[i] = np.array([[1, 1], [2, 2]])
            texts[i] = ["a" * (i + 1)] * 2
        m = Texts(offsets=offsets, texts=texts)

        assert m._is_iterating
        for nav_position in range(4):
            assert len(m.kwargs["offsets"][nav_position]) == 2
            assert len(m.kwargs["texts"][nav_position]) == 2
            assert m.kwargs["texts"][nav_position] == ["a" * (nav_position + 1)] * 2
        m.add_items(
            offsets=np.array([[0, 1]]),
            texts=["new_text"],
            navigation_indices=(1,),
        )

        # marker added only in nav coordinates (1, )
        for nav_position in [0, 2, 3]:
            assert len(m.kwargs["offsets"][nav_position]) == 2
            assert len(m.kwargs["texts"][nav_position]) == 2
            assert m.kwargs["texts"][nav_position] == ["a" * (nav_position + 1)] * 2

        assert len(m.kwargs["offsets"][1]) == 3
        assert len(m.kwargs["texts"][1]) == 3
        assert m.kwargs["texts"][1][2] == "new_text"
        assert m.kwargs["texts"][1][2] == "new_text"

    def test_remove_items_iterable_navigation_indices2(self):
        offsets = np.empty(4, dtype=object)
        texts = np.empty(4, dtype=object)
        for i in range(len(offsets)):
            offsets[i] = np.array([[1, 1], [2, 2]])
            texts[i] = ["a" * (i + 1)] * 2
        m = Texts(offsets=offsets, texts=texts)

        assert m._is_iterating
        for nav_position in range(4):
            assert len(m.kwargs["offsets"][nav_position]) == 2
            assert len(m.kwargs["texts"][nav_position]) == 2
        m.remove_items(1, navigation_indices=(1,))

        # marker removed only in nav coordinates (1, )
        for nav_position in [0, 2, 3]:
            assert len(m.kwargs["offsets"][nav_position]) == 2
            assert len(m.kwargs["texts"][nav_position]) == 2
        assert len(m.kwargs["offsets"][1]) == 1
        assert len(m.kwargs["texts"][1]) == 1

    def test_remove_items(self):
        offsets = np.empty(2, dtype=object)
        texts = np.empty(2, dtype=object)
        for i in range(2):
            offsets[i] = np.array([[1, 1], [2, 2]])
            texts[i] = ["a" * i] * 2
        m = Texts(offsets=offsets, texts=texts)
        assert len(m.kwargs["offsets"][0]) == 2
        assert len(m.kwargs["texts"][0]) == 2
        m.remove_items(indices=1, keys="offsets")
        assert len(m.kwargs["offsets"][0]) == 1
        assert len(m.kwargs["texts"][0]) == 2

    def test_remove_items_None(self):
        offsets = np.empty(2, dtype=object)
        texts = ["a"]
        for i in range(2):
            offsets[i] = np.array([[1, 1], [2, 2]])
        m = Texts(offsets=offsets, texts=texts)
        assert len(m.kwargs["offsets"][0]) == 2
        assert len(m.kwargs["texts"]) == 1
        m.remove_items(indices=1)
        assert len(m.kwargs["offsets"][0]) == 1
        assert len(m.kwargs["texts"]) == 1

    def test_remove_items_None_iterable(self):
        nav_length = 4
        offsets = np.empty(nav_length, dtype=object)
        texts = np.empty(nav_length, dtype=object)
        for i in range(len(offsets)):
            # 3 markers
            offsets[i] = np.array([[1, 1], [2, 2], [3, 3]])
            texts[i] = ["a" * (i + 1)] * 3
        m = Texts(offsets=offsets, texts=texts)
        for nav_indices in range(nav_length):
            assert len(m.kwargs["offsets"][nav_indices]) == 3
            assert len(m.kwargs["texts"][nav_indices]) == 3
            np.testing.assert_allclose(m.kwargs["offsets"][nav_indices][0], [1, 1])
            np.testing.assert_allclose(m.kwargs["offsets"][nav_indices][1], [2, 2])
            np.testing.assert_allclose(m.kwargs["offsets"][nav_indices][2], [3, 3])
        m.remove_items(indices=1)
        for nav_indices in range(nav_length):
            assert len(m.kwargs["offsets"][nav_indices]) == 2
            assert len(m.kwargs["texts"][nav_indices]) == 2
            np.testing.assert_allclose(m.kwargs["offsets"][nav_indices][0], [1, 1])
            np.testing.assert_allclose(m.kwargs["offsets"][nav_indices][1], [3, 3])

    def test_remove_items_slice(self):
        offsets = np.stack([np.arange(0, 100, 10)] * 2).T + np.array(
            [
                5,
            ]
            * 2
        )
        texts = np.array(["a", "b", "c", "d", "e", "f", "g", "f", "h", "i"])
        m = Texts(offsets=offsets, texts=texts)

        assert len(m.kwargs["offsets"]) == 10
        assert len(m.kwargs["texts"]) == 10
        m.remove_items(indices=[1, 2])
        assert len(m.kwargs["offsets"]) == 8
        assert len(m.kwargs["texts"]) == 8

        np.testing.assert_allclose(m.kwargs["offsets"][0], [5, 5])
        np.testing.assert_allclose(m.kwargs["offsets"][1], [35, 35])
        assert m.kwargs["texts"][0] == "a"
        assert m.kwargs["texts"][1] == "d"

    def test_remove_items_navigation_indices(self):
        offsets = np.array([[1, 1], [2, 2]])
        m = Points(offsets=offsets)
        assert not m._is_iterating
        assert len(m) == 2
        with pytest.raises(ValueError):
            m.remove_items(1, navigation_indices=(1,))


class TestMarkersDictToMarkers:
    @pytest.fixture
    def iter_data(self):
        data = {
            "x1": np.arange(10),
            "y1": np.arange(10),
            "x2": np.arange(10),
            "y2": np.arange(10),
            "size": np.arange(10),
            "text": np.array(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]),
        }
        return data

    @pytest.fixture
    def static_data(self):
        data = {
            "x1": 1,
            "y1": 2,
            "x2": 3,
            "y2": 4,
            "text": "a",
            "size": 5,
        }
        return data

    @pytest.fixture
    def static_and_iter_data(self):
        data = {
            "x1": 1,
            "y1": np.arange(10),
            "x2": np.arange(10),
            "y2": 4,
            "text": np.array(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]),
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
        "marker_type",
        (
            "Point",
            "HorizontalLineSegment",
            "LineSegment",
            "Ellipse",
            "HorizontalLine",
            "VerticalLine",
            "Arrow",
            "Rectangle",
            "VerticalLineSegment",
            "Text",
        ),
    )
    def test_marker_hs17_API(self, request, marker_type, data, signal):
        d = request.getfixturevalue(data)
        test_dict = {}
        test_dict["data"] = d
        test_dict["marker_type"] = marker_type
        test_dict["marker_properties"] = {"color": "black"}
        test_dict["plot_on_signal"] = True
        if marker_type in ["Ellipse", "Rectangle"]:
            test_dict["marker_properties"]["fill"] = None
        markers = markers_dict_to_markers(test_dict)

        signal.add_marker(
            markers,
        )
        signal.plot()

    @pytest.mark.parametrize(
        "data", ("iter_data", "static_data", "static_and_iter_data")
    )
    @pytest.mark.parametrize("marker_type", ("NotAValidMarker",))
    def test_marker_hs17_API_fail(self, request, marker_type, data):
        d = request.getfixturevalue(data)
        test_dict = {}
        test_dict["data"] = d
        test_dict["marker_type"] = marker_type
        test_dict["marker_properties"] = {"color": "black"}
        test_dict["plot_on_signal"] = True
        with pytest.raises(AttributeError):
            markers_dict_to_markers(test_dict)

    def test_marker2collection_empty(self):
        with pytest.raises(ValueError):
            markers_dict_to_markers({})


def _test_marker_collection_close():
    signal = Signal2D(np.ones((10, 10)))
    segments = np.ones((10, 2, 2))
    markers = Lines(segments=segments)
    signal.add_marker(markers)
    return signal


@update_close_figure()
def test_marker_collection_close():
    return _test_marker_collection_close()


class TestMarkersTransform:
    @pytest.mark.parametrize(
        "offset_transform",
        (
            "data",
            "display",
            "xaxis",
            "yaxis",
            "axes",
            "relative",
        ),
    )
    def test_set_offset_transform(self, offset_transform):
        m = Points(
            offsets=[[1, 1], [4, 4]],
            sizes=(10,),
            color=("black",),
            offset_transform=offset_transform,
        )
        assert m.offset_transform == offset_transform
        signal = Signal1D((np.arange(100) + 1).reshape(10, 10))

        signal.plot()
        signal.add_marker(m)
        mapping = {
            "data": m.ax.transData.__class__,
            "display": IdentityTransform,
            "xaxis": m.ax.get_yaxis_transform().__class__,
            "yaxis": m.ax.get_xaxis_transform().__class__,
            "axes": m.ax.transAxes.__class__,
            "relative": CompositeGenericTransform,
        }

        assert isinstance(m.offset_transform, mapping[offset_transform])

    def test_set_plotted_transform(
        self,
    ):
        markers = Points(
            offsets=[[1, 1], [4, 4]],
            sizes=(10,),
            color=("black",),
            offset_transform="display",
        )
        assert markers.offset_transform == "display"
        signal = Signal1D((np.arange(100) + 1).reshape(10, 10))
        signal.plot()
        signal.add_marker(markers)
        assert isinstance(markers.transform, IdentityTransform)
        assert isinstance(markers.offset_transform, IdentityTransform)
        markers.offset_transform = "data"
        assert isinstance(markers.offset_transform, CompositeGenericTransform)
        assert markers._collection.get_transform() == markers.transform

    def test_unknown_tranform(self):
        with pytest.raises(ValueError):
            _ = Points(
                offsets=[[1, 1], [4, 4]],
                sizes=(10,),
                color=("black",),
                transform="test",
                offset_transform="test",
            )


class TestRelativeMarkers:
    def test_relative_marker_collection(self):
        signal = Signal1D((np.arange(100) + 1).reshape(10, 10))
        segments = np.zeros((10, 2, 2))
        segments[:, 1, 1] = 1  # set y values end
        segments[:, 0, 0] = np.arange(10).reshape(10)  # set x values
        segments[:, 1, 0] = np.arange(10).reshape(10)  # set x values

        markers = Lines(segments=segments, transform="relative")
        texts = Texts(offsets=segments[:, 1], texts="a", offset_transform="relative")
        signal.plot()
        signal.add_marker(markers)
        signal.add_marker(texts)
        signal.axes_manager.navigation_axes[0].index = 1
        segs = markers._collection.get_segments()
        offs = texts._collection.get_offsets()
        assert segs[0][0][0] == 0
        assert segs[0][1][1] == 11
        assert offs[0][1] == 11

    def test_relative_marker_collection_with_shifts(self):
        signal = Signal1D((np.arange(100) + 1).reshape(10, 10))
        segments = np.zeros((10, 2, 2))
        segments[:, 1, 1] = 1  # set y values end
        segments[:, 0, 0] = np.arange(10).reshape(10)  # set x values
        segments[:, 1, 0] = np.arange(10).reshape(10)  # set x values

        markers = Lines(segments=segments, shift=1 / 9, transform="relative")
        texts = Texts(
            offsets=segments[:, 1], shift=1 / 9, texts="a", offset_transform="relative"
        )
        signal.plot()
        signal.add_marker(markers)
        signal.add_marker(texts)
        signal.axes_manager.navigation_axes[0].index = 1
        segs = markers._collection.get_segments()
        offs = texts._collection.get_offsets()
        assert segs[0][0][0] == 0
        assert segs[0][1][1] == 12
        assert offs[0][1] == 12


class TestLines:
    @pytest.fixture
    def offsets(self):
        d = np.empty((3,), dtype=object)
        for i in np.ndindex(d.shape):
            d[i] = np.arange(i[0] + 1)
        return d

    def test_vertical_line_collection(self, offsets):
        vert = VerticalLines(offsets=offsets)
        s = Signal2D(np.zeros((3, 3, 3)))
        # s.axes_manager.signal_axes[0].offset = 0
        # s.axes_manager.signal_axes[1].offset = 0
        s.plot()
        s.add_marker(vert)
        segments = vert.get_current_kwargs()["segments"]
        # Offsets --> segments for vertical lines
        assert segments.shape == (1, 2, 2)  # one line
        np.testing.assert_array_equal(segments, np.array([[[0, 0], [0, 1]]]))

        # change position to navigation coordinate (1, )
        s.axes_manager.indices = (1,)
        segments = vert.get_current_kwargs()["segments"]
        assert segments.shape == (2, 2, 2)  # two lines
        np.testing.assert_array_equal(
            segments, np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]])
        )

    def test_horizontal_line_collection(self, offsets):
        hor = HorizontalLines(offsets=offsets)
        s = Signal2D(np.zeros((3, 3, 3)))
        s.axes_manager.signal_axes[0].offset = 0
        s.axes_manager.signal_axes[1].offset = 0
        s.plot(interpolation=None)
        s.add_marker(hor)
        kwargs = hor.get_current_kwargs()
        np.testing.assert_array_equal(kwargs["segments"], [[[0.0, 0], [1, 0]]])

    def test_horizontal_vertical_line_error(self, offsets):
        with pytest.raises(ValueError):
            _ = HorizontalLines(offsets=offsets, transform="data")
        with pytest.raises(ValueError):
            _ = VerticalLines(offsets=offsets, transform="data")


def test_marker_collection_close_render():
    signal = Signal2D(np.ones((2, 10, 10)))
    markers = Points(offsets=[[1, 1], [4, 4]], sizes=(10,), color=("black",))
    signal.plot()
    signal.add_marker(markers, render_figure=True)
    markers.close(render_figure=True)


class TestMarkers2:
    @pytest.fixture
    def offsets(self):
        d = np.empty((3,), dtype=object)
        for i in np.ndindex(d.shape):
            d[i] = np.stack([np.arange(3), np.ones(3) * i], axis=1)
        return d

    @pytest.fixture
    def extra_kwargs(self):
        widths = np.empty((3,), dtype=object)
        for i in np.ndindex(widths.shape):
            widths[i] = np.ones(3)

        heights = np.empty((3,), dtype=object)
        for i in np.ndindex(heights.shape):
            heights[i] = np.ones(3)
        angles = np.empty((3,), dtype=object)
        for i in np.ndindex(angles.shape):
            angles[i] = np.ones(3)

        kwds = {
            Points: {},
            Circles: {"sizes": (1,)},
            Arrows: {"U": 1, "V": 1},
            Ellipses: {"widths": widths, "heights": heights, "angles": angles},
        }
        return kwds

    @pytest.fixture
    def signal(self):
        return Signal2D(np.ones((3, 10, 10)))

    @pytest.mark.parametrize("MarkerClass", [Points, Circles, Ellipses, Arrows])
    def test_offsest_markers(self, extra_kwargs, MarkerClass, offsets, signal):
        markers = MarkerClass(offsets=offsets, **extra_kwargs[MarkerClass])
        signal.plot()
        signal.add_marker(markers)
        signal.axes_manager.navigation_axes[0].index = 1

    def test_arrows(self, signal):
        arrows = Arrows(offsets=[[1, 1], [4, 4]], U=1, V=1, C=(2, 2))
        signal.plot()
        signal.add_marker(arrows)
        signal.axes_manager.navigation_axes[0].index = 1

    @pytest.mark.parametrize("MarkerClass", [Points, Circles, Ellipses, Arrows])
    def test_markers_length_offsets(self, extra_kwargs, MarkerClass, offsets, signal):
        m = MarkerClass(offsets=offsets, **extra_kwargs[MarkerClass])
        # variable length markers: needs axes_manager, etc.
        with pytest.raises(RuntimeError):
            len(m)

        signal.add_marker(m)

        assert len(m) == 3


def test_polygons():
    s = Signal2D(np.zeros((100, 100)))
    poylgon1 = [[1, 1], [20, 20], [1, 20], [25, 5]]
    poylgon2 = [[50, 60], [90, 40], [60, 40], [23, 60]]
    verts = [poylgon1, poylgon2]
    m = Polygons(verts=verts)
    s.plot()
    s.add_marker(m)


def test_warning_logger():
    s = Signal2D(np.ones((10, 10)))
    m = Points(
        offsets=[
            [1, 1],
        ],
        sizes=10,
    )
    s.plot()
    with pytest.warns(UserWarning):
        s.add_marker(m, plot_marker=False, permanent=False)


@pytest.mark.mpl_image_compare(
    baseline_dir=BASELINE_DIR, tolerance=5.0, style=STYLE_PYTEST_MPL
)
def test_load_old_markers():
    """
    File generated using

    import hyperspy.api as hs
    import numpy as np

    s = hs.signals.Signal2D(np.ones((14, 14)))

    m_point = hs.plot.markers.point(x=2, y=2, color='C0')
    m_line = hs.plot.markers.line_segment(x1=4, x2=6, y1=2, y2=4, color='C1')
    m_vline = hs.plot.markers.vertical_line(x=12, color='C1')
    m_vline_segment = hs.plot.markers.vertical_line_segment(x=8, y1=0, y2=4, color='C1')
    m_hline = hs.plot.markers.horizontal_line(y=5, color='C1')
    m_hline_segment = hs.plot.markers.horizontal_line_segment(x1=1, x2=9, y=6, color='C1')
    m_arrow = hs.plot.markers.arrow(x1=4, y1=7, x2=6, y2=8, arrowstyle='<->')
    m_text = hs.plot.markers.text(x=1, y=4, text="test", color='C2')
    m_rect = hs.plot.markers.rectangle(x1=1, x2=3, y1=7, y2=12, edgecolor='C3', fill=True, facecolor='C4')
    m_ellipse = hs.plot.markers.ellipse(x=8, y=10, width=4, height=5, edgecolor='C5')

    marker_list = [
        m_point,
        m_line,
        m_vline,
        m_vline_segment,
        m_hline,
        m_hline_segment,
        m_rect,
        m_text,
        m_arrow,
        m_ellipse
        ]

    s.add_marker(marker_list, permanent=True)
    s.plot(axes_ticks=True)

    import matplotlib.pyplot as plt
    plt.savefig('test.png', dpi=300)
    s.save("signal_markers_hs1_7_5.hspy")
    """
    s = hs.load(FILE_PATH / "data" / "signal_markers_hs1_7_5.hspy")
    s.metadata.General.original_filename = ""
    s.tmp_parameters.filename = ""
    s.plot(axes_ticks=True)
    return s._plot.signal_plot.figure


@pytest.mark.mpl_image_compare(
    baseline_dir=BASELINE_DIR, tolerance=5.0, style=STYLE_PYTEST_MPL
)
def test_colorbar_collection():
    s = Signal2D(np.ones((100, 100)))
    rng = np.random.default_rng(0)
    sizes = rng.random((10,)) * 20 + 5
    offsets = rng.random((10, 2)) * 100
    m = hs.plot.markers.Circles(
        sizes=sizes,
        offsets=offsets,
        linewidth=2,
    )

    with pytest.raises(RuntimeError):
        m.plot_colorbar()

    s.plot()
    s.add_marker(m)
    m.set_ScalarMappable_array(sizes.ravel() / 2)
    cbar = m.plot_colorbar()
    cbar.set_label("Circle radius")
    return s._plot.signal_plot.figure


def test_collection_error():
    with pytest.raises(ValueError):
        Markers(offsets=[[1, 1], [2, 2]], collection="NotACollection")

    with pytest.raises(ValueError):
        Markers(offsets=[[1, 1], [2, 2]], collection=object)

    m = Points(offsets=[[1, 1], [2, 2]])
    with pytest.raises(ValueError):
        m._set_transform(value="test")


def test_permanent_markers_close_open_cycle():
    s = Signal2D(np.ones((100, 100)))
    rng = np.random.default_rng(0)
    offsets = rng.random((10, 2)) * 100
    m = hs.plot.markers.Points(offsets=offsets)
    assert m._signal is None
    assert m._axes_manager is None

    s.add_marker(m, permanent=True)
    assert m._signal is s
    assert m._axes_manager is s.axes_manager

    s._plot.close()
    assert m._signal is None
    assert m._axes_manager is None

    s.plot()
    assert m._signal is s
    assert m._axes_manager is s.axes_manager


def test_variable_length_markers_navigation_shape():
    nav_dim = 2
    rng = np.random.default_rng(0)

    nav_shape = np.arange(10, 10 * (nav_dim + 1), step=10)
    data = np.ones(tuple(nav_shape) + (100, 100))
    s = hs.signals.Signal2D(data)

    offsets = np.empty(s.axes_manager.navigation_shape, dtype=object)
    for ind in np.ndindex(offsets.shape):
        num = rng.integers(3, 10)
        offsets[ind] = rng.random((num, 2)) * 100

    m = hs.plot.markers.Points(
        offsets=offsets,
        color="orange",
    )

    s.plot()
    s.add_marker(m, permanent=True)
    # go to last indices to check that the shape of `offsets` and
    # navigation are aligned and plotting/getting currnet kwargs works fine
    s.axes_manager.indices = np.array(s.axes_manager.navigation_shape) - 1


@pytest.mark.mpl_image_compare(
    baseline_dir=BASELINE_DIR, tolerance=5.0, style=STYLE_PYTEST_MPL
)
def test_position_texts_with_mathtext():
    s = hs.signals.Signal2D(np.arange(25).reshape((5, 5)))

    s.plot()

    offset = [
        [3, 3],
    ]
    raw_text = "$\\bar{1}$"

    point_marker = hs.plot.markers.Points(offset)
    text_marker = hs.plot.markers.Texts(
        offset,
        texts=[
            raw_text,
        ],
        color="red",
    )

    s.add_marker([point_marker, text_marker])

    return s._plot.signal_plot.figure
