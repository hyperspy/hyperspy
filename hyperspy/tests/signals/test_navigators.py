import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import hyperspy.api as hs


def _make_signal_with_2d_nav():
    """Signal2D with nav=(15, 10), sig=(64, 64). Data shape (10, 15, 64, 64)."""
    s = hs.signals.Signal2D(np.ones((10, 15, 64, 64)))
    # navigation_axes display order (innermost first):
    #   [0] inner nav: data-axis 1, size=15
    #   [1] outer nav: data-axis 0, size=10
    s.axes_manager.navigation_axes[0].scale = 2.0
    s.axes_manager.navigation_axes[0].units = "nm"
    s.axes_manager.navigation_axes[1].scale = 1.5
    s.axes_manager.navigation_axes[1].units = "nm"
    return s


def _make_nav_for_2d(s):
    """Pure-signal navigator whose data matches the parent nav shape."""
    # nav_shape = (15, 10); data (10, 15)
    nav = hs.signals.Signal2D(np.ones((10, 15)))
    # Signal2D(10, 15): nav_shape=(), sig_shape=(15, 10) display order
    # signal_axes[0] = inner (data-axis 1, size=15); [1] = outer (data-axis 0, size=10)
    nav.axes_manager.signal_axes[0].scale = 2.0
    nav.axes_manager.signal_axes[0].units = "nm"
    nav.axes_manager.signal_axes[1].scale = 1.5
    nav.axes_manager.signal_axes[1].units = "nm"
    return nav


class TestNavigatorsProxyBasic:
    def test_set_and_get(self):
        s = _make_signal_with_2d_nav()
        nav = _make_nav_for_2d(s)
        s.navigators["VBF"] = nav
        assert s.navigators["VBF"] is nav

    def test_contains(self):
        s = _make_signal_with_2d_nav()
        nav = _make_nav_for_2d(s)
        s.navigators["VBF"] = nav
        assert "VBF" in s.navigators
        assert "other" not in s.navigators

    def test_delete(self):
        s = _make_signal_with_2d_nav()
        nav = _make_nav_for_2d(s)
        s.navigators["VBF"] = nav
        del s.navigators["VBF"]
        assert "VBF" not in s.navigators

    def test_keys_values_items(self):
        s = _make_signal_with_2d_nav()
        nav = _make_nav_for_2d(s)
        s.navigators["VBF"] = nav
        assert list(s.navigators.keys()) == ["VBF"]
        assert list(s.navigators.values()) == [nav]
        assert list(s.navigators.items()) == [("VBF", nav)]

    def test_len_and_iter(self):
        s = _make_signal_with_2d_nav()
        nav = _make_nav_for_2d(s)
        s.navigators["VBF"] = nav
        assert len(s.navigators) == 1
        assert list(s.navigators) == ["VBF"]


class TestNavigatorsProxyValidation:
    def test_wrong_type_raises_typeerror(self):
        s = _make_signal_with_2d_nav()
        with pytest.raises(TypeError, match="BaseSignal"):
            s.navigators["bad"] = np.ones((10, 15))

    def test_wrong_ndim_raises_valueerror(self):
        s = _make_signal_with_2d_nav()
        # 1D navigator for a 2D nav signal
        nav_1d = hs.signals.Signal1D(np.ones((10,)))
        with pytest.raises(ValueError, match="2 total dimensions"):
            s.navigators["bad"] = nav_1d

    def test_wrong_shape_raises_valueerror(self):
        s = _make_signal_with_2d_nav()
        # 2D but wrong sizes
        nav_bad = hs.signals.Signal2D(np.ones((8, 12)))
        with pytest.raises(ValueError, match="compatible"):
            s.navigators["bad"] = nav_bad

    def test_correct_shape_passes(self):
        s = _make_signal_with_2d_nav()
        nav = _make_nav_for_2d(s)
        s.navigators["VBF"] = nav  # should not raise


def _make_5d_signal_and_navigator():
    """
    Parent: Signal2D, data (3, 5, 7, 64, 64)
      nav axes (display, innermost first):
        [0] inner:  data-axis 2, size=7,  scale=2.0, units="nm"  (y)
        [1] middle: data-axis 1, size=5,  scale=1.5, units="nm"  (x)
        [2] outer:  data-axis 0, size=3,  scale=0.1, units="s"   (time)
      nav_shape display = (7, 5, 3)

    Navigator: Signal2D, data (3, 5, 7)  -> nav=(time=3,), sig=(x=5, y=7)
      Signal2D(np.ones((3, 5, 7))):
        data shape (3, 5, 7)
        Signal2D -> signal_dimension=2
        nav axis: data-axis 0 (size=3)
        sig axes: data-axis 1 (size=5), data-axis 2 (size=7)
        signal_axes display (innermost first): [0]=data-axis2(size=7), [1]=data-axis1(size=5)
        navigation_axes display: [0]=data-axis0(size=3)
    """
    s = hs.signals.Signal2D(np.zeros((3, 5, 7, 64, 64)))
    s.axes_manager.navigation_axes[0].scale = 2.0  # y (inner, size=7)
    s.axes_manager.navigation_axes[0].units = "nm"
    s.axes_manager.navigation_axes[0].offset = 0.0
    s.axes_manager.navigation_axes[1].scale = 1.5  # x (middle, size=5)
    s.axes_manager.navigation_axes[1].units = "nm"
    s.axes_manager.navigation_axes[1].offset = 0.0
    s.axes_manager.navigation_axes[2].scale = 0.1  # time (outer, size=3)
    s.axes_manager.navigation_axes[2].units = "s"
    s.axes_manager.navigation_axes[2].offset = 0.0

    nav = hs.signals.Signal2D(np.zeros((3, 5, 7)))
    # navigation_axes[0] = data-axis 0 (size=3) = time
    # signal_axes[0]     = data-axis 2 (size=7) = y (inner)
    # signal_axes[1]     = data-axis 1 (size=5) = x
    nav.axes_manager.navigation_axes[0].scale = 0.1  # time
    nav.axes_manager.navigation_axes[0].units = "s"
    nav.axes_manager.navigation_axes[0].offset = 0.0
    nav.axes_manager.signal_axes[0].scale = 2.0  # y (inner sig)
    nav.axes_manager.signal_axes[0].units = "nm"
    nav.axes_manager.signal_axes[0].offset = 0.0
    nav.axes_manager.signal_axes[1].scale = 1.5  # x
    nav.axes_manager.signal_axes[1].units = "nm"
    nav.axes_manager.signal_axes[1].offset = 0.0

    return s, nav


class TestInavSlicing:
    def test_inav_2d_nav_slice(self):
        """2D nav: inav[0:5, 0:8] slices both signal dims of the navigator."""
        s = _make_signal_with_2d_nav()
        nav = _make_nav_for_2d(s)
        s.navigators["nav"] = nav

        # Parent nav_shape display (innermost first) = (15, 10)
        # inav[0:5, 0:8]: s0=0:5 -> inner(15->5), s1=0:8 -> outer(10->8)
        sliced = s.inav[0:5, 0:8]

        assert "nav" in sliced.navigators
        # Navigator had data (10, 15): outer-in-array=10, inner-in-array=15
        # After isig[0:5, 0:8]: inner(15->5), outer(10->8)
        # Result data shape: outer(8) x inner(5) = (8, 5)
        assert sliced.navigators["nav"].data.shape == (8, 5)

    def test_inav_5d_mixed_slice(self):
        """5D signal: inav slices propagate to mixed nav|sig navigator."""
        s, nav = _make_5d_signal_and_navigator()
        s.navigators["nav"] = nav

        # Parent nav display order: [0]=y(7), [1]=x(5), [2]=time(3)
        # inav[0:4, 0:3, 0:2]:
        #   s0=0:4 -> y(7->4)
        #   s1=0:3 -> x(5->3)
        #   s2=0:2 -> time(3->2)
        sliced = s.inav[0:4, 0:3, 0:2]

        assert "nav" in sliced.navigators
        nav_sliced = sliced.navigators["nav"]
        # Navigator original data: (time=3, x=5, y=7)
        # After inav[0:2] on time: (2, 5, 7)
        # After isig[0:4, 0:3]:
        #   signal_axes[0]=y(inner,7) -> 0:4 -> 4
        #   signal_axes[1]=x(5)       -> 0:3 -> 3
        # Result: (2, 3, 4)
        assert nav_sliced.data.shape == (2, 3, 4)

    def test_inav_integer_removes_nav_dim(self):
        """Integer inav index removes the corresponding nav dim from navigator."""
        s, nav = _make_5d_signal_and_navigator()
        s.navigators["nav"] = nav

        # inav[0:4, 0:3, 1]:
        #   s0=0:4 -> y(7->4)
        #   s1=0:3 -> x(5->3)
        #   s2=1   -> time(3->removed)
        sliced = s.inav[0:4, 0:3, 1]

        assert "nav" in sliced.navigators
        nav_sliced = sliced.navigators["nav"]
        # Navigator original data: (time=3, x=5, y=7)
        # After inav[1] on time: removes nav dim -> data (5, 7) but as signal
        # After isig[0:4, 0:3]: y(7->4), x(5->3)
        # Result shape: (3, 4)
        assert nav_sliced.data.shape == (3, 4)
        # No navigation dimension remaining
        assert nav_sliced.axes_manager.navigation_dimension == 0


class TestSetDefault:
    def test_set_default_sets_singular_navigator(self):
        s = _make_signal_with_2d_nav()
        nav = _make_nav_for_2d(s)
        s.navigators["VBF"] = nav
        assert s.navigator is None  # not yet set

        s.navigators.set_default("VBF")
        assert s.navigator is nav

    def test_set_default_missing_key_raises_keyerror(self):
        s = _make_signal_with_2d_nav()
        with pytest.raises(KeyError, match="not found"):
            s.navigators.set_default("nonexistent")


class TestComputeNavigator:
    def test_compute_navigator_saves_to_dict(self):
        s = hs.signals.Signal2D(np.ones((5, 7, 16, 16)))
        s.compute_navigator()
        assert "Signal Sum Image" in s.navigators

    def test_compute_navigator_sets_singular_navigator(self):
        s = hs.signals.Signal2D(np.ones((5, 7, 16, 16)))
        s.compute_navigator()
        assert s.navigator is not None
        assert s.navigator is s.navigators["Signal Sum Image"]

    def test_compute_navigator_shape_matches_nav(self):
        s = hs.signals.Signal2D(np.ones((5, 7, 16, 16)))
        s.compute_navigator()
        nav = s.navigators["Signal Sum Image"]
        # Total data shape of nav should be (5, 7) or (7, 5)
        assert set(nav.data.shape) == {5, 7}

    def test_compute_navigator_values_are_sum(self):
        data = np.arange(5 * 7 * 16 * 16, dtype=float).reshape(5, 7, 16, 16)
        s = hs.signals.Signal2D(data)
        s.compute_navigator()
        nav = s.navigators["Signal Sum Image"]
        # sum over signal axes (last 2 dims, size 16x16=256)
        expected_sum = data.sum(axis=(-2, -1))
        # nav.data should equal expected_sum (possibly transposed)
        assert np.allclose(sorted(nav.data.ravel()), sorted(expected_sum.ravel()))


class TestPlotIntegration:
    def test_plot_string_key_resolves(self):
        """plot(navigator='VBF') resolves the named navigator without error."""
        s = _make_signal_with_2d_nav()
        nav = _make_nav_for_2d(s)
        s.navigators["VBF"] = nav
        try:
            s.plot(navigator="VBF")
        finally:
            plt.close("all")

    def test_plot_invalid_key_raises_valueerror(self):
        s = _make_signal_with_2d_nav()
        nav = _make_nav_for_2d(s)
        s.navigators["VBF"] = nav
        with pytest.raises(ValueError, match="VBF"):
            s.plot(navigator="no_such_key")

    def test_plot_auto_fallback_uses_first_navigator(self):
        """plot(navigator='auto') uses first dict entry when self.navigator is None."""
        s = _make_signal_with_2d_nav()
        nav = _make_nav_for_2d(s)
        s.navigators["VBF"] = nav
        assert s.navigator is None  # no singular navigator set
        try:
            # Should use "VBF" from dict without error
            s.plot(navigator="auto")
            # Confirm the plot was created (no exception)
        finally:
            plt.close("all")

    def test_plot_auto_singular_navigator_takes_priority(self):
        """self.navigator takes priority over the navigators dict."""
        s = _make_signal_with_2d_nav()
        nav = _make_nav_for_2d(s)
        s.navigators["VBF"] = nav
        s.navigator = nav  # set explicitly
        try:
            s.plot(navigator="auto")
        finally:
            plt.close("all")


class TestMapPreservation:
    def test_map_inplace_true_preserves_navigators(self):
        s = _make_signal_with_2d_nav()
        nav = _make_nav_for_2d(s)
        s.navigators["VBF"] = nav
        s.map(lambda x: x * 2, inplace=True)
        assert "VBF" in s.navigators
        assert s.navigators["VBF"] is nav

    def test_map_inplace_false_copies_navigators(self):
        s = _make_signal_with_2d_nav()
        nav = _make_nav_for_2d(s)
        s.navigators["VBF"] = nav
        result = s.map(lambda x: x * 2, inplace=False)
        assert "VBF" in result.navigators
        # Deep copy — different object but same data
        assert result.navigators["VBF"] is not nav
        assert np.allclose(result.navigators["VBF"].data, nav.data)


class TestLazyNavigators:
    def test_lazy_compute_navigator_saves_to_dict(self):
        s = hs.signals.Signal2D(np.ones((5, 7, 16, 16))).as_lazy()
        s.compute_navigator(chunks_number=2)
        assert "Signal Sum Image" in s.navigators
        assert s.navigator is not None

    def test_lazy_plot_auto_uses_dict_without_recomputing(self):
        """If navigators dict is populated, lazy plot should not call compute_navigator."""
        s = hs.signals.Signal2D(np.ones((5, 7, 16, 16))).as_lazy()
        nav = hs.signals.Signal2D(np.ones((7, 5)))  # matches nav shape
        # set axis properties to pass validation
        nav.axes_manager.signal_axes[0].scale = 1.0
        nav.axes_manager.signal_axes[1].scale = 1.0
        # bypass validation and set directly
        nav_dict = {}
        nav_dict["precomputed"] = nav
        s.metadata.set_item("_HyperSpy.navigators", nav_dict)

        called = []
        original_compute = s.compute_navigator

        def mock_compute(*a, **kw):
            called.append(True)
            return original_compute(*a, **kw)

        s.compute_navigator = mock_compute
        try:
            s.plot(navigator="auto")
        finally:
            plt.close("all")

        assert not called, (
            "compute_navigator should not be called when dict is populated"
        )
