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

"""Unit tests for the PlottingBackend protocol and the matplotlib backend."""

import inspect

import numpy as np
import pytest

REQUIRED_METHODS = [
    "create_figure",
    "close_figure",
    "draw_idle",
    "supports_blit",
    "copy_background",
    "restore_background",
    "blit",
    "connect_draw_event",
    "disconnect_event",
    "create_axes",
    "set_xlabel",
    "set_ylabel",
    "set_title",
    "set_xlim",
    "set_ylim",
    "get_xlim",
    "get_ylim",
    "get_xbound",
    "set_axis_off",
    "set_aspect",
    "add_right_axis",
    "remove_right_axis",
    "plot_line",
    "update_line",
    "remove_line",
    "set_line_props",
    "line_get_xdata",
    "line_get_color",
    "line_get_linewidth",
    "add_text",
    "update_text",
    "remove_text",
    "text_set_color",
    "text_get_color",
    "artist_set_animated",
    "plot_image",
    "plot_mesh",
    "image_set_data",
    "image_set_extent",
    "image_set_clim",
    "image_set_norm",
    "get_image_handle",
    "add_colorbar",
    "colorbar_set_label",
    "colorbar_remove",
    "colorbar_redraw",
    "connect_key_press",
    "connect_mouse_move",
    "connect_mouse_press",
    "connect_mouse_release",
    "connect_pick",
    # BlitMixin methods
    "draw_animated_artists",
    "render_figure_from_ax",
    "invalidate_blit_background",
    "supports_blit_from_ax",
    # PointerMixin — consolidated pointer API
    "create_line_pointer",
    "update_line_pointer",
    "create_rect_pointer",
    "update_rect_pointer",
    "remove_pointer",
    "set_pointer_style",
    "add_artist",
    "create_rect_patch",
    "simulate_pick",
    "connect_widget_drag",
    "create_span_selector",
    "create_polygon_selector",
    "convert_coords",
    "create_markers",
    "update_markers",
    "remove_markers",
    "plot_step",
    "create_line2d_patch",
    "create_circle_patch",
    "set_autoscale",
    "set_xticklabels",
    "set_yticklabels",
    # Core PlottingBackend
    "add_collection",
    "collection_update",
    "collection_remove",
    "create_combined_figure_panels",
    "ensure_displayed",
    "connect_close_event",
    "get_explorer",
    "tight_layout",
    "get_figure_from_ax",
    "create_signal1d_figure",
    "create_image_figure",
    "create_scalebar",
    "remove_scalebar",
    "get_image_cmap_name",
]


def test_protocol_declares_all_required_methods():
    from hyperspy.drawing.backends._protocol import PlottingBackend

    members = {name for name, _ in inspect.getmembers(PlottingBackend)}
    for name in REQUIRED_METHODS:
        assert name in members, f"PlottingBackend missing: {name}"


def test_mpl_backend_satisfies_protocol():
    from hyperspy.drawing.backends._protocol import PlottingBackend
    from hyperspy.drawing.backends.mpl import MplBackend

    assert isinstance(MplBackend(), PlottingBackend)


def test_anyplotlib_backend_satisfies_protocol():
    pytest.importorskip("anyplotlib")
    from hyperspy.drawing.backends._protocol import PlottingBackend
    from hyperspy.drawing.backends.anyplotlib import AnyplotlibBackend

    assert isinstance(AnyplotlibBackend(), PlottingBackend)


def test_backend_capability_error_is_notimplementederror():
    from hyperspy.drawing.backends._protocol import BackendCapabilityError

    assert issubclass(BackendCapabilityError, NotImplementedError)


def test_default_backend_get_explorer_all_dims():
    import hyperspy.drawing  # noqa: F401 — ensure backend is registered
    from hyperspy.drawing.backends import get_backend
    from hyperspy.drawing.he import HyperExplorer

    b = get_backend()
    for dim in (0, 1, 2):
        cls = b.get_explorer(dim)
        assert issubclass(cls, HyperExplorer), f"dim={dim} returned non-HyperExplorer"


def test_mpl_backend_create_combined_returns_none_by_default():
    from hyperspy.defaults_parser import preferences
    from hyperspy.drawing.backends.mpl import MplBackend

    original = preferences.Plot.use_subfigure
    try:
        preferences.Plot.use_subfigure = False
        result = MplBackend().create_combined_figure_panels()
        assert result is None
    finally:
        preferences.Plot.use_subfigure = original


def test_old_mpl_explorer_import_paths_still_work():
    """hyperspy.drawing.mpl_h{e,se,ie} are back-compat shims re-exporting the
    classes now living under hyperspy.drawing.backends.mpl.*."""
    from hyperspy.drawing.backends.mpl.mpl_he import MPL_HyperExplorer
    from hyperspy.drawing.backends.mpl.mpl_hie import MPL_HyperImage_Explorer
    from hyperspy.drawing.backends.mpl.mpl_hse import MPL_HyperSignal1D_Explorer
    from hyperspy.drawing.mpl_he import MPL_HyperExplorer as OldHE
    from hyperspy.drawing.mpl_hie import MPL_HyperImage_Explorer as OldHIE
    from hyperspy.drawing.mpl_hse import MPL_HyperSignal1D_Explorer as OldHSE

    assert OldHE is MPL_HyperExplorer
    assert OldHIE is MPL_HyperImage_Explorer
    assert OldHSE is MPL_HyperSignal1D_Explorer


def test_mpl_backend_get_explorer_correct_classes():
    from hyperspy.drawing.backends.mpl import MplBackend
    from hyperspy.drawing.backends.mpl.mpl_he import MPL_HyperExplorer
    from hyperspy.drawing.backends.mpl.mpl_hie import MPL_HyperImage_Explorer
    from hyperspy.drawing.backends.mpl.mpl_hse import MPL_HyperSignal1D_Explorer

    b = MplBackend()
    assert b.get_explorer(0) is MPL_HyperExplorer
    assert b.get_explorer(1) is MPL_HyperSignal1D_Explorer
    assert b.get_explorer(2) is MPL_HyperImage_Explorer


@pytest.fixture
def bare_backend():
    """A backend that overrides nothing: exercises every Mixin/Protocol default."""
    from hyperspy.drawing.backends._protocol import PlottingBackend

    class _BareBackend(PlottingBackend):
        pass

    return _BareBackend()


class TestBlitMixinDefaults:
    def test_supports_blit_false(self, bare_backend):
        assert bare_backend.supports_blit(None) is False

    def test_copy_background_none(self, bare_backend):
        assert bare_backend.copy_background(None) is None

    def test_restore_background_noop(self, bare_backend):
        assert bare_backend.restore_background(None, None) is None

    def test_blit_noop(self, bare_backend):
        assert bare_backend.blit(None) is None

    def test_connect_draw_event_none(self, bare_backend):
        assert bare_backend.connect_draw_event(None, lambda *a: None) is None

    def test_draw_animated_artists_noop(self, bare_backend):
        assert bare_backend.draw_animated_artists(None) is None

    def test_render_figure_from_ax_calls_draw_idle(self, bare_backend):
        calls = []
        bare_backend.draw_idle = lambda fig: calls.append(fig)

        class _Ax:
            figure = "the-figure"

        bare_backend.render_figure_from_ax(_Ax())
        assert calls == ["the-figure"]

    def test_render_figure_from_ax_no_figure_attr(self, bare_backend):
        calls = []
        bare_backend.draw_idle = lambda fig: calls.append(fig)

        class _Ax:
            pass

        bare_backend.render_figure_from_ax(_Ax())
        assert calls == [None]

    def test_invalidate_blit_background_noop(self, bare_backend):
        assert bare_backend.invalidate_blit_background(None) is None

    def test_supports_blit_from_ax_false(self, bare_backend):
        assert bare_backend.supports_blit_from_ax(None) is False


class TestPointerMixinDefaults:
    def test_create_line_pointer_raises(self, bare_backend):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        with pytest.raises(BackendCapabilityError):
            bare_backend.create_line_pointer(None, "x", 0.0)

    def test_update_line_pointer_raises(self, bare_backend):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        with pytest.raises(BackendCapabilityError):
            bare_backend.update_line_pointer(None, "x", 0.0)

    def test_create_rect_pointer_raises(self, bare_backend):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        with pytest.raises(BackendCapabilityError):
            bare_backend.create_rect_pointer(None, 0.0, 0.0, 1.0, 1.0)

    def test_update_rect_pointer_raises(self, bare_backend):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        with pytest.raises(BackendCapabilityError):
            bare_backend.update_rect_pointer(None, 0.0, 0.0, 1.0, 1.0)

    def test_remove_pointer_noop(self, bare_backend):
        assert bare_backend.remove_pointer(None, None) is None

    def test_set_pointer_style_noop(self, bare_backend):
        assert bare_backend.set_pointer_style(None, color="red") is None

    def test_add_artist_noop(self, bare_backend):
        assert bare_backend.add_artist(None, None) is None

    def test_create_rect_patch_raises(self, bare_backend):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        with pytest.raises(BackendCapabilityError):
            bare_backend.create_rect_patch((0, 0), 1.0, 1.0)

    def test_get_data_transform_inverse_raises(self, bare_backend):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        with pytest.raises(BackendCapabilityError):
            bare_backend.get_data_transform_inverse(None)

    def test_transform_point_raises(self, bare_backend):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        with pytest.raises(BackendCapabilityError):
            bare_backend.transform_point(None, (0, 0))

    def test_simulate_pick_noop(self, bare_backend):
        assert bare_backend.simulate_pick(None, None) is None

    def test_connect_widget_drag_noop(self, bare_backend):
        assert bare_backend.connect_widget_drag(None, lambda *a: None) is None

    def test_create_span_selector_raises(self, bare_backend):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        with pytest.raises(BackendCapabilityError):
            bare_backend.create_span_selector(None)

    def test_create_polygon_selector_raises(self, bare_backend):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        with pytest.raises(BackendCapabilityError):
            bare_backend.create_polygon_selector(None)

    def test_get_ax_transform_raises(self, bare_backend):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        with pytest.raises(BackendCapabilityError, match="data"):
            bare_backend.get_ax_transform(None, "data")

    def test_create_line2d_patch_raises(self, bare_backend):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        with pytest.raises(BackendCapabilityError):
            bare_backend.create_line2d_patch([0, 1], [0, 1])

    def test_create_circle_patch_raises(self, bare_backend):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        with pytest.raises(BackendCapabilityError):
            bare_backend.create_circle_patch((0, 0), 1.0)

    def test_convert_coords_raises(self, bare_backend):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        with pytest.raises(BackendCapabilityError):
            bare_backend.convert_coords(None, [(0, 0)], "data", "axes")

    def test_create_markers_raises(self, bare_backend):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        with pytest.raises(BackendCapabilityError):
            bare_backend.create_markers(None, "points")

    def test_update_markers_raises(self, bare_backend):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        with pytest.raises(BackendCapabilityError):
            bare_backend.update_markers(None)

    def test_remove_markers_noop(self, bare_backend):
        assert bare_backend.remove_markers(None, None) is None

    def test_plot_step_raises(self, bare_backend):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        with pytest.raises(BackendCapabilityError):
            bare_backend.plot_step(None, [0, 1], [0, 1])

    def test_set_autoscale_noop(self, bare_backend):
        assert bare_backend.set_autoscale(None, True) is None

    def test_set_xticklabels_noop(self, bare_backend):
        assert bare_backend.set_xticklabels(None, []) is None

    def test_set_yticklabels_noop(self, bare_backend):
        assert bare_backend.set_yticklabels(None, []) is None

    def test_set_xticks_noop(self, bare_backend):
        assert bare_backend.set_xticks(None, []) is None

    def test_set_yticks_noop(self, bare_backend):
        assert bare_backend.set_yticks(None, []) is None


class TestPlottingBackendDefaults:
    def test_get_figure_from_ax_raises(self, bare_backend):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        with pytest.raises(BackendCapabilityError):
            bare_backend.get_figure_from_ax(None)

    def test_create_combined_figure_panels_none(self, bare_backend):
        assert bare_backend.create_combined_figure_panels() is None

    def test_connect_close_event_none(self, bare_backend):
        assert bare_backend.connect_close_event(None, lambda: None) is None

    def test_get_explorer_default_returns_hyperexplorer(self, bare_backend):
        from hyperspy.drawing.he import HyperExplorer

        for dim in (0, 1, 2):
            assert bare_backend.get_explorer(dim) is HyperExplorer

    def test_create_signal1d_figure_raises(self, bare_backend):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        with pytest.raises(BackendCapabilityError):
            bare_backend.create_signal1d_figure()

    def test_create_image_figure_raises(self, bare_backend):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        with pytest.raises(BackendCapabilityError):
            bare_backend.create_image_figure()

    def test_create_scalebar_raises(self, bare_backend):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        with pytest.raises(BackendCapabilityError):
            bare_backend.create_scalebar(None, "nm")

    def test_get_image_cmap_name_raises(self, bare_backend):
        from hyperspy.drawing.backends._protocol import BackendCapabilityError

        with pytest.raises(BackendCapabilityError):
            bare_backend.get_image_cmap_name(None)


@pytest.fixture
def mpl_backend():
    from hyperspy.drawing.backends.mpl import MplBackend

    return MplBackend()


@pytest.fixture
def mpl_fig_ax(mpl_backend):
    fig = mpl_backend.create_figure()
    ax = mpl_backend.create_axes(fig)
    return fig, ax


class TestMplBackend:
    """Direct unit tests for MplBackend methods not exercised by the broader
    signal-plotting integration tests (edge cases, exception-swallowing
    branches, and thin wrappers)."""

    def test_disconnect_event_swallows_exception(self, mpl_backend):
        class Dummy:
            pass

        # No "canvas"/"figure" attribute -> AttributeError inside _canvas,
        # caught and silenced.
        assert mpl_backend.disconnect_event(Dummy(), 1) is None

    def test_remove_right_axis_swallows_exception(self, mpl_backend, mpl_fig_ax):
        class Dummy:
            pass

        _, ax = mpl_fig_ax
        assert mpl_backend.remove_right_axis(ax, Dummy()) is None

    def test_to_mpl_norm_linear(self, mpl_backend):
        from matplotlib.colors import Normalize

        from hyperspy.drawing.norm import LinearNorm

        norm = LinearNorm(vmin=0.1, vmax=0.9, clip=True)
        mpl_norm = mpl_backend._to_mpl_norm(norm)
        assert isinstance(mpl_norm, Normalize)
        assert mpl_norm.vmin == 0.1
        assert mpl_norm.vmax == 0.9

    def test_to_mpl_norm_unknown_hypernorm_returns_none(self, mpl_backend):
        from hyperspy.drawing.norm import HyperNorm

        # A bare HyperNorm is not one of the four known subclasses.
        assert mpl_backend._to_mpl_norm(HyperNorm()) is None

    def test_line_get_xdata(self, mpl_backend, mpl_fig_ax):
        _, ax = mpl_fig_ax
        line = mpl_backend.plot_line(ax, [0, 1, 2], [0, 1, 2])
        np.testing.assert_array_equal(mpl_backend.line_get_xdata(line), [0, 1, 2])

    def test_line_get_linewidth(self, mpl_backend, mpl_fig_ax):
        _, ax = mpl_fig_ax
        line = mpl_backend.plot_line(ax, [0, 1, 2], [0, 1, 2], linewidth=2.5)
        assert mpl_backend.line_get_linewidth(line) == 2.5

    def test_text_get_color_none_handle(self, mpl_backend):
        assert mpl_backend.text_get_color(None) == "black"

    def test_text_get_color_real_handle(self, mpl_backend, mpl_fig_ax):
        _, ax = mpl_fig_ax
        t = mpl_backend.add_text(ax, 0.1, 0.1, "hi", color="red")
        assert mpl_backend.text_get_color(t) == "red"

    def test_image_set_data_falls_back_to_set_array(self, mpl_backend, mpl_fig_ax):
        """A mesh handle (no set_data) must go through the set_array branch."""
        _, ax = mpl_fig_ax
        x, y = np.arange(4), np.arange(4)
        data = np.random.rand(3, 3)
        mesh = mpl_backend.plot_mesh(ax, x, y, data)
        assert not hasattr(mesh, "set_data")
        new_data = np.ones((3, 3))
        mpl_backend.image_set_data(mesh, new_data)  # must not raise

    def test_colorbar_remove(self, mpl_backend, mpl_fig_ax):
        fig, ax = mpl_fig_ax
        im = mpl_backend.plot_image(ax, np.random.rand(4, 4))
        cb = mpl_backend.add_colorbar(fig, im, ax)
        mpl_backend.colorbar_remove(cb)  # must not raise

    def test_remove_pointer_success(self, mpl_backend, mpl_fig_ax):
        _, ax = mpl_fig_ax
        pointer = mpl_backend.create_line_pointer(ax, "x", 0.5)
        assert mpl_backend.remove_pointer(ax, pointer) is None

    def test_remove_pointer_swallows_exception(self, mpl_backend):
        class Dummy:
            pass

        assert mpl_backend.remove_pointer(None, Dummy()) is None

    def test_set_pointer_style_color_and_alpha(self, mpl_backend, mpl_fig_ax):
        _, ax = mpl_fig_ax
        pointer = mpl_backend.create_line_pointer(ax, "x", 0.5)
        mpl_backend.set_pointer_style(pointer, color="blue", alpha=0.5)
        assert pointer.get_color() == "blue"
        assert pointer.get_alpha() == 0.5

    def test_get_data_transform_inverse(self, mpl_backend, mpl_fig_ax):
        _, ax = mpl_fig_ax
        assert mpl_backend.get_data_transform_inverse(ax) is not None

    def test_transform_point(self, mpl_backend, mpl_fig_ax):
        _, ax = mpl_fig_ax
        pt = mpl_backend.transform_point(ax.transData, (0.5, 0.5))
        assert len(pt) == 2

    def test_collection_update(self, mpl_backend, mpl_fig_ax):
        from matplotlib.collections import LineCollection

        _, ax = mpl_fig_ax
        coll = LineCollection([])
        mpl_backend.add_collection(ax, coll)
        mpl_backend.collection_update(coll, color="red")  # must not raise

    def test_collection_remove_swallows_exception(self, mpl_backend):
        class Dummy:
            pass

        assert mpl_backend.collection_remove(None, Dummy()) is None

    def test_connect_close_event_none_fig(self, mpl_backend):
        assert mpl_backend.connect_close_event(None, lambda: None) is None

    def test_connect_close_event_no_canvas(self, mpl_backend):
        class Dummy:
            pass

        assert mpl_backend.connect_close_event(Dummy(), lambda: None) is None

    def test_render_figure_from_ax_falls_back_to_canvas_draw_idle(
        self, mpl_backend, mpl_fig_ax
    ):
        """When ax has a .figure but no hspy_fig, draw via the raw canvas."""
        fig, _ = mpl_fig_ax

        class FakeAx:
            figure = fig

        assert mpl_backend.render_figure_from_ax(FakeAx()) is None

    def test_get_ax_transform_unknown_kind_raises(self, mpl_backend, mpl_fig_ax):
        _, ax = mpl_fig_ax
        with pytest.raises(ValueError, match="Unknown coordinate space"):
            mpl_backend.get_ax_transform(ax, "bogus")

    def test_space_transform_unknown_raises(self, mpl_backend, mpl_fig_ax):
        _, ax = mpl_fig_ax
        with pytest.raises(ValueError, match="Unknown coordinate space"):
            mpl_backend._space_transform(ax, "bogus")

    def test_remove_markers_swallows_exception(self, mpl_backend):
        class Dummy:
            pass

        assert mpl_backend.remove_markers(None, Dummy()) is None

    def test_remove_scalebar(self, mpl_backend, mpl_fig_ax):
        from unittest.mock import MagicMock

        _, ax = mpl_fig_ax
        handle = MagicMock()
        mpl_backend.remove_scalebar(ax, handle)
        handle.remove.assert_called_once()

    def test_get_image_cmap_name(self, mpl_backend, mpl_fig_ax):
        _, ax = mpl_fig_ax
        im = mpl_backend.plot_image(ax, np.random.rand(4, 4), cmap="viridis")
        assert mpl_backend.get_image_cmap_name(im) == "viridis"

    def test_simulate_pick_processes_pick_event(
        self, mpl_backend, mpl_fig_ax, monkeypatch
    ):
        """With a truthy mouse button, the inner try block dispatches a
        PickEvent through the canvas callbacks."""
        import matplotlib.backend_bases as mbb
        import matplotlib.patches as mpatches

        class _ForceButtonMouseEvent(mbb.MouseEvent):
            def __init__(self, *args, **kwargs):
                kwargs.setdefault("button", 1)
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(mbb, "MouseEvent", _ForceButtonMouseEvent)
        _, ax = mpl_fig_ax
        patch = mpatches.Rectangle((0, 0), 1, 1)
        ax.add_patch(patch)
        assert mpl_backend.simulate_pick(ax, patch) is None

    def test_simulate_pick_falls_back_when_pickevent_fails(
        self, mpl_backend, mpl_fig_ax, monkeypatch
    ):
        """If constructing/dispatching the *first* PickEvent raises, fall
        back to canvas.pick_event(). On matplotlib versions where that
        deprecated method still exists, it builds its own PickEvent
        internally, so only the first construction must be forced to fail
        (a global failure would also break the fallback, since it calls
        the same PickEvent constructor)."""
        import matplotlib.backend_bases as mbb
        import matplotlib.patches as mpatches

        real_pick_event = mbb.PickEvent
        calls = []

        def _fail_once(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("boom")
            return real_pick_event(*args, **kwargs)

        class _ForceButtonMouseEvent(mbb.MouseEvent):
            def __init__(self, *args, **kwargs):
                kwargs.setdefault("button", 1)
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(mbb, "MouseEvent", _ForceButtonMouseEvent)
        monkeypatch.setattr(mbb, "PickEvent", _fail_once)
        _, ax = mpl_fig_ax
        patch = mpatches.Rectangle((0, 0), 1, 1)
        ax.add_patch(patch)
        assert mpl_backend.simulate_pick(ax, patch) is None
        assert len(calls) >= 1

    def test_simulate_pick_swallows_attribute_error(self, mpl_backend, mpl_fig_ax):
        """A patch missing get_transform() hits the outer except clause."""
        _, ax = mpl_fig_ax
        assert mpl_backend.simulate_pick(ax, None) is None
