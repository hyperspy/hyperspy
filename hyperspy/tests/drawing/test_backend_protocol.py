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

"""Unit tests for the PlottingBackend protocol (Phases 1–4)."""

import inspect

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
    "add_text",
    "update_text",
    "remove_text",
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
    "get_data_transform_inverse",
    "transform_point",
    "simulate_pick",
    "connect_widget_drag",
    "create_span_selector",
    "create_polygon_selector",
    "get_ax_transform",
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


def test_mpl_backend_get_explorer_correct_classes():
    from hyperspy.drawing.backends.mpl import MplBackend
    from hyperspy.drawing.backends.mpl.mpl_he import MPL_HyperExplorer
    from hyperspy.drawing.backends.mpl.mpl_hie import MPL_HyperImage_Explorer
    from hyperspy.drawing.backends.mpl.mpl_hse import MPL_HyperSignal1D_Explorer

    b = MplBackend()
    assert b.get_explorer(0) is MPL_HyperExplorer
    assert b.get_explorer(1) is MPL_HyperSignal1D_Explorer
    assert b.get_explorer(2) is MPL_HyperImage_Explorer
