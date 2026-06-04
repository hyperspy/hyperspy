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

from hyperspy.defaults_parser import preferences
from hyperspy.utils import show_keybindings


def test_text_output(capsys):
    show_keybindings()
    captured = capsys.readouterr()
    assert "Navigation" in captured.out
    assert "Plot Interaction" in captured.out
    assert "Widget Resize" in captured.out
    assert "Model Plot" in captured.out
    assert "<p>" not in captured.out


def test_contains_navigation_keys(capsys):
    show_keybindings()
    captured = capsys.readouterr()
    assert "Navigate dimension 0" in captured.out
    assert "Navigate dimension 1" in captured.out
    assert "Navigate dimension 2" in captured.out


def test_contains_plot_keys(capsys):
    show_keybindings()
    captured = capsys.readouterr()
    assert "Toggle second pointer" in captured.out
    assert "Launch contrast adjustment tool" in captured.out
    assert "Toggle log/linear scale" in captured.out


def test_contains_widget_keys(capsys):
    show_keybindings()
    captured = capsys.readouterr()
    assert "Increase widget size" in captured.out
    assert "Decrease widget size" in captured.out
    assert "Jump-to-click on span/cursor" in captured.out
    assert "Snap to 30" in captured.out


def test_contains_model_keys(capsys):
    show_keybindings()
    captured = capsys.readouterr()
    assert "Toggle adjust-position lines" in captured.out
    assert "Toggle plot-components visibility" in captured.out
    assert "Toggle residual display" in captured.out


def test_contains_step_keys(capsys):
    show_keybindings()
    captured = capsys.readouterr()
    assert "step multiplier" in captured.out.lower()


def test_no_html_in_output(capsys):
    show_keybindings()
    captured = capsys.readouterr()
    assert "<table>" not in captured.out
    assert "<tr>" not in captured.out
    assert "<td>" not in captured.out


def test_platform_macos_shows_symbols(capsys):
    """platform='macos' forces macOS symbols regardless of host OS."""
    try:
        old = preferences.Plot.platform_shortcuts
        preferences.Plot.platform_shortcuts = "standard"
        show_keybindings(platform="macos")
        captured = capsys.readouterr()
        # Standard preset: modifiers are ctrl/shift/alt → symbols ⌃⇧⌥
        assert "\u2303" in captured.out  # ⌃
    finally:
        preferences.Plot.platform_shortcuts = old


def test_platform_standard_shows_raw(capsys):
    """platform='standard' forces raw ASCII regardless of host OS."""
    try:
        old = preferences.Plot.platform_shortcuts
        preferences.Plot.platform_shortcuts = "macos"
        show_keybindings(platform="standard")
        captured = capsys.readouterr()
        # No macOS symbols must appear
        assert "\u2303" not in captured.out  # ⌃
        assert "\u2325" not in captured.out  # ⌥
        # MacBook aliases (fn+↑) must not appear
        assert "fn+" not in captured.out
    finally:
        preferences.Plot.platform_shortcuts = old


def test_platform_none_respects_preference(capsys):
    """platform=None uses platform_shortcuts preference for display."""
    try:
        old = preferences.Plot.platform_shortcuts
        preferences.Plot.platform_shortcuts = "macos"
        show_keybindings()
        captured = capsys.readouterr()
        # macOS preference → MacBook aliases (fn+↑ for pageup)
        assert "fn+" in captured.out
    finally:
        preferences.Plot.platform_shortcuts = old
