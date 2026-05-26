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

import pytest

import hyperspy.api as hs
from hyperspy.ui_registry import KNOWN_TOOLKITS, _toolkits_to_string, get_gui


class TestKnownToolkits:
    def test_ipywidgets_in_known_toolkits(self):
        assert "ipywidgets" in KNOWN_TOOLKITS

    def test_traitsui_in_known_toolkits(self):
        assert "traitsui" in KNOWN_TOOLKITS

    def test_anywidget_in_known_toolkits(self):
        assert "anywidget" in KNOWN_TOOLKITS


class TestAnywidgetPreference:
    def test_enable_anywidget_gui_default(self):
        assert hs.preferences.GUIs.enable_anywidget_gui is True

    def test_enable_anywidget_gui_setter(self):
        hs.preferences.GUIs.enable_anywidget_gui = False
        assert hs.preferences.GUIs.enable_anywidget_gui is False
        hs.preferences.GUIs.enable_anywidget_gui = True
        assert hs.preferences.GUIs.enable_anywidget_gui is True


class TestToolkitsToString:
    def test_single_toolkit(self):
        assert _toolkits_to_string({"anywidget"}) == "anywidget toolkit"

    def test_string_input(self):
        assert _toolkits_to_string("anywidget") == "anywidget toolkit"

    def test_two_toolkits(self):
        result = _toolkits_to_string({"ipywidgets", "anywidget"})
        assert "anywidget" in result
        assert "ipywidgets" in result
        assert " and " in result
        assert result.endswith(" toolkits")

    def test_three_toolkits(self):
        toolkits = {"ipywidgets", "traitsui", "anywidget"}
        result = _toolkits_to_string(toolkits)
        assert "anywidget" in result
        assert "ipywidgets" in result
        assert "traitsui" in result
        assert ", " in result
        assert " and " in result
        assert result.endswith(" toolkits")


class TestGetGuiToolkitSelection:
    """Verify get_gui respects enable_anywidget_gui when selecting toolkits.

    These tests monkeypatch TOOLKIT_REGISTRY to control which toolkits
    appear registered, so no GUI packages need to be installed.
    """

    @pytest.fixture
    def _mock_signal(self):
        return object()

    def test_anywidget_disabled_raises_correct_error(self, monkeypatch, _mock_signal):
        monkeypatch.setattr("hyperspy.ui_registry.TOOLKIT_REGISTRY", {"anywidget"})
        hs.preferences.GUIs.enable_anywidget_gui = False

        with pytest.raises(ValueError, match="No toolkit available"):
            get_gui(
                _mock_signal,
                toolkey="hyperspy.SimpleMessage",
                toolkit=None,
            )

        hs.preferences.GUIs.enable_anywidget_gui = True

    def test_anywidget_not_registered_select_by_name_raises(
        self, monkeypatch, _mock_signal
    ):
        monkeypatch.setattr("hyperspy.ui_registry.TOOLKIT_REGISTRY", {"ipywidgets"})

        with pytest.raises(ValueError, match="anywidget is not a registered toolkit"):
            get_gui(
                _mock_signal,
                toolkey="hyperspy.SimpleMessage",
                toolkit="anywidget",
            )

    def test_anywidget_registered_enabled_no_other_toolkit(
        self, monkeypatch, _mock_signal
    ):
        monkeypatch.setattr("hyperspy.ui_registry.TOOLKIT_REGISTRY", {"anywidget"})
        assert hs.preferences.GUIs.enable_anywidget_gui is True

        with pytest.raises(NotImplementedError, match="not available"):
            get_gui(
                _mock_signal,
                toolkey="hyperspy.SimpleMessage",
                toolkit="anywidget",
            )
