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

"""Tests for the backend entry-point registry and extensible preference (Phase 3)."""

import importlib.metadata

import pytest
import traits.api as t


def test_available_backends_includes_builtins():
    from hyperspy.drawing.backends._registry import available_backends

    names = available_backends()
    assert "matplotlib" in names
    assert "anyplotlib" in names


def test_load_backend_matplotlib():
    from hyperspy.drawing.backends._registry import load_backend
    from hyperspy.drawing.backends.mpl import MplBackend

    backend = load_backend("matplotlib")
    assert isinstance(backend, MplBackend)


def test_load_backend_unknown_raises_valueerror():
    from hyperspy.drawing.backends._registry import load_backend

    with pytest.raises(ValueError, match="Unknown plotting backend"):
        load_backend("__nonexistent__")


def test_external_backend_via_entry_point(monkeypatch):
    """A fake backend injected via entry-point mocking can be loaded."""
    from hyperspy.drawing.backends._registry import load_backend

    class _FakeBackend:
        pass

    class _FakeEP:
        name = "fake_test_backend"

        def load(self):
            return _FakeBackend

    real_entry_points = importlib.metadata.entry_points

    def _patched(group=None, **kwargs):
        result = real_entry_points(group=group, **kwargs)
        if group == "hyperspy.backends":
            return list(result) + [_FakeEP()]
        return result

    monkeypatch.setattr(importlib.metadata, "entry_points", _patched)
    backend = load_backend("fake_test_backend")
    assert isinstance(backend, _FakeBackend)


def test_plot_config_accepts_known_backend():
    from hyperspy.defaults_parser import preferences

    original = preferences.Plot.backend
    try:
        preferences.Plot.backend = "matplotlib"
    finally:
        preferences.Plot.backend = original


def test_plot_config_rejects_unknown_backend():
    """An unknown backend name fails at load time (not at assignment time)."""
    from hyperspy.drawing.backends._registry import load_backend

    with pytest.raises(ValueError, match="Unknown plotting backend"):
        load_backend("__not_a_real_backend__")


def test_plot_config_backend_is_str_not_enum():
    from hyperspy.defaults_parser import PlotConfig

    # class_traits() returns CTrait objects; check the underlying trait_type
    trait = PlotConfig.class_traits()["backend"]
    assert isinstance(trait.trait_type, t.Str)
