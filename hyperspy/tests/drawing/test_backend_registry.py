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
import subprocess
import sys
import textwrap

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


@pytest.mark.parametrize("backend_name", ["matplotlib", "anyplotlib"])
def test_preference_set_before_drawing_import_is_honoured(backend_name):
    """The preference must win even when set before ``hyperspy.drawing`` loads.

    ``hyperspy.drawing`` is imported lazily, on the first ``plot()`` call, so a
    preference set beforehand (in a script, or restored from the user's config
    file) predates the traits observer that reacts to later changes.  If
    ``drawing/__init__`` hardcoded matplotlib, the preference would be silently
    ignored for the whole session.  Run in a subprocess because the test
    session has already imported ``hyperspy.drawing``.
    """
    pytest.importorskip("anyplotlib")
    script = textwrap.dedent(
        f"""
        import sys
        import matplotlib
        matplotlib.use("Agg")
        import hyperspy.api as hs

        assert "hyperspy.drawing" not in sys.modules, "drawing imported too early"
        hs.preferences.Plot.backend = "{backend_name}"

        from hyperspy.drawing.backends import get_backend
        print(type(get_backend()).__name__)
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    expected = {"matplotlib": "MplBackend", "anyplotlib": "AnyplotlibBackend"}
    assert out.stdout.strip().splitlines()[-1] == expected[backend_name]


def test_plot_config_backend_is_str_not_enum():
    from hyperspy.defaults_parser import PlotConfig

    # class_traits() returns CTrait objects; check the underlying trait_type
    trait = PlotConfig.class_traits()["backend"]
    assert isinstance(trait.trait_type, t.Str)
