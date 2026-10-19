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

"""Tests for the %anyplotlib IPython magic (hyperspy.ipython_magic)."""

import importlib

import pytest

IPython = pytest.importorskip("IPython")


@pytest.fixture
def ip():
    from IPython.testing.globalipapp import get_ipython

    return get_ipython()


@pytest.fixture(autouse=True)
def _restore_backend_pref():
    from hyperspy.defaults_parser import preferences

    original = preferences.Plot.backend
    yield
    preferences.Plot.backend = original


def test_anyplotlib_magic_switches_backend(ip, capsys):
    from hyperspy.defaults_parser import preferences
    from hyperspy.ipython_magic import _register_anyplotlib_magic

    preferences.Plot.backend = "matplotlib"
    _register_anyplotlib_magic(ip)
    ip.run_line_magic("anyplotlib", "")

    assert preferences.Plot.backend == "anyplotlib"
    assert "switched plotting backend to anyplotlib" in capsys.readouterr().out


def test_load_ipython_extension_registers_magic(ip):
    from hyperspy.ipython_magic import load_ipython_extension

    load_ipython_extension(ip)
    assert ip.find_line_magic("anyplotlib") is not None


def test_drawing_registers_magic_when_ipython_active(ip, monkeypatch):
    """hyperspy.drawing registers %anyplotlib at import time when running
    inside an IPython session (IPython.get_ipython() is not None)."""
    import hyperspy.drawing

    monkeypatch.setattr("IPython.get_ipython", lambda: ip)
    importlib.reload(hyperspy.drawing)
    try:
        assert ip.find_line_magic("anyplotlib") is not None
    finally:
        # Restore the module to its normal (no-IPython-detected) state.
        importlib.reload(hyperspy.drawing)


def test_drawing_import_survives_missing_ipython(monkeypatch):
    """hyperspy.drawing must not fail to import in environments without
    IPython installed at all (the whole magic-registration block is wrapped
    in a bare try/except ImportError)."""
    import sys

    import hyperspy.drawing

    with monkeypatch.context() as m:
        m.setitem(sys.modules, "IPython", None)
        importlib.reload(hyperspy.drawing)  # must not raise
    importlib.reload(hyperspy.drawing)


def test_api_registers_magic_when_ipython_active(ip, monkeypatch):
    """%anyplotlib must be usable right after ``import hyperspy.api as hs``,
    without first touching anything that lazily imports hyperspy.drawing
    (e.g. signal.plot()) — see hyperspy.ipython_magic for why."""
    import hyperspy.api

    monkeypatch.setattr("IPython.get_ipython", lambda: ip)
    importlib.reload(hyperspy.api)
    try:
        assert ip.find_line_magic("anyplotlib") is not None
    finally:
        importlib.reload(hyperspy.api)


def test_api_import_does_not_import_drawing(monkeypatch):
    """Registering the magic from hyperspy.api must not pull in the heavier,
    lazily-loaded hyperspy.drawing package tree (which eagerly registers the
    matplotlib backend) — that would defeat the point of lazy loading."""
    import sys

    for mod in list(sys.modules):
        if mod == "hyperspy.drawing" or mod.startswith("hyperspy.drawing."):
            monkeypatch.delitem(sys.modules, mod)

    import hyperspy.api

    importlib.reload(hyperspy.api)
    assert "hyperspy.drawing" not in sys.modules
