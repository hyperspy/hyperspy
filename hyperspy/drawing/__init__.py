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

import importlib

# Register the backend named by the preferences (matplotlib by default).  This
# module is imported lazily, so a preference set before the first plot()
# predates the observer below and is only picked up here.
from hyperspy.defaults_parser import preferences as _pref
from hyperspy.drawing.backends import register_backend as _register_backend
from hyperspy.drawing.backends._registry import load_backend as _load_backend

_register_backend(_load_backend(_pref.Plot.backend))


def _on_backend_pref_change(change=None):
    name = change.new if change is not None else _pref.Plot.backend
    _register_backend(_load_backend(name))


_pref.Plot.observe(_on_backend_pref_change, "backend")

from hyperspy.ipython_magic import (  # noqa: E402
    _register_if_active as _register_anyplotlib_magic_now,
)

_register_anyplotlib_magic_now()

__all__ = [
    "mpl_he",
    "mpl_hie",
    "mpl_hse",
    "signal",
    "signal1d",
    "utils",
    "widgets",
]


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    if name in __all__:  # pragma: no cover
        # We can't get this block covered in the test suite because it is
        # already imported, when running the test suite.
        # If this is broken, a lot of things will be broken!
        return importlib.import_module("." + name, "hyperspy.drawing")

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
