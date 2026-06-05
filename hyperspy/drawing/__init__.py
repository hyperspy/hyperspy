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
