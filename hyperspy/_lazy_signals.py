# -*- coding: utf-8 -*-
# Copyright 2007-2025 The HyperSpy developers
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

# ruff: noqa: F822

__all__ = [
    "LazyComplexSignal",
    "LazyComplexSignal1D",
    "LazyComplexSignal2D",
    "LazySignal",
    "LazySignal1D",
    "LazySignal2D",
]

_import_mapping = {
    "LazyComplexSignal": "complex_signal",
    "LazyComplexSignal1D": "complex_signal1d",
    "LazyComplexSignal2D": "complex_signal2d",
    "LazySignal": "lazy",
    "LazySignal1D": "signal1d",
    "LazySignal2D": "signal2d",
}


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    if name in __all__:
        import_name = f"hyperspy._signals.{_import_mapping[name]}"
        return getattr(importlib.import_module(import_name), name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
