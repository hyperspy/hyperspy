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

# ruff: noqa: F822

__all__ = [
    "Model1D",
    "Model2D",
]

_import_mapping = {
    "Model1D": "model1d",
    "Model2D": "model2d",
}


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    if name in __all__:
        import_path = "hyperspy.models." + _import_mapping.get(name)
        return getattr(importlib.import_module(import_path), name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
