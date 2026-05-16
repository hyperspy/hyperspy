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

from hyperspy import signals
from hyperspy._signals.common_signal1d import CommonSignal1D
from hyperspy.misc._utils import lazy_signal_import_deprecation_warning


class ComplexSignal1D(signals.ComplexSignal, CommonSignal1D):
    """Signal class for complex 1-dimensional data."""

    _signal_dimension = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# ruff: noqa: F822

__all__ = [
    "ComplexSignal1D",
    "LazyComplexSignal1D",
]


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    if "Lazy" in name:
        lazy_signal_import_deprecation_warning(name, __name__)
        return getattr(importlib.import_module("hyperspy.signals"), name)
    if name in __all__:
        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
