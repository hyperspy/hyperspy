# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
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

"""
The Signal class and its specialized subclasses:

BaseSignal
    For generic data with arbitrary signal_dimension. All other signal
    classes inherit from this one. It should only be used with none of
    the others is appropriated.
Signal1D
    For generic data with signal_dimension equal 1, i.e. spectral data of
    n-dimensions. The signal is unbinned by default.
Signal2D
    For generic data with signal_dimension equal 2, i.e. image data of
    n-dimensions. The signal is unbinned by default.
ComplexSignal
    For generic complex data with arbitrary signal_dimension.
ComplexSignal1D
    For generic complex data with signal_dimension equal 1, i.e. spectral
    data of n-dimensions. The signal is unbinned by default.
ComplexSignal2D
    For generic complex data with signal_dimension equal 2, i.e. image
    data of n-dimensions. The signal is unbinned by default.
"""

import importlib

from hyperspy.extensions import EXTENSIONS as EXTENSIONS_

__all__ = [
    signal_ for signal_, specs_ in EXTENSIONS_["signals"].items() if not specs_["lazy"]
]


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    if name in __all__:
        spec = EXTENSIONS_["signals"][name]
        return getattr(importlib.import_module(spec["module"]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
