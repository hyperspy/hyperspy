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
import warnings

from hyperspy.exceptions import VisibleDeprecationWarning
from hyperspy.extensions import EXTENSIONS as EXTENSIONS_

__all__ = [
    signal_ for signal_, specs_ in EXTENSIONS_["signals"].items() if specs_["lazy"]
]


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    warnings.warn(
        "The private module `_lazy_signals` is deprecated and will be removed "
        "in the HyperSpy 3.0 release. Please use the public module "
        "`hyperspy.signals` instead.",
        VisibleDeprecationWarning,
    )
    if name in __all__:
        spec = EXTENSIONS_["signals"][name]
        return getattr(importlib.import_module(spec["module"]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
