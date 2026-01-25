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

from hyperspy import signals
from hyperspy._signals.complex_signal1d import ComplexSignal1D
from hyperspy.docstrings.signal import LAZYSIGNAL_DOC


class LazyComplexSignal1D(ComplexSignal1D, signals.LazyComplexSignal):
    """Lazy signal class for complex 1-dimensional data."""

    __doc__ += LAZYSIGNAL_DOC.replace("__BASECLASS__", "ComplexSignal1D")
