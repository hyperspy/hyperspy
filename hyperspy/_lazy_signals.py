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

from hyperspy._signals.complex_signal import LazyComplexSignal
from hyperspy._signals.complex_signal1d import LazyComplexSignal1D
from hyperspy._signals.complex_signal2d import LazyComplexSignal2D
from hyperspy._signals.lazy import LazySignal
from hyperspy._signals.signal1d import LazySignal1D
from hyperspy._signals.signal2d import LazySignal2D

__all__ = [
    "LazyComplexSignal",
    "LazyComplexSignal1D",
    "LazyComplexSignal2D",
    "LazySignal",
    "LazySignal1D",
    "LazySignal2D",
]


def __dir__():
    return sorted(__all__)
