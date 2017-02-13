# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.


from hyperspy._signals.common_signal1d import CommonSignal1D
from hyperspy._signals.complex_signal import (ComplexSignal, LazyComplexSignal)


class ComplexSignal1D(ComplexSignal, CommonSignal1D):

    """BaseSignal subclass for complex 1-dimensional data."""

    _signal_dimension = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.axes_manager.signal_dimension != 1:
            self.axes_manager.set_signal_dimension(1)
        self.metadata.Signal.binned = False


class LazyComplexSignal1D(LazyComplexSignal, CommonSignal1D):

    """BaseSignal subclass for lazy complex 1-dimensional data."""

    _signal_dimension = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.axes_manager.signal_dimension != 1:
            self.axes_manager.set_signal_dimension(1)
        self.metadata.Signal.binned = False
