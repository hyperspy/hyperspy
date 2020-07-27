# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
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

import numpy as np
import pytest

import hyperspy.signals
from hyperspy.misc.utils import find_subclasses
from hyperspy.signal import BaseSignal


@pytest.mark.parametrize("signal",
                         find_subclasses(hyperspy.signals, BaseSignal))
def test_lazy_signal_inheritance(signal):
    bs = getattr(hyperspy.signals, signal)
    s = bs(np.empty((2,) * bs._signal_dimension))
    ls = s.as_lazy()
    assert isinstance(ls, bs)
