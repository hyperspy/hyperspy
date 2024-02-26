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
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/>.

import pytest

from hyperspy.misc.signal_tools import broadcast_signals
from hyperspy.signals import Signal1D


def test_boardcast_signals_error():
    with pytest.raises(ValueError):
        broadcast_signals([0, 1], [2, 3])
    with pytest.raises(ValueError):
        broadcast_signals(Signal1D([0, 1]))
