# -*- coding: utf-8 -*-
# Copyright 2007-2023 The HyperSpy developers
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

import pytest

from holospy import __version__
from holospy.signals import HologramImage

def test_hologram_image():
    s = HologramImage([[10, 10], [10, 10]])
    s.axes_manager[0].convert_to_non_uniform_axis()
    s.axes_manager[1].convert_to_non_uniform_axis()
    with pytest.raises(NotImplementedError):
        s.estimate_sideband_position()
    with pytest.raises(NotImplementedError):
        s.estimate_sideband_size(s)
    with pytest.raises(NotImplementedError):
        s.reconstruct_phase()
    with pytest.raises(NotImplementedError):
        s.statistics()