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
import numpy as np

from hyperspy.misc.eds.utils import _get_element_and_line
from hyperspy.misc.eds.utils import _get_xray_lines


@pytest.mark.parametrize("get_offsets", [True, False])
def test_get_xray_lines(get_offsets):
    x = np.arange(1, 11, 1)
    line_index = [3, 4]
    line_real_index = [2, 3]
    line_relative_factor = [0.5, 0.5]
    norm = 1.0
    minimum_intensity = 0.0

    lines = _get_xray_lines(x,
                            line_index,
                            line_real_index,
                            line_relative_factor,
                            norm,
                            minimum_intensity,
                            get_offsets=get_offsets,
                            factor=1.0,
                            )
    if get_offsets:
        np.testing.assert_array_equal(lines, np.array([[2, 2],
                                                       [3, 2.5]])
                                      )
    else:
        np.testing.assert_array_equal(lines, np.array([[[2, 0], [2, 2]],
                                                       [[3, 0], [3, 2.5]]])
                                      )




def test_get_element_and_line():
    assert _get_element_and_line('Mn_Ka') == ('Mn', 'Ka')

    with pytest.raises(ValueError):
        _get_element_and_line('MnKa') == -1
