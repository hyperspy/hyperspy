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

from unittest.mock import Mock

import numpy as np
import matplotlib
import pytest

from hyperspy.drawing.utils import create_figure, contrast_stretching


def test_create_figure():
    if matplotlib.get_backend() == "agg":
        pytest.xfail("{} backend does not support on_close event.".format(
            matplotlib.get_backend()))

    dummy_function = Mock()
    fig = create_figure(window_title="test title",
                        _on_figure_window_close=dummy_function)
    assert isinstance(fig, matplotlib.figure.Figure) == True
    matplotlib.pyplot.close(fig)
    dummy_function.assert_called_once_with()


def test_contrast_stretching():
    data = np.arange(100)
    assert contrast_stretching(data, 1, 99) == (1, 99)
    assert contrast_stretching(data, 1.0, 99.0) == (1, 99)
    assert contrast_stretching(data, '1th', '99th') == (0.99, 98.01)
    assert contrast_stretching(data, '0.05th', '99.95th') == (0.0495, 98.9505)
    # vmin, vmax are to set in conftest.py
    assert contrast_stretching(data, None, None) == (0.0, 99.0)
