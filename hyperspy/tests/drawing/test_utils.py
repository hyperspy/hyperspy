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

import warnings
import sys
from unittest.mock import Mock

import matplotlib
from matplotlib.testing.decorators import cleanup
import pytest

import hyperspy.drawing.utils as utils
from hyperspy.misc.test_utils import assert_warns


@pytest.mark.skipif(sys.platform == 'darwin',
                    reason="Plot testing not supported on osx by travis-ci")
@cleanup
def test_create_figure():
    dummy_warning = 'dummy_function have been called after closing windows'
    if matplotlib.get_backend() not in ("GTKAgg", "WXAgg", "TkAgg", "Qt4Agg"):
        pytest.xfail("{} backend does not support on_close event.".format(
            matplotlib.get_backend()))

    dummy_function = Mock()
    fig = utils.create_figure(window_title="test title",
                              _on_figure_window_close=dummy_function)
    assert isinstance(fig, matplotlib.figure.Figure) == True
    matplotlib.pyplot.close(fig)
    dummy_function.assert_called_once_with()
