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

import matplotlib
from matplotlib.testing.decorators import cleanup
import warnings
import sys
import pytest

import hyperspy.drawing.utils as utils
from hyperspy.misc.test_utils import assert_warns


@pytest.mark.skipif(sys.platform == 'darwin',
                    reason="Plot testing not supported on osx by travis-ci")
@cleanup
def test_create_figure():
    dummy_warning = 'dummy_function have been called after closing windows'

    original_backend = matplotlib.get_backend()
    if original_backend == 'agg':
        matplotlib.pyplot.switch_backend('TkAgg')

    def dummy_function():
        # raise a warning to check if this function have been called
        warnings.warn(dummy_warning, UserWarning)

    with assert_warns(message=dummy_warning, category=UserWarning):
        window_title = 'test title'
        fig = utils.create_figure(window_title=window_title,
                                  _on_figure_window_close=dummy_function)
        assert isinstance(fig, matplotlib.figure.Figure) == True
        matplotlib.pyplot.close(fig)
        
    if original_backend == 'agg':  # switch back to the original backend
        matplotlib.pyplot.switch_backend(original_backend)

