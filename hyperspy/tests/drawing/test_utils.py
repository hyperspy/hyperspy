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
import nose.tools as nt
import warnings

import hyperspy.drawing.utils as utils
from hyperspy.misc.test_utils import assert_warns


@cleanup
def test_create_figure():
    def dummy_function():
        global test
        test = 10
        print('dummy_function have been called after closing windows')
    window_title = 'test title'
    fig = utils.create_figure(window_title=window_title,
                              _on_figure_window_close=dummy_function)
    nt.assert_true(isinstance(fig, matplotlib.figure.Figure))

    matplotlib.pyplot.close(fig)
    nt.assert_equal(test, 10)


@cleanup
def test_create_figure2():
    dummy_warning = 'Dummy_function have been called after closing windows'

    def dummy_function():
        warnings.warn(dummy_warning, UserWarning)
        print(dummy_warning)

    with assert_warns(
            message=dummy_warning,
            category=UserWarning):
        window_title = 'test title'
        fig = utils.create_figure(window_title=window_title,
                                  _on_figure_window_close=dummy_function)
        nt.assert_true(isinstance(fig, matplotlib.figure.Figure))
        matplotlib.pyplot.close(fig)
