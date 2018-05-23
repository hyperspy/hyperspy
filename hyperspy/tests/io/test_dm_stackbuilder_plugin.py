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


import os

from hyperspy.io import load

my_path = os.path.dirname(__file__)


class TestStackBuilder:

    def test_load_stackbuilder_imagestack(self):
        image_stack = load(
            my_path +
            "/dm_stackbuilder_plugin/test_stackbuilder_imagestack.dm3")
        data_dimensions = image_stack.data.ndim
        am = image_stack.axes_manager
        axes_dimensions = am.signal_dimension + am.navigation_dimension
        assert data_dimensions == axes_dimensions
