# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
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

from hyperspy.axes import VectorDataAxis
import pytest
import traits.api as t


class TestVectorAxis:
    def setup_method(self, method):
        self.axis = VectorDataAxis()

    def test_initialisation_VectorDataAxis_default(self):
        with pytest.raises(AttributeError):
            assert self.axis.index_in_array is None
        assert self.axis.name is t.Undefined
        assert self.axis.units is t.Undefined
        assert not self.axis.navigate
        assert not self.axis.is_binned
        assert self.axis.is_uniform == True
        assert self.axis.axis is t.Undefined
        assert self.axis.size is t.Undefined
