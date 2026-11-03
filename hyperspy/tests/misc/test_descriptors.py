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


from hyperspy.signal import BaseSignal
from hyperspy.misc.descriptors import AttributeDescriptor
import pytest
import numpy as np


class TestSignal(BaseSignal):
    title = AttributeDescriptor("General.title")
    test = AttributeDescriptor("General.test")
    test_layered = AttributeDescriptor("testbranch1.testbranch2")


class TestAttributeDescriptor:
    @pytest.fixture
    def s(self):
        return TestSignal(np.ones((10, 10, 10)))

    def test_get_title(self, s):
        assert s.title == ""
        s.title = "Test"
        assert s.title == "Test"
        assert s.metadata.General.title == "Test"

    def test_get_set_test(self, s):
        assert not hasattr(s.metadata.General, "test")
        assert s.test is None
        s.test = "Test"
        assert s.test == "Test"
        assert s.metadata.General.test == "Test"

    def test_get_set_test_layered(self, s):
        assert not hasattr(s.metadata, "testbranch1")
        assert s.test_layered is None
        s.test_layered = "Test"
        assert s.test_layered == "Test"
        assert s.metadata.testbranch1.testbranch2 == "Test"
