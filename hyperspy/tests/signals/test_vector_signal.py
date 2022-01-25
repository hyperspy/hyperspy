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
# along with HyperSpy. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import pytest

from hyperspy.decorators import lazifyTestClass
from hyperspy.signals import BaseVectorSignal, BaseSignal


class TestVectorSignal:
    @pytest.fixture
    def two_d_vector(self):
        x = np.empty(shape=(4, 3), dtype=object)
        for i in np.ndindex(x.shape):
            x[i] = np.random.random((4, 2))
        s = BaseSignal(x).T
        s.vector = True
        s.set_signal_type("vector")
        for ax, name in zip(s.axes_manager._axes, "abcd"):
            ax.name = name
        return s

    def test_all_to_vector(self, two_d_vector):
        new = two_d_vector.nav_to_vector(inplace=False)
        assert len(new.axes_manager.signal_axes) ==4
        assert new.data.shape == (1,)

    def test_signal_iterable_int_transpose(self):
        t = self.s.transpose(signal_axes=[0, 5, 4])
        var = t.metadata.Signal.Noise_properties.variance
        assert t.axes_manager.signal_shape == (6, 1, 2)
        assert var.axes_manager.signal_shape == (6, 1, 2)
        assert [ax.name for ax in t.axes_manager.signal_axes] == ["f", "a", "b"]

    def test_signal_iterable_names_transpose(self):
        t = self.s.transpose(signal_axes=["f", "a", "b"])
        var = t.metadata.Signal.Noise_properties.variance
        assert t.axes_manager.signal_shape == (6, 1, 2)
        assert var.axes_manager.signal_shape == (6, 1, 2)
        assert [ax.name for ax in t.axes_manager.signal_axes] == ["f", "a", "b"]

    def test_signal_iterable_axes_transpose(self):
        t = self.s.transpose(signal_axes=self.s.axes_manager.signal_axes[:2])
        var = t.metadata.Signal.Noise_properties.variance
        assert t.axes_manager.signal_shape == (6, 5)
        assert var.axes_manager.signal_shape == (6, 5)
        assert [ax.name for ax in t.axes_manager.signal_axes] == ["f", "e"]

    def test_signal_one_name(self):
        with pytest.raises(ValueError):
            self.s.transpose(signal_axes="a")