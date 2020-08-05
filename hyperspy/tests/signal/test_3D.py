# -*- coding: utf-8 -*-
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

import numpy as np
import pytest

from hyperspy.decorators import lazifyTestClass
from hyperspy.signals import Signal1D


def _test_default_navigation_signal_operations_over_many_axes(self, op):
    s = getattr(self.signal, op)()
    ar = getattr(self.data, op)(axis=(0, 1))
    np.testing.assert_array_equal(ar, s.data)
    assert s.data.ndim == 1
    assert s.axes_manager.signal_dimension == 1
    assert s.axes_manager.navigation_dimension == 0


@lazifyTestClass
class Test3D:
    def setup_method(self, method):
        self.signal = Signal1D(np.arange(2 * 4 * 6).reshape(2, 4, 6))
        self.signal.axes_manager[0].name = "x"
        self.signal.axes_manager[1].name = "y"
        self.signal.axes_manager[2].name = "E"
        self.signal.axes_manager[0].scale = 0.5
        self.data = self.signal.data.copy()

    def test_indexmin(self):
        s = self.signal.indexmin("E")
        ar = self.data.argmin(2)
        np.testing.assert_array_equal(ar, s.data)
        assert s.data.ndim == 2
        assert s.axes_manager.signal_dimension == 0
        assert s.axes_manager.navigation_dimension == 2

    def test_indexmax(self):
        s = self.signal.indexmax("E")
        ar = self.data.argmax(2)
        np.testing.assert_array_equal(ar, s.data)
        assert s.data.ndim == 2
        assert s.axes_manager.signal_dimension == 0
        assert s.axes_manager.navigation_dimension == 2

    def test_valuemin(self):
        s = self.signal.valuemin("x")
        ar = self.signal.axes_manager["x"].index2value(self.data.argmin(1))
        np.testing.assert_array_equal(ar, s.data)
        assert s.data.ndim == 2
        assert s.axes_manager.signal_dimension == 1
        assert s.axes_manager.navigation_dimension == 1

    def test_valuemax(self):
        s = self.signal.valuemax("x")
        ar = self.signal.axes_manager["x"].index2value(self.data.argmax(1))
        np.testing.assert_array_equal(ar, s.data)
        assert s.data.ndim == 2
        assert s.axes_manager.signal_dimension == 1
        assert s.axes_manager.navigation_dimension == 1

    def test_default_navigation_sum(self):
        _test_default_navigation_signal_operations_over_many_axes(self, "sum")

    def test_default_navigation_max(self):
        _test_default_navigation_signal_operations_over_many_axes(self, "max")

    def test_default_navigation_min(self):
        _test_default_navigation_signal_operations_over_many_axes(self, "min")

    def test_default_navigation_mean(self):
        _test_default_navigation_signal_operations_over_many_axes(self, "mean")

    def test_default_navigation_std(self):
        _test_default_navigation_signal_operations_over_many_axes(self, "std")

    def test_default_navigation_var(self):
        _test_default_navigation_signal_operations_over_many_axes(self, "var")

    def test_rebin(self):
        self.signal.estimate_poissonian_noise_variance()
        new_s = self.signal.rebin(scale=(2, 2, 1))
        var = new_s.metadata.Signal.Noise_properties.variance
        assert new_s.data.shape == (1, 2, 6)
        assert var.data.shape == (1, 2, 6)
        from hyperspy.misc.array_tools import rebin

        np.testing.assert_array_equal(
            rebin(self.signal.data, scale=(2, 2, 1)), var.data
        )
        np.testing.assert_array_equal(
            rebin(self.signal.data, scale=(2, 2, 1)), new_s.data
        )
        if self.signal._lazy:
            new_s = self.signal.rebin(scale=(2, 2, 1), rechunk=False)
            np.testing.assert_array_equal(
                rebin(self.signal.data, scale=(2, 2, 1)), var.data
            )
            np.testing.assert_array_equal(
                rebin(self.signal.data, scale=(2, 2, 1)), new_s.data
            )

    def test_rebin_no_variance(self):
        new_s = self.signal.rebin(scale=(2, 2, 1))
        with pytest.raises(AttributeError):
            _ = new_s.metadata.Signal.Noise_properties

    def test_rebin_const_variance(self):
        self.signal.metadata.set_item("Signal.Noise_properties.variance", 0.3)
        new_s = self.signal.rebin(scale=(2, 2, 1))
        assert new_s.metadata.Signal.Noise_properties.variance == 0.3

    def test_rebin_dtype(self):
        s = Signal1D(np.arange(1000).reshape(10, 10, 10))
        s.change_dtype(np.uint8)
        s2 = s.rebin(scale=(3, 3, 1), crop=False)
        assert s.sum() == s2.sum()

    def test_swap_axes_simple(self):
        s = self.signal
        assert s.swap_axes(0, 1).data.shape == (4, 2, 6)
        assert s.swap_axes(0, 2).axes_manager.shape == (6, 2, 4)
        if not s._lazy:
            assert not s.swap_axes(0, 2).data.flags["C_CONTIGUOUS"]
            assert s.swap_axes(0, 2, optimize=True).data.flags["C_CONTIGUOUS"]
        else:
            cks = s.data.chunks
            assert s.swap_axes(0, 1).data.chunks == (cks[1], cks[0], cks[2])
            # This data shape does not require rechunking
            assert s.swap_axes(0, 1, optimize=True).data.chunks == (
                cks[1],
                cks[0],
                cks[2],
            )

    def test_swap_axes_iteration(self):
        s = self.signal
        s = s.swap_axes(0, 2)
        assert s.axes_manager._getitem_tuple[:2] == (0, 0)
        s.axes_manager.indices = (2, 1)
        assert s.axes_manager._getitem_tuple[:2] == (1, 2)

    def test_get_navigation_signal_nav_dim0(self):
        s = self.signal
        s = s.transpose(signal_axes=3)
        ns = s._get_navigation_signal()
        assert ns.axes_manager.signal_dimension == 1
        assert ns.axes_manager.signal_size == 1
        assert ns.axes_manager.navigation_dimension == 0

    def test_get_navigation_signal_nav_dim1(self):
        s = self.signal
        s = s.transpose(signal_axes=2)
        ns = s._get_navigation_signal()
        assert ns.axes_manager.signal_shape == s.axes_manager.navigation_shape
        assert ns.axes_manager.navigation_dimension == 0

    def test_get_navigation_signal_nav_dim2(self):
        s = self.signal
        s = s.transpose(signal_axes=1)
        ns = s._get_navigation_signal()
        assert ns.axes_manager.signal_shape == s.axes_manager.navigation_shape
        assert ns.axes_manager.navigation_dimension == 0

    def test_get_navigation_signal_nav_dim3(self):
        s = self.signal
        s = s.transpose(signal_axes=0)
        ns = s._get_navigation_signal()
        assert ns.axes_manager.signal_shape == s.axes_manager.navigation_shape
        assert ns.axes_manager.navigation_dimension == 0

    def test_get_navigation_signal_wrong_data_shape(self):
        s = self.signal
        s = s.transpose(signal_axes=1)
        with pytest.raises(ValueError):
            s._get_navigation_signal(data=np.zeros((3, 2)))

    def test_get_navigation_signal_wrong_data_shape_dim0(self):
        s = self.signal
        s = s.transpose(signal_axes=3)
        with pytest.raises(ValueError):
            s._get_navigation_signal(data=np.asarray(0))

    def test_get_navigation_signal_given_data(self):
        s = self.signal
        s = s.transpose(signal_axes=1)
        data = np.zeros(s.axes_manager._navigation_shape_in_array)
        ns = s._get_navigation_signal(data=data)
        assert ns.data is data

    def test_get_signal_signal_nav_dim0(self):
        s = self.signal
        s = s.transpose(signal_axes=0)
        ns = s._get_signal_signal()
        assert ns.axes_manager.navigation_dimension == 0
        assert ns.axes_manager.navigation_size == 0
        assert ns.axes_manager.signal_dimension == 1

    def test_get_signal_signal_nav_dim1(self):
        s = self.signal
        s = s.transpose(signal_axes=1)
        ns = s._get_signal_signal()
        assert ns.axes_manager.signal_shape == s.axes_manager.signal_shape
        assert ns.axes_manager.navigation_dimension == 0

    def test_get_signal_signal_nav_dim2(self):
        s = self.signal
        s = s.transpose(signal_axes=2)
        s._assign_subclass()
        ns = s._get_signal_signal()
        assert ns.axes_manager.signal_shape == s.axes_manager.signal_shape
        assert ns.axes_manager.navigation_dimension == 0

    def test_get_signal_signal_nav_dim3(self):
        s = self.signal
        s = s.transpose(signal_axes=3)
        s._assign_subclass()
        ns = s._get_signal_signal()
        assert ns.axes_manager.signal_shape == s.axes_manager.signal_shape
        assert ns.axes_manager.navigation_dimension == 0

    def test_get_signal_signal_wrong_data_shape(self):
        s = self.signal
        s = s.transpose(signal_axes=1)
        with pytest.raises(ValueError):
            s._get_signal_signal(data=np.zeros((3, 2)))

    def test_get_signal_signal_wrong_data_shape_dim0(self):
        s = self.signal
        s = s.transpose(signal_axes=0)
        with pytest.raises(ValueError):
            s._get_signal_signal(data=np.asarray(0))

    def test_get_signal_signal_given_data(self):
        s = self.signal
        s = s.transpose(signal_axes=2)
        data = np.zeros(s.axes_manager._signal_shape_in_array)
        ns = s._get_signal_signal(data=data)
        assert ns.data is data

    def test_get_navigation_signal_dtype(self):
        s = self.signal
        assert s._get_navigation_signal().data.dtype.name == s.data.dtype.name

    def test_get_signal_signal_dtype(self):
        s = self.signal
        assert s._get_signal_signal().data.dtype.name == s.data.dtype.name

    def test_get_navigation_signal_given_dtype(self):
        s = self.signal
        assert s._get_navigation_signal(dtype="bool").data.dtype.name == "bool"

    def test_get_signal_signal_given_dtype(self):
        s = self.signal
        assert s._get_signal_signal(dtype="bool").data.dtype.name == "bool"
