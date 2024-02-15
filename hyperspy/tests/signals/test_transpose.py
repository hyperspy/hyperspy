# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
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

import dask.array as da
import numpy as np
import pytest

from hyperspy.decorators import lazifyTestClass
from hyperspy.signals import BaseSignal, Signal2D


@lazifyTestClass
class TestTranspose:
    def setup_method(self, method):
        s = BaseSignal(np.random.rand(1, 2, 3, 4, 5, 6))
        for ax, name in zip(s.axes_manager._axes, "abcdef"):
            ax.name = name
        # just to make sure in case default changes
        assert s.axes_manager.signal_dimension == 6
        s.estimate_poissonian_noise_variance()
        self.s = s

    def test_signal_int_transpose(self):
        t = self.s.transpose(signal_axes=2)
        var = t.metadata.Signal.Noise_properties.variance
        assert t.axes_manager.signal_shape == (6, 5)
        assert var.axes_manager.signal_shape == (6, 5)
        assert [ax.name for ax in t.axes_manager.signal_axes] == ["f", "e"]
        assert isinstance(t, Signal2D)
        assert isinstance(t.metadata.Signal.Noise_properties.variance, Signal2D)

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

    def test_too_many_signal_axes(self):
        with pytest.raises(ValueError):
            self.s.transpose(signal_axes=10)

    def test_navigation_int_transpose(self):
        t = self.s.transpose(navigation_axes=2)
        var = t.metadata.Signal.Noise_properties.variance
        assert t.axes_manager.navigation_shape == (2, 1)
        assert var.axes_manager.navigation_shape == (2, 1)
        assert [ax.name for ax in t.axes_manager.navigation_axes] == ["b", "a"]

    def test_navigation_iterable_int_transpose(self):
        t = self.s.transpose(navigation_axes=[0, 5, 4])
        var = t.metadata.Signal.Noise_properties.variance
        assert t.axes_manager.navigation_shape == (6, 1, 2)
        assert var.axes_manager.navigation_shape == (6, 1, 2)
        assert [ax.name for ax in t.axes_manager.navigation_axes] == ["f", "a", "b"]

    def test_navigation_iterable_names_transpose(self):
        t = self.s.transpose(navigation_axes=["f", "a", "b"])
        var = t.metadata.Signal.Noise_properties.variance
        assert var.axes_manager.navigation_shape == (6, 1, 2)
        assert t.axes_manager.navigation_shape == (6, 1, 2)
        assert [ax.name for ax in t.axes_manager.navigation_axes] == ["f", "a", "b"]

    def test_navigation_iterable_axes_transpose(self):
        t = self.s.transpose(navigation_axes=self.s.axes_manager.signal_axes[:2])
        var = t.metadata.Signal.Noise_properties.variance
        assert t.axes_manager.navigation_shape == (6, 5)
        assert var.axes_manager.navigation_shape == (6, 5)
        assert [ax.name for ax in t.axes_manager.navigation_axes] == ["f", "e"]

    def test_navigation_one_name(self):
        with pytest.raises(ValueError):
            self.s.transpose(navigation_axes="a")

    def test_too_many_navigation_axes(self):
        with pytest.raises(ValueError):
            self.s.transpose(navigation_axes=10)

    def test_transpose_shortcut(self):
        s = self.s.transpose(signal_axes=2)
        t = s.T
        assert t.axes_manager.navigation_shape == (6, 5)
        assert [ax.name for ax in t.axes_manager.navigation_axes] == ["f", "e"]

    def test_optimize(self):
        if self.s._lazy:
            pytest.skip(
                "LazySignal optimization is tested in test_lazy_tranpose_rechunk"
            )
        t = self.s.transpose(signal_axes=["f", "a", "b"], optimize=False)
        assert t.data.base is self.s.data

        t = self.s.transpose(signal_axes=["f", "a", "b"], optimize=True)
        assert t.data.base is not self.s.data


def test_lazy_transpose_rechunks():
    ar = da.ones((50, 50, 256, 256), chunks=(5, 5, 256, 256))
    s = Signal2D(ar).as_lazy()
    s1 = s.T  # By default it does not rechunk
    cks = s.data.chunks
    assert s1.data.chunks == (cks[2], cks[3], cks[0], cks[1])
    s2 = s.transpose(optimize=True)
    assert s2.data.chunks != s1.data.chunks


def test_transpose_nav0_sig0():
    s = BaseSignal([0.0])
    assert s.axes_manager.signal_dimension == 0
    assert s.axes_manager.navigation_dimension == 0
    assert s.axes_manager.signal_axes[0].size == 1
    assert s.axes_manager.navigation_axes == ()

    s2 = s.T
    assert s2.axes_manager.signal_dimension == 0
    assert s2.axes_manager.navigation_dimension == 1
    assert s2.axes_manager.signal_axes == ()
    assert s2.axes_manager.navigation_axes[0].size == 1
