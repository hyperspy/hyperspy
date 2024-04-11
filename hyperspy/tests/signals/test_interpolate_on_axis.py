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


import gc
from copy import deepcopy

import numpy as np
import pytest

from hyperspy import signals
from hyperspy.axes import DataAxis, FunctionalDataAxis, UniformDataAxis


def _assert_equal_dimensions(s1, s2):
    assert s1.axes_manager.navigation_dimension == s2.axes_manager.navigation_dimension
    assert s1.axes_manager.signal_dimension == s2.axes_manager.signal_dimension


def _assert_axes_equality(ax1, ax2, type):
    _type = type.lower().strip()
    assert _type in ["data", "uniform", "functional"]
    assert ax1.name == ax2.name
    if _type == "uniform":
        np.testing.assert_almost_equal(ax1.offset, ax2.offset)
        np.testing.assert_almost_equal(ax1.scale, ax2.scale)
        assert ax1.size == ax2.size
    elif _type == "data":
        np.testing.assert_almost_equal(ax1.axis, ax2.axis)
    elif _type == "functional":
        assert ax1.x.size == ax2.x.size
        assert ax1.x.offset == ax2.x.offset
        assert ax1.x.scale == ax2.x.scale
        assert ax1.size == ax2.size
        np.testing.assert_almost_equal(ax1.axis, ax2.axis)
    else:
        raise ValueError(
            f"Invalid argument {type}, only data, functional and uniform are accepted"
        )


class TestInterpolateAxis1D:
    @classmethod
    def setup_class(cls):
        d = np.arange(0, 100)
        x_initial = {"offset": 0, "scale": 1, "size": 100, "name": "X"}
        cls.s0 = signals.Signal1D(d, axes=[x_initial])

    @classmethod
    def teardown_class(cls):
        del cls.s0
        gc.collect()

    def test_interpolate_uniform_axis(self):
        x_uniform = UniformDataAxis(
            offset=10, scale=5, size=10, navigate=False, name="XU"
        )
        s1 = self.s0.interpolate_on_axis(x_uniform, 0, inplace=False)
        _assert_equal_dimensions(s1, self.s0)
        _assert_axes_equality(s1.axes_manager[0], x_uniform, "uniform")
        np.testing.assert_almost_equal(s1.data, np.arange(10, 60, 5))

        # test inplace=True and default axes_manager_index
        s2 = self.s0.deepcopy()
        s2.interpolate_on_axis(x_uniform, inplace=True)
        _assert_equal_dimensions(s2, self.s0)
        _assert_axes_equality(s2.axes_manager[0], x_uniform, "uniform")
        np.testing.assert_almost_equal(s2.data, s1.data)

        # test index by string
        s3 = self.s0.interpolate_on_axis(x_uniform, "X", inplace=False)
        _assert_equal_dimensions(s3, self.s0)
        _assert_axes_equality(s3.axes_manager[0], x_uniform, "uniform")
        np.testing.assert_almost_equal(s3.data, np.arange(10, 60, 5))

    def test_interpolate_functional_axis(self):
        x_functional = FunctionalDataAxis(
            expression="x^2", size=10, navigate=False, name="XF"
        )
        s3 = self.s0.interpolate_on_axis(x_functional, 0, inplace=False)
        _assert_equal_dimensions(s3, self.s0)
        _assert_axes_equality(s3.axes_manager[0], x_functional, "functional")
        np.testing.assert_almost_equal(s3.data, np.arange(0, 10) ** 2)

    def test_interpolate_data_axis(self):
        x_data = DataAxis(axis=(np.arange(0, 10) ** 2), navigate=False, name="XD")
        s2 = self.s0.interpolate_on_axis(x_data, 0, inplace=False)
        _assert_equal_dimensions(s2, self.s0)
        _assert_axes_equality(s2.axes_manager[0], x_data, "data")
        np.testing.assert_almost_equal(s2.data, np.arange(0, 10) ** 2)

    def test_extrapolation(self):
        x_new = UniformDataAxis(
            offset=30, scale=30, size=10, navigate=False, name="X1", units="µm"
        )
        s2 = self.s0.interpolate_on_axis(x_new, inplace=False)
        _assert_equal_dimensions(s2, self.s0)
        _assert_axes_equality(s2.axes_manager[0], x_new, "uniform")
        np.testing.assert_almost_equal(s2.data, np.arange(30, 330, 30))

    def test_interpolate_error_navigate(self):
        x_uniform = UniformDataAxis(
            offset=10, scale=5, size=10, navigate=True, name="XU"
        )
        with pytest.raises(ValueError):
            self.s0.interpolate_on_axis(x_uniform, 0)


def test_interpolate_on_axis_2D():
    d = np.arange(0, 140).reshape(7, 20)
    x_initial = {"offset": 0, "scale": 1, "size": 20, "name": "X"}
    y_initial = {"offset": 5, "scale": 0.5, "size": 7, "name": "Y"}
    s0 = signals.Signal1D(d, axes=[y_initial, x_initial])

    y_new = UniformDataAxis(offset=5.5, scale=1, size=3, navigate=True, name="Y1")
    x_new = UniformDataAxis(offset=11, scale=2, size=3, navigate=False, name="X1")

    # switch y-axis
    s1 = s0.interpolate_on_axis(y_new, 0, inplace=False)
    _assert_equal_dimensions(s1, s0)
    _assert_axes_equality(s1.axes_manager[0], y_new, "uniform")
    _assert_axes_equality(s1.axes_manager[1], s0.axes_manager[1], "uniform")
    np.testing.assert_almost_equal(
        s1.data,
        np.stack((np.arange(20, 40), np.arange(60, 80), np.arange(100, 120))),
    )

    # switch also x-axis inplace for s1, such that both axes are now replaced
    s1.interpolate_on_axis(x_new, 1, inplace=True)
    _assert_equal_dimensions(s1, s0)
    _assert_axes_equality(s1.axes_manager[0], y_new, "uniform")
    _assert_axes_equality(s1.axes_manager[1], x_new, "uniform")
    np.testing.assert_almost_equal(
        s1.data,
        np.stack((np.arange(31, 37, 2), np.arange(71, 77, 2), np.arange(111, 117, 2))),
    )

    # switch both axes inplace in different order and compare with previous result
    s0.interpolate_on_axis(x_new, 1, inplace=True)
    s0.interpolate_on_axis(y_new, 0, inplace=True)
    _assert_equal_dimensions(s1, s0)
    _assert_axes_equality(s0.axes_manager[0], y_new, "uniform")
    _assert_axes_equality(s0.axes_manager[1], x_new, "uniform")
    np.testing.assert_almost_equal(s0.data, s1.data)


class TestInterpolateAxis3D:
    @classmethod
    def setup_class(cls):
        x_initial = {"offset": 0, "scale": 1, "size": 10, "name": "X"}
        y_initial = {"offset": 0, "scale": 1, "size": 10, "name": "Y"}
        e_initial = {"offset": 400, "scale": 1, "size": 100, "name": "E"}
        d = np.arange(0, 10000).reshape(10, 10, 100)
        cls.s0 = signals.Signal1D(d, axes=[y_initial, x_initial, e_initial])

        cls.y_new = UniformDataAxis(offset=3, scale=1, size=3, navigate=True, name="Y1")
        cls.x_new = UniformDataAxis(offset=3, scale=1, size=3, navigate=True, name="X1")
        cls.e_new = UniformDataAxis(
            offset=420, scale=1, size=50, navigate=False, name="E1"
        )

    @classmethod
    def teardown_class(cls):
        del cls.s0
        gc.collect()

    def test_interpolate_1s2n(self):
        s1 = self.s0.interpolate_on_axis(self.y_new, 1, inplace=False)
        _assert_equal_dimensions(s1, self.s0)
        _assert_axes_equality(s1.axes_manager[1], self.y_new, "uniform")
        _assert_axes_equality(s1.axes_manager[0], self.s0.axes_manager[0], "uniform")
        _assert_axes_equality(s1.axes_manager[2], self.s0.axes_manager[2], "uniform")
        s1.interpolate_on_axis(self.x_new, 0, inplace=True)
        _assert_equal_dimensions(s1, self.s0)
        _assert_axes_equality(s1.axes_manager[0], self.x_new, "uniform")
        _assert_axes_equality(s1.axes_manager[1], self.y_new, "uniform")
        _assert_axes_equality(s1.axes_manager[2], self.s0.axes_manager[2], "uniform")
        s1.interpolate_on_axis(self.e_new, 2, inplace=True)
        _assert_equal_dimensions(s1, self.s0)
        _assert_axes_equality(s1.axes_manager[2], self.e_new, "uniform")
        _assert_axes_equality(s1.axes_manager[1], self.y_new, "uniform")
        _assert_axes_equality(s1.axes_manager[0], self.x_new, "uniform")
        np.testing.assert_almost_equal(s1.inav[0, 0].data, np.arange(3320, 3370, 1))
        np.testing.assert_almost_equal(s1.inav[1, 0].data, np.arange(3420, 3470, 1))
        np.testing.assert_almost_equal(s1.inav[0, 1].data, np.arange(4320, 4370, 1))

    def test_interpolate_2s1n(self):
        s2 = self.s0.T
        e_new = deepcopy(self.e_new)
        x_new = deepcopy(self.x_new)
        y_new = deepcopy(self.y_new)
        e_new.navigate = True
        x_new.navigate = False
        y_new.navigate = False
        s3 = s2.interpolate_on_axis(e_new, 0, inplace=False)
        _assert_equal_dimensions(s3, s2)
        _assert_axes_equality(s3.axes_manager[0], e_new, "uniform")
        _assert_axes_equality(s3.axes_manager[1], s2.axes_manager[1], "uniform")
        _assert_axes_equality(s3.axes_manager[2], s2.axes_manager[2], "uniform")
        s3.interpolate_on_axis(x_new, 1, inplace=True)
        _assert_equal_dimensions(s3, s2)
        _assert_axes_equality(s3.axes_manager[1], x_new, "uniform")
        _assert_axes_equality(s3.axes_manager[0], e_new, "uniform")
        _assert_axes_equality(s3.axes_manager[2], s2.axes_manager[2], "uniform")
        s3.interpolate_on_axis(y_new, 2, inplace=True)
        _assert_equal_dimensions(s3, s2)
        _assert_axes_equality(s3.axes_manager[2], y_new, "uniform")
        _assert_axes_equality(s3.axes_manager[1], x_new, "uniform")
        _assert_axes_equality(s3.axes_manager[0], e_new, "uniform")
        np.testing.assert_almost_equal(s3.isig[0, 0].data, np.arange(3320, 3370, 1))
        np.testing.assert_almost_equal(s3.isig[1, 0].data, np.arange(3420, 3470, 1))
        np.testing.assert_almost_equal(s3.isig[0, 1].data, np.arange(4320, 4370, 1))


@pytest.mark.parametrize("dim", [1, 2, 3, 4, 5])
def test_interpolate_on_axis_random_switch(dim):
    seed = np.random.randint(0, 1000000)
    rng = np.random.default_rng(seed)
    nav_dim = rng.integers(0, dim + 1)
    axes_list = []
    data_sizes = np.ones(dim, dtype="int")
    for i in range(dim):
        size = 3 + i
        data_sizes[i] = size
        ax_dict = {"name": f"{i}", "size": size, "offset": 0, "scale": 1}
        if i < nav_dim:
            ax_dict["navigate"] = True
        else:
            ax_dict["navigate"] = False
        axes_list.append(ax_dict)
    data = rng.random(int(data_sizes.prod())).reshape(data_sizes)
    s = signals.BaseSignal(data, axes=axes_list)
    switch_idx = rng.integers(0, dim)
    navigate = s.axes_manager[switch_idx].navigate
    new_ax = UniformDataAxis(offset=1, scale=2, size=2, navigate=navigate, name="NEW")
    try:
        s.interpolate_on_axis(new_ax, switch_idx, inplace=True)
        _assert_axes_equality(s.axes_manager[switch_idx], new_ax, "uniform")
    except Exception as e:
        print(
            f"{e}\n\n seed: {seed}, nav_dim: {nav_dim}, dim: {dim}, switch_idx: {switch_idx}"
        )
