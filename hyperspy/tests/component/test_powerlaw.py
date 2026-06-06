# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
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

import itertools

import numpy as np
import pytest

import hyperspy.api as hs

TRUE_FALSE_2_TUPLE = [p for p in itertools.product((True, False), repeat=2)]
TRUE_FALSE_3_TUPLE = [p for p in itertools.product((True, False), repeat=3)]


def test_function():
    g = hs.model.components1D.PowerLaw()
    g.A.value = 1
    g.r.value = 2
    g.origin.value = 3
    assert g.function(2) == 1
    assert g.function(1) == 0.25


def test_linear_override():
    g = hs.model.components1D.PowerLaw()
    for para in g.parameters:
        if para is g.A:
            assert para._linear
        else:
            assert not para._linear


@pytest.mark.parametrize(("lazy"), (True, False))
@pytest.mark.parametrize(("only_current", "binned", "split"), TRUE_FALSE_3_TUPLE)
def test_estimate_parameters_binned(only_current, binned, split, lazy):
    s = hs.signals.Signal1D(np.empty((100,)))
    s.axes_manager.signal_axes[0].is_binned = binned
    axis = s.axes_manager.signal_axes[0]
    axis.scale = 0.02
    axis.offset = 1
    g1 = hs.model.components1D.PowerLaw(50015.156, 1.2)
    s.data = g1.function(axis.axis)
    if lazy:
        s = s.as_lazy()
    g2 = hs.model.components1D.PowerLaw()
    factor = axis.scale if binned else 1
    if split:
        intervals = [
            (axis.low_value, (axis.low_value + axis.high_value) / 2),
            ((axis.low_value + axis.high_value) / 2, axis.high_value),
        ]
        assert g2.estimate_parameters(s, intervals=intervals, only_current=only_current)
    else:
        assert g2.estimate_parameters(
            s, intervals=[(axis.low_value, axis.high_value)], only_current=only_current
        )
    assert g2._axes_manager[-1].is_binned == binned
    np.testing.assert_allclose(g1.A.value, g2.A.value * factor, rtol=0.05)
    np.testing.assert_allclose(g1.r.value, g2.r.value, rtol=0.05)


@pytest.mark.parametrize(("lazy"), (True, False))
def test_estimate_parameters_intervals_with_roi(lazy):
    s = hs.signals.Signal1D(np.empty((100,)))
    axis = s.axes_manager.signal_axes[0]
    axis.scale = 0.02
    axis.offset = 1
    g1 = hs.model.components1D.PowerLaw(50015.156, 1.2)
    s.data = g1.function(axis.axis)
    if lazy:
        s = s.as_lazy()
    g2 = hs.model.components1D.PowerLaw()
    roi1 = hs.roi.SpanROI(axis.low_value, (axis.low_value + axis.high_value) / 2)
    roi2 = hs.roi.SpanROI((axis.low_value + axis.high_value) / 2, axis.high_value)
    assert g2.estimate_parameters(s, intervals=[roi1, roi2], only_current=True)
    np.testing.assert_allclose(g1.A.value, g2.A.value, rtol=0.05)
    np.testing.assert_allclose(g1.r.value, g2.r.value, rtol=0.05)


def test_estimate_parameters_intervals_validation():
    s = hs.signals.Signal1D(np.empty((100,)))
    axis = s.axes_manager.signal_axes[0]
    axis.scale = 0.02
    axis.offset = 1
    g = hs.model.components1D.PowerLaw()
    with pytest.raises(ValueError, match="requires exactly 2 intervals"):
        g.estimate_parameters(s, intervals=[(1, 2), (3, 4), (5, 6)])
    with pytest.raises(ValueError, match="Invalid interval format"):
        g.estimate_parameters(s, intervals=["invalid"])


@pytest.mark.parametrize(("lazy"), (True, False))
@pytest.mark.parametrize(("binned"), (True, False))
def test_function_nd(binned, lazy):
    s = hs.signals.Signal1D(np.empty((100,)))
    axis = s.axes_manager.signal_axes[0]
    axis.scale = 0.02
    axis.offset = 1
    g1 = hs.model.components1D.PowerLaw(50015.156, 1.2)
    s.data = g1.function(axis.axis)
    s.axes_manager.signal_axes[0].is_binned = binned
    s2 = hs.stack([s] * 2)
    if lazy:
        s = s.as_lazy()
    g2 = hs.model.components1D.PowerLaw()
    factor = axis.scale if binned else 1
    g2.estimate_parameters(s2, axis.low_value, axis.high_value, only_current=False)
    assert g2._axes_manager[-1].is_binned == binned
    np.testing.assert_allclose(g2.function_nd(axis.axis) * factor, s2.data, rtol=0.05)


def test_estimate_parameters_intervals_not_list():
    s = hs.signals.Signal1D(np.empty((100,)))
    g = hs.model.components1D.PowerLaw()
    with pytest.raises(ValueError, match="must be a list of tuples or SpanROI"):
        g.estimate_parameters(s, intervals="not_a_list")


def test_estimate_parameters_intervals_empty_tuple():
    s = hs.signals.Signal1D(np.empty((100,)))
    g = hs.model.components1D.PowerLaw()
    with pytest.raises(ValueError, match="requires exactly 2 intervals"):
        g.estimate_parameters(s, intervals=())


def test_estimate_parameters_intervals_bare_tuple():
    s = hs.signals.Signal1D(np.empty((100,)))
    axis = s.axes_manager.signal_axes[0]
    axis.scale = 0.02
    axis.offset = 1
    g1 = hs.model.components1D.PowerLaw(50015.156, 1.2)
    s.data = g1.function(axis.axis)
    g2 = hs.model.components1D.PowerLaw()
    assert g2.estimate_parameters(
        s, intervals=(axis.low_value, axis.high_value), only_current=True
    )
    np.testing.assert_allclose(g1.A.value, g2.A.value, rtol=0.05)
    np.testing.assert_allclose(g1.r.value, g2.r.value, rtol=0.05)


def test_estimate_parameters_intervals_tuple_of_spanrois():
    s = hs.signals.Signal1D(np.empty((100,)))
    axis = s.axes_manager.signal_axes[0]
    axis.scale = 0.02
    axis.offset = 1
    g1 = hs.model.components1D.PowerLaw(50015.156, 1.2)
    s.data = g1.function(axis.axis)
    g2 = hs.model.components1D.PowerLaw()
    roi1 = hs.roi.SpanROI(axis.low_value, (axis.low_value + axis.high_value) / 2)
    roi2 = hs.roi.SpanROI((axis.low_value + axis.high_value) / 2, axis.high_value)
    assert g2.estimate_parameters(s, intervals=(roi1, roi2), only_current=True)
    np.testing.assert_allclose(g1.A.value, g2.A.value, rtol=0.05)
    np.testing.assert_allclose(g1.r.value, g2.r.value, rtol=0.05)


def test_estimate_parameters_x1_without_x2():
    s = hs.signals.Signal1D(np.empty((100,)))
    g = hs.model.components1D.PowerLaw()
    with pytest.raises(ValueError, match="x2 must be provided when using x1"):
        g.estimate_parameters(s, x1=2.0)


def test_estimate_parameters_default_range():
    s = hs.signals.Signal1D(np.empty((100,)))
    axis = s.axes_manager.signal_axes[0]
    axis.scale = 0.02
    axis.offset = 1
    g1 = hs.model.components1D.PowerLaw(50015.156, 1.2)
    s.data = g1.function(axis.axis)
    g2 = hs.model.components1D.PowerLaw()
    assert g2.estimate_parameters(s)  # no args — uses full axis range
    np.testing.assert_allclose(g1.A.value, g2.A.value, rtol=0.05)
    np.testing.assert_allclose(g1.r.value, g2.r.value, rtol=0.05)


def test_estimate_parameters_x2_le_x1():
    s = hs.signals.Signal1D(np.empty((100,)))
    g = hs.model.components1D.PowerLaw()
    with pytest.raises(ValueError, match="x2 must be greater than x1"):
        g.estimate_parameters(s, x1=5.0, x2=5.0)


def test_estimate_parameters_two_intervals_not_adjacent():
    s = hs.signals.Signal1D(np.empty((100,)))
    g = hs.model.components1D.PowerLaw()
    with pytest.raises(ValueError, match="x3 must be greater than x2"):
        g.estimate_parameters(s, intervals=[(4.0, 6.0), (3.0, 5.0)])


def test_estimate_parameters_two_intervals_x4_le_x3():
    s = hs.signals.Signal1D(np.empty((100,)))
    axis = s.axes_manager.signal_axes[0]
    axis.scale = 0.02
    axis.offset = 1
    g = hs.model.components1D.PowerLaw()
    with pytest.raises(ValueError, match="x4 must be greater than x3"):
        g.estimate_parameters(s, intervals=[(1.0, 1.5), (1.5, 1.5)])


def test_estimate_parameters_two_intervals_too_narrow():
    s = hs.signals.Signal1D(np.empty((100,)))
    axis = s.axes_manager.signal_axes[0]
    axis.scale = 0.02
    axis.offset = 1
    g = hs.model.components1D.PowerLaw()
    with pytest.raises(ValueError, match="at least 2 points"):
        g.estimate_parameters(s, intervals=[(1.001, 1.003), (2.001, 2.003)])


def test_estimate_parameters_out_true():
    # ALL-DIFFERENT dims: (7, 11, 50) → nav=(7,11), sig=(50,)
    data = np.empty((7, 11, 50))
    s = hs.signals.Signal1D(data)
    axis = s.axes_manager.signal_axes[0]
    axis.scale = 0.02
    axis.offset = 1
    g1 = hs.model.components1D.PowerLaw(50015.156, 1.2)
    for i in range(7):
        for j in range(11):
            s.data[i, j, :] = g1.function(axis.axis)
    g2 = hs.model.components1D.PowerLaw()
    A, r = g2.estimate_parameters(
        s, axis.low_value, axis.high_value, only_current=False, out=True
    )
    assert A.shape == (7, 11)
    assert r.shape == (7, 11)
    np.testing.assert_allclose(A, g1.A.value, rtol=0.05)
    np.testing.assert_allclose(r, g1.r.value, rtol=0.05)


def test_estimate_parameters_intervals_tuple_of_tuples():
    s = hs.signals.Signal1D(np.empty((100,)))
    axis = s.axes_manager.signal_axes[0]
    axis.scale = 0.02
    axis.offset = 1
    g1 = hs.model.components1D.PowerLaw(50015.156, 1.2)
    s.data = g1.function(axis.axis)
    g2 = hs.model.components1D.PowerLaw()
    left = axis.low_value
    mid = (axis.low_value + axis.high_value) / 2
    right = axis.high_value
    assert g2.estimate_parameters(
        s, intervals=((left, mid), (mid, right)), only_current=True
    )
    np.testing.assert_allclose(g1.A.value, g2.A.value, rtol=0.05)
    np.testing.assert_allclose(g1.r.value, g2.r.value, rtol=0.05)


def test_estimate_parameters_single_interval_index_edge():
    s = hs.signals.Signal1D(np.empty((201,)))
    axis = s.axes_manager.signal_axes[0]
    axis.scale = 0.001
    axis.offset = 1.0
    g_ref = hs.model.components1D.PowerLaw(50015.156, 1.2)
    s.data = g_ref.function(axis.axis)

    # Even-sum indices (i1=0, i4=2): the odd/even adjustment is skipped,
    # covering the False branch at line 200→202.
    g1 = hs.model.components1D.PowerLaw()
    g1.estimate_parameters(s, x1=1.0003, x2=1.0017, only_current=True)

    # Consecutive indices (i1=0, i4=1): i4-=1 then i4==i1 triggers
    # i4+=2, covering line 203.
    g2 = hs.model.components1D.PowerLaw()
    g2.estimate_parameters(s, x1=1.0005, x2=1.0015, only_current=True)
