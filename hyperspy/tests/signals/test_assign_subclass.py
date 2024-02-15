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

import logging
from collections import namedtuple

import numpy as np
import pytest

import hyperspy.api as hs
from hyperspy import _lazy_signals
from hyperspy.decorators import lazifyTestClass
from hyperspy.io import assign_signal_subclass

testcase = namedtuple("testcase", ["dtype", "sig_dim", "sig_type", "cls"])

subclass_cases = (
    testcase("float", 1000, "", "BaseSignal"),
    testcase("float", 1, "", "Signal1D"),
    testcase("float", 2, "", "Signal2D"),
    testcase("complex", 1000, "", "ComplexSignal"),
    testcase("complex", 1, "", "ComplexSignal1D"),
    testcase("complex", 2, "", "ComplexSignal2D"),
    testcase("float", 1000, "DefinitelyNotAHyperSpySignal", "BaseSignal"),
    testcase("float", 1, "DefinitelyNotAHyperSpySignal", "Signal1D"),
    testcase("complex", 1000, "DefinitelyNotAHyperSpySignal", "ComplexSignal"),
    testcase("float", 1, "DefinitelyNotAHyperSpySignal", "Signal1D"),
)


def test_assignment_class(caplog):
    for case in subclass_cases:
        with caplog.at_level(logging.WARNING):
            new_subclass = assign_signal_subclass(
                dtype=np.dtype(case.dtype),
                signal_dimension=case.sig_dim,
                signal_type=case.sig_type,
                lazy=False,
            )

        assert new_subclass is getattr(hs.signals, case.cls)

        warn_msg = "not understood. See `hs.print_known_signal_types()` for a list"
        if case.sig_type == "DefinitelyNotAHyperSpySignal":
            assert warn_msg in caplog.text
        else:
            assert warn_msg not in caplog.text

        lazyclass = "Lazy" + case.cls if case.cls != "BaseSignal" else "LazySignal"
        new_subclass = assign_signal_subclass(
            dtype=np.dtype(case.dtype),
            signal_dimension=case.sig_dim,
            signal_type=case.sig_type,
            lazy=True,
        )

        assert new_subclass is getattr(_lazy_signals, lazyclass)


def test_id_set_signal_type():
    s = hs.signals.BaseSignal(np.zeros((3, 3)))
    id_events = id(s.events)
    id_metadata = id(s.metadata)
    id_om = id(s.original_metadata)
    s.set_signal_type()
    assert id_events == id(s.events)
    assert id_metadata == id(s.metadata)
    assert id_om == id(s.original_metadata)


@lazifyTestClass
class TestToBaseSignalScalar:
    def setup_method(self, method):
        self.s = hs.signals.Signal1D(np.array([0]))

    def test_simple(self):
        self.s._assign_subclass()
        assert isinstance(self.s, hs.signals.BaseSignal)
        assert self.s.axes_manager.signal_dimension == 0
        assert self.s.axes_manager.signal_shape == (1,)
        if self.s._lazy:
            assert isinstance(self.s, _lazy_signals.LazySignal)


class TestConvertBaseSignal:
    def setup_method(self, method):
        self.s = hs.signals.BaseSignal(np.zeros((3, 3)))

    def test_base_to_lazy(self):
        assert not self.s._lazy
        self.s._lazy = True
        self.s._assign_subclass()
        assert isinstance(self.s, _lazy_signals.LazySignal)
        assert self.s._lazy

    def test_base_to_1d(self):
        self.s.axes_manager._set_signal_dimension(1)
        self.s._assign_subclass()
        assert isinstance(self.s, hs.signals.Signal1D)
        self.s.metadata.Signal.record_by = ""
        self.s._assign_subclass()
        assert isinstance(self.s, hs.signals.BaseSignal)

    def test_base_to_2d(self):
        self.s.axes_manager._set_signal_dimension(2)
        self.s._assign_subclass()
        assert isinstance(self.s, hs.signals.Signal2D)

    def test_base_to_complex(self):
        self.s.change_dtype(complex)
        assert isinstance(self.s, hs.signals.ComplexSignal)
        # Going back from ComplexSignal to BaseSignal is not possible!
        # If real data is required use `real`, `imag`, `amplitude` or `phase`!


class TestConvertComplexSignal:
    def setup_method(self, method):
        self.s = hs.signals.ComplexSignal(np.zeros((3, 3)))

    def test_complex_to_complex1d(self):
        self.s.axes_manager._set_signal_dimension(1)
        self.s._assign_subclass()
        assert isinstance(self.s, hs.signals.ComplexSignal1D)

    def test_complex_to_complex2d(self):
        self.s.axes_manager._set_signal_dimension(2)
        self.s._assign_subclass()
        assert isinstance(self.s, hs.signals.ComplexSignal2D)


def test_create_lazy_signal():
    # Check that this syntax is working
    _ = hs.signals.BaseSignal([0, 1, 2], attributes={"_lazy": True})


@pytest.mark.parametrize("signal_dim", (0, 1, 2, 4))
def test_setting_signal_dimension(signal_dim):
    s = hs.signals.BaseSignal(np.random.random(size=(10, 20, 30, 40)))
    nav_dim = s.data.ndim - signal_dim

    s.axes_manager._set_signal_dimension(signal_dim)

    if signal_dim == 0:
        expected_sig_shape = ()
        expected_nav_shape = s.data.shape[::-1]
    else:
        expected_sig_shape = s.data.shape[-signal_dim:][::-1]
        expected_nav_shape = s.data.shape[:-signal_dim][::-1]

    def expected_size(expected_shape):
        return np.prod(expected_shape) if expected_shape else 0

    assert s.axes_manager.signal_dimension == signal_dim
    assert s.axes_manager.navigation_dimension == nav_dim
    assert len(s.axes_manager.signal_axes) == signal_dim
    assert len(s.axes_manager.navigation_axes) == nav_dim
    assert s.axes_manager.signal_shape == expected_sig_shape
    assert s.axes_manager.navigation_shape == expected_nav_shape
    assert s.axes_manager.signal_size == expected_size(expected_sig_shape)
    assert s.axes_manager.navigation_size == expected_size(expected_nav_shape)

    s._assign_subclass()
    class_mapping = {
        0: hs.signals.BaseSignal,
        1: hs.signals.Signal1D,
        2: hs.signals.Signal2D,
        4: hs.signals.BaseSignal,
    }
    assert isinstance(s, class_mapping[signal_dim])
