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

import numpy as np

import hyperspy.api as hs


def test_spectrum_binned_default():
    s = hs.signals.Signal1D([0])
    assert not s.axes_manager[-1].is_binned


def test_image_binned_default():
    s = hs.signals.Signal2D(np.zeros((2, 2)))
    assert not s.axes_manager[-1].is_binned


def test_signal_binned_default():
    s = hs.signals.BaseSignal([0])
    assert not s.axes_manager[-1].is_binned


class TestModelBinned:
    def setup_method(self, method):
        s = hs.signals.Signal1D([1])
        s.axes_manager[0].scale = 0.1
        m = s.create_model()
        m.append(hs.model.components1D.Offset())
        m[0].offset.value = 1
        self.m = m

    def test_unbinned(self):
        self.m.signal.axes_manager[-1].is_binned = False
        assert self.m._get_current_data() == 1

    def test_binned(self):
        self.m.signal.axes_manager[-1].is_binned = True
        assert self.m._get_current_data() == 0.1
