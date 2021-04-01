# -*- coding: utf-8 -*-
# Copyright 2007-2021 The HyperSpy developers
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

import hyperspy.api as hs


def test_spectrum_binned_default():
    s = hs.signals.Signal1D([0])
    assert not s.metadata.Signal.binned


def test_image_binned_default():
    s = hs.signals.Signal2D(np.zeros((2, 2)))
    assert not s.metadata.Signal.binned


def test_dielectric_function_binned_default():
    s = hs.signals.DielectricFunction([0])
    assert not s.metadata.Signal.binned


def test_signal_binned_default():
    s = hs.signals.BaseSignal([0])
    assert not s.metadata.Signal.binned


def test_eels_spectrum_binned_default():
    s = hs.signals.EELSSpectrum([0])
    assert s.metadata.Signal.binned


def test_eds_tem_binned_default():
    s = hs.signals.EDSTEMSpectrum([0])
    assert s.metadata.Signal.binned


def test_eds_sem_binned_default():
    s = hs.signals.EDSSEMSpectrum([0])
    assert s.metadata.Signal.binned


class TestModelBinned:

    def setup_method(self, method):
        s = hs.signals.Signal1D([1])
        s.axes_manager[0].scale = 0.1
        m = s.create_model()
        m.append(hs.model.components1D.Offset())
        m[0].offset.value = 1
        self.m = m

    def test_unbinned(self):
        self.m.signal.metadata.Signal.binned = False
        assert self.m() == 1

    def test_binned(self):
        self.m.signal.metadata.Signal.binned = True
        assert self.m() == 0.1
