# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
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

from hyperspy import signals
from hyperspy import components1d
from hyperspy.decorators import lazifyTestClass


@lazifyTestClass
class TestRemoveBackground1DGaussian:

    def setup_method(self, method):
        gaussian = components1d.Gaussian()
        gaussian.A.value = 10
        gaussian.centre.value = 10
        gaussian.sigma.value = 1
        self.signal = signals.Signal1D(
            gaussian.function(np.arange(0, 20, 0.01)))
        self.signal.axes_manager[0].scale = 0.01
        self.signal.metadata.Signal.binned = False

    def test_background_remove_gaussian(self):
        s1 = self.signal.remove_background(
            signal_range=(None, None),
            background_type='Gaussian',
            show_progressbar=None)
        assert np.allclose(s1.data, np.zeros(len(s1.data)))

    def test_background_remove_gaussian_full_fit(self):
        s1 = self.signal.remove_background(
            signal_range=(None, None),
            background_type='Gaussian',
            fast=False)
        assert np.allclose(s1.data, np.zeros(len(s1.data)))


@lazifyTestClass
class TestRemoveBackground1DPowerLaw:

    def setup_method(self, method):
        pl = components1d.PowerLaw()
        pl.A.value = 1e10
        pl.r.value = 3
        self.signal = signals.Signal1D(
            pl.function(np.arange(100, 200)))
        self.signal.axes_manager[0].offset = 100
        self.signal.metadata.Signal.binned = False

        self.signal_noisy = self.signal.deepcopy()
        self.signal_noisy.add_gaussian_noise(1)

        self.atol = 0.04 * abs(self.signal.data).max()
        self.atol_zero_fill = 0.04 * abs(self.signal.isig[10:].data).max()

    def test_background_remove_pl(self):
        s1 = self.signal.remove_background(
            signal_range=(None, None),
            background_type='PowerLaw',
            show_progressbar=None)
        # since we compare to zero, rtol can't be used (see np.allclose doc)
        assert np.allclose(s1.data, np.zeros(len(s1.data)), atol=self.atol)
        assert s1.axes_manager.navigation_dimension == 0

    def test_background_remove_pl_zero(self):
        s1 = self.signal_noisy.remove_background(
            signal_range=(110.0, 190.0),
            background_type='PowerLaw',
            zero_fill=True,
            show_progressbar=None)
        # since we compare to zero, rtol can't be used (see np.allclose doc)
        assert np.allclose(s1.isig[10:], np.zeros(len(s1.data[10:])),
                           atol=self.atol_zero_fill)
        assert np.allclose(s1.data[:10], np.zeros(10))

    def test_background_remove_pl_int(self):
        self.signal.change_dtype("int")
        s1 = self.signal.remove_background(
            signal_range=(None, None),
            background_type='PowerLaw',
            show_progressbar=None)
        # since we compare to zero, rtol can't be used (see np.allclose doc)
        assert np.allclose(s1.data, np.zeros(len(s1.data)), atol=self.atol)

    def test_background_remove_pl_int_zero(self):
        self.signal_noisy.change_dtype("int")
        s1 = self.signal_noisy.remove_background(
            signal_range=(110.0, 190.0),
            background_type='PowerLaw',
            zero_fill=True,
            show_progressbar=None)
        # since we compare to zero, rtol can't be used (see np.allclose doc)
        assert np.allclose(s1.isig[10:], np.zeros(len(s1.data[10:])),
                           atol=self.atol_zero_fill)
        assert np.allclose(s1.data[:10], np.zeros(10))
