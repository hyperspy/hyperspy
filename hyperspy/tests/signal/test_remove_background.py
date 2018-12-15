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


def compare_axes_manager_metadata(s0, s1):
    assert s0.data.shape == s1.data.shape
    assert s0.axes_manager.shape == s1.axes_manager.shape
    for iaxis in range(len(s0.axes_manager._axes)):
        a0, a1 = s0.axes_manager[iaxis], s1.axes_manager[iaxis]
        assert a0.name == a1.name
        assert a0.units == a1.units
        assert a0.scale == a1.scale
        assert a0.offset == a1.offset
    assert s0.metadata.General.title == s1.metadata.General.title


class TestRemoveBackgroundMetadataAxesManagerCopy1D:

    def setup_method(self, method):
        s = signals.Signal1D(np.arange(10, 100)[::-1])
        s.axes_manager[0].name = 'axis0'
        s.axes_manager[0].units = 'units0'
        s.axes_manager[0].scale = 0.9
        s.axes_manager[0].offset = 1.
        s.metadata.General.title = "atitle"
        self.s = s

    def test_non_fast(self):
        s = self.s
        s_r = s.remove_background(signal_range=(2, 50), fast=False)
        compare_axes_manager_metadata(s, s_r)

    def test_fast(self):
        s = self.s
        s_r = s.remove_background(signal_range=(2, 50), fast=True)
        compare_axes_manager_metadata(s, s_r)

    def test_non_zero_fill(self):
        s = self.s
        s_r = s.remove_background(signal_range=(2, 50), zero_fill=False)
        compare_axes_manager_metadata(s, s_r)

    def test_zero_fill(self):
        s = self.s
        s_r = s.remove_background(signal_range=(2, 50), zero_fill=True)
        compare_axes_manager_metadata(s, s_r)

    def test_non_show_progressbar(self):
        s = self.s
        s_r = s.remove_background(signal_range=(2, 50), show_progressbar=False)
        compare_axes_manager_metadata(s, s_r)

    def test_show_progressbar(self):
        s = self.s
        s_r = s.remove_background(signal_range=(2, 50), show_progressbar=True)
        compare_axes_manager_metadata(s, s_r)

    def test_non_plot_remainder(self):
        s = self.s
        s_r = s.remove_background(signal_range=(2, 50), plot_remainder=False)
        compare_axes_manager_metadata(s, s_r)

    def test_background_type_powerlaw(self):
        s = self.s
        s_r = s.remove_background(signal_range=(2, 50),
                                  background_type='Power Law')
        compare_axes_manager_metadata(s, s_r)

    def test_background_type_offset(self):
        s = self.s
        s_r = s.remove_background(signal_range=(2, 50),
                                  background_type='Offset')
        compare_axes_manager_metadata(s, s_r)

    def test_background_type_Polynomial_order_2(self):
        s = self.s
        s_r = s.remove_background(signal_range=(2, 50),
                                  background_type='Polynomial',
                                  polynomial_order=2)
        compare_axes_manager_metadata(s, s_r)

    def test_background_type_Polynomial_order_3(self):
        s = self.s
        s_r = s.remove_background(signal_range=(2, 50),
                                  background_type='Polynomial',
                                  polynomial_order=3)
        compare_axes_manager_metadata(s, s_r)


class TestRemoveBackgroundMetadataAxesManagerCopy2D:

    def setup_method(self, method):
        s = signals.Signal1D(np.arange(10, 210)[::-1].reshape(2, 100))
        s.axes_manager[0].name = 'axis0'
        s.axes_manager[1].name = 'axis1'
        s.axes_manager[0].units = 'units0'
        s.axes_manager[1].units = 'units1'
        s.axes_manager[0].scale = 0.9
        s.axes_manager[1].scale = 1.1
        s.axes_manager[0].offset = 1.
        s.axes_manager[1].offset = 1.2
        s.metadata.General.title = "atitle"
        self.s = s

    def test_non_fast(self):
        s = self.s
        s_r = s.remove_background(signal_range=(2, 50), fast=False)
        compare_axes_manager_metadata(s, s_r)

    def test_fast(self):
        s = self.s
        s_r = s.remove_background(signal_range=(2, 50), fast=True)
        compare_axes_manager_metadata(s, s_r)

    def test_non_zero_fill(self):
        s = self.s
        s_r = s.remove_background(signal_range=(2, 50), zero_fill=False)
        compare_axes_manager_metadata(s, s_r)

    def test_zero_fill(self):
        s = self.s
        s_r = s.remove_background(signal_range=(2, 50), zero_fill=True)
        compare_axes_manager_metadata(s, s_r)

    def test_non_show_progressbar(self):
        s = self.s
        s_r = s.remove_background(signal_range=(2, 50), show_progressbar=False)
        compare_axes_manager_metadata(s, s_r)

    def test_show_progressbar(self):
        s = self.s
        s_r = s.remove_background(signal_range=(2, 50), show_progressbar=True)
        compare_axes_manager_metadata(s, s_r)

    def test_non_plot_remainder(self):
        s = self.s
        s_r = s.remove_background(signal_range=(2, 50), plot_remainder=False)
        compare_axes_manager_metadata(s, s_r)

    def test_background_type_powerlaw(self):
        s = self.s
        s_r = s.remove_background(signal_range=(2, 50),
                                  background_type='Power Law')
        compare_axes_manager_metadata(s, s_r)

    def test_background_type_offset(self):
        s = self.s
        s_r = s.remove_background(signal_range=(2, 50),
                                  background_type='Offset')
        compare_axes_manager_metadata(s, s_r)

    def test_background_type_Polynomial_order_2(self):
        s = self.s
        s_r = s.remove_background(signal_range=(2, 50),
                                  background_type='Polynomial',
                                  polynomial_order=2)
        compare_axes_manager_metadata(s, s_r)

    def test_background_type_Polynomial_order_3(self):
        s = self.s
        s_r = s.remove_background(signal_range=(2, 50),
                                  background_type='Polynomial',
                                  polynomial_order=3)
        compare_axes_manager_metadata(s, s_r)
