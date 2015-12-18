# -*- coding: utf-8 -*-
# Copyright 2007-2015 The HyperSpy developers
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


import nose.tools as nt
import numpy as np
from hyperspy.components import GaussianHF
from hyperspy.signals import Spectrum

sqrt2pi = np.sqrt(2 * np.pi)
sigma2fwhm = 2 * np.sqrt(2 * np.log(2))


class TestGaussianHF(object):
    def setUp(self):
        self.s = Spectrum(np.zeros((10, 5, 100)))

    def test_bad_multifit(self):
        # Simply test that no errors get raised while fitting component
        g = GaussianHF()
        m = self.s.create_model()
        m.append(g)
        m.multifit()

    def test_estimate_varying_centre_single(self):
        s = self.s
        g1 = GaussianHF()
        for c in np.linspace(0, 90, 13):
            g1.centre.value = c
            s.data[0, 0, :] = g1.function(s.axes_manager.signal_axes[0].axis)
            g2 = GaussianHF()
            g2.estimate_parameters(s, 0, 100, True)
            nt.assert_less(np.abs(g2.centre.value - g1.centre.value), 0.06)
            nt.assert_less(np.abs(g2.fwhm.value - g1.fwhm.value), 0.5)
            nt.assert_less(np.abs(g2.height.value - g1.height.value), 0.6)

    def test_estimate_varying_centre_multi(self):
        s = self.s
        g1 = GaussianHF()
        for d, c in zip(s._iterate_signal(),
                        np.linspace(0, 90, s.axes_manager.navigation_size)):
            g1.centre.value = c
            d[:] = g1.function(s.axes_manager.signal_axes[0].axis)
        g2 = GaussianHF()
        g1._axes_manager = g2._axes_manager = s.axes_manager
        g2.estimate_parameters(s, 0, 100, False)
        ref = np.linspace(0, 90, s.axes_manager.navigation_size).reshape(
            s.axes_manager.navigation_shape[::-1])
        np.testing.assert_array_less(
            np.abs(ref - g2.centre.map['values']), 0.5)
        np.testing.assert_array_less(
            np.abs(g2.fwhm.map['values'] - g1.fwhm.value), 0.5)
        np.testing.assert_array_less(
            np.abs(g2.height.map['values'] - g1.height.value), 0.6)

    def test_fit_all_param(self):
        s = self.s
        g1 = GaussianHF(50015.156, 23, 10)
        s.data[0, 0, :] = g1.function(s.axes_manager.signal_axes[0].axis)
        g2 = GaussianHF()
        g2.estimate_parameters(s, 0, 100, True)
        nt.assert_less(np.abs(g2.centre.value - g1.centre.value), 2.6)
        nt.assert_less(np.abs(g2.fwhm.value - g1.fwhm.value), 4.5)
        nt.assert_less(np.abs(g2.height.value - g1.height.value), 0.1)

    def test_binned(self):
        s = self.s
        s.axes_manager.signal_axes[0].scale = 0.3
        s.metadata.Signal.binned = True
        g1 = GaussianHF(50015.156, 23, 10)
        s.data[0, 0, :] = g1.function(s.axes_manager.signal_axes[0].axis)
        g2 = GaussianHF()
        g2.estimate_parameters(s, 0, 100, True)
        nt.assert_less(np.abs(
            g1.height.value/s.axes_manager.signal_axes[0].scale -
            g2.height.value),
            12.)
        # TODO: For some reason, this is more accurate than above...
        nt.assert_less(np.abs(g2.centre.value - g1.centre.value), 2.2)
        nt.assert_less(np.abs(g2.fwhm.value - g1.fwhm.value), 6.5)

    def test_integral_as_signal(self):
        s = self.s
        g1 = GaussianHF(fwhm=3.33, centre=20.)
        h_ref = np.linspace(0.1, 3.0, s.axes_manager.navigation_size)
        for d, h in zip(s._iterate_signal(), h_ref):
            g1.height.value = h
            d[:] = g1.function(s.axes_manager.signal_axes[0].axis)
        m = self.s.create_model()
        g2 = GaussianHF()
        m.append(g2)
        g2.estimate_parameters(s, 0, 100, True)
        m.multifit()
        s_out = g2.integral_as_signal()
        ref = (h_ref * 3.33 * sqrt2pi / sigma2fwhm).reshape(s_out.data.shape)
        np.testing.assert_almost_equal(
            s_out.data, ref)


def test_util_sigma_set():
    g1 = GaussianHF()
    g1.sigma = 1.0
    nt.assert_almost_equal(g1.fwhm.value, 1.0 * sigma2fwhm)


def test_util_sigma_get():
    g1 = GaussianHF()
    g1.fwhm.value = 1.0
    nt.assert_almost_equal(g1.sigma, 1.0 / sigma2fwhm)


def test_util_sigma_getset():
    g1 = GaussianHF()
    g1.sigma = 1.0
    nt.assert_almost_equal(g1.sigma, 1.0)


def test_util_fwhm_set():
    g1 = GaussianHF(fwhm=0.33)
    g1.A = 1.0
    nt.assert_almost_equal(g1.height.value, 1.0 * sigma2fwhm / (
        0.33 * sqrt2pi))


def test_util_fwhm_get():
    g1 = GaussianHF(fwhm=0.33)
    g1.height.value = 1.0
    nt.assert_almost_equal(g1.A, 1.0 * sqrt2pi * 0.33 / sigma2fwhm)


def test_util_fwhm_getset():
    g1 = GaussianHF(fwhm=0.33)
    g1.A = 1.0
    nt.assert_almost_equal(g1.A, 1.0)
