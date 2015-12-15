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


class TestGaussianHF(object):
    def setUp(self):
        self.s = Spectrum(np.zeros((10, 5, 100)))
        self.m = self.s.create_model()

    def test_bad_multifit(self):
        g = GaussianHF()
        self.m.append(g)
        self.m.multifit()

    def test_fit_varying_centre_single(self):
        s = self.s
        g1 = GaussianHF()
        for c in np.linspace(0, 90, 13):
            g1.centre.value = c
            s.data[0, 0, :] = g1.function(s.axes_manager.signal_axes[0].axis)
            g2 = GaussianHF()
            g2.estimate_parameters(s, 0, 100, True)
            self.m.append(g2)
            self.m.fit()
            for p1, p2 in zip(g1.parameters, g2.parameters):
                nt.assert_almost_equal(p1.value, p2.value)
            self.m.remove(g2)

    def test_fit_varying_centre_multi(self):
        s = self.s
        g1 = GaussianHF()
        for d, c in zip(s._iterate_signal(),
                        np.linspace(0, 90, s.axes_manager.navigation_size)):
            g1.centre.value = c
            d[:] = g1.function(s.axes_manager.signal_axes[0].axis)
        g2 = GaussianHF()
        g1._axes_manager = g2._axes_manager = s.axes_manager
        g2.estimate_parameters(s, 0, 100, False)
        self.m.append(g2)
        self.m.multifit()
        ref = np.linspace(0, 90, s.axes_manager.navigation_size).reshape(
            s.axes_manager.navigation_shape[::-1])
        np.testing.assert_almost_equal(ref, g2.centre.map['values'])
        for p1, p2 in zip([g1.fwhm, g1.height], [g2.fwhm, g2.height]):
            np.testing.assert_almost_equal(p1.value, p2.map['values'])
