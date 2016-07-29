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
import nose.tools as nt

from hyperspy.signals import SEDPattern
from hyperspy.defaults_parser import preferences


class Test_metadata:

    def setUp(self):
        # Create an empty diffraction pattern
        dp = SEDPattern(np.ones((2, 2, 2, 2)))
        dp.axes_manager.signal_axes[0].scale = 1e-3
        dp.metadata.Acquisition_instrument.TEM.accelerating_voltage = 200
        dp.metadata.Acquisition_instrument.TEM.convergence_angle = 15.0
        dp.metadata.Acquisition_instrument.TEM.precession_angle = 18.0
        dp.metadata.Acquisition_instrument.TEM.precession_frequency = 63
        dp.metadata.Acquisition_instrument.TEM.Detector.exposure_time = 35
        self.signal = dp

    def test_default_param(self):
        dp = self.signal
        md = dp.metadata
        nt.assert_equal(md.Acquisition_instrument.TEM.precession_angle,
                        preferences.SED.sed_precession_angle)

class Test_direct_beam_methods:

    def setUp(self):
        dp = SEDPattern(np.ones((3, 6, 6)))
        self.signal = dp

    def test_get_direct_beam_position(self):
        dp = self.signal

    def test_get_direct_beam_subpixel(self):
        dp = self.signal

    def test_direct_beam_shifts(self):
        dp = self.signal

class Test_masking:

    def setUp(self):
        dp = SEDPattern(np.ones((3, 6, 6)))
        self.signal = dp

    def test_direct_beam_mask(self):
        dp = self.signal

    def test_vacuum_mask(self):
        dp = self.signal
