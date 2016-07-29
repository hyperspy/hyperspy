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
        dp = SEDPattern(np.ones((4, 2, 512, 512)))
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
        nt.assert_equal(
            md.Acquisition_instrument.TEM.precession_angle,
            preferences.SED.sed_precession_angle)
