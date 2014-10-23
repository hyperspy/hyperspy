# Copyright 2007-2011 The HyperSpy developers
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
import nose.tools

from hyperspy.signals import EDSSEMSpectrum
from hyperspy.defaults_parser import preferences
from hyperspy.components import Gaussian
from hyperspy import utils


class Test_metadata:

    def setUp(self):
        # Create an empty spectrum
        s = EDSSEMSpectrum(np.ones((4, 2, 1024)))
        s.axes_manager.signal_axes[0].scale = 1e-3
        s.axes_manager.signal_axes[0].units = "keV"
        s.axes_manager.signal_axes[0].name = "Energy"
        s.metadata.Acquisition_instrument.SEM.Detector.EDS.live_time = 3.1
        s.metadata.Acquisition_instrument.SEM.beam_energy = 15.0
        s.metadata.Acquisition_instrument.SEM.tilt_stage = -38
        s.metadata.Acquisition_instrument.SEM.Detector.EDS.azimuth_angle = 63
        s.metadata.Acquisition_instrument.SEM.Detector.EDS.elevation_angle = 35
        self.signal = s

    def test_sum_live_time(self):
        s = self.signal
        sSum = s.sum(0)
        nose.tools.assert_equal(
            sSum.metadata.Acquisition_instrument.SEM.Detector.EDS.live_time,
            3.1 *
            2)

    def test_rebin_live_time(self):
        s = self.signal
        dim = s.axes_manager.shape
        s = s.rebin([dim[0] / 2, dim[1] / 2, dim[2]])
        nose.tools.assert_equal(
            s.metadata.Acquisition_instrument.SEM.Detector.EDS.live_time,
            3.1 *
            2 *
            2)

    def test_add_elements(self):
        s = self.signal
        s.add_elements(['Al', 'Ni'])
        nose.tools.assert_equal(s.metadata.Sample.elements, ['Al', 'Ni'])
        s.add_elements(['Al', 'Ni'])
        nose.tools.assert_equal(s.metadata.Sample.elements, ['Al', 'Ni'])
        s.add_elements(["Fe", ])
        nose.tools.assert_equal(s.metadata.Sample.elements, ['Al', "Fe", 'Ni'])
        s.set_elements(['Al', 'Ni'])
        nose.tools.assert_equal(s.metadata.Sample.elements, ['Al', 'Ni'])

    def test_add_lines(self):
        s = self.signal
        s.add_lines(lines=())
        nose.tools.assert_equal(s.metadata.Sample.xray_lines, [])
        s.add_lines(("Fe_Ln",))
        nose.tools.assert_equal(s.metadata.Sample.xray_lines, ["Fe_Ln"])
        s.add_lines(("Fe_Ln",))
        nose.tools.assert_equal(s.metadata.Sample.xray_lines, ["Fe_Ln"])
        s.add_elements(["Ti", ])
        s.add_lines(())
        nose.tools.assert_equal(
            s.metadata.Sample.xray_lines, ['Fe_Ln', 'Ti_La'])
        s.set_lines((), only_one=False, only_lines=False)
        nose.tools.assert_equal(s.metadata.Sample.xray_lines,
                                ['Fe_La', 'Fe_Lb3', 'Fe_Ll', 'Fe_Ln', 'Ti_La',
                                 'Ti_Lb3', 'Ti_Ll', 'Ti_Ln'])
        s.metadata.Acquisition_instrument.SEM.beam_energy = 0.4
        s.set_lines((), only_one=False, only_lines=False)
        nose.tools.assert_equal(s.metadata.Sample.xray_lines, ['Ti_Ll'])

    def test_add_lines_auto(self):
        s = self.signal
        s.axes_manager.signal_axes[0].scale = 1e-2
        s.set_elements(["Ti", "Al"])
        s.set_lines(['Al_Ka'])
        nose.tools.assert_equal(
            s.metadata.Sample.xray_lines, ['Al_Ka', 'Ti_Ka'])

        del s.metadata.Sample.xray_lines
        s.set_elements(['Al', 'Ni'])
        s.add_lines()
        nose.tools.assert_equal(
            s.metadata.Sample.xray_lines, ['Al_Ka', 'Ni_Ka'])
        s.metadata.Acquisition_instrument.SEM.beam_energy = 10.0
        s.set_lines([])
        nose.tools.assert_equal(
            s.metadata.Sample.xray_lines, ['Al_Ka', 'Ni_La'])

    def test_default_param(self):
        s = self.signal
        mp = s.metadata
        nose.tools.assert_equal(mp.Acquisition_instrument.SEM.Detector.EDS.energy_resolution_MnKa,
                                preferences.EDS.eds_mn_ka)

    def test_SEM_to_TEM(self):
        s = self.signal[0, 0]
        signal_type = 'EDS_TEM'
        mp = s.metadata
        mp.Acquisition_instrument.SEM.Detector.EDS.energy_resolution_MnKa = 125.3
        sTEM = s.deepcopy()
        sTEM.set_signal_type(signal_type)
        mpTEM = sTEM.metadata
        results = [
            mp.Acquisition_instrument.SEM.Detector.EDS.energy_resolution_MnKa]
        results.append(signal_type)
        resultsTEM = [
            mpTEM.Acquisition_instrument.TEM.Detector.EDS.energy_resolution_MnKa]
        resultsTEM.append(mpTEM.Signal.signal_type)
        nose.tools.assert_equal(results, resultsTEM)

    def test_get_calibration_from(self):
        s = self.signal
        scalib = EDSSEMSpectrum(np.ones((1024)))
        energy_axis = scalib.axes_manager.signal_axes[0]
        energy_axis.scale = 0.01
        energy_axis.offset = -0.10
        s.get_calibration_from(scalib)
        nose.tools.assert_equal(s.axes_manager.signal_axes[0].scale,
                                energy_axis.scale)

    def test_take_off_angle(self):
        s = self.signal
        nose.tools.assert_equal(s.get_take_off_angle(), 12.886929785732487)


class Test_get_lines_intentisity:

    def setUp(self):
        # Create an empty spectrum
        s = EDSSEMSpectrum(np.zeros((2, 2, 3, 100)))
        energy_axis = s.axes_manager.signal_axes[0]
        energy_axis.scale = 0.04
        energy_axis.units = 'keV'
        energy_axis.name = "Energy"
        g = Gaussian()
        g.sigma.value = 0.05
        g.centre.value = 1.487
        s.data[:] = g.function(energy_axis.axis)
        s.metadata.Acquisition_instrument.SEM.Detector.EDS.live_time = 3.1
        s.metadata.Acquisition_instrument.SEM.beam_energy = 15.0
        self.signal = s

    def test(self):
        s = self.signal
        sAl = s.get_lines_intensity(["Al_Ka"],
                                    plot_result=False,
                                    integration_window_factor=5)[0]
        nose.tools.assert_true(
            np.allclose(24.99516, sAl.data[0, 0, 0], atol=1e-3))
        sAl = s[0].get_lines_intensity(["Al_Ka"],
                                       plot_result=False,
                                       integration_window_factor=5)[0]
        nose.tools.assert_true(
            np.allclose(24.99516, sAl.data[0, 0], atol=1e-3))
        sAl = s[0, 0].get_lines_intensity(["Al_Ka"],
                                          plot_result=False,
                                          integration_window_factor=5)[0]
        nose.tools.assert_true(np.allclose(24.99516, sAl.data[0], atol=1e-3))
        sAl = s[0, 0, 0].get_lines_intensity(["Al_Ka"],
                                             plot_result=False,
                                             integration_window_factor=5)[0]
        nose.tools.assert_true(np.allclose(24.99516, sAl.data, atol=1e-3))

    def test_eV(self):
        s = self.signal
        energy_axis = s.axes_manager.signal_axes[0]
        energy_axis.scale = 40
        energy_axis.units = 'eV'

        sAl = s.get_lines_intensity(["Al_Ka"],
                                    plot_result=False,
                                    integration_window_factor=5)[0]
        nose.tools.assert_true(
            np.allclose(24.99516, sAl.data[0, 0, 0], atol=1e-3))


class Test_tools_bulk:

    def setUp(self):
        s = EDSSEMSpectrum(np.ones(1024))
        s.metadata.Acquisition_instrument.SEM.beam_energy = 5.0
        energy_axis = s.axes_manager.signal_axes[0]
        energy_axis.scale = 0.01
        energy_axis.units = 'keV'
        s.set_elements(['Al', 'Zn'])
        s.add_lines()
        self.signal = s

    def test_electron_range(self):
        s = self.signal
        mp = s.metadata
        elec_range = utils.eds.electron_range(
            mp.Sample.elements[0],
            mp.Acquisition_instrument.SEM.beam_energy,
            density='auto',
            tilt=mp.Acquisition_instrument.SEM.tilt_stage)
        nose.tools.assert_equal(elec_range, 0.41350651162374225)

    def test_xray_range(self):
        s = self.signal
        mp = s.metadata
        xr_range = utils.eds.xray_range(
            mp.Sample.xray_lines[0],
            mp.Acquisition_instrument.SEM.beam_energy,
            density=4.37499648818)
        nose.tools.assert_equal(xr_range, 0.1900368800933955)


class Test_energy_units:

    def setUp(self):
        s = EDSSEMSpectrum(np.ones(1024))
        s.metadata.Acquisition_instrument.SEM.beam_energy = 5.0
        s.axes_manager.signal_axes[0].units = 'keV'
        s.set_microscope_parameters(energy_resolution_MnKa=130)
        self.signal = s

    def test_beam_energy(self):
        s = self.signal
        nose.tools.assert_equal(s._get_beam_energy(), 5.0)
        s.axes_manager.signal_axes[0].units = 'eV'
        nose.tools.assert_equal(s._get_beam_energy(), 5000.0)
        s.axes_manager.signal_axes[0].units = 'keV'

    def test_line_energy(self):
        s = self.signal
        nose.tools.assert_equal(s._get_line_energy('Al_Ka'), 1.4865)
        s.axes_manager.signal_axes[0].units = 'eV'
        nose.tools.assert_equal(s._get_line_energy('Al_Ka'), 1486.5)
        s.axes_manager.signal_axes[0].units = 'keV'

        nose.tools.assert_equal(s._get_line_energy('Al_Ka', FWHM_MnKa='auto'),
                                (1.4865, 0.07661266213883969))
        nose.tools.assert_equal(s._get_line_energy('Al_Ka', FWHM_MnKa=128),
                                (1.4865, 0.073167615787314))
