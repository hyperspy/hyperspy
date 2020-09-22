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

import sys

import numpy as np

from numpy.testing import assert_allclose

from hyperspy.signals import EDSSEMSpectrum
from hyperspy.defaults_parser import preferences
from hyperspy.components1d import Gaussian
from hyperspy import utils
from hyperspy.misc.test_utils import assert_warns
from hyperspy.decorators import lazifyTestClass


@lazifyTestClass
class Test_metadata:

    def setup_method(self, method):
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
        old_metadata = s.metadata.deepcopy()
        sSum = s.sum(0)
        assert (
            sSum.metadata.Acquisition_instrument.SEM.Detector.EDS.live_time ==
            3.1 * 2)
        # Check that metadata is unchanged
        print(old_metadata, s.metadata)      # Capture for comparison on error
        assert (old_metadata.as_dictionary() ==
                s.metadata.as_dictionary()), "Source metadata changed"

    def test_sum_live_time2(self):
        s = self.signal
        old_metadata = s.metadata.deepcopy()
        sSum = s.sum((0, 1))
        assert (
            sSum.metadata.Acquisition_instrument.SEM.Detector.EDS.live_time ==
            3.1 *
            2 * 4)
        # Check that metadata is unchanged
        print(old_metadata, s.metadata)      # Capture for comparison on error
        assert (old_metadata.as_dictionary() ==
                s.metadata.as_dictionary()), "Source metadata changed"

    def test_sum_live_time_out_arg(self):
        s = self.signal
        sSum = s.sum(0)
        s.metadata.Acquisition_instrument.SEM.Detector.EDS.live_time = 4.2
        s_resum = s.sum(0)
        r = s.sum(0, out=sSum)
        assert r is None
        assert (
            s_resum.metadata.Acquisition_instrument.SEM.Detector.EDS.live_time ==
            sSum.metadata.Acquisition_instrument.SEM.Detector.EDS.live_time)
        np.testing.assert_allclose(s_resum.data, sSum.data)

    def test_rebin_live_time(self):
        s = self.signal
        old_metadata = s.metadata.deepcopy()
        dim = s.axes_manager.shape
        s = s.rebin([dim[0] / 2, dim[1] / 2, dim[2]])
        assert (
            s.metadata.Acquisition_instrument.SEM.Detector.EDS.live_time ==
            3.1 *
            2 *
            2)
        # Check that metadata is unchanged
        print(old_metadata, self.signal.metadata)    # Captured on error
        assert (old_metadata.as_dictionary() ==
                self.signal.metadata.as_dictionary()), "Source metadata changed"

    def test_add_elements(self):
        s = self.signal
        s.add_elements(['Al', 'Ni'])
        assert s.metadata.Sample.elements == ['Al', 'Ni']
        s.add_elements(['Al', 'Ni'])
        assert s.metadata.Sample.elements == ['Al', 'Ni']
        s.add_elements(["Fe", ])
        assert s.metadata.Sample.elements == ['Al', "Fe", 'Ni']
        s.set_elements(['Al', 'Ni'])
        assert s.metadata.Sample.elements == ['Al', 'Ni']

    def test_add_lines(self):
        s = self.signal
        s.add_lines(lines=())
        assert s.metadata.Sample.xray_lines == []
        s.add_lines(("Fe_Ln",))
        assert s.metadata.Sample.xray_lines == ["Fe_Ln"]
        s.add_lines(("Fe_Ln",))
        assert s.metadata.Sample.xray_lines == ["Fe_Ln"]
        s.add_elements(["Ti", ])
        s.add_lines(())
        assert (
            s.metadata.Sample.xray_lines == ['Fe_Ln', 'Ti_La'])
        s.set_lines((), only_one=False, only_lines=False)
        assert (s.metadata.Sample.xray_lines ==
                ['Fe_La', 'Fe_Lb3', 'Fe_Ll', 'Fe_Ln', 'Ti_La',
                 'Ti_Lb3', 'Ti_Ll', 'Ti_Ln'])
        s.metadata.Acquisition_instrument.SEM.beam_energy = 0.4
        s.set_lines((), only_one=False, only_lines=False)
        assert s.metadata.Sample.xray_lines == ['Ti_Ll']

    def test_add_lines_auto(self):
        s = self.signal
        s.axes_manager.signal_axes[0].scale = 1e-2
        s.set_elements(["Ti", "Al"])
        s.set_lines(['Al_Ka'])
        assert (
            s.metadata.Sample.xray_lines == ['Al_Ka', 'Ti_Ka'])

        del s.metadata.Sample.xray_lines
        s.set_elements(['Al', 'Ni'])
        s.add_lines()
        assert (
            s.metadata.Sample.xray_lines == ['Al_Ka', 'Ni_Ka'])
        s.metadata.Acquisition_instrument.SEM.beam_energy = 10.0
        s.set_lines([])
        assert (
            s.metadata.Sample.xray_lines == ['Al_Ka', 'Ni_La'])
        s.metadata.Acquisition_instrument.SEM.beam_energy = 200
        s.set_elements(['Au', 'Ni'])
        s.set_lines([])
        assert (s.metadata.Sample.xray_lines ==
                ['Au_La', 'Ni_Ka'])

    def test_default_param(self):
        s = self.signal
        mp = s.metadata
        assert (
            mp.Acquisition_instrument.SEM.Detector.EDS.energy_resolution_MnKa ==
            preferences.EDS.eds_mn_ka)

    def test_SEM_to_TEM(self):
        s = self.signal.inav[0, 0]
        signal_type = 'EDS_TEM'
        mp = s.metadata
        mp.Acquisition_instrument.SEM.Detector.EDS.energy_resolution_MnKa = \
            125.3
        sTEM = s.deepcopy()
        sTEM.set_signal_type(signal_type)
        mpTEM = sTEM.metadata
        results = [
            mp.Acquisition_instrument.SEM.Detector.EDS.energy_resolution_MnKa,
            signal_type]
        resultsTEM = [
            (mpTEM.Acquisition_instrument.TEM.Detector.EDS.
             energy_resolution_MnKa),
            mpTEM.Signal.signal_type]
        assert results == resultsTEM

    def test_get_calibration_from(self):
        s = self.signal
        scalib = EDSSEMSpectrum(np.ones(1024))
        energy_axis = scalib.axes_manager.signal_axes[0]
        energy_axis.scale = 0.01
        energy_axis.offset = -0.10
        s.get_calibration_from(scalib)
        assert s.axes_manager.signal_axes[0].scale == energy_axis.scale

    def test_take_off_angle(self):
        s = self.signal
        assert_allclose(s.get_take_off_angle(), 12.886929785732487,
                        atol=10**-(sys.float_info.dig - 2))


@lazifyTestClass
class Test_get_lines_intentisity:

    def setup_method(self, method):
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
                                    integration_windows=5)[0]
        assert sAl.axes_manager.signal_dimension == 0
        np.testing.assert_allclose(24.99516, sAl.data[0, 0, 0], atol=1e-3)
        sAl = s.inav[0].get_lines_intensity(
            ["Al_Ka"], plot_result=False, integration_windows=5)[0]
        np.testing.assert_allclose(24.99516, sAl.data[0, 0], atol=1e-3)
        sAl = s.inav[0, 0].get_lines_intensity(
            ["Al_Ka"], plot_result=False, integration_windows=5)[0]
        np.testing.assert_allclose(24.99516, sAl.data[0], atol=1e-3)
        sAl = s.inav[0, 0, 0].get_lines_intensity(
            ["Al_Ka"], plot_result=False, integration_windows=5)[0]
        np.testing.assert_allclose(24.99516, sAl.data, atol=1e-3)
        s.axes_manager[-1].offset = 1.0
        with assert_warns(message="C_Ka is not in the data energy range."):
            sC = s.get_lines_intensity(["C_Ka"], plot_result=False)
        assert len(sC) == 0
        assert sAl.metadata.Sample.elements == ["Al"]
        assert sAl.metadata.Sample.xray_lines == ["Al_Ka"]

    def test_eV(self):
        s = self.signal
        energy_axis = s.axes_manager.signal_axes[0]
        energy_axis.scale = 40
        energy_axis.units = 'eV'

        sAl = s.get_lines_intensity(["Al_Ka"],
                                    plot_result=False,
                                    integration_windows=5)[0]
        np.testing.assert_allclose(24.99516, sAl.data[0, 0, 0], atol=1e-3)

    def test_background_substraction(self):
        s = self.signal
        intens = s.get_lines_intensity(["Al_Ka"], plot_result=False)[0].data
        s = s + 1.
        np.testing.assert_allclose(s.estimate_background_windows(
            xray_lines=["Al_Ka"])[0, 0], 1.25666201, atol=1e-3)
        np.testing.assert_allclose(
            s.get_lines_intensity(
                ["Al_Ka"],
                background_windows=s.estimate_background_windows(
                    [4, 4], xray_lines=["Al_Ka"]),
                plot_result=False)[0].data,
            intens, atol=1e-3)

    def test_estimate_integration_windows(self):
        s = self.signal
        np.testing.assert_allclose(
            s.estimate_integration_windows(3.0, ["Al_Ka"]),
            [[1.371, 1.601]], atol=1e-2)

    def test_with_signals_examples(self):
        from hyperspy.misc.example_signals_loading import \
            load_1D_EDS_SEM_spectrum as EDS_SEM_Spectrum
        s = EDS_SEM_Spectrum()
        np.testing.assert_allclose(
            utils.stack(s.get_lines_intensity()).data.squeeze(),
            np.array([84163, 89063, 96117, 96700, 99075]))


@lazifyTestClass
class Test_tools_bulk:

    def setup_method(self, method):
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
        np.testing.assert_allclose(elec_range, 0.41350651162374225)

    def test_xray_range(self):
        s = self.signal
        mp = s.metadata
        xr_range = utils.eds.xray_range(
            mp.Sample.xray_lines[0],
            mp.Acquisition_instrument.SEM.beam_energy,
            density=4.37499648818)
        np.testing.assert_allclose(xr_range, 0.1900368800933955)


@lazifyTestClass
class Test_energy_units:

    def setup_method(self, method):
        s = EDSSEMSpectrum(np.ones(1024))
        s.metadata.Acquisition_instrument.SEM.beam_energy = 5.0
        s.axes_manager.signal_axes[0].units = 'keV'
        s.set_microscope_parameters(energy_resolution_MnKa=130)
        self.signal = s

    def test_beam_energy(self):
        s = self.signal
        assert s._get_beam_energy() == 5.0
        s.axes_manager.signal_axes[0].units = 'eV'
        assert s._get_beam_energy() == 5000.0
        s.axes_manager.signal_axes[0].units = 'keV'

    def test_line_energy(self):
        s = self.signal
        assert s._get_line_energy('Al_Ka') == 1.4865
        s.axes_manager.signal_axes[0].units = 'eV'
        assert s._get_line_energy('Al_Ka') == 1486.5
        s.axes_manager.signal_axes[0].units = 'keV'

        np.testing.assert_allclose(s._get_line_energy('Al_Ka', FWHM_MnKa='auto'),
                                   (1.4865, 0.07661266213883969))
        np.testing.assert_allclose(s._get_line_energy('Al_Ka', FWHM_MnKa=128),
                                   (1.4865, 0.073167615787314))
