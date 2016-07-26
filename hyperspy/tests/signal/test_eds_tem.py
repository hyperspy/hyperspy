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

from hyperspy.signals import EDSTEMSpectrum
from hyperspy.defaults_parser import preferences
from hyperspy.components1d import Gaussian
from hyperspy.misc.eds import utils as utils_eds
from hyperspy.misc.test_utils import ignore_warning


class Test_metadata:

    def setUp(self):
        # Create an empty spectrum
        s = EDSTEMSpectrum(np.ones((4, 2, 1024)))
        s.metadata.Acquisition_instrument.TEM.Detector.EDS.live_time = 3.1
        s.metadata.Acquisition_instrument.TEM.beam_energy = 15.0
        self.signal = s

    def test_sum_live_time1(self):
        s = self.signal
        old_metadata = s.metadata.deepcopy()
        sSum = s.sum(0)
        nt.assert_equal(
            sSum.metadata.Acquisition_instrument.TEM.Detector.EDS.live_time,
            3.1 * 2)
        # Check that metadata is unchanged
        print(old_metadata, s.metadata)      # Capture for comparison on error
        nt.assert_dict_equal(old_metadata.as_dictionary(),
                             s.metadata.as_dictionary(),
                             "Source metadata changed")

    def test_sum_live_time2(self):
        s = self.signal
        old_metadata = s.metadata.deepcopy()
        sSum = s.sum((0, 1))
        nt.assert_equal(
            sSum.metadata.Acquisition_instrument.TEM.Detector.EDS.live_time,
            3.1 * 2 * 4)
        # Check that metadata is unchanged
        print(old_metadata, s.metadata)      # Capture for comparison on error
        nt.assert_dict_equal(old_metadata.as_dictionary(),
                             s.metadata.as_dictionary(),
                             "Source metadata changed")

    def test_sum_live_time_out_arg(self):
        s = self.signal
        sSum = s.sum(0)
        s.metadata.Acquisition_instrument.TEM.Detector.EDS.live_time = 4.2
        s_resum = s.sum(0)
        r = s.sum(0, out=sSum)
        nt.assert_is_none(r)
        nt.assert_equal(
            s_resum.metadata.Acquisition_instrument.TEM.Detector.EDS.live_time,
            sSum.metadata.Acquisition_instrument.TEM.Detector.EDS.live_time)
        np.testing.assert_allclose(s_resum.data, sSum.data)

    def test_rebin_live_time(self):
        s = self.signal
        old_metadata = s.metadata.deepcopy()
        dim = s.axes_manager.shape
        s = s.rebin([dim[0] / 2, dim[1] / 2, dim[2]])
        nt.assert_equal(
            s.metadata.Acquisition_instrument.TEM.Detector.EDS.live_time,
            3.1 * 2 * 2)
        # Check that metadata is unchanged
        print(old_metadata, self.signal.metadata)    # Captured on error
        nt.assert_dict_equal(old_metadata.as_dictionary(),
                             self.signal.metadata.as_dictionary(),
                             "Source metadata changed")

    def test_add_elements(self):
        s = self.signal
        s.add_elements(['Al', 'Ni'])
        nt.assert_equal(s.metadata.Sample.elements, ['Al', 'Ni'])
        s.add_elements(['Al', 'Ni'])
        nt.assert_equal(s.metadata.Sample.elements, ['Al', 'Ni'])
        s.add_elements(["Fe", ])
        nt.assert_equal(s.metadata.Sample.elements, ['Al', "Fe", 'Ni'])
        s.set_elements(['Al', 'Ni'])
        nt.assert_equal(s.metadata.Sample.elements, ['Al', 'Ni'])

    def test_default_param(self):
        s = self.signal
        mp = s.metadata
        nt.assert_equal(
            mp.Acquisition_instrument.TEM.Detector.EDS.energy_resolution_MnKa,
            preferences.EDS.eds_mn_ka)

    def test_TEM_to_SEM(self):
        s = self.signal.inav[0, 0]
        signal_type = 'EDS_SEM'
        mp = s.metadata.Acquisition_instrument.TEM.Detector.EDS
        mp.energy_resolution_MnKa = 125.3
        sSEM = s.deepcopy()
        sSEM.set_signal_type(signal_type)
        mpSEM = sSEM.metadata.Acquisition_instrument.SEM.Detector.EDS
        results = [
            mp.energy_resolution_MnKa,
            signal_type]
        resultsSEM = [
            mpSEM.energy_resolution_MnKa,
            sSEM.metadata.Signal.signal_type]
        nt.assert_equal(results, resultsSEM)

    def test_get_calibration_from(self):
        s = self.signal
        scalib = EDSTEMSpectrum(np.ones(1024))
        energy_axis = scalib.axes_manager.signal_axes[0]
        energy_axis.scale = 0.01
        energy_axis.offset = -0.10
        s.get_calibration_from(scalib)
        nt.assert_equal(s.axes_manager.signal_axes[0].scale,
                        energy_axis.scale)


class Test_quantification:

    def setUp(self):
        s = EDSTEMSpectrum(np.ones([2, 2, 1024]))
        energy_axis = s.axes_manager.signal_axes[0]
        energy_axis.scale = 1e-2
        energy_axis.units = 'keV'
        energy_axis.name = "Energy"
        s.set_microscope_parameters(beam_energy=200,
                                    live_time=3.1, tilt_stage=0.0,
                                    azimuth_angle=None, elevation_angle=35,
                                    energy_resolution_MnKa=130)
        s.metadata.Acquisition_instrument.TEM.Detector.EDS.real_time = 2.5
        s.metadata.Acquisition_instrument.TEM.beam_current = 0.05
        elements = ['Al', 'Zn']
        xray_lines = ['Al_Ka', 'Zn_Ka']
        intensities = [300, 500]
        for i, xray_line in enumerate(xray_lines):
            gauss = Gaussian()
            line_energy, FWHM = s._get_line_energy(xray_line, FWHM_MnKa='auto')
            gauss.centre.value = line_energy
            gauss.A.value = intensities[i]
            gauss.sigma.value = FWHM
            s.data[:] += gauss.function(energy_axis.axis)

        s.set_elements(elements)
        s.add_lines(xray_lines)
        s.axes_manager[0].scale = 0.5
        s.axes_manager[1].scale = 0.5
        self.signal = s

    def test_quant_lorimer(self):
        s = self.signal
        method = 'CL'
        kfactors = [1, 2.0009344042484134]
        composition_units = 'weight'
        intensities = s.get_lines_intensity()
        res = s.quantification(intensities, method, kfactors,
                               composition_units)
        np.testing.assert_allclose(res[0].data, np.array([
            [22.70779, 22.70779],
            [22.70779, 22.70779]]), atol=1e-3)

    def test_quant_zeta(self):
        s = self.signal
        method = 'zeta'
        compositions_units = 'weight'
        factors = [20, 50]
        intensities = s.get_lines_intensity()
        res = s.quantification(intensities, method, factors,
                               compositions_units)
        np.testing.assert_allclose(res[1].data, np.array(
            [[2.7125736e-03, 2.7125736e-03],
             [2.7125736e-03, 2.7125736e-03]]), atol=1e-3)
        np.testing.assert_allclose(res[0][1].data, np.array(
            [[80.962287987, 80.962287987],
             [80.962287987, 80.962287987]]), atol=1e-3)

    def test_quant_cross_section(self):
        s = self.signal
        method = 'cross_section'
        factors = [3, 5]
        intensities = s.get_lines_intensity()
        res = s.quantification(intensities, method, factors)
        np.testing.assert_allclose(res[1][0].data, np.array(
            [[21517.1647074, 21517.1647074],
                [21517.1647074, 21517.1647074]]), atol=1e-3)
        np.testing.assert_allclose(res[1][1].data, np.array(
            [[21961.616621, 21961.616621],
             [21961.616621, 21961.616621]]), atol=1e-3)
        np.testing.assert_allclose(res[0][0].data, np.array(
            [[49.4888856823, 49.4888856823],
                [49.4888856823, 49.4888856823]]), atol=1e-3)

    def test_quant_zeros(self):
        intens = np.array([[0.5, 0.5, 0.5],
                           [0.0, 0.5, 0.5],
                           [0.5, 0.0, 0.5],
                           [0.5, 0.5, 0.0],
                           [0.5, 0.0, 0.0]]).T
        with ignore_warning(message="divide by zero encountered",
                            category=RuntimeWarning):
            quant = utils_eds.quantification_cliff_lorimer(
                intens, [1, 1, 3]).T
        np.testing.assert_allclose(
            quant,
            np.array([[0.2, 0.2, 0.6],
                      [0.0, 0.25, 0.75],
                      [0.25, 0.0, 0.75],
                      [0.5, 0.5, 0.0],
                      [1.0, 0.0, 0.0]]))

    def test_edx_cross_section_to_zeta(self):
        cs = [3, 6]
        elements = ['Pt', 'Ni']
        res = utils_eds.edx_cross_section_to_zeta(cs, elements)
        np.testing.assert_allclose(res, [1079.815272, 162.4378035], atol=1e-3)

    def test_zeta_to_edx_cross_section(self):
        factors = [1079.815272, 162.4378035]
        elements = ['Pt', 'Ni']
        res = utils_eds.zeta_to_edx_cross_section(factors, elements)
        np.testing.assert_allclose(res, [3, 6], atol=1e-3)


class Test_vacum_mask:

    def setUp(self):
        s = EDSTEMSpectrum(np.array([np.linspace(0.001, 0.5, 20)] * 100).T)
        s.add_poissonian_noise()
        self.signal = s

    def test_vacuum_mask(self):
        s = self.signal
        nt.assert_true(s.vacuum_mask().data[0])
        nt.assert_false(s.vacuum_mask().data[-1])


class Test_simple_model:

    def setUp(self):
        s = utils_eds.xray_lines_model(elements=['Al', 'Zn'],
                                       weight_percents=[50, 50])
        self.signal = s

    def test_intensity(self):
        s = self.signal
        np.testing.assert_allclose(
            [i.data[0] for i in s.get_lines_intensity(
                integration_window_factor=5.0)],
            [0.5, 0.5],
            atol=1e-1)


class Test_get_lines_intentisity:

    def test_with_signals_examples(self):
        from hyperspy.misc.example_signals_loading import \
            load_1D_EDS_TEM_spectrum as EDS_TEM_Spectrum
        s = EDS_TEM_Spectrum()
        np.testing.assert_allclose(
            np.array([res.data[0] for res in s.get_lines_intensity()]),
            np.array([3710, 15872]))


class Test_eds_markers:

    def setUp(self):
        s = utils_eds.xray_lines_model(elements=['Al', 'Zn'],
                                       weight_percents=[50, 50])
        self.signal = s

    def test_plot_auto_add(self):
        s = self.signal
        s.plot(xray_lines=True)
        # Should contain 6 lines
        nt.assert_sequence_equal(
            sorted(s._xray_markers.keys()),
            ['Al_Ka', 'Al_Kb', 'Zn_Ka', 'Zn_Kb', 'Zn_La', 'Zn_Lb1'])

    def test_manual_add_line(self):
        s = self.signal
        s.add_xray_lines_markers(['Zn_La'])
        nt.assert_sequence_equal(
            list(s._xray_markers.keys()),
            ['Zn_La'])
        nt.assert_equal(len(s._xray_markers), 1)
        # Check that the line has both a vertical line marker and text marker:
        nt.assert_equal(len(s._xray_markers['Zn_La']), 2)

    def test_manual_remove_element(self):
        s = self.signal
        s.add_xray_lines_markers(['Zn_Ka', 'Zn_Kb', 'Zn_La'])
        s.remove_xray_lines_markers(['Zn_Kb'])
        nt.assert_sequence_equal(
            sorted(s._xray_markers.keys()),
            ['Zn_Ka', 'Zn_La'])
