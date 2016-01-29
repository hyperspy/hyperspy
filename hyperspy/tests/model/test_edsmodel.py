import nose.tools as nt
import numpy as np
from hyperspy.misc.test_utils import assert_warns

from hyperspy.misc.eds import utils as utils_eds
from hyperspy.misc.elements import elements as elements_db


class TestlineFit:

    def setUp(self):
        s = utils_eds.xray_lines_model(elements=['Fe', 'Cr', 'Zn'],
                                       beam_energy=200,
                                       weight_percents=[20, 50, 30],
                                       energy_resolution_MnKa=130,
                                       energy_axis={'units': 'keV',
                                                    'size': 400,
                                                    'scale': 0.01,
                                                    'name': 'E',
                                                    'offset': 5.})
        s = s + 0.002
        self.s = s

    def test_fit(self):
        s = self.s
        m = s.create_model()
        m.fit()
        nt.assert_true(np.allclose([i.data for i in
                                   m.get_lines_intensity()],
                                   [[0.5], [0.2], [0.3]], atol=10 - 4))

    def test_calibrate_energy_resolution(self):
        s = self.s
        m = s.create_model()
        m.fit()
        m.fit_background()
        reso = s.metadata.Acquisition_instrument.TEM.Detector.EDS.\
            energy_resolution_MnKa,
        s.set_microscope_parameters(energy_resolution_MnKa=150)
        with assert_warns(message=r"Energy resolution \(FWHM at Mn Ka\) "
                          "changed from"):
            m.calibrate_energy_axis(calibrate='resolution')
        nt.assert_true(np.allclose(
            s.metadata.Acquisition_instrument.TEM.Detector.EDS.
            energy_resolution_MnKa, reso, atol=1))

    def test_calibrate_energy_scale(self):
        s = self.s
        m = s.create_model()
        m.fit()
        scale = s.axes_manager[-1].scale
        s.axes_manager[-1].scale += 0.0004
        m.calibrate_energy_axis('scale')
        nt.assert_true(np.allclose(s.axes_manager[-1].scale,
                                   scale, atol=1e-3))

    def test_calibrate_energy_offset(self):
        s = self.s
        m = s.create_model()
        m.fit()
        offset = s.axes_manager[-1].offset
        s.axes_manager[-1].offset += 0.04
        m.calibrate_energy_axis('offset')
        nt.assert_true(np.allclose(s.axes_manager[-1].offset,
                                   offset, atol=1e-1))

    def test_calibrate_xray_energy(self):
        s = self.s
        m = s.create_model()
        m.fit()
        m['Fe_Ka'].centre.value = 6.39
        m.calibrate_xray_lines(calibrate='energy', xray_lines=['Fe_Ka'],
                               bound=100)
        nt.assert_true(np.allclose(
            m['Fe_Ka'].centre.value, elements_db['Fe']['Atomic_properties'][
                'Xray_lines']['Ka']['energy (keV)'], atol=1e-6))

    def test_calibrate_xray_weight(self):
        s = self.s
        s1 = utils_eds.xray_lines_model(
            elements=['Co'],
            weight_percents=[50],
            energy_axis={'units': 'keV', 'size': 400,
                         'scale': 0.01, 'name': 'E',
                         'offset': 4.9})
        s = (s + s1 / 50)
        m = s.create_model()
        m.fit()
        with assert_warns(message='The X-ray line expected to be in the model '
                          'was not found'):
            m.calibrate_xray_lines(calibrate='sub_weight',
                                   xray_lines=['Fe_Ka'], bound=100)

        nt.assert_true(np.allclose(0.0347, m['Fe_Kb'].A.value,
                       atol=1e-3))

    def test_calibrate_xray_width(self):
        s = self.s
        m = s.create_model()
        m.fit()
        sigma = m['Fe_Ka'].sigma.value
        m['Fe_Ka'].sigma.value = 0.065
        m.calibrate_xray_lines(calibrate='energy', xray_lines=['Fe_Ka'],
                               bound=10)
        nt.assert_true(np.allclose(sigma, m['Fe_Ka'].sigma.value,
                                   atol=1e-2))
