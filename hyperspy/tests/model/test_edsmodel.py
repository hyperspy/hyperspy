import nose.tools as nt
import numpy as np

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
        s = s+0.002
        self.s = s

    def test_fit(self):
        s = self.s
        m = s.create_model()
        m.fit()
        np.testing.assert_allclose([i.data for i in
                                   m.get_lines_intensity()],
                                   [[0.5], [0.2], [0.3]], atol=10-4)

    def test_calibrate_energy_resolution(self):
        s = self.s
        m = s.create_model()
        m.fit()
        m.fit_background()
        reso = s.metadata.Acquisition_instrument.TEM.Detector.EDS.\
            energy_resolution_MnKa,
        s.set_microscope_parameters(energy_resolution_MnKa=150)
        m.calibrate_energy_axis(calibrate='resolution')
        np.testing.assert_allclose(
            s.metadata.Acquisition_instrument.TEM.Detector.EDS.
            energy_resolution_MnKa, reso, atol=1)

    def test_calibrate_energy_scale(self):
        s = self.s
        m = s.create_model()
        m.fit()
        ax = s.axes_manager[-1]
        scale = ax.scale
        ax.scale += 0.0004
        m.calibrate_energy_axis('scale')
        np.testing.assert_allclose(ax.scale, scale, atol=1e-5)

    def test_calibrate_energy_offset(self):
        s = self.s
        m = s.create_model()
        m.fit()
        ax = s.axes_manager[-1]
        offset = ax.offset
        ax.offset += 0.04
        m.calibrate_energy_axis('offset')
        np.testing.assert_allclose(ax.offset, offset, atol=1e-5)

    def test_calibrate_xray_energy(self):
        s = self.s
        m = s.create_model()
        m.fit()
        m['Fe_Ka'].centre.value = 6.39
        m.calibrate_xray_lines(calibrate='energy', xray_lines=['Fe_Ka'],
                               bound=100)
        np.testing.assert_allclose(
            m['Fe_Ka'].centre.value, elements_db['Fe']['Atomic_properties'][
                'Xray_lines']['Ka']['energy (keV)'], atol=1e-6)

    def test_calibrate_xray_weight(self):
        s = self.s
        s1 = utils_eds.xray_lines_model(
            elements=['Co'],
            weight_percents=[50],
            energy_axis={'units': 'keV', 'size': 400,
                         'scale': 0.01, 'name': 'E',
                         'offset': 4.9})
        s = (s+s1/50)
        m = s.create_model()
        m.fit()
        m.calibrate_xray_lines(calibrate='sub_weight',
                               xray_lines=['Fe_Ka'], bound=100)
        np.testing.assert_allclose(0.0347, m['Fe_Kb'].A.value,
                                   atol=1e-3)

    def test_calibrate_xray_width(self):
        s = self.s
        m = s.create_model()
        m.fit()
        sigma = m['Fe_Ka'].sigma.value
        m['Fe_Ka'].sigma.value = 0.065
        m.calibrate_xray_lines(calibrate='energy', xray_lines=['Fe_Ka'],
                               bound=10)
        np.testing.assert_allclose(sigma, m['Fe_Ka'].sigma.value,
                                   atol=1e-2)


class TestMaps:

    def setUp(self):
        beam_energy = 200
        energy_resolution_MnKa = 130
        energy_axis = {'units': 'keV', 'size': 1200, 'scale': 0.01,
                       'name': 'E'}
        s1 = utils_eds.xray_lines_model(
            elements=['Fe', 'Cr'], weight_percents=[30, 70],
            beam_energy=beam_energy,
            energy_resolution_MnKa=energy_resolution_MnKa,
            energy_axis=energy_axis)
        s2 = utils_eds.xray_lines_model(
            elements=['Ga', 'As'], weight_percents=[50, 50],
            beam_energy=beam_energy,
            energy_resolution_MnKa=energy_resolution_MnKa,
            energy_axis=energy_axis)
        # Make a circular mixing pattern with diffuse edge
        mix = np.mgrid[-10:10, -10:10]
        mix = np.sqrt(mix[0, ...]**2 + mix[1, ...]**2)
        mix = np.clip((mix-4), 0., 5.0) / 5.0
        mix_data = np.tile(s1.data, mix.shape + (1,))
        s = s1._deepcopy_with_new_data(mix_data)
        a = s.axes_manager._axes.pop(0).get_axis_dictionary()
        s.axes_manager.create_axes([{'size': mix.shape[0],
                                     'navigate': True}]*2 + [a])
        s.add_elements(s2.metadata.Sample.elements)

        for d, m in zip(s._iterate_signal(), mix.flatten()):
            d[:] = d*m + (1-m)*s2.data
        self.mix = mix
        self.s = s

    def test_lines_intensity(self):
        s = self.s
        m = s.create_model()
        m.fit()
        w1 = np.array([0.3, 0.7])
        w2 = np.array([0.5, 0.5])
        w = np.zeros((4,) + self.mix.shape)
        for x in xrange(self.mix.shape[0]):
            for y in xrange(self.mix.shape[1]):
                w[0:2, x, y] = w1 * self.mix[x, y]
                w[2:4, x, y] = w2 * (1-self.mix[x, y])
        xray_lines = s._get_lines_from_elements(
            s.metadata.Sample.elements, only_lines=('Ka',))
        np.testing.assert_allclose([i.data for i in
                                   m.get_lines_intensity(xray_lines)],
                                   w, atol=10-4)
