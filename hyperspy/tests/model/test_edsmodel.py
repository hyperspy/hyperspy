
import numpy as np
from hyperspy.misc.test_utils import assert_warns

from hyperspy.misc.eds import utils as utils_eds
from hyperspy.misc.elements import elements as elements_db


class TestlineFit:

    def setup_method(self, method):
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
        np.testing.assert_allclose([i.data for i in
                                    m.get_lines_intensity()],
                                   [[0.5], [0.2], [0.3]], atol=10 - 4)

    def _check_model_creation(self):
        s = self.s
        # Default:
        m = s.create_model()
        assert (
            [c.name for c in m] ==
            ['background_order_6', 'Cr_Ka', 'Cr_Kb',
             'Fe_Ka', 'Fe_Kb', 'Zn_Ka'])
        # No auto componentes:
        m = s.create_model(False, False)
        assert [c.name for c in m] == []

    def test_model_creation(self):
        self._check_model_creation()

    def test_semmodel_creation(self):
        self.s.set_signal_type("EDS_SEM")
        self._check_model_creation()

    def test_temmodel_creation(self):
        self.s.set_signal_type("EDS_TEM")
        self._check_model_creation()

    def _check_model_store(self):
        # Simply check that storing/restoring of EDSModels work
        # This also checks that creation from dictionary works
        s = self.s
        # Default:
        m = s.create_model()
        m.store()
        m1 = s.models.a.restore()
        assert (
            [c.name for c in m] == [c.name for c in m1])
        assert ([c.name for c in m.xray_lines] ==
                [c.name for c in m1.xray_lines])

    def test_edsmodel_store(self):
        self._check_model_store()

    def test_semmodel_store(self):
        self.s.set_signal_type("EDS_SEM")
        self._check_model_store()

    def test_temmodel_store(self):
        self.s.set_signal_type("EDS_TEM")
        self._check_model_store()

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
        s = (s + s1 / 50)
        m = s.create_model()
        m.fit()
        with assert_warns(message='The X-ray line expected to be in the model '
                          'was not found'):
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

    def test_enable_adjust_position(self):
        m = self.s.create_model()
        m.enable_adjust_position()
        assert len(m._position_widgets) == 5
        # Check that both line and label was added
        assert len(list(m._position_widgets.values())[0]) == 2
        lbls = [p[1].string for p in m._position_widgets.values()]
        assert sorted(lbls) == [
            '$\\mathrm{Cr}_{\\mathrm{Ka}}$',
            '$\\mathrm{Cr}_{\\mathrm{Kb}}$',
            '$\\mathrm{Fe}_{\\mathrm{Ka}}$',
            '$\\mathrm{Fe}_{\\mathrm{Kb}}$',
            '$\\mathrm{Zn}_{\\mathrm{Ka}}$']


class TestMaps:

    def setup_method(self, method):
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

        mix = np.linspace(0., 1., 4).reshape(2, 2)
        mix_data = np.tile(s1.data, mix.shape + (1,))
        s = s1._deepcopy_with_new_data(mix_data)
        a = s.axes_manager._axes.pop(0).get_axis_dictionary()
        s.axes_manager.create_axes([{'size': mix.shape[0],
                                     'navigate': True}] * 2 + [a])
        s.add_elements(s2.metadata.Sample.elements)

        for d, m in zip(s._iterate_signal(), mix.flatten()):
            d[:] = d * m + (1 - m) * s2.data
        self.mix = mix
        self.s = s

    def test_lines_intensity(self):
        s = self.s
        m = s.create_model()
        m.multifit()    # m.fit() is just too inaccurate
        ws = np.array([0.5, 0.7, 0.3, 0.5])
        w = np.zeros((4,) + self.mix.shape)
        for x in range(self.mix.shape[0]):
            for y in range(self.mix.shape[1]):
                for i in range(4):
                    mix = self.mix[x, y] if i in (1, 2) else 1 - self.mix[x, y]
                    w[i, x, y] = ws[i] * mix
        xray_lines = s._get_lines_from_elements(
            s.metadata.Sample.elements, only_lines=('Ka',))
        np.testing.assert_allclose(
            [i.data for i in m.get_lines_intensity(xray_lines)], w, atol=1e-7)
