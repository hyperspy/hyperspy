# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import numpy as np
import pytest

from hyperspy.datasets.example_signals import EDS_TEM_Spectrum
from hyperspy.decorators import lazifyTestClass
from hyperspy.misc import utils
from hyperspy.misc.eds import utils as utils_eds
from hyperspy.misc.elements import elements as elements_db


# Create this outside the test class to
# reduce computation in test suite by ~10seconds
s = utils_eds.xray_lines_model(
    elements=["Fe", "Cr", "Zn"],
    beam_energy=200,
    weight_percents=[20, 50, 30],
    energy_resolution_MnKa=130,
    energy_axis={
        "units": "keV",
        "size": 400,
        "scale": 0.01,
        "name": "E",
        "offset": 5.0,
    },
)
s = s + 0.002


@lazifyTestClass
class TestlineFit:

    def setup_method(self, method):
        self.s = s.deepcopy()

    def test_fit(self):
        s = self.s
        m = s.create_model()
        m.fit()
        np.testing.assert_allclose([i.data for i in
                                    m.get_lines_intensity()],
                                   [[0.5], [0.2], [0.3]], atol=1E-4)

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
        m.remove(["Cr_Ka", "background_order_6"])
        m.store()
        m1 = s.models.a.restore()
        assert (
            [c.name for c in m] == [c.name for c in m1])
        assert ([c.name for c in m.xray_lines] ==
                [c.name for c in m1.xray_lines])
        assert "Cr_Ka" not in m1.xray_lines
        assert "background_order_6" not in m1.background_components

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

        with pytest.warns(
            UserWarning,
            match="X-ray line expected to be in the model was not found"
        ):
            m.calibrate_xray_lines(calibrate='sub_weight',
                                   xray_lines=['Fe_Ka'], bound=100)

        np.testing.assert_allclose(0.0347, m['Fe_Kb'].A.value, atol=1e-3)

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

    def test_quantification(self):
        s = self.s
        # to have enough intensities, so that the default values for
        # `navigation_mask` of 1.0 in `quantification` doesn't mask the data
        s = s * 1000
        m = s.create_model()
        m.fit()
        intensities = m.get_lines_intensity()
        quant = s.quantification(intensities, method='CL',
                                 factors=[1.0, 1.0, 1.0],
                                 composition_units='weight')
        np.testing.assert_allclose(utils.stack(quant, axis=0), [50, 20, 30])

    def test_quantification_2_elements(self):
        s = self.s
        m = s.create_model()
        m.fit()
        intensities = m.get_lines_intensity(['Fe_Ka', 'Cr_Ka'])
        _ = s.quantification(intensities, method='CL', factors=[1.0, 1.0])


def test_comparison_quantification():
    kfactors = [1.450226, 5.075602]  # For Fe Ka and Pt La

    s = EDS_TEM_Spectrum()
    s.add_elements(['Cu'])  # to get good estimation of the background
    m = s.create_model(True)
    m.set_signal_range(5.5, 10.0)  # to get good fit
    m.fit()
    intensities_m = m.get_lines_intensity(['Fe_Ka', "Pt_La"])
    quant_model = s.quantification(intensities_m, method='CL',
                                   factors=kfactors)

    # Background substracted EDS quantification
    s2 = EDS_TEM_Spectrum()
    s2.add_lines()
    bw = s2.estimate_background_windows(line_width=[5.0, 2.0])
    intensities = s2.get_lines_intensity(background_windows=bw)
    atomic_percent = s2.quantification(intensities, method='CL',
                                       factors=kfactors)

    np.testing.assert_allclose([q.data for q in quant_model],
                               [q.data for q in atomic_percent],
                               rtol=0.5E-1)


@lazifyTestClass
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
        # HyperSpy 2.0: remove setting iterpath='serpentine'
        m.multifit(iterpath='serpentine')
        ws = np.array([0.5, 0.7, 0.3, 0.5])
        w = np.zeros((4,) + self.mix.shape)
        for x in range(self.mix.shape[0]):
            for y in range(self.mix.shape[1]):
                for i in range(4):
                    mix = self.mix[x, y] if i in (1, 2) else 1 - self.mix[x, y]
                    w[i, x, y] = ws[i] * mix
        xray_lines = s._get_lines_from_elements(
            s.metadata.Sample.elements, only_lines=('Ka',))
        if s._lazy:
            s.compute()
        for fitted, expected in zip(m.get_lines_intensity(xray_lines), w):
            np.testing.assert_allclose(fitted, expected, atol=1e-7)

        m_single_fit = s.create_model()
        # make sure we fit the navigation indices (0, 0) and don't rely on
        # the current index of the axes_manager.
        m_single_fit.inav[0, 0].fit()

        for fitted, expected in zip(
                m.inav[0, 0].get_lines_intensity(xray_lines),
                m_single_fit.inav[0, 0].get_lines_intensity(xray_lines)):
            np.testing.assert_allclose(fitted, expected, atol=1e-7)
