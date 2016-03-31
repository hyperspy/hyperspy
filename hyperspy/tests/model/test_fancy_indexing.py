# Copyright 2007-2011 The Hyperspy developers
#
# This file is part of  Hyperspy.
#
#  Hyperspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  Hyperspy.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from nose.tools import assert_true, assert_false

from hyperspy._signals.spectrum_simulation import SpectrumSimulation
from hyperspy._signals.eels import EELSSpectrum
from hyperspy.components import Gaussian


class TestModelIndexing:

    def setUp(self):
        np.random.seed(1)
        axes = np.array([[100 * np.random.random() + np.arange(0., 600, 1)
                          for i in range(3)] for j in range(4)])
        g = Gaussian()
        g.A.value = 30000.
        g.centre.value = 300.
        g.sigma.value = 150.
        data = g.function(axes)
        s = SpectrumSimulation(data)
        s.axes_manager[-1].offset = -150.
        s.axes_manager[-1].scale = 0.5
        s.add_gaussian_noise(2.0)
        m = s.create_model()
        g = Gaussian()
        g.A.ext_force_positive = True
        g.A.ext_bounded = True
        m.append(g)
        g.active_is_multidimensional = True
        for index in m.axes_manager:
            m.fit()
        self.model = m

    def test_model_signal_indexer_slice(self):
        s = self.model.spectrum.isig[:300]
        m = self.model.isig[:300]
        m1 = self.model.isig[300:]
        m2 = self.model.isig[:0.]
        assert_true(m1[0].A.ext_bounded ==
                    m[0].A.ext_bounded)
        assert_true((s.data == m.spectrum.data).all())
        assert_true((s.data == m2.spectrum.data).all())
        assert_true((m.dof.data == self.model.dof.data).all())
        for ic, c in enumerate(m):
            for p_new, p_old in zip(c.parameters, self.model[ic].parameters):
                assert_true((p_old.map == p_new.map).all())
        np.testing.assert_array_almost_equal(m.chisq.data + m1.chisq.data,
                                             self.model.chisq.data)

        self.model.channel_switches[0] = False
        m = self.model.isig[:-100.]
        assert_true(m.channel_switches[0] == False)
        assert_true(np.all(m.channel_switches[1:] == True))

    def test_model_navigation_indexer_slice(self):
        self.model.axes_manager.indices = (0, 0)
        self.model[0].active = False
        m = self.model.inav[0::2]
        assert_true((m.chisq.data == self.model.chisq.data[:, 0::2]).all())
        assert_true((m.dof.data == self.model.dof.data[:, 0::2]).all())
        assert_true(m.inav[:2][0].A.ext_force_positive ==
                    m[0].A.ext_force_positive)
        assert_true(m.chisq.data.shape == (4, 2))
        assert_false(m[0]._active_array[0, 0])
        for ic, c in enumerate(m):
            np.testing.assert_equal(
                c._active_array,
                self.model[ic]._active_array[:, 0::2])
            for p_new, p_old in zip(c.parameters, self.model[ic].parameters):
                assert_true((p_old.map[:, 0::2] == p_new.map).all())


class TestModelIndexingClass:

    def setUp(self):
        s_eels = EELSSpectrum([list(range(10))] * 3)
        s_eels.metadata.set_item(
            'Acquisition_instrument.TEM.Detector.EELS.collection_angle',
            3.0)
        s_eels.metadata.set_item('Acquisition_instrument.TEM.beam_energy', 1.0)
        s_eels.metadata.set_item(
            'Acquisition_instrument.TEM.convergence_angle',
            2.0)
        self.eels_m = s_eels.create_model(auto_background=False)

    def test_model_class(self):
        m_eels = self.eels_m
        assert_true(isinstance(m_eels, type(m_eels.isig[1:])))
        assert_true(isinstance(m_eels, type(m_eels.inav[1:])))


class TestEELSModelSlicing:

    def setUp(self):
        data = np.random.random((10, 10, 600))
        s = EELSSpectrum(data)
        s.axes_manager[-1].offset = -150.
        s.axes_manager[-1].scale = 0.5
        s.metadata.set_item(
            'Acquisition_instrument.TEM.Detector.EELS.collection_angle',
            3.0)
        s.metadata.set_item('Acquisition_instrument.TEM.beam_energy', 1.0)
        s.metadata.set_item(
            'Acquisition_instrument.TEM.convergence_angle',
            2.0)
        m = s.create_model(
            ll=s + 1,
            auto_background=False,
            auto_add_edges=False)
        g = Gaussian()
        m.append(g)
        self.model = m

    def test_slicing_low_loss_inav(self):
        m = self.model
        m1 = m.inav[::2]
        assert_true(m1.spectrum.data.shape == m1.low_loss.data.shape)

    def test_slicing_low_loss_isig(self):
        m = self.model
        m1 = m.isig[::2]
        assert_true(m.spectrum.data.shape == m1.low_loss.data.shape)
