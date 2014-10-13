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
from nose.tools import assert_true

from hyperspy._signals.spectrum_simulation import SpectrumSimulation
from hyperspy.hspy import create_model
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
        m = create_model(s)
        g = Gaussian()
        g.A.ext_force_positive = True
        g.A.ext_bounded = True
        m.append(g)
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
        assert_true(
            np.allclose(
                m.chisq.data +
                m1.chisq.data,
                self.model.chisq.data))

    def test_model_navigation_indexer_slice(self):
        m = self.model.inav[0::2]
        assert_true((m.chisq.data == self.model.chisq.data[:, 0::2]).all())
        assert_true((m.dof.data == self.model.dof.data[:, 0::2]).all())
        assert_true(m.inav[:2][0].A.ext_force_positive ==
                    m[0].A.ext_force_positive)
        assert_true(m.chisq.data.shape == (4, 2))
        for ic, c in enumerate(m):
            for p_new, p_old in zip(c.parameters, self.model[ic].parameters):
                assert_true((p_old.map[:, 0::2] == p_new.map).all())
