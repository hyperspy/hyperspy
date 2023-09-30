# -*- coding: utf-8 -*-
# Copyright 2007-2023 The exspy developers
#
# This file is part of exspy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# exspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with exspy. If not, see <https://www.gnu.org/licenses/#GPL>.

from hyperspy.decorators import lazifyTestClass
import numpy as np

import exspy

@lazifyTestClass
class TestLinearEELSFitting:

    def setup_method(self, method):
        ll = exspy.data.EELS_low_loss(navigation_shape=())
        cl = exspy.data.EELS_MnFe(add_powerlaw=False, navigation_shape=())
        m = cl.create_model(auto_background=False)
        m[0].onset_energy.value = 637.
        m_convolved = cl.create_model(auto_background=False, ll=ll)
        m_convolved[0].onset_energy.value = 637.
        self.ll, self.cl = ll, cl
        self.m, self.m_convolved = m, m_convolved

    def test_convolved_and_std_error(self):
        m = self.m_convolved
        m.fit(optimizer='lstsq')
        linear = m.as_signal()
        std_linear = m.p_std
        m.fit(optimizer='lm')
        lm = m.as_signal()
        std_lm = m.p_std
        diff = linear - lm
        np.testing.assert_allclose(diff.data.sum(), 0.0, atol=5E-6)
        np.testing.assert_allclose(std_linear, std_lm)

    def test_nonconvolved(self):
        m = self.m
        m.fit(optimizer='lstsq')
        linear = m.as_signal()
        m.fit(optimizer='lm')
        lm = m.as_signal()
        diff = linear - lm
        np.testing.assert_allclose(diff.data.sum(), 0.0, atol=1E-6)


class TestTwinnedComponents:

    def setup_method(self):
        s = exspy.data.EDS_SEM_TM002()
        m = s.create_model()
        m2 = s.isig[5.:15.].create_model()
        self.m = m
        self.m2 = m2

    def test_fixed_chained_twinned_components(self):
        m = self.m
        m.fit(optimizer="lstsq")
        A = m.as_signal()

        m[4].A.free = False
        m.fit(optimizer="lstsq")
        B = m.as_signal()
        np.testing.assert_allclose(A.data, B.data, rtol=5E-5)

    def test_fit_fixed_twinned_components_and_std(self):
        m = self.m2
        m[1].A.free = False
        m.fit(optimizer='lstsq')
        lstsq_fit = m.as_signal()
        nonlinear_parameters = [p for c in m for p in c.parameters
                                if not p._linear]
        linear_std = [para.std for para in nonlinear_parameters if para.std]

        m.fit()
        nonlinear_fit = m.as_signal()
        nonlinear_std = [para.std for para in nonlinear_parameters if para.std]

        np.testing.assert_allclose(nonlinear_fit.data, lstsq_fit.data)
        np.testing.assert_allclose(nonlinear_std, linear_std)
