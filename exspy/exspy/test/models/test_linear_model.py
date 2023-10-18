# -*- coding: utf-8 -*-
# Copyright 2007-2023 The exSpy developers
#
# This file is part of exSpy.
#
# exSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# exSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with exSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import numpy as np
import pytest
from hyperspy.decorators import lazifyTestClass
from hyperspy._components.gaussian import Gaussian
from hyperspy._components.lorentzian import Lorentzian
import hyperspy.api as hs

import exspy
from exspy.signals import EELSSpectrum



@lazifyTestClass
class TestLinearEELSFitting:

    def setup_method(self, method):
        ll = exspy.data.EELS_low_loss(navigation_shape=())
        cl = exspy.data.EELS_MnFe(add_powerlaw=False, navigation_shape=())
        m = cl.create_model(auto_background=False)
        m[0].onset_energy.value = 637.
        m_convolved = cl.create_model(auto_background=False, low_loss=ll)
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
        np.testing.assert_allclose(diff.data.sum(), 0.0, atol=5E-6)


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

class TestWarningSlowMultifit:

    def setup_method(self, method):
        s = hs.datasets.two_gaussians().inav[0]
        s.set_signal_type("EELS")
        m = s.create_model(auto_background=False, auto_add_edges=False)
        g1 = Gaussian(centre=40)
        g2 = Gaussian(centre=55)
        m.extend([g1, g2])

        # make dummy twinning
        g2.centre.twin = g1.centre
        g2.centre.twin_function_expr = '15 + x'
        g2.A.twin = g1.A
        g2.centre.twin_function_expr = '2 * x'

        m.set_parameters_not_free(only_nonlinear=True)

        self.m = m

    def test_convolved(self):
        m = self.m
        s2 = hs.signals.Signal1D(np.ones(m.signal.data.shape))
        m.low_loss = s2
        m.convolved = True
        with pytest.warns(UserWarning, match="convolution is not supported"):
            m.multifit(optimizer='lstsq')


@pytest.mark.parametrize('multiple_free_parameters', (True, False))
@pytest.mark.parametrize('nav_dim', (0, 1, 2))
def test_expression_convolved(nav_dim, multiple_free_parameters):
    s_ref = EELSSpectrum(np.ones(100))

    # Create signal to convolve
    to_convolve_component = hs.model.components1D.Gaussian(A=100, sigma=5, centre=10)
    to_convolve = EELSSpectrum(to_convolve_component.function(np.arange(100)))
    to_convolve.axes_manager[-1].offset = -to_convolve_component.centre.value

    # Create reference signal from model with convolution
    l_ref = Lorentzian(A=100, centre=20, gamma=4)
    m_ref = s_ref.create_model(auto_add_edges=False, auto_background=False)
    m_ref.append(l_ref)
    m_ref.low_loss = to_convolve
    s = m_ref.as_signal()

    if nav_dim >= 1:
        s = hs.stack([s]*2)
        to_convolve = hs.stack([to_convolve]*2)
    if nav_dim == 2:
        s = hs.stack([s]*3)
        to_convolve = hs.stack([to_convolve]*3)

    m = s.create_model(auto_add_edges=False, auto_background=False)
    l = Lorentzian(centre=20, gamma=4)
    m.append(l)
    assert not m.convolved
    m.low_loss = to_convolve
    assert m.convolved
    m.set_parameters_not_free(only_nonlinear=True)
    with pytest.warns(UserWarning):
        m.multifit(optimizer='lstsq')

    np.testing.assert_allclose(l_ref.A.value, l.A.value)
    np.testing.assert_allclose(l_ref.centre.value, l.centre.value)
    np.testing.assert_allclose(l_ref.gamma.value, l.gamma.value)
    np.testing.assert_allclose(m.as_signal().data, s.data)
    if nav_dim in (1, 2):
        np.testing.assert_allclose(l.A.map['values'].mean(), l_ref.A.value)
        np.testing.assert_allclose(l.centre.map['values'].mean(), l_ref.centre.value)
        np.testing.assert_allclose(l.gamma.map['values'].mean(), l_ref.gamma.value)


@pytest.mark.parametrize("nav_dim", (0, 1, 2))
@pytest.mark.parametrize("convolve", (True, False))
def test_expression_multiple_linear_parameter(nav_dim, convolve):
    """
    This test checks that linear fitting works with convolution with
     - single and multidimensional fit (warning raise)
     - multiple free parameters for the same component (different code path)
    """
    s_ref = EELSSpectrum(np.ones(20))
    p_ref = hs.model.components1D.Polynomial(order=2, a0=25, a1=-50, a2=2.5)

    # Create signal to convolve
    to_convolve_component = Gaussian(A=100, sigma=5, centre=10)
    to_convolve = hs.signals.Signal1D(to_convolve_component.function(np.arange(1000)))
    to_convolve.axes_manager[-1].offset = -to_convolve_component.centre.value

    m_ref = s_ref.create_model(auto_add_edges=False, auto_background=False)
    m_ref.extend([p_ref])
    if convolve:
        m_ref.low_loss = to_convolve
    s = m_ref.as_signal()

    if nav_dim >= 1:
        s = hs.stack([s]*2)
        if convolve:
            to_convolve = hs.stack([to_convolve]*2)
    if nav_dim == 2:
        s = hs.stack([s]*3)
        if convolve:
            to_convolve = hs.stack([to_convolve]*3)

    m = s.create_model(auto_add_edges=False, auto_background=False)
    p = hs.model.components1D.Polynomial(order=2)
    m.append(p)
    assert not m.convolved
    if convolve:
        m.low_loss = to_convolve
        with pytest.warns(UserWarning):
            m.multifit(optimizer='lstsq')
    else:
        m.multifit(optimizer='lstsq')

    np.testing.assert_allclose(p_ref.a0.value, p.a0.value)
    np.testing.assert_allclose(p_ref.a1.value, p.a1.value)
    np.testing.assert_allclose(p_ref.a2.value, p.a2.value)
    np.testing.assert_allclose(m.as_signal().data, s.data)
    if nav_dim >= 1:
        np.testing.assert_allclose(p.a0.map['values'].mean(), p_ref.a0.value)
        np.testing.assert_allclose(p.a1.map['values'].mean(), p_ref.a1.value)
        np.testing.assert_allclose(p.a2.map['values'].mean(), p_ref.a2.value)


@pytest.mark.parametrize('nav_dim', (0, 1, 2))
def test_multiple_linear_parameters_convolution(nav_dim):
    s_ref = EELSSpectrum(np.ones(1000))

    # Create signal to convolve
    to_convolve_component = Gaussian(A=1000, sigma=50, centre=100)
    to_convolve = EELSSpectrum(to_convolve_component.function(np.arange(1000)))
    to_convolve.axes_manager[-1].offset = -to_convolve_component.centre.value

    l_ref1 = Lorentzian(A=100, centre=200, gamma=10)
    l_ref2 = Lorentzian(A=100, centre=600, gamma=20)

    m_ref = s_ref.create_model(auto_add_edges=False, auto_background=False)
    m_ref.extend([l_ref1, l_ref2])
    m_ref.low_loss = to_convolve
    s = m_ref.as_signal()

    if nav_dim >= 1:
        s = hs.stack([s]*2)
        to_convolve = hs.stack([to_convolve]*2)
    if nav_dim == 2:
        s = hs.stack([s]*3)
        to_convolve = hs.stack([to_convolve]*3)

    m = s.create_model(auto_add_edges=False, auto_background=False)
    l1 = Lorentzian(centre=200, gamma=10)
    l2 = Lorentzian(centre=600, gamma=20)
    m.extend([l1, l2])
    assert not m.convolved
    m.low_loss = to_convolve
    assert m.convolved
    m.set_parameters_not_free(only_nonlinear=True)
    with pytest.warns(UserWarning):
        m.multifit(optimizer='lstsq')

    np.testing.assert_allclose(l_ref1.A.value, l1.A.value)
    np.testing.assert_allclose(l_ref1.centre.value, l1.centre.value)
    np.testing.assert_allclose(l_ref1.gamma.value, l1.gamma.value)
    np.testing.assert_allclose(l_ref2.A.value, l2.A.value)
    np.testing.assert_allclose(l_ref2.centre.value, l2.centre.value)
    np.testing.assert_allclose(l_ref2.gamma.value, l2.gamma.value)
    np.testing.assert_allclose(m.as_signal().data, s.data)
    if nav_dim >= 1:
        np.testing.assert_allclose(l1.A.map['values'].mean(), l_ref1.A.value)
        np.testing.assert_allclose(l1.centre.map['values'].mean(), l_ref1.centre.value)
        np.testing.assert_allclose(l1.gamma.map['values'].mean(), l_ref1.gamma.value)
        np.testing.assert_allclose(l2.A.map['values'].mean(), l_ref2.A.value)
        np.testing.assert_allclose(l2.centre.map['values'].mean(), l_ref2.centre.value)
        np.testing.assert_allclose(l2.gamma.map['values'].mean(), l_ref2.gamma.value)
