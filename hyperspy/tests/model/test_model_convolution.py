# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
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

import hyperspy.api as hs
from hyperspy.misc.axis_tools import calculate_convolution1D_axis
from hyperspy.misc.utils import dummy_context_manager
from hyperspy.models.model1d import Model1D


class ConvolvedModel1D(Model1D):
    # Copy and paste example from the gallery in the test suite
    # to measure coverage of the code

    def __init__(self, signal1D, detector_response=None, **kwargs):
        super().__init__(signal1D, **kwargs)
        self._convolved = False
        self._detector_response = None
        self._convolution_axis = None
        self.detector_response = detector_response
        self._whitelist.update(
            {
                "_convolved": None,
                "detector_response": ("sig", None),
            }
        )

    def _set_convolution_axis(self):
        """
        Set the convolution axis used to add padding before taking
        the convolution.
        """
        # Used during model fitting
        self._convolution_axis = calculate_convolution1D_axis(
            self.signal.axes_manager.signal_axes[0],
            self.detector_response.axes_manager.signal_axes[0],
        )

    @property
    def detector_response(self):
        return self._detector_response

    @detector_response.setter
    def detector_response(self, signal):
        if signal is not None:
            self._detector_response = signal
            self._set_convolution_axis()
            self._convolved = True
        else:
            self._detector_response = None
            self._convolution_axis = None
            self._convolved = False

    @property
    def _signal_to_convolve(self):
        # Used during model fitting
        return self.detector_response

    @property
    def convolved(self):
        # Used during model fitting
        return self._convolved


@pytest.mark.parametrize("nav_dim", (0, 1))
@pytest.mark.parametrize("optimizer", ("lm", "lstsq"))
def test_convolved_model(nav_dim, optimizer):
    # example of detector response
    g = hs.model.components1D.Gaussian(sigma=3)
    g_signal = hs.signals.Signal1D(g.function(np.arange(-20, 20)))
    g_signal.axes_manager.signal_axes.set(offset=-20)

    # example signal
    f = hs.model.components1D.Lorentzian(A=2.5, centre=220)
    f_signal = hs.signals.Signal1D(f.function(np.arange(200, 300)))
    f_signal.axes_manager.signal_axes.set(offset=200)
    convolution_axis = calculate_convolution1D_axis(
        f_signal.axes_manager.signal_axes[0], g_signal.axes_manager.signal_axes[0]
    )
    f_padded_data = f.function(convolution_axis)
    f_signal.data = np.convolve(f_padded_data, g_signal.data, mode="valid") + 10

    if nav_dim == 1:
        g_signal = hs.stack([g_signal] * 2)
        f_signal = hs.stack([f_signal] * 2)

    m = ConvolvedModel1D(f_signal, detector_response=g_signal)
    lorentzian_component = hs.model.components1D.Lorentzian()
    lorentzian_component.estimate_parameters(f_signal, 210, 230)
    offset_component = hs.model.components1D.Offset()
    m.extend([lorentzian_component, offset_component])

    lorentzian_component.convolved = True
    offset_component.convolved = False
    if optimizer == "lstsq":
        lorentzian_component.centre.value = 220
        lorentzian_component.gamma.value = 1
        m.assign_current_values_to_all()
        m.set_parameters_not_free(only_nonlinear=True)
        cm = pytest.warns
    else:
        cm = dummy_context_manager
    with cm():
        m.multifit(optimizer=optimizer)
    np.testing.assert_allclose(lorentzian_component.A.value, 2.5, rtol=2e-3)
    np.testing.assert_allclose(lorentzian_component.centre.value, 220, rtol=1e-5)
    np.testing.assert_allclose(lorentzian_component.gamma.value, 1, rtol=5e-3)

    s = m.as_signal()
    np.testing.assert_allclose(s.data, f_signal.data, rtol=1e-5)


@pytest.mark.parametrize("nav_dim", (0, 1))
@pytest.mark.parametrize("optimizer", ("lm", "lstsq"))
def test_convolved_model_polynomial(nav_dim, optimizer):
    # Test with polynomial to cover the `Expression._compute_expression_part` code
    # which is used when a components has several free linear parameters
    # example of detector response
    g = hs.model.components1D.Gaussian(sigma=3)
    g_signal = hs.signals.Signal1D(g.function(np.arange(-20, 20)))
    g_signal.axes_manager.signal_axes.set(offset=-20)

    # example signal
    polynomial_values = dict(a0=0.5, a1=0.1, a2=-0.05)
    f = hs.model.components1D.Polynomial(order=2, **polynomial_values)
    f_signal = hs.signals.Signal1D(f.function(np.arange(200, 300)))
    f_signal.axes_manager.signal_axes.set(offset=200)
    convolution_axis = calculate_convolution1D_axis(
        f_signal.axes_manager.signal_axes[0], g_signal.axes_manager.signal_axes[0]
    )
    f_padded_data = f.function(convolution_axis)
    f_signal.data = np.convolve(f_padded_data, g_signal.data, mode="valid")

    polynomial = hs.model.components1D.Polynomial(order=2)
    f_signal.data += polynomial.function(f_signal.axes_manager[-1].axis)

    lorentzian = hs.model.components1D.Lorentzian(centre=260, A=1e5)
    f_signal.data += lorentzian.function(f_signal.axes_manager[-1].axis)

    if nav_dim == 1:
        g_signal = hs.stack([g_signal] * 2)
        f_signal = hs.stack([f_signal] * 2)

    m = ConvolvedModel1D(f_signal, detector_response=g_signal)
    polynomial_component = hs.model.components1D.Polynomial(order=2)
    lorentzian_component = hs.model.components1D.Lorentzian()
    lorentzian_component.estimate_parameters(f_signal, 250, 270)
    m.extend([polynomial_component, lorentzian_component])

    polynomial_component.convolved = True
    lorentzian_component.convolved = False
    if optimizer == "lstsq":
        lorentzian_component.centre.value = 260
        lorentzian_component.gamma.value = 1
        m.assign_current_values_to_all()
        m.set_parameters_not_free(only_nonlinear=True)
        cm = pytest.warns
    else:
        cm = dummy_context_manager
    with cm():
        m.multifit(optimizer=optimizer)
    np.testing.assert_allclose(lorentzian_component.A.value, 1e5, rtol=2e-3)
    np.testing.assert_allclose(lorentzian_component.centre.value, 260, rtol=1e-5)
    np.testing.assert_allclose(lorentzian_component.gamma.value, 1, rtol=5e-3)
    np.testing.assert_allclose(
        polynomial_component.a0.value, polynomial_values["a0"], rtol=2e-3
    )
    np.testing.assert_allclose(
        polynomial_component.a1.value, polynomial_values["a1"], rtol=1e-5
    )
    np.testing.assert_allclose(
        polynomial_component.a2.value, polynomial_values["a2"], rtol=5e-3
    )

    s = m.as_signal()
    np.testing.assert_allclose(s.data, f_signal.data, rtol=1e-5)


def test_convolve_jacobian():
    g = hs.model.components1D.Gaussian(sigma=3)
    g_signal = hs.signals.Signal1D(g.function(np.arange(-20, 20)))
    g_signal.axes_manager.signal_axes.set(offset=-20)

    # example signal
    f = hs.model.components1D.Lorentzian(A=2.5, centre=220)
    f_signal = hs.signals.Signal1D(f.function(np.arange(200, 300)))
    f_signal.axes_manager.signal_axes.set(offset=200)
    convolution_axis = calculate_convolution1D_axis(
        f_signal.axes_manager.signal_axes[0], g_signal.axes_manager.signal_axes[0]
    )
    f_padded_data = f.function(convolution_axis)
    f_signal.data = np.convolve(f_padded_data, g_signal.data, mode="valid")

    m = ConvolvedModel1D(f_signal, detector_response=g_signal)
    lorentzian_component = hs.model.components1D.Lorentzian()
    lorentzian_component.convolved = True
    lorentzian_component.estimate_parameters(f_signal, 210, 230)
    m.extend([lorentzian_component])

    m.fit(optimizer="lm", grad="analytical")
    np.testing.assert_allclose(lorentzian_component.A.value, 2.5)
    np.testing.assert_allclose(lorentzian_component.centre.value, 220)
    np.testing.assert_allclose(lorentzian_component.gamma.value, 1)

    s = m.as_signal()
    np.testing.assert_allclose(s.data, f_signal.data)


def test_convolve_jacobian_twin():
    g = hs.model.components1D.Gaussian(sigma=3)
    g_signal = hs.signals.Signal1D(g.function(np.arange(-20, 20)))
    g_signal.axes_manager.signal_axes.set(offset=-20)

    # example signal
    f = hs.model.components1D.Lorentzian(A=2.5, centre=220)
    f_signal = hs.signals.Signal1D(f.function(np.arange(200, 300)))
    f_signal.axes_manager.signal_axes.set(offset=200)

    convolution_axis = calculate_convolution1D_axis(
        f_signal.axes_manager.signal_axes[0], g_signal.axes_manager.signal_axes[0]
    )

    f_padded_data = f.function(convolution_axis)
    f_signal.data = np.convolve(f_padded_data, g_signal.data, mode="valid")

    f_twinned = hs.model.components1D.Lorentzian(centre=260)
    f_twinned.A.twin = f.A
    f_twinned_padded_data = f_twinned.function(convolution_axis)
    f_signal.data += np.convolve(f_twinned_padded_data, g_signal.data, mode="valid")

    m = ConvolvedModel1D(f_signal, detector_response=g_signal)
    lorentzian_component = hs.model.components1D.Lorentzian()
    lorentzian_component.convolved = True
    lorentzian_component.estimate_parameters(f_signal, 210, 230)
    lorentzian2_component = hs.model.components1D.Lorentzian()
    lorentzian2_component.convolved = True
    lorentzian2_component.A.twin = lorentzian_component.A
    lorentzian2_component.estimate_parameters(f_signal, 250, 270)
    m.extend([lorentzian_component, lorentzian2_component])

    m.fit(optimizer="lm", grad="analytical")
    np.testing.assert_allclose(lorentzian_component.A.value, 2.5)
    np.testing.assert_allclose(lorentzian_component.centre.value, 220)
    np.testing.assert_allclose(lorentzian_component.gamma.value, 1)

    np.testing.assert_allclose(lorentzian2_component.A.value, 2.5)
    np.testing.assert_allclose(lorentzian2_component.centre.value, 260)
    np.testing.assert_allclose(lorentzian2_component.gamma.value, 1)

    s = m.as_signal()
    np.testing.assert_allclose(s.data, f_signal.data)
