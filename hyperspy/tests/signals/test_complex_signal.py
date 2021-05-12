# -*- coding: utf-8 -*-
# Copyright 2007-2021 The HyperSpy developers
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
import pytest

import hyperspy.api as hs
from hyperspy.decorators import lazifyTestClass


@lazifyTestClass
class TestComplexProperties:

    real_ref = np.arange(9).reshape((3, 3))
    imag_ref = np.arange(9).reshape((3, 3)) + 9
    comp_ref = real_ref + 1j * imag_ref
    phase_ref = np.angle(comp_ref)
    amplitude_ref = np.abs(comp_ref)

    def setup_method(self, method):
        test = self.real_ref + 1j * self.imag_ref
        self.s = hs.signals.ComplexSignal(test)
        self.s.axes_manager.set_signal_dimension(1)

    def test_get_real(self):
        np.testing.assert_allclose(self.s.real.data, self.real_ref)

    def test_set_real(self):
        real = np.random.random((3, 3))
        self.s.real = real
        np.testing.assert_allclose(self.s.real.data, real)

    def test_get_imag(self):
        np.testing.assert_allclose(self.s.imag.data, self.imag_ref)

    def test_set_imag(self):
        imag = np.random.random((3, 3))
        self.s.imag = imag
        np.testing.assert_allclose(self.s.imag.data, imag)

    def test_get_amplitude(self):
        np.testing.assert_allclose(self.s.amplitude.data, self.amplitude_ref)

    def test_set_amplitude(self):
        amplitude = np.random.random((3, 3))
        self.s.amplitude = amplitude
        np.testing.assert_allclose(self.s.amplitude, amplitude)

    def test_get_phase(self):
        np.testing.assert_allclose(self.s.phase.data, self.phase_ref)

    def test_set_phase(self):
        phase = np.random.random((3, 3))
        self.s.phase = phase
        np.testing.assert_allclose(self.s.phase, phase)

    def test_angle(self):
        np.testing.assert_allclose(self.s.angle(deg=False), self.phase_ref)
        np.testing.assert_allclose(
            self.s.angle(
                deg=True),
            self.phase_ref *
            180 /
            np.pi)


@pytest.mark.parametrize('parallel,lazy', [(True, False),
                                           (False, False),
                                           (False, True)])
def test_get_unwrapped_phase_1D(parallel, lazy):
    phase = 6 * (1 - np.abs(np.indices((9,)) - 4) / 4)
    s = hs.signals.ComplexSignal(np.ones_like(phase) * np.exp(1j * phase))
    if lazy:
        s = s.as_lazy()
    s.axes_manager.set_signal_dimension(1)
    phase_unwrapped = s.unwrapped_phase(seed=42, parallel=parallel)
    assert (
        phase_unwrapped.metadata.General.title ==
        'unwrapped phase(Untitled Signal)')
    np.testing.assert_allclose(phase_unwrapped.data, phase)


@pytest.mark.parametrize('parallel,lazy', [(True, False),
                                           (False, False),
                                           (False, True)])
def test_get_unwrapped_phase_2D(parallel, lazy):
    phase = 5 * (1 - np.abs(np.indices((9, 9)) - 4).sum(axis=0) / 8)
    s = hs.signals.ComplexSignal(np.ones_like(phase) * np.exp(1j * phase))
    if lazy:
        s = s.as_lazy()
    phase_unwrapped = s.unwrapped_phase(seed=42, parallel=parallel)
    assert (
        phase_unwrapped.metadata.General.title ==
        'unwrapped phase(Untitled Signal)')
    np.testing.assert_allclose(phase_unwrapped.data, phase)


@pytest.mark.parametrize('parallel,lazy', [(True, False),
                                           (False, False),
                                           (False, True)])
def test_get_unwrapped_phase_3D(parallel, lazy):
    phase = 4 * (1 - np.abs(np.indices((9, 9, 9)) - 4).sum(axis=0) / 12)
    s = hs.signals.ComplexSignal(np.ones_like(phase) * np.exp(1j * phase))
    if lazy:
        s = s.as_lazy()
    phase_unwrapped = s.unwrapped_phase(seed=42, parallel=parallel)
    assert (
        phase_unwrapped.metadata.General.title ==
        'unwrapped phase(Untitled Signal)')
    np.testing.assert_allclose(phase_unwrapped.data, phase)


def test_argand_diagram():
    # 0. Set up phase and amplitude and real and imaginary parts
    amp = np.random.rand() * 10. * np.random.rand(64)
    phase = np.random.rand(64) * 3. * np.pi
    re = amp * np.cos(phase)
    im = amp * np.sin(phase)

    # 1. Test ComplexSignal1D
    s1d = hs.signals.ComplexSignal((amp * np.exp(1j * phase)))
    s1d.metadata.General.title = 'Test signal'
    s1d.metadata.Signal.quantity = 'Test quantity (Test units)'
    ap1d = s1d.argand_diagram(size=[7, 7])
    ap1_ref = np.histogram2d(re, im, bins=[7, 7])
    np.testing.assert_allclose(ap1d.data, ap1_ref[0].T)
    assert ap1d.metadata.General.title == 'Argand diagram of Test signal'

    # 2. Test ComplexSignal1D with specified range
    s2d = hs.signals.ComplexSignal((amp * np.exp(1j * phase)).reshape((4, 16)))
    s2d.metadata.Signal.quantity = 'Test quantity (Test units)'
    ap2d = s2d.argand_diagram(size=[7, 7], range=[-12., 13.])
    ap2d_a = s2d.argand_diagram(size=[7, 7], range=[[-12., 11.], [-10., 13.]])
    ap2_ref = np.histogram2d(re, im, bins=[7, 7], range=[[-12., 13.], [-12., 13.]])
    ap2_ref_a = np.histogram2d(re, im, bins=[7, 7], range=[[-12., 11.], [-10., 13.]])

    x_axis = ap2d_a.axes_manager.signal_axes[0]
    y_axis = ap2d_a.axes_manager.signal_axes[1]

    np.testing.assert_allclose(ap2d.data, ap2_ref[0].T)
    np.testing.assert_allclose(ap2d_a.data, ap2_ref_a[0].T)

    assert x_axis.offset == -12.
    np.testing.assert_allclose(x_axis.scale, np.gradient(ap2_ref_a[2]))

    assert y_axis.offset == -10.
    np.testing.assert_allclose(y_axis.scale, np.gradient(ap2_ref_a[1]))

    assert x_axis.units == 'Test units'
    assert y_axis.units == 'Test units'

    # 3. Test raises:
    with pytest.raises(ValueError):
        s1d.argand_diagram(range=[-12., 11., -10., 13.])
    with pytest.raises(NotImplementedError):
        s1d = s1d.as_lazy()
        s1d.argand_diagram()
