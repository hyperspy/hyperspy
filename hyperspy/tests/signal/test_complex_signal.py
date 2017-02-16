# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
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
import numpy.testing as nt
from numpy.testing import assert_allclose
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
        assert_allclose(self.s.real.data, self.real_ref)

    def test_set_real(self):
        real = np.random.random((3, 3))
        self.s.real = real
        assert_allclose(self.s.real.data, real)

    def test_get_imag(self):
        assert_allclose(self.s.imag.data, self.imag_ref)

    def test_set_imag(self):
        imag = np.random.random((3, 3))
        self.s.imag = imag
        assert_allclose(self.s.imag.data, imag)

    def test_get_amplitude(self):
        assert_allclose(self.s.amplitude.data, self.amplitude_ref)

    def test_set_amplitude(self):
        amplitude = np.random.random((3, 3))
        self.s.amplitude = amplitude
        assert_allclose(self.s.amplitude, amplitude)

    def test_get_phase(self):
        assert_allclose(self.s.phase.data, self.phase_ref)

    def test_set_phase(self):
        phase = np.random.random((3, 3))
        self.s.phase = phase
        assert_allclose(self.s.phase, phase)

    def test_angle(self):
        assert_allclose(self.s.angle(deg=False), self.phase_ref)
        assert_allclose(
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
    phase_unwrapped = s.unwrapped_phase(seed=42, show_progressbar=False,
                                        parallel=parallel)
    assert (
        phase_unwrapped.metadata.General.title ==
        'unwrapped phase(Untitled Signal)')
    assert_allclose(phase_unwrapped.data, phase)


@pytest.mark.parametrize('parallel,lazy', [(True, False),
                                           (False, False),
                                           (False, True)])
def test_get_unwrapped_phase_2D(parallel, lazy):
    phase = 5 * (1 - np.abs(np.indices((9, 9)) - 4).sum(axis=0) / 8)
    s = hs.signals.ComplexSignal(np.ones_like(phase) * np.exp(1j * phase))
    if lazy:
        s = s.as_lazy()
    phase_unwrapped = s.unwrapped_phase(seed=42, show_progressbar=False,
                                        parallel=parallel)
    assert (
        phase_unwrapped.metadata.General.title ==
        'unwrapped phase(Untitled Signal)')
    assert_allclose(phase_unwrapped.data, phase)


@pytest.mark.parametrize('parallel,lazy', [(True, False),
                                           (False, False),
                                           (False, True)])
def test_get_unwrapped_phase_3D(parallel, lazy):
    phase = 4 * (1 - np.abs(np.indices((9, 9, 9)) - 4).sum(axis=0) / 12)
    s = hs.signals.ComplexSignal(np.ones_like(phase) * np.exp(1j * phase))
    if lazy:
        s = s.as_lazy()
    phase_unwrapped = s.unwrapped_phase(seed=42, show_progressbar=False,
                                        parallel=parallel)
    assert (
        phase_unwrapped.metadata.General.title ==
        'unwrapped phase(Untitled Signal)')
    assert_allclose(phase_unwrapped.data, phase)


if __name__ == '__main__':

    import pytest
    pytest.main(__name__)
