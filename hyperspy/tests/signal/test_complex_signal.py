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

import hyperspy.api as hs


class TestComplexProperties:

    real_ref = np.arange(9).reshape((3, 3))
    imag_ref = np.arange(9).reshape((3, 3)) + 9
    comp_ref = real_ref + 1j * imag_ref
    phase_ref = np.angle(comp_ref)
    amplitude_ref = np.abs(comp_ref)

    def setUp(self):
        test = self.real_ref + 1j * self.imag_ref
        self.s = hs.signals.ComplexSignal(test)
        self.s.axes_manager.set_signal_dimension(1)

    def test_get_real(self):
        nt.assert_almost_equal(self.s.real.data, self.real_ref)

    def test_set_real(self):
        real = np.random.random((3, 3))
        self.s.real = real
        nt.assert_almost_equal(self.s.real.data, real)

    def test_get_imag(self):
        nt.assert_almost_equal(self.s.imag.data, self.imag_ref)

    def test_set_imag(self):
        imag = np.random.random((3, 3))
        self.s.imag = imag
        nt.assert_almost_equal(self.s.imag.data, imag)

    def test_get_amplitude(self):
        nt.assert_almost_equal(self.s.amplitude.data, self.amplitude_ref)

    def test_set_amplitude(self):
        amplitude = np.random.random((3, 3))
        self.s.amplitude = amplitude
        nt.assert_almost_equal(self.s.amplitude, amplitude)

    def test_get_phase(self):
        nt.assert_almost_equal(self.s.phase.data, self.phase_ref)

    def test_set_phase(self):
        phase = np.random.random((3, 3))
        self.s.phase = phase
        nt.assert_almost_equal(self.s.phase, phase)

    def test_angle(self):
        nt.assert_almost_equal(self.s.angle(deg=False), self.phase_ref)
        nt.assert_almost_equal(
            self.s.angle(
                deg=True),
            self.phase_ref *
            180 /
            np.pi)


def test_get_unwrapped_phase_1D():
    phase = 6 * (1 - np.abs(np.indices((9,)) - 4) / 4)
    s = hs.signals.ComplexSignal(np.ones_like(phase) * np.exp(1j * phase))
    s.axes_manager.set_signal_dimension(1)
    phase_unwrapped = s.unwrapped_phase(seed=42, show_progressbar=False)
    nt.assert_equal(
        phase_unwrapped.metadata.General.title,
        'unwrapped phase(Untitled Signal)')
    nt.assert_almost_equal(phase_unwrapped.data, phase)


def test_get_unwrapped_phase_2D():
    phase = 5 * (1 - np.abs(np.indices((9, 9)) - 4).sum(axis=0) / 8)
    s = hs.signals.ComplexSignal(np.ones_like(phase) * np.exp(1j * phase))
    phase_unwrapped = s.unwrapped_phase(seed=42, show_progressbar=False)
    nt.assert_equal(
        phase_unwrapped.metadata.General.title,
        'unwrapped phase(Untitled Signal)')
    nt.assert_almost_equal(phase_unwrapped.data, phase)


def test_get_unwrapped_phase_3D():
    phase = 4 * (1 - np.abs(np.indices((9, 9, 9)) - 4).sum(axis=0) / 12)
    s = hs.signals.ComplexSignal(np.ones_like(phase) * np.exp(1j * phase))
    phase_unwrapped = s.unwrapped_phase(seed=42, show_progressbar=False)
    nt.assert_equal(
        phase_unwrapped.metadata.General.title,
        'unwrapped phase(Untitled Signal)')
    nt.assert_almost_equal(phase_unwrapped.data, phase)


if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)
