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

from hyperspy.decorators import lazifyTestClass
from hyperspy.signals import (
    BaseSignal,
    ComplexSignal1D,
    ComplexSignal2D,
    Signal1D,
    Signal2D,
)


def test_null_signal():
    rng = np.random.RandomState(123)
    s = BaseSignal(rng.random_sample())
    with pytest.raises(AttributeError):
        s.T.fft()
    with pytest.raises(AttributeError):
        s.T.ifft()


@lazifyTestClass
class TestFFTSignal2D:
    def setup_method(self, method):
        rng = np.random.RandomState(123)
        self.im = Signal2D(rng.random_sample(size=(2, 3, 4, 5)))
        self.im.axes_manager.signal_axes[0].units = "nm"
        self.im.axes_manager.signal_axes[1].units = "nm"
        self.im.axes_manager.signal_axes[0].scale = 10.0
        self.im.axes_manager.signal_axes[1].scale = 10.0

    def test_fft(self):
        im_fft = self.im.fft()
        assert im_fft.axes_manager.signal_axes[0].units == "1 / nm"
        assert im_fft.axes_manager.signal_axes[1].units == "1 / nm"

        np.testing.assert_allclose(im_fft.axes_manager.signal_axes[0].scale, 0.02)
        np.testing.assert_allclose(im_fft.axes_manager.signal_axes[1].scale, 0.025)
        np.testing.assert_allclose(im_fft.axes_manager.signal_axes[0].offset, 0.0)
        np.testing.assert_allclose(im_fft.axes_manager.signal_axes[1].offset, 0.0)

    def test_ifft(self):
        im_fft = self.im.fft()
        im_ifft = im_fft.ifft()
        assert im_ifft.axes_manager.signal_axes[0].units == "nm"
        assert im_ifft.axes_manager.signal_axes[1].units == "nm"
        np.testing.assert_allclose(im_ifft.axes_manager.signal_axes[0].scale, 10.0)
        np.testing.assert_allclose(im_ifft.axes_manager.signal_axes[1].scale, 10.0)
        np.testing.assert_allclose(im_ifft.axes_manager.signal_axes[0].offset, 0.0)
        np.testing.assert_allclose(im_ifft.axes_manager.signal_axes[1].offset, 0.0)

        assert im_fft.metadata.Signal.FFT.shifted is False
        assert im_ifft.metadata.has_item("Signal.FFT") is False

        assert isinstance(im_fft, ComplexSignal2D)
        assert isinstance(im_ifft, Signal2D)

        np.testing.assert_allclose(self.im.data, im_ifft.data)

    def test_ifft_nav_0(self):
        im_fft = self.im.inav[0].fft()
        im_ifft = im_fft.ifft()
        np.testing.assert_allclose(self.im.inav[0].data, im_ifft.data)

    def test_ifft_nav_0_0(self):
        im_fft = self.im.inav[0, 0].fft()
        im_ifft = im_fft.ifft()
        np.testing.assert_allclose(self.im.inav[0, 0].data, im_ifft.data)
        np.testing.assert_allclose(im_fft.data, np.fft.fft2(self.im.inav[0, 0]).data)

    def test_fft_shift(self):
        im_fft = self.im.inav[0, 0].fft(shift=True)
        axis = im_fft.axes_manager.signal_axes[0]
        np.testing.assert_allclose(axis.offset, -axis.high_value)

        im_ifft = im_fft.ifft()
        np.testing.assert_allclose(im_ifft.axes_manager.signal_axes[0].offset, 0.0)

        assert im_fft.metadata.Signal.FFT.shifted is True
        assert im_ifft.metadata.has_item("Signal.FFT") is False

        np.testing.assert_allclose(self.im.inav[0, 0].data, im_ifft.data)
        np.testing.assert_allclose(
            im_fft.data, np.fft.fftshift(np.fft.fft2(self.im.inav[0, 0]).data)
        )

    def test_apodization_no_window(self):
        assert self.im.fft(apodization=True) == self.im.apply_apodization().fft()

    @pytest.mark.parametrize("apodization", ["hann", "hamming", "tukey"])
    def test_apodization(self, apodization):
        assert (
            self.im.fft(apodization=apodization)
            == self.im.apply_apodization(window=apodization).fft()
        )


@lazifyTestClass
class TestFFTSignal1D:
    def setup_method(self, method):
        rng = np.random.RandomState(123)
        self.s = Signal1D(rng.random_sample(size=(2, 3, 4, 5)))
        self.s.axes_manager.signal_axes[0].scale = 6.0

    def test_fft(self):
        s_fft = self.s.fft()
        s_fft.axes_manager.signal_axes[0].units = "mrad"
        assert s_fft.axes_manager.signal_axes[0].scale == 1.0 / 5.0 / 6.0

    def test_ifft(self):
        s_fft = self.s.fft()
        s_fft.axes_manager.signal_axes[0].units = "mrad"

        s_ifft = s_fft.ifft()
        assert s_ifft.axes_manager.signal_axes[0].units == "1 / mrad"
        np.testing.assert_allclose(s_ifft.axes_manager.signal_axes[0].scale, 6.0)

        assert isinstance(s_fft, ComplexSignal1D)
        assert isinstance(s_ifft, Signal1D)

        np.testing.assert_allclose(self.s.data, s_ifft.data)

    def test_ifft_nav_0(self):
        s_fft = self.s.inav[0].fft()
        s_ifft = s_fft.ifft()
        np.testing.assert_allclose(self.s.inav[0].data, s_ifft.data)

    def test_ifft_nav_0_0(self):
        s_fft = self.s.inav[0, 0].fft()
        s_ifft = s_fft.ifft()
        np.testing.assert_allclose(self.s.inav[0, 0].data, s_ifft.data)

    def test_ifft_nav_0_0_0(self):
        s_fft = self.s.inav[0, 0, 0].fft()
        s_ifft = s_fft.ifft()
        np.testing.assert_allclose(self.s.inav[0, 0, 0].data, s_ifft.data)
        np.testing.assert_allclose(np.fft.fft(self.s.inav[0, 0, 0].data), s_fft.data)

    def test_fft_shift(self):
        s_fft = self.s.inav[0, 0, 0].fft(shift=True)
        s_ifft = s_fft.ifft(shift=True)
        np.testing.assert_allclose(self.s.inav[0, 0, 0].data, s_ifft.data)
        np.testing.assert_allclose(
            np.fft.fftshift(np.fft.fft(self.s.inav[0, 0, 0].data)), s_fft.data
        )

    def test_apodization_no_window(self):
        assert self.s.fft(apodization=True) == self.s.apply_apodization().fft()

    @pytest.mark.parametrize("apodization", ["hann", "hamming", "tukey"])
    def test_apodization(self, apodization):
        assert (
            self.s.fft(apodization=apodization)
            == self.s.apply_apodization(window=apodization).fft()
        )
