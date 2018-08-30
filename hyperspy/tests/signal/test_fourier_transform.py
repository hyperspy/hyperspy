# Copyright 2007-2016 The Hyperspy developers
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

import pytest
import numpy as np
from numpy.testing import assert_allclose

from hyperspy.signals import Signal1D, Signal2D, ComplexSignal1D, ComplexSignal2D, BaseSignal


@pytest.mark.parametrize('lazy', [True, False])
def test_fft_signal2d(lazy):
    im = Signal2D(np.random.random((2, 3, 4, 5)))
    if lazy:
        im = im.as_lazy()
    im.axes_manager.signal_axes[0].units = 'nm'
    im.axes_manager.signal_axes[1].units = 'nm'
    im.axes_manager.signal_axes[0].scale = 10.
    im.axes_manager.signal_axes[1].scale = 10.

    im_fft = im.fft()
    assert im_fft.axes_manager.signal_axes[0].units == '1 / nm'
    assert im_fft.axes_manager.signal_axes[1].units == '1 / nm'
    assert im_fft.axes_manager.signal_axes[0].scale == 1. / 5. / 10.
    assert im_fft.axes_manager.signal_axes[1].scale == 1. / 4. / 10.
    assert im_fft.axes_manager.signal_axes[0].offset == 0.
    assert im_fft.axes_manager.signal_axes[1].offset == 0.

    im_ifft = im_fft.ifft()
    assert im_ifft.axes_manager.signal_axes[0].units == 'nm'
    assert im_ifft.axes_manager.signal_axes[1].units == 'nm'
    assert im_ifft.axes_manager.signal_axes[0].scale == 10.
    assert im_ifft.axes_manager.signal_axes[1].scale == 10.
    assert im_ifft.axes_manager.signal_axes[0].offset == 0.
    assert im_ifft.axes_manager.signal_axes[1].offset == 0.

    assert im_fft.metadata.Signal.FFT.shifted is False
    assert im_ifft.metadata.has_item('Signal.FFT') is False

    assert isinstance(im_fft, ComplexSignal2D)
    assert isinstance(im_ifft, Signal2D)

    assert_allclose(im.data, im_ifft.data, atol=1e-3)

    im_fft = im.inav[0].fft()
    im_ifft = im_fft.ifft()
    assert_allclose(im.inav[0].data, im_ifft.data, atol=1e-3)

    im_fft = im.inav[0, 0].fft()
    im_ifft = im_fft.ifft()
    assert_allclose(im.inav[0, 0].data, im_ifft.data, atol=1e-3)
    assert_allclose(im_fft.data, np.fft.fft2(im.inav[0, 0]).data)

    im_fft = im.inav[0, 0].fft(shift=True)
    axis = im_fft.axes_manager.signal_axes[0]
    assert axis.offset == -axis.high_value

    im_ifft = im_fft.ifft()
    assert im_ifft.axes_manager.signal_axes[0].offset == 0.

    assert im_fft.metadata.Signal.FFT.shifted is True
    assert im_ifft.metadata.has_item('Signal.FFT') is False

    assert_allclose(im.inav[0, 0].data, im_ifft.data, atol=1e-3)
    assert_allclose(im_fft.data, np.fft.fftshift(
        np.fft.fft2(im.inav[0, 0]).data))


@pytest.mark.parametrize('lazy', [True, False])
def test_fft_signal1d(lazy):
    s = Signal1D(np.random.random((2, 3, 4, 5)))
    if lazy:
        s = s.as_lazy()

    s.axes_manager.signal_axes[0].scale = 6.

    s_fft = s.fft()
    s_fft.axes_manager.signal_axes[0].units = 'mrad'

    assert s_fft.axes_manager.signal_axes[0].scale == 1. / 5. / 6.

    s_ifft = s_fft.ifft()
    assert s_ifft.axes_manager.signal_axes[0].units == '1 / mrad'
    assert s_ifft.axes_manager.signal_axes[0].scale == 6.
    assert isinstance(s_fft, ComplexSignal1D)
    assert isinstance(s_ifft, Signal1D)
    assert_allclose(s.data, s_ifft.data, atol=1e-3)

    s_fft = s.inav[0].fft()
    s_ifft = s_fft.ifft()
    assert_allclose(s.inav[0].data, s_ifft.data, atol=1e-3)

    s_fft = s.inav[0, 0].fft()
    s_ifft = s_fft.ifft()
    assert_allclose(s.inav[0, 0].data, s_ifft.data, atol=1e-3)

    s_fft = s.inav[0, 0, 0].fft()
    s_ifft = s_fft.ifft()
    assert_allclose(s.inav[0, 0, 0].data, s_ifft.data, atol=1e-3)
    assert_allclose(np.fft.fft(s.inav[0, 0, 0].data), s_fft.data)

    s_fft = s.inav[0, 0, 0].fft(shift=True)
    s_ifft = s_fft.ifft(shift=True)
    assert_allclose(s.inav[0, 0, 0].data, s_ifft.data, atol=1e-3)
    assert_allclose(np.fft.fftshift(
        np.fft.fft(s.inav[0, 0, 0].data)), s_fft.data)


def test_nul_signal():
    s = BaseSignal(np.random.random())
    with pytest.raises(AttributeError):
        s.T.fft()
    with pytest.raises(AttributeError):
        s.T.ifft()
