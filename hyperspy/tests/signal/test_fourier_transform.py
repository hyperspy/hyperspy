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

from hyperspy.signals import Signal1D, Signal2D, ComplexSignal1D, ComplexSignal2D


@pytest.mark.parametrize('lazy', [True, False])
def test_fft_signal2d(lazy):
    im = Signal2D(np.random.random((2, 3, 4, 5)))
    if lazy:
        im = im.as_lazy()
    im_fft = im.fft()
    im_ifft = im_fft.ifft()
    assert isinstance(im_fft, ComplexSignal2D)
    assert isinstance(im_ifft, Signal2D)
    assert_allclose(im.data,  im_ifft.data, atol=1e-3)

    im_fft = im.inav[0].fft()
    im_ifft = im_fft.ifft()
    assert_allclose(im.inav[0].data,  im_ifft.data, atol=1e-3)

    im_fft = im.inav[0, 0].fft()
    im_ifft = im_fft.ifft()
    assert_allclose(im.inav[0, 0].data,  im_ifft.data, atol=1e-3)
    assert_allclose(im_fft.data, np.fft.fftshift(np.fft.fft2(im.inav[0, 0]).data))

    im_fft = im.inav[0, 0].fft(shifted=False)
    im_ifft = im_fft.ifft(shifted=False)
    assert_allclose(im.inav[0, 0].data, im_ifft.data, atol=1e-3)
    assert_allclose(im_fft.data, np.fft.fft2(im.inav[0, 0]).data)


@pytest.mark.parametrize('lazy', [True, False])
def test_fft_signal1d(lazy):
    s = Signal1D(np.random.random((2, 3, 4, 5)))
    if lazy:
        s = s.as_lazy()

    s_fft = s.fft()
    s_ifft = s_fft.ifft()
    assert isinstance(s_fft, ComplexSignal1D)
    assert isinstance(s_ifft, Signal1D)
    assert_allclose(s.data,  s_ifft.data, atol=1e-3)

    s_fft = s.inav[0].fft()
    s_ifft = s_fft.ifft()
    assert_allclose(s.inav[0].data,  s_ifft.data, atol=1e-3)

    s_fft = s.inav[0, 0].fft()
    s_ifft = s_fft.ifft()
    assert_allclose(s.inav[0, 0].data,  s_ifft.data, atol=1e-3)

    s_fft = s.inav[0, 0, 0].fft()
    s_ifft = s_fft.ifft()
    assert_allclose(s.inav[0, 0, 0].data,  s_ifft.data, atol=1e-3)
    assert_allclose(np.fft.fftshift(np.fft.fft(s.inav[0, 0, 0].data)), s_fft.data)

    s_fft = s.inav[0, 0, 0].fft(shifted=False)
    s_ifft = s_fft.ifft(shifted=False)
    assert_allclose(s.inav[0, 0, 0].data, s_ifft.data, atol=1e-3)
    assert_allclose(np.fft.fft(s.inav[0, 0, 0].data), s_fft.data)
