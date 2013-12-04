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

from nose.tools import assert_true, assert_equal, raises
import numpy as np

from hyperspy.signals import Spectrum, Image, Signal, FourierTransformSignal

class TestImageFFT():
    def setUp(self):
        im = Image(np.random.random((2, 3, 4, 5)))
        self.signal = im
        
    def test_fft_ifft(self):
        im = self.signal
        
        im_fft = im.fft()
        im_ifft = im_fft.ifft()
        assert_true(isinstance(im_fft, FourierTransformSignal))
        assert_true(isinstance(im_ifft, Signal))
        assert_true(np.allclose(im.data,  im_ifft.data, atol=1e-3))
        
        im_fft = im[0].fft()
        im_ifft = im_fft.ifft()
        assert_true(np.allclose(im[0].data,  im_ifft.data, atol=1e-3))
        
        im_fft = im[0,0].fft()
        im_ifft = im_fft.ifft()
        assert_true(np.allclose(im[0,0].data,  im_ifft.data, atol=1e-3))
        
class TestSpectrumFFT():
    def setUp(self):
        s = Spectrum(np.random.random((2, 3, 4, 5)))
        self.signal = s
        
    def test_fft_ifft(self):
        s = self.signal
        
        s_fft = s.fft()
        s_ifft = s_fft.ifft()
        assert_true(isinstance(s_fft, FourierTransformSignal))
        assert_true(isinstance(s_ifft, Signal))
        assert_true(np.allclose(s.data,  s_ifft.data, atol=1e-3))
        
        s_fft = s[0].fft()
        s_ifft = s_fft.ifft()
        assert_true(np.allclose(s[0].data,  s_ifft.data, atol=1e-3))
        
        s_fft = s[0,0].fft()
        s_ifft = s_fft.ifft()
        assert_true(np.allclose(s[0,0].data,  s_ifft.data, atol=1e-3))
        
        s_fft = s[0,0,0].fft()
        s_ifft = s_fft.ifft()
        assert_true(np.allclose(s[0,0,0].data,  s_ifft.data, atol=1e-3))



         


