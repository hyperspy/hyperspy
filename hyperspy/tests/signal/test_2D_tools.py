# Copyright 2007-2016 The HyperSpy developers
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
# along with HyperSpy. If not, see <http://www.gnu.org/licenses/>.


import numpy as np
from scipy.misc import ascent
from scipy.ndimage import fourier_shift

import hyperspy.api as hs


class TestSubPixelAlign:

    def setUp(self):
        ref_image = ascent()
        center = np.array((256, 256))
        shifts = np.array([(0.0, 0.0), (4.3, 2.13), (1.65, 3.58),
                           (-2.3, 2.9), (5.2, -2.1), (2.7, 2.9),
                           (5.0, 6.8), (-9.1, -9.5), (-9.0, -9.9),
                           (-6.3, -9.2)])
        s = hs.signals.Signal2D(np.zeros((10, 100, 100)))
        for i in range(10):
            # Apply each sup-pixel shift using FFT and InverseFFT
            offset_image = fourier_shift(np.fft.fftn(ref_image), shifts[i])
            offset_image = np.fft.ifftn(offset_image).real

            # Crop central regions of shifted images to avoid wrap around
            s.data[i, ...] = offset_image[center[0]:center[0] + 100,
                                          center[1]:center[1] + 100]

            self.spectrum = s
            self.shifts = shifts

    def test_align_subpix(self):
        # Align signal
        s = self.spectrum
        shifts = self.shifts
        s.align2D(shifts=shifts)
        # Compare by broadcasting
        np.testing.assert_allclose(s.data[4], s.data[0], rtol=1)
