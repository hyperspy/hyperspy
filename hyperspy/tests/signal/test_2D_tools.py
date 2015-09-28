# Copyright 2007-2015 The HyperSpy developers
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
import nose.tools as nt
from scipy.misc import lena

import hyperspy.hspy as hs


class TestAlignTools:

    def setUp(self):
        im = lena()
        self.lena_offset = np.array((256, 256))
        s = hs.signals.Image(np.zeros((10, 100, 100)))
        self.scales = np.array((0.1, 0.3))
        self.offsets = np.array((-2, -3))
        izlp = []
        for ax, offset, scale in zip(
                s.axes_manager.signal_axes, self.offsets, self.scales):
            ax.scale = scale
            ax.offset = offset
            izlp.append(ax.value2index(0))
        self.izlp = izlp
        self.ishifts = np.array([(0, 0), (4, 2), (1, 3), (-2, 2), (5, -2),
                                 (2, 2), (5, 6), (-9, -9), (-9, -9), (-6, -9)])
        self.new_offsets = self.offsets - self.ishifts.min(0) * self.scales
        zlp_pos = self.ishifts + self.izlp
        for i in xrange(10):
            slices = self.lena_offset - zlp_pos[i, ...]
            s.data[i, ...] = im[slices[0]:slices[0] + 100,
                                slices[1]:slices[1] + 100]
        self.spectrum = s

        # How image should be after successfull alignment
        smin = self.ishifts.min(0)
        smax = self.ishifts.max(0)
        offsets = self.lena_offset + self.offsets / self.scales - smin
        size = np.array((100, 100)) - (smax - smin)
        self.aligned = im[offsets[0]:offsets[0] + size[0],
                          offsets[1]:offsets[1] + size[1]]

    def test_estimate_shift(self):
        s = self.spectrum
        shifts = s.estimate_shift2D()
        print shifts
        print self.ishifts
        nt.assert_true(np.allclose(shifts, self.ishifts))

    def test_align(self):
        # Align signal
        s = self.spectrum
        s.align2D()
        # Compare by broadcasting
        nt.assert_true(np.all(s.data == self.aligned))

    def test_align_expand(self):
        s = self.spectrum
        s.align2D(expand=True)

        # Check the numbers of NaNs to make sure expansion happened properly
        ds = self.ishifts.max(0) - self.ishifts.min(0)
        Nnan = np.sum(ds) * 100 + np.prod(ds)
        Nnan_data = np.sum(1*np.isnan(s.data), axis=(1, 2))
        # Due to interpolation, the number of NaNs in the data might
        # be 2 higher (left and right side) than expected
        nt.assert_true(np.all(Nnan_data - Nnan <= 2))

        # Check alignment is correct
        d_al = s.data[:, ds[0]:-ds[0], ds[1]:-ds[1]]
        nt.assert_true(np.all(d_al == self.aligned))
