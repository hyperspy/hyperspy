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

from unittest import mock

import numpy.testing as npt
import numpy as np
from scipy.misc import face, ascent
from scipy.ndimage import fourier_shift
import pytest

import hyperspy.api as hs
from hyperspy.decorators import lazifyTestClass


def _generate_parameters():
    parameters = []
    for normalize_corr in [False, True]:
        for reference in ['current', 'cascade', 'stat']:
            parameters.append([normalize_corr, reference])
    return parameters


@lazifyTestClass
class TestSubPixelAlign:

    def setup_method(self, method):
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

        self.signal = s
        self.shifts = shifts

    def test_align_subpix(self):
        # Align signal
        s = self.signal
        shifts = self.shifts
        s.align2D(shifts=shifts)
        # Compare by broadcasting
        np.testing.assert_allclose(s.data[4], s.data[0], rtol=0.5)

    @pytest.mark.parametrize(("normalize_corr", "reference"),
                             _generate_parameters())
    def test_estimate_subpix(self, normalize_corr, reference):
        s = self.signal
        shifts = s.estimate_shift2D(sub_pixel_factor=200,
                                    normalize_corr=normalize_corr)
        np.testing.assert_allclose(shifts, self.shifts, rtol=0.2, atol=0.2,
                                   verbose=True)

    @pytest.mark.parametrize(("plot"), [True, 'reuse'])
    def test_estimate_subpix_plot(self, mpl_cleanup, plot):
        # To avoid this function plotting many figures and holding the test, we
        # make sure the backend is set to `agg` in case it is set to something
        # else in the testing environment
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')
        s = self.signal
        s.estimate_shift2D(sub_pixel_factor=200, plot=plot)


@lazifyTestClass
class TestAlignTools:

    def setup_method(self, method):
        im = face(gray=True)
        self.ascent_offset = np.array((256, 256))
        s = hs.signals.Signal2D(np.zeros((10, 100, 100)))
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
        for i in range(10):
            slices = self.ascent_offset - zlp_pos[i, ...]
            s.data[i, ...] = im[slices[0]:slices[0] + 100,
                                slices[1]:slices[1] + 100]
        self.signal = s

        # How image should be after successfull alignment
        smin = self.ishifts.min(0)
        smax = self.ishifts.max(0)
        offsets = self.ascent_offset + self.offsets / self.scales - smin
        size = np.array((100, 100)) - (smax - smin)
        self.aligned = im[int(offsets[0]):int(offsets[0] + size[0]),
                          int(offsets[1]):int(offsets[1] + size[1])]

    def test_estimate_shift(self):
        s = self.signal
        shifts = s.estimate_shift2D()
        print(shifts)
        print(self.ishifts)
        assert np.allclose(shifts, self.ishifts)

    def test_align(self):
        # Align signal
        m = mock.Mock()
        s = self.signal
        s.events.data_changed.connect(m.data_changed)
        s.align2D()
        # Compare by broadcasting
        assert np.all(s.data == self.aligned)
        assert m.data_changed.called

    def test_align_expand(self):
        s = self.signal
        s.align2D(expand=True)

        # Check the numbers of NaNs to make sure expansion happened properly
        ds = self.ishifts.max(0) - self.ishifts.min(0)
        Nnan = np.sum(ds) * 100 + np.prod(ds)
        Nnan_data = np.sum(1 * np.isnan(s.data), axis=(1, 2))
        # Due to interpolation, the number of NaNs in the data might
        # be 2 higher (left and right side) than expected
        assert np.all(Nnan_data - Nnan <= 2)

        # Check alignment is correct
        d_al = s.data[:, ds[0]:-ds[0], ds[1]:-ds[1]]
        assert np.all(d_al == self.aligned)


def test_add_ramp():
    s = hs.signals.Signal2D(np.indices((3, 3)).sum(axis=0) + 4)
    s.add_ramp(-1, -1, -4)
    npt.assert_allclose(s.data, 0)


def test_add_ramp_lazy():
    s = hs.signals.Signal2D(np.indices((3, 3)).sum(axis=0) + 4).as_lazy()
    s.add_ramp(-1, -1, -4)
    npt.assert_almost_equal(s.data.compute(), 0)


if __name__ == '__main__':
    import pytest
    pytest.main(__name__)
