# -*- coding: utf-8 -*-
#  Copyright 2007-2016 The HyperSpy developers
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
from scipy.interpolate import interp2d


class TestCaseHologramImage(object):

    def setUp(self):
        self.img_size = 1024
        fringe_direction = -np.pi/6
        x, y = np.meshgrid(np.linspace(-self.img_size/2, self.img_size/2-1, self.img_size),
                           np.linspace(-self.img_size/2, self.img_size/2-1, self.img_size))
        self.phase_ref = np.pi*x/(self.img_size/2.2)
        self.amp_ref = np.ones((self.img_size, self.img_size))
        holo = 2 * (1 + np.cos(self.phase_ref+5.23/(2*np.pi)*(x*np.cos(fringe_direction) + y*np.sin(fringe_direction))))
        self.holo_image = hs.signals.HologramImage(holo)
        self.ref = 2 * (1 + np.cos(5.23/(2*np.pi)*(x*np.cos(fringe_direction) + y*np.sin(fringe_direction))))
        self.ref_image = hs.signals.HologramImage(self.ref)

    def test_reconstruct_phase(self):
        wave_image = self.holo_image.reconstruct_phase(self.ref)
        rec_param_cc = [self.img_size-wave_image.rec_param[2], self.img_size-wave_image.rec_param[3],
                        self.img_size-wave_image.rec_param[0], self.img_size-wave_image.rec_param[1],
                        wave_image.rec_param[4]]
        wave_image_cc = self.holo_image.reconstruct_phase(self.ref_image, rec_param=rec_param_cc)
        x_start = int(wave_image.rec_param[4]*2/10)
        x_stop = int(wave_image.rec_param[4]*2*9/10)
        wave_crop = wave_image.data[x_start:x_stop, x_start:x_stop]
        wave_cc_crop = wave_image_cc.data[x_start:x_stop, x_start:x_stop]

        nt.assert_allclose(wave_crop, np.conj(wave_cc_crop), rtol=1e-3)  # asserts that waves from different
        # sidebands are complex conjugate; this also tests possibility of reconstruction with given rec_param

        # interpolate reconstructed phase to compare with the input (reference phase):
        interp_x = np.arange(wave_image.rec_param[4]*2)
        phase_interp = interp2d(interp_x, interp_x, wave_image.unwrapped_phase().data, kind='cubic')
        phase_new = phase_interp(np.linspace(0, wave_image.rec_param[4]*2, self.img_size),
                                 np.linspace(0, wave_image.rec_param[4]*2, self.img_size))
        x_start = int(self.img_size/10)
        x_stop = self.img_size-1-int(self.img_size/10)
        phase_new_crop = phase_new[x_start:x_stop, x_start:x_stop]
        phase_ref_crop = self.phase_ref[x_start:x_stop, x_start:x_stop]
        nt.assert_almost_equal(phase_new_crop, phase_ref_crop, decimal=2)


if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)