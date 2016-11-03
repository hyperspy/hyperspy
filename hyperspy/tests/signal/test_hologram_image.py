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
        x2, z2, y2 = np.meshgrid(np.linspace(-self.img_size/2, self.img_size/2-1, self.img_size),
                                 np.array([0, 1]),
                                 np.linspace(-self.img_size/2, self.img_size/2-1, self.img_size))
        self.phase_ref = np.pi*x/(self.img_size/2.2)
        # self.phase_ref2 = np.pi*(x2+self.img_size/2)**z2/(self.img_size/2.2)
        self.phase_ref2 = np.pi*x2*(1-z2)/(self.img_size/2.2) + np.pi*y2*z2/(self.img_size/2.2)
        holo = 2 * (1 + np.cos(self.phase_ref+5.23/(2*np.pi)*(x*np.cos(fringe_direction) + y*np.sin(fringe_direction))))
        holo2 = 2 * (1 + np.cos(self.phase_ref2+5.23/(2*np.pi)*(x2*np.cos(fringe_direction) + y2*np.sin(fringe_direction))))
        self.holo_image = hs.signals.HologramImage(holo)
        self.holo_image2 = hs.signals.HologramImage(holo2)
        self.ref = 2 * (1 + np.cos(5.23/(2*np.pi)*(x*np.cos(fringe_direction) + y*np.sin(fringe_direction))))
        self.ref2 = 2 * (1 + np.cos(5.23/(2*np.pi)*(x2*np.cos(fringe_direction) + y2*np.sin(fringe_direction))))
        self.ref_image = hs.signals.HologramImage(self.ref)
        self.ref_image2 = hs.signals.HologramImage(self.ref2)

    def test_reconstruct_phase(self):

        # Testing reconstruction of a single hologram with a reference (as a np array an as HologramImage) with default
        # output size with an without input sideband parameters:

        wave_image = self.holo_image.reconstruct_phase(self.ref)
        sb_pos_cc = [self.img_size, self.img_size] - wave_image.reconstruction_parameters[0]
        sb_size_cc = wave_image.reconstruction_parameters[1]
        sb_smoothness_cc = wave_image.reconstruction_parameters[2]
        sb_units_cc = wave_image.reconstruction_parameters[3]
        wave_image_cc = self.holo_image.reconstruct_phase(self.ref_image, sb_position=sb_pos_cc, sb_size=sb_size_cc,
                                                          sb_smooth=sb_smoothness_cc, sb_unit=sb_units_cc)
        x_start = int(wave_image.reconstruction_parameters[1]*2/10)
        x_stop = int(wave_image.reconstruction_parameters[1]*2*9/10)
        wave_crop = wave_image.data[x_start:x_stop, x_start:x_stop]
        wave_cc_crop = wave_image_cc.data[x_start:x_stop, x_start:x_stop]

        nt.assert_allclose(wave_crop, np.conj(wave_cc_crop), rtol=1e-3)  # asserts that waves from different
        # sidebands are complex conjugate; this also tests possibility of reconstruction with given sideband parameters

        # Cropping the reconstructed and original phase images and comparing:
        x_start = int(self.img_size/10)
        x_stop = self.img_size-1-int(self.img_size/10)
        phase_new_crop = wave_image.unwrapped_phase().data[x_start:x_stop, x_start:x_stop]
        phase_ref_crop = self.phase_ref[x_start:x_stop, x_start:x_stop]
        nt.assert_almost_equal(phase_new_crop, phase_ref_crop, decimal=2)

        # Testing reconstruction with non-standard output size for stacked images:
        sb_position2 = self.ref_image2.find_sideband_position(sb='upper')
        sb_size2 = self.ref_image2.find_sideband_size(sb_position2)
        output_shape = (np.int(sb_size2[0]*2), np.int(sb_size2[0]*2))

        wave_image2 = self.holo_image2.reconstruct_phase(reference=self.ref_image2, sb_position=sb_position2,
                                                         sb_size=sb_size2, sb_smooth=sb_size2*0.05,
                                                         output_shape=output_shape)

        # interpolate reconstructed phase to compare with the input (reference phase):
        interp_x = np.arange(output_shape[0])
        phase_interp0 = interp2d(interp_x, interp_x, wave_image2.inav[0].unwrapped_phase().data, kind='cubic')
        phase_new0 = phase_interp0(np.linspace(0, output_shape[0], self.img_size),
                                   np.linspace(0, output_shape[0], self.img_size))

        phase_interp1 = interp2d(interp_x, interp_x, wave_image2.inav[1].unwrapped_phase().data, kind='cubic')
        phase_new1 = phase_interp1(np.linspace(0, output_shape[0], self.img_size),
                                   np.linspace(0, output_shape[0], self.img_size))

        x_start = int(self.img_size/10)
        x_stop = self.img_size-1-int(self.img_size/10)
        phase_new_crop0 = phase_new0[x_start:x_stop, x_start:x_stop]
        phase_new_crop1 = phase_new1[x_start:x_stop, x_start:x_stop]
        phase_ref_crop0 = self.phase_ref2[0, x_start:x_stop, x_start:x_stop]
        phase_ref_crop1 = self.phase_ref2[1, x_start:x_stop, x_start:x_stop]
        nt.assert_almost_equal(phase_new_crop0, phase_ref_crop0, decimal=2)
        nt.assert_almost_equal(phase_new_crop1, phase_ref_crop1, decimal=2)





if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)
