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
        self.img_size3x = 768
        self.img_size3y = 512
        fringe_direction = -np.pi/6
        fringe_spacing = 5.23
        fringe_direction3 = np.pi/3
        fringe_spacing3 = 6.11
        x, y = np.meshgrid(np.linspace(-self.img_size/2, self.img_size/2-1, self.img_size),
                           np.linspace(-self.img_size/2, self.img_size/2-1, self.img_size))
        x2, z2, y2 = np.meshgrid(np.linspace(-self.img_size/2, self.img_size/2-1, self.img_size),
                                 np.array([0, 1]),
                                 np.linspace(-self.img_size/2, self.img_size/2-1, self.img_size))

        x3, z3, y3 = np.meshgrid(np.linspace(-self.img_size3x/2, self.img_size3x/2, self.img_size3x),
                                 np.arange(6),
                                 np.linspace(-self.img_size3y/2, self.img_size3y/2, self.img_size3y))

        self.phase_ref = np.pi*x/(self.img_size/2.2)
        # self.phase_ref2 = np.pi*(x2+self.img_size/2)**z2/(self.img_size/2.2)
        self.phase_ref2 = np.pi*x2*(1-z2)/(self.img_size/2.2) + np.pi*y2*z2/(self.img_size/2.2)
        self.phase_ref3 = np.pi*x3*(1-z3/6)/(self.img_size3x*2) + np.pi*y3*z3/6/(self.img_size3y*2)

        # self.phase_ref3 = np.append(np.append(self.phase_ref2[:, 0:512, 0:512], self.phase_ref2[:, 0:512, 512:1024], 0),
        #                             self.phase_ref2[:, 512:1024, 0:512], 0).reshape((3, 2, 512, 512))
        holo = 2 * (1 + np.cos(self.phase_ref+fringe_spacing/(2*np.pi)*(x*np.cos(fringe_direction) +
                                                                        y*np.sin(fringe_direction))))
        holo2 = 2 * (1 + np.cos(self.phase_ref2+fringe_spacing/(2*np.pi)*(x2*np.cos(fringe_direction) +
                                                                          y2*np.sin(fringe_direction))))
        holo3 = 2 * (1 + np.cos(self.phase_ref3+fringe_spacing3/(2*np.pi)*(x3*np.cos(fringe_direction3) +
                                                                           y3*np.sin(fringe_direction3))))
        # holo3 = np.append(np.append(holo2[:, 0:512, 0:512], holo2[:, 0:512, 512:1024], 0),
        #                   holo2[:, 512:1024, 0:512], 0).reshape((3, 2, 512, 512))

        self.ref = 2 * (1 + np.cos(5.23/(2*np.pi)*(x*np.cos(fringe_direction) + y*np.sin(fringe_direction))))
        self.ref2 = 2 * (1 + np.cos(5.23/(2*np.pi)*(x2*np.cos(fringe_direction) + y2*np.sin(fringe_direction))))
        self.ref3 = 2 * (1 + np.cos(fringe_spacing3/(2*np.pi)*(x3*np.cos(fringe_direction3) +
                                                               y3*np.sin(fringe_direction3))))

        # Creating HologramImage signals for test:
        self.holo_image = hs.signals.HologramImage(holo)
        self.holo_image2 = hs.signals.HologramImage(holo2)  # (2|1024, 1024) signal

        self.ref_image = hs.signals.HologramImage(self.ref)
        self.ref_image2 = hs.signals.HologramImage(self.ref2)
        self.ref_image3 = hs.signals.HologramImage(self.ref3.reshape(2, 3, self.img_size3x, self.img_size3y))

        self.holo_image3 = hs.signals.HologramImage(holo3.reshape(2, 3, self.img_size3x, self.img_size3y))

    def test_set_microscope_parameters(self):
        self.holo_image.set_microscope_parameters(beam_energy=300., biprism_voltage=80.5, tilt_stage=2.2)
        nt.assert_equal(self.holo_image.metadata.Acquisition_instrument.TEM.beam_energy, 300.)
        nt.assert_equal(self.holo_image.metadata.Acquisition_instrument.TEM.Biprism.voltage, 80.5)
        nt.assert_equal(self.holo_image.metadata.Acquisition_instrument.TEM.tilt_stage, 2.2)

    def test_reconstruct_phase(self):

        # 1. Testing reconstruction of a single hologram with a reference (as a np array and as HologramImage) with
        # default output size with an without input sideband parameters:

        wave_image = self.holo_image.reconstruct_phase(self.ref, store_parameters=True)
        sb_pos_cc = wave_image.metadata.Signal.Holography.Reconstruction_parameters.sb_position * (-1) + \
                    [self.img_size, self.img_size]

        sb_size_cc = wave_image.metadata.Signal.Holography.Reconstruction_parameters.sb_size
        sb_smoothness_cc = wave_image.metadata.Signal.Holography.Reconstruction_parameters.sb_smoothness
        sb_units_cc = wave_image.metadata.Signal.Holography.Reconstruction_parameters.sb_units
        wave_image_cc = self.holo_image.reconstruct_phase(self.ref_image, sb_position=sb_pos_cc, sb_size=sb_size_cc,
                                                          sb_smoothness=sb_smoothness_cc, sb_unit=sb_units_cc)
        x_start = int(wave_image.axes_manager.signal_shape[0] / 10)
        x_stop = int(wave_image.axes_manager.signal_shape[0] * 9 / 10)
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

        # 2. Testing reconstruction with non-standard output size for stacked images:
        sb_position2 = self.ref_image2.estimate_sideband_position(sb='upper')
        sb_size2 = self.ref_image2.estimate_sideband_size(sb_position2)
        output_shape = (np.int(sb_size2.inav[0].data*2), np.int(sb_size2.inav[0].data*2))

        wave_image2 = self.holo_image2.reconstruct_phase(reference=self.ref_image2, sb_position=sb_position2,
                                                         sb_size=sb_size2, sb_smoothness=sb_size2 * 0.05,
                                                         output_shape=output_shape)
        #   a. Reconstruction with parameters provided as ndarrays should be identical to above:
        wave_image2a = self.holo_image2.reconstruct_phase(reference=self.ref_image2.data, sb_position=sb_position2.data,
                                                          sb_size=sb_size2.data, sb_smoothness=sb_size2.data * 0.05,
                                                          output_shape=output_shape)
        nt.assert_equal(wave_image2, wave_image2a)

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

        # 3. Testing reconstruction with multidimensional images (3, 2| 512, 768) using 1d image as a reference:
        wave_image3 = self.holo_image3.reconstruct_phase(self.ref_image3.inav[0, 0], sb='upper')

        # Cropping the reconstructed and original phase images and comparing:
        x_start = int(self.img_size3x/9)
        x_stop = self.img_size3x-1-int(self.img_size3x/9)  # larger fringes require larger crop
        y_start = int(self.img_size3y/9)
        y_stop = self.img_size3y-1-int(self.img_size3y/9)
        phase3_new_crop = wave_image3.unwrapped_phase()
        phase3_new_crop.crop(2, y_start, y_stop)
        phase3_new_crop.crop(3, x_start, x_stop)
        phase3_ref_crop = self.phase_ref3.reshape(2, 3, self.img_size3x, self.img_size3y)[:, :, x_start:x_stop,
                          y_start:y_stop]
        nt.assert_almost_equal(phase3_new_crop.data, phase3_ref_crop, decimal=2)

        # 3a. Testing reconstruction with input parameters in 'nm' and with multiple parameter input,
        # but reference ndim=0:

        sb_position3 = self.ref_image3.estimate_sideband_position(sb='upper')
        f_sampling = np.divide(1, [a * b for a, b in zip(self.ref_image3.axes_manager.signal_shape,
                                                         (self.ref_image3.axes_manager.signal_axes[0].scale,
                                                          self.ref_image3.axes_manager.signal_axes[1].scale))])
        sb_size3 = self.ref_image3.estimate_sideband_size(sb_position3) * np.mean(f_sampling)
        sb_smoothness3 = sb_size3 * 0.05
        sb_units3 = 'nm'

        wave_image3a = self.holo_image3.reconstruct_phase(self.ref_image3.inav[0, 0], sb_position=sb_position3,
                                                          sb_size=sb_size3, sb_smoothness=sb_smoothness3,
                                                          sb_unit=sb_units3)
        phase3a_new_crop = wave_image3a.unwrapped_phase()
        phase3a_new_crop.crop(2, y_start, y_stop)
        phase3a_new_crop.crop(3, x_start, x_stop)
        nt.assert_almost_equal(phase3a_new_crop.data, phase3_ref_crop, decimal=2)



        # 4. Testing raises:
        #   a. Mismatch of navigation dimensions of object and reference holograms, except if reference hologram ndim=0
        nt.assert_raises(ValueError, self.holo_image3.reconstruct_phase, self.ref_image3.inav[0, :])
        reference4a = self.ref_image3.inav[0, :]
        reference4a.set_signal_type('signal2d')
        nt.assert_raises(ValueError, self.holo_image3.reconstruct_phase, reference=reference4a)
        #   b. Mismatch of signal shapes of object and reference holograms
        nt.assert_raises(ValueError, self.holo_image3.reconstruct_phase,
                         self.ref_image3.inav[:, :].isig[y_start:y_stop, x_start:x_stop])

        #   c. Mismatch of signal shape of sb_position
        sb_position_mismatched = hs.signals.Signal2D(np.arange(9).reshape((3, 3)))
        nt.assert_raises(ValueError, self.holo_image3.reconstruct_phase, sb_position=sb_position_mismatched)

        #   d. Mismatch of navigation dimensions of reconstruction parameters
        sb_position_mismatched = hs.signals.Signal1D(np.arange(16).reshape((8, 2)))
        sb_size_mismatched = hs.signals.BaseSignal(np.arange(9)).T
        sb_smoothness_mismatched = hs.signals.BaseSignal(np.arange(9)).T
        nt.assert_raises(ValueError, self.holo_image3.reconstruct_phase, sb_position=sb_position_mismatched)
        nt.assert_raises(ValueError, self.holo_image3.reconstruct_phase, sb_size=sb_size_mismatched)
        nt.assert_raises(ValueError, self.holo_image3.reconstruct_phase, sb_smoothness=sb_smoothness_mismatched)

        #   e. Beam energy is not assigned, while 'mrad' units selected
        nt.assert_raises(AttributeError, self.holo_image3.reconstruct_phase, sb_size=40, sb_unit='mrad')

if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)
