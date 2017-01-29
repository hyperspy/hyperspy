# -*- coding: utf-8 -*-
# Copyright 2007-2015 The HyperSpy developers
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

import os

import nose.tools as nt
import numpy as np
import traits.api as t

from hyperspy.io import load
from hyperspy.io_plugins.fei import load_ser_file
from hyperspy.misc.test_utils import assert_warns

MY_PATH = os.path.dirname(__file__)


class TestFEIReader():

    def setUp(self):
        self.dirpathold = os.path.join(MY_PATH, 'FEI_old')
        self.dirpathnew = os.path.join(MY_PATH, 'FEI_new')

    def test_load_emi_old_new_format(self):
        # TIA old format
        fname0 = os.path.join(self.dirpathold, '64x64_TEM_images_acquire.emi')
        load(fname0)
        # TIA new format
        fname1 = os.path.join(self.dirpathnew, '128x128_TEM_acquire-sum1.emi')
        load(fname1)

    def test_load_image_content(self):
        # TEM image of the beam stop
        fname0 = os.path.join(self.dirpathold, '64x64_TEM_images_acquire.emi')
        s0 = load(fname0)
        data = np.load(fname0.replace('emi', 'npy'))
        np.testing.assert_array_equal(s0.data, data)

    def test_load_ser_reader_old_new_format(self):
        # test TIA old format
        fname0 = os.path.join(
            self.dirpathold, '64x64_TEM_images_acquire_1.ser')
        header0, data0 = load_ser_file(fname0)
        nt.assert_equal(header0['SeriesVersion'], 528)
        # test TIA new format
        fname1 = os.path.join(
            self.dirpathnew, '128x128_TEM_acquire-sum1_1.ser')
        header1, data1 = load_ser_file(fname1)
        nt.assert_equal(header1['SeriesVersion'], 544)

    def test_load_diffraction_point(self):
        fname0 = os.path.join(self.dirpathold, '64x64_diffraction_acquire.emi')
        s0 = load(fname0)
        nt.assert_equal(s0.data.shape, (64, 64))
        nt.assert_equal(s0.axes_manager.signal_dimension, 2)
        nt.assert_equal(
            s0.metadata.Acquisition_instrument.TEM.acquisition_mode, 'TEM')
        nt.assert_almost_equal(s0.axes_manager[0].scale, 0.10157, places=5)
        nt.assert_equal(s0.axes_manager[0].units, '1/nm')
        nt.assert_equal(s0.axes_manager[0].name, 'x')
        nt.assert_almost_equal(s0.axes_manager[1].scale, 0.10157, places=5)
        nt.assert_equal(s0.axes_manager[1].units, '1/nm')
        nt.assert_equal(s0.axes_manager[1].name, 'y')

    def test_load_diffraction_line_scan(self):
        fname0 = os.path.join(
            self.dirpathnew, '16x16-line_profile_horizontal_5x128x128_EDS.emi')
        s0 = load(fname0)
        # s0[0] contains EDS
        nt.assert_equal(s0[0].data.shape, (5, 4000))
        nt.assert_equal(s0[0].axes_manager.signal_dimension, 1)
        nt.assert_equal(
            s0[0].metadata.Acquisition_instrument.TEM.acquisition_mode, 'STEM')
        nt.assert_almost_equal(s0[0].axes_manager[0].scale, 3.68864, places=5)
        nt.assert_equal(s0[0].axes_manager[0].units, 'nm')
        nt.assert_equal(s0[0].axes_manager[0].name,
                        'y')  # Should this name be y?
        nt.assert_almost_equal(s0[0].axes_manager[1].scale, 5.0, places=5)
        nt.assert_equal(s0[0].axes_manager[1].units, 'eV')
        nt.assert_equal(s0[0].axes_manager[1].name, 'Energy')
        # s0[1] contains diffraction patterns
        nt.assert_equal(s0[1].data.shape, (5, 128, 128))
        nt.assert_equal(s0[1].axes_manager.signal_dimension, 2)
        nt.assert_equal(
            s0[1].metadata.Acquisition_instrument.TEM.acquisition_mode, 'STEM')
        nt.assert_almost_equal(s0[1].axes_manager[0].scale, 3.68864, places=5)
        nt.assert_equal(s0[1].axes_manager[0].units, 'nm')
        nt.assert_equal(s0[1].axes_manager[0].name,
                        'y')  # Should this name be y?
        nt.assert_almost_equal(s0[1].axes_manager[1].scale, 0.17435, places=5)
        nt.assert_equal(s0[1].axes_manager[1].units, '1/nm')
        nt.assert_equal(s0[1].axes_manager[1].name, 'x')
        nt.assert_almost_equal(s0[1].axes_manager[2].scale, 0.17435, places=5)
        nt.assert_equal(s0[1].axes_manager[2].units, '1/nm')
        nt.assert_equal(s0[1].axes_manager[2].name, 'y')

    def test_load_diffraction_area_scan(self):
        fname0 = os.path.join(
            self.dirpathnew, '16x16-diffraction_imagel_5x5x256x256_EDS.emi')
        s0 = load(fname0)
        # s0[0] contains EDS
        nt.assert_equal(s0[0].data.shape, (5, 5, 4000))
        nt.assert_equal(s0[0].axes_manager.signal_dimension, 1)
        nt.assert_equal(
            s0[0].metadata.Acquisition_instrument.TEM.acquisition_mode, 'STEM')
        nt.assert_almost_equal(s0[0].axes_manager[0].scale, 1.87390, places=5)
        nt.assert_equal(s0[0].axes_manager[0].units, 'nm')
        nt.assert_equal(s0[0].axes_manager[0].name, 'x')
        nt.assert_almost_equal(s0[0].axes_manager[1].scale, -1.87390, places=5)
        nt.assert_equal(s0[0].axes_manager[1].units, 'nm')
        nt.assert_equal(s0[0].axes_manager[1].name, 'y')
        nt.assert_almost_equal(s0[0].axes_manager[2].scale, 5.0, places=5)
        nt.assert_equal(s0[0].axes_manager[2].units, 'eV')
        nt.assert_equal(s0[0].axes_manager[2].name, 'Energy')
        # s0[1] contains diffraction patterns
        nt.assert_equal(s0[1].data.shape, (5, 5, 256, 256))
        nt.assert_equal(s0[1].axes_manager.signal_dimension, 2)
        nt.assert_equal(
            s0[1].metadata.Acquisition_instrument.TEM.acquisition_mode, 'STEM')
        nt.assert_almost_equal(s0[1].axes_manager[0].scale, 1.87390, places=5)
        nt.assert_equal(s0[1].axes_manager[0].units, 'nm')
        nt.assert_equal(s0[1].axes_manager[0].name, 'x')
        nt.assert_almost_equal(s0[1].axes_manager[1].scale, -1.87390, places=5)
        nt.assert_equal(s0[1].axes_manager[1].units, 'nm')
        nt.assert_equal(s0[1].axes_manager[1].name, 'y')
        nt.assert_almost_equal(s0[1].axes_manager[2].scale, 0.17435, places=5)
        nt.assert_equal(s0[1].axes_manager[2].units, '1/nm')
        nt.assert_equal(s0[1].axes_manager[2].name, 'x')
        nt.assert_almost_equal(s0[1].axes_manager[3].scale, 0.17435, places=5)
        nt.assert_equal(s0[1].axes_manager[3].units, '1/nm')
        nt.assert_equal(s0[1].axes_manager[3].name, 'y')

    def test_load_spectrum_point(self):
        fname0 = os.path.join(
            self.dirpathold, '16x16-point_spectrum-1x1024.emi')
        s0 = load(fname0)
        nt.assert_equal(s0.data.shape, (1, 1024))
        nt.assert_equal(s0.axes_manager.signal_dimension, 1)
        nt.assert_equal(
            s0.metadata.Acquisition_instrument.TEM.acquisition_mode, 'STEM')
        # single spectrum should be imported as 1D data, not 2D
        # TODO: the following calibration is wrong because it parse the
        # 'Dim-1_CalibrationDelta' from the ser header, which is not correct in
        # case of point spectra. However, the position seems to be saved in
        # 'PositionX' and 'PositionY' arrays of the ser header, so it should
        # be possible to workaround using the position arrays.
#        nt.assert_almost_equal(
#            s0.axes_manager[0].scale, 1.0, places=5)
#        nt.assert_equal(s0.axes_manager[0].units, '')
#        nt.assert_is(s0.axes_manager[0].name, 'Position index')
        nt.assert_almost_equal(s0.axes_manager[1].scale, 0.2, places=5)
        nt.assert_equal(s0.axes_manager[1].units, 'eV')
        nt.assert_equal(s0.axes_manager[1].name, 'Energy')

        fname1 = os.path.join(
            self.dirpathold, '16x16-2_point-spectra-2x1024.emi')
        s1 = load(fname1)
        nt.assert_equal(s1.data.shape, (2, 1024))
        nt.assert_equal(s1.axes_manager.signal_dimension, 1)
        nt.assert_equal(
            s1.metadata.Acquisition_instrument.TEM.acquisition_mode, 'STEM')
        nt.assert_almost_equal(s0.axes_manager[1].scale, 0.2, places=5)
        nt.assert_equal(s0.axes_manager[1].units, 'eV')
        nt.assert_equal(s0.axes_manager[1].name, 'Energy')

    def test_load_spectrum_line_scan(self):
        fname0 = os.path.join(
            self.dirpathold, '16x16-line_profile_horizontal_10x1024.emi')
        s0 = load(fname0)
        nt.assert_equal(s0.data.shape, (10, 1024))
        nt.assert_equal(s0.axes_manager.signal_dimension, 1)
        nt.assert_equal(
            s0.metadata.Acquisition_instrument.TEM.acquisition_mode, 'STEM')
        nt.assert_almost_equal(s0.axes_manager[0].scale, 0.12303, places=5)
        nt.assert_equal(s0.axes_manager[0].units, 'nm')
        nt.assert_equal(s0.axes_manager[0].name, 'y')
        nt.assert_almost_equal(s0.axes_manager[1].scale, 0.2, places=5)
        nt.assert_equal(s0.axes_manager[1].units, 'eV')
        nt.assert_equal(s0.axes_manager[1].name, 'Energy')

        fname1 = os.path.join(
            self.dirpathold, '16x16-line_profile_diagonal_10x1024.emi')
        s1 = load(fname1)
        nt.assert_equal(s1.data.shape, (10, 1024))
        nt.assert_equal(s1.axes_manager.signal_dimension, 1)
        nt.assert_equal(
            s1.metadata.Acquisition_instrument.TEM.acquisition_mode, 'STEM')
        nt.assert_almost_equal(s1.axes_manager[0].scale, 0.166318, places=5)
        nt.assert_equal(s1.axes_manager[0].units, 'nm')
        nt.assert_almost_equal(s1.axes_manager[1].scale, 0.2, places=5)
        nt.assert_equal(s1.axes_manager[1].units, 'eV')

    def test_load_spectrum_area_scan(self):
        fname0 = os.path.join(
            self.dirpathold, '16x16-spectrum_image-5x5x1024.emi')
        s0 = load(fname0)
        nt.assert_equal(s0.data.shape, (5, 5, 1024))
        nt.assert_equal(s0.axes_manager.signal_dimension, 1)
        nt.assert_equal(
            s0.metadata.Acquisition_instrument.TEM.acquisition_mode, 'STEM')
        nt.assert_almost_equal(s0.axes_manager[0].scale, 0.120539, places=5)
        nt.assert_equal(s0.axes_manager[0].units, 'nm')
        nt.assert_equal(s0.axes_manager[0].name, 'x')
        nt.assert_almost_equal(s0.axes_manager[1].scale, -0.120539, places=5)
        nt.assert_equal(s0.axes_manager[1].units, 'nm')
        nt.assert_equal(s0.axes_manager[1].name, 'y')
        nt.assert_almost_equal(s0.axes_manager[2].scale, 0.2, places=5)
        nt.assert_equal(s0.axes_manager[2].units, 'eV')
        nt.assert_equal(s0.axes_manager[2].name, 'Energy')

    def test_load_spectrum_area_scan_not_square(self):
        fname0 = os.path.join(
            self.dirpathnew, '16x16-spectrum_image_5x5x4000-not_square.emi')
        s0 = load(fname0)
        nt.assert_equal(s0.data.shape, (5, 5, 4000))
        nt.assert_equal(s0.axes_manager.signal_dimension, 1)
        nt.assert_equal(
            s0.metadata.Acquisition_instrument.TEM.acquisition_mode, 'STEM')
        nt.assert_almost_equal(s0.axes_manager[0].scale, 1.98591, places=5)
        nt.assert_equal(s0.axes_manager[0].units, 'nm')
        nt.assert_almost_equal(s0.axes_manager[1].scale, -4.25819, places=5)
        nt.assert_equal(s0.axes_manager[1].units, 'nm')
        nt.assert_almost_equal(s0.axes_manager[2].scale, 5.0, places=5)
        nt.assert_equal(s0.axes_manager[2].units, 'eV')

    def test_load_search(self):
        fname0 = os.path.join(self.dirpathnew, '128x128-TEM_search.emi')
        s0 = load(fname0)
        nt.assert_equal(s0.data.shape, (128, 128))
        nt.assert_almost_equal(s0.axes_manager[0].scale, 5.26121, places=5)
        nt.assert_equal(s0.axes_manager[0].units, 'nm')
        nt.assert_almost_equal(s0.axes_manager[1].scale, 5.26121, places=5)
        nt.assert_equal(s0.axes_manager[1].units, 'nm')

        fname1 = os.path.join(self.dirpathold, '16x16_STEM_BF_DF_search.emi')
        s1 = load(fname1)
        nt.assert_equal(len(s1), 2)
        for s in s1:
            nt.assert_equal(s.data.shape, (16, 16))
            nt.assert_equal(
                s.metadata.Acquisition_instrument.TEM.acquisition_mode, 'STEM')
            nt.assert_almost_equal(
                s.axes_manager[0].scale, 22.026285, places=5)
            nt.assert_equal(s.axes_manager[0].units, 'nm')
            nt.assert_almost_equal(
                s.axes_manager[1].scale, 22.026285, places=5)
            nt.assert_equal(s.axes_manager[1].units, 'nm')

    def test_load_stack_image_preview(self):
        fname0 = os.path.join(self.dirpathold, '64x64x5_TEM_preview.emi')
        s0 = load(fname0)
        nt.assert_equal(s0.data.shape, (5, 64, 64))
        nt.assert_equal(s0.axes_manager.signal_dimension, 2)
        nt.assert_equal(
            s0.metadata.Acquisition_instrument.TEM.acquisition_mode, 'TEM')
        nt.assert_almost_equal(s0.axes_manager[0].scale, 1.0, places=5)
        nt.assert_is(s0.axes_manager[0].units, t.Undefined)
        nt.assert_equal(s0.axes_manager[0].scale, 1.0)
        nt.assert_is(s0.axes_manager[0].name, t.Undefined)
        nt.assert_almost_equal(s0.axes_manager[1].scale, 6.281833, places=5)
        nt.assert_equal(s0.axes_manager[1].units, 'nm')
        nt.assert_equal(s0.axes_manager[1].name, 'x')
        nt.assert_almost_equal(s0.axes_manager[2].scale, 6.281833, places=5)
        nt.assert_equal(s0.axes_manager[2].units, 'nm')
        nt.assert_equal(s0.axes_manager[2].name, 'y')

        fname2 = os.path.join(
            self.dirpathnew, '128x128x5-diffraction_preview.emi')
        s2 = load(fname2)
        nt.assert_equal(s2.data.shape, (5, 128, 128))
        nt.assert_almost_equal(s2.axes_manager[1].scale, 0.042464, places=5)
        nt.assert_is(s0.axes_manager[0].units, t.Undefined)
        nt.assert_equal(s2.axes_manager[1].units, '1/nm')
        nt.assert_almost_equal(s2.axes_manager[2].scale, 0.042464, places=5)
        nt.assert_equal(s2.axes_manager[2].units, '1/nm')

        fname1 = os.path.join(
            self.dirpathold, '16x16x5_STEM_BF_DF_preview.emi')
        s1 = load(fname1)
        nt.assert_equal(len(s1), 2)
        for s in s1:
            nt.assert_equal(s.data.shape, (5, 16, 16))
            nt.assert_equal(
                s.metadata.Acquisition_instrument.TEM.acquisition_mode, 'STEM')
            nt.assert_almost_equal(s.axes_manager[0].scale, 1.0, places=5)
            nt.assert_equal(s.axes_manager[1].units, 'nm')
            nt.assert_almost_equal(
                s.axes_manager[1].scale, 21.510044, places=5)

    def test_load_acquire(self):
        fname0 = os.path.join(self.dirpathold, '64x64_TEM_images_acquire.emi')
        s0 = load(fname0)
        nt.assert_equal(s0.axes_manager.signal_dimension, 2)
        nt.assert_equal(
            s0.metadata.Acquisition_instrument.TEM.acquisition_mode, 'TEM')
        nt.assert_almost_equal(s0.axes_manager[0].scale, 6.281833, places=5)
        nt.assert_equal(s0.axes_manager[0].units, 'nm')
        nt.assert_equal(s0.axes_manager[0].name, 'x')
        nt.assert_almost_equal(s0.axes_manager[1].scale, 6.281833, places=5)
        nt.assert_equal(s0.axes_manager[1].units, 'nm')
        nt.assert_equal(s0.axes_manager[1].name, 'y')

        fname1 = os.path.join(self.dirpathold, '16x16_STEM_BF_DF_acquire.emi')
        s1 = load(fname1)
        nt.assert_equal(len(s1), 2)
        for s in s1:
            nt.assert_equal(s.data.shape, (16, 16))
            nt.assert_equal(
                s.metadata.Acquisition_instrument.TEM.acquisition_mode, 'STEM')
            nt.assert_almost_equal(
                s.axes_manager[0].scale, 21.510044, places=5)
            nt.assert_equal(s.axes_manager[0].units, 'nm')
            nt.assert_equal(s.axes_manager[0].name, 'x')
            nt.assert_almost_equal(
                s.axes_manager[1].scale, 21.510044, places=5)
            nt.assert_equal(s.axes_manager[1].units, 'nm')
            nt.assert_equal(s.axes_manager[1].name, 'y')

    def test_read_STEM_TEM_mode(self):
        # TEM image
        fname0 = os.path.join(self.dirpathold, '64x64_TEM_images_acquire.emi')
        s0 = load(fname0)
        nt.assert_equal(
            s0.metadata.Acquisition_instrument.TEM.acquisition_mode, "TEM")
        # TEM diffraction
        fname1 = os.path.join(self.dirpathold, '64x64_diffraction_acquire.emi')
        s1 = load(fname1)
        nt.assert_equal(
            s1.metadata.Acquisition_instrument.TEM.acquisition_mode, "TEM")
        fname2 = os.path.join(self.dirpathold, '16x16_STEM_BF_DF_acquire.emi')
        # STEM diffraction
        s2 = load(fname2)
        nt.assert_equal(
            s2[0].metadata.Acquisition_instrument.TEM.acquisition_mode, "STEM")
        nt.assert_equal(
            s2[1].metadata.Acquisition_instrument.TEM.acquisition_mode, "STEM")

    def test_load_units_scale(self):
        # TEM image
        fname0 = os.path.join(self.dirpathold, '64x64_TEM_images_acquire.emi')
        s0 = load(fname0)
        nt.assert_almost_equal(s0.axes_manager[0].scale, 6.28183, places=5)
        nt.assert_equal(s0.axes_manager[0].units, 'nm')
        nt.assert_almost_equal(s0.metadata.Acquisition_instrument.TEM.magnification,
                               19500.0, places=4)
        # TEM diffraction
        fname1 = os.path.join(self.dirpathold, '64x64_diffraction_acquire.emi')
        s1 = load(fname1)
        nt.assert_almost_equal(s1.axes_manager[0].scale, 0.10157, places=4)
        nt.assert_equal(s1.axes_manager[0].units, '1/nm')
        nt.assert_almost_equal(s1.metadata.Acquisition_instrument.TEM.camera_length,
                               490.0, places=4)
        # STEM diffraction
        fname2 = os.path.join(self.dirpathold, '16x16_STEM_BF_DF_acquire.emi')
        s2 = load(fname2)
        nt.assert_equal(s2[0].axes_manager[0].units, 'nm')
        nt.assert_almost_equal(s2[0].axes_manager[0].scale, 21.5100, places=4)
        nt.assert_almost_equal(s2[0].metadata.Acquisition_instrument.TEM.magnification,
                               10000.0, places=4)

    def test_guess_units_from_mode(self):
        from hyperspy.io_plugins.fei import _guess_units_from_mode, \
            convert_xml_to_dict, get_xml_info_from_emi
        fname0_emi = os.path.join(
            self.dirpathold, '64x64_TEM_images_acquire.emi')
        fname0_ser = os.path.join(
            self.dirpathold, '64x64_TEM_images_acquire_1.ser')
        objects = get_xml_info_from_emi(fname0_emi)
        header0, data0 = load_ser_file(fname0_ser)
        objects_dict = convert_xml_to_dict(objects[0])

        unit = _guess_units_from_mode(objects_dict, header0)
        nt.assert_equal(unit, 'meters')

        # objects is empty dictionary
        with assert_warns(
                message="The navigation axes units could not be determined.",
                category=UserWarning):
            unit = _guess_units_from_mode({}, header0)
        nt.assert_equal(unit, 'meters')

    def test_date_time(self):
        fname0 = os.path.join(self.dirpathold, '64x64_TEM_images_acquire.emi')
        s = load(fname0)
        nt.assert_equal(s.metadata.General.date, "2016-02-21")
        nt.assert_equal(s.metadata.General.time, "17:50:18")
        nt.assert_equal(s.metadata.General.authors, "ERIC")

        fname1 = os.path.join(self.dirpathold,
                              '16x16-line_profile_horizontal_10x1024.emi')
        s = load(fname1)
        nt.assert_false(s.metadata.has_item('General.date'))
        nt.assert_false(s.metadata.has_item('General.time'))

    def test_metadata_TEM(self):
        fname0 = os.path.join(self.dirpathold, '64x64_TEM_images_acquire.emi')
        s = load(fname0)
        nt.assert_equal(
            s.metadata.Acquisition_instrument.TEM.beam_energy, 200.0)
        nt.assert_equal(
            s.metadata.Acquisition_instrument.TEM.magnification,
            19500.0)
        nt.assert_equal(
            s.metadata.Acquisition_instrument.TEM.microscope,
            "Tecnai 200 kV D2267 SuperTwin")
        nt.assert_almost_equal(
            s.metadata.Acquisition_instrument.TEM.tilt_stage,
            0.00,
            places=2)

    def test_metadata_STEM(self):
        fname0 = os.path.join(self.dirpathold, '16x16_STEM_BF_DF_acquire.emi')
        s = load(fname0)[0]
        nt.assert_equal(
            s.metadata.Acquisition_instrument.TEM.beam_energy, 200.0)
        nt.assert_equal(
            s.metadata.Acquisition_instrument.TEM.camera_length, 40.0)
        nt.assert_equal(
            s.metadata.Acquisition_instrument.TEM.magnification,
            10000.0)
        nt.assert_equal(
            s.metadata.Acquisition_instrument.TEM.microscope,
            "Tecnai 200 kV D2267 SuperTwin")
        nt.assert_almost_equal(
            s.metadata.Acquisition_instrument.TEM.tilt_stage,
            0.00,
            places=2)

    def test_metadata_diffraction(self):
        fname0 = os.path.join(self.dirpathold, '64x64_diffraction_acquire.emi')
        s = load(fname0)
        nt.assert_equal(
            s.metadata.Acquisition_instrument.TEM.beam_energy, 200.0)
        nt.assert_equal(
            s.metadata.Acquisition_instrument.TEM.camera_length,
            490.0)
        nt.assert_equal(
            s.metadata.Acquisition_instrument.TEM.microscope,
            "Tecnai 200 kV D2267 SuperTwin")
        nt.assert_almost_equal(
            s.metadata.Acquisition_instrument.TEM.tilt_stage,
            0.00,
            places=2)
