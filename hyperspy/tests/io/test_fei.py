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

from hyperspy.io import load
from hyperspy.io_plugins.fei import load_ser_file

MY_PATH = os.path.dirname(__file__)


class TestFEIReader():

    def setUp(self):
        self.dirpathold = os.path.join(MY_PATH, 'FEI_old')
        self.dirpathnew = os.path.join(MY_PATH, 'FEI_new')

    def test_load_emi_old_new_format(self, verbose=True):
        # TIA old format
        fname0 = os.path.join(self.dirpathold, '64x64_TEM_images_acquire.emi')
        load(fname0, verbose=verbose)
        # TIA new format
        fname1 = os.path.join(self.dirpathnew, '128x128_TEM_acquire-sum1.emi')
        load(fname1, verbose=verbose)
        # TIA new format 1Go file
#        fname2= os.path.join('/mnt/data/test/64bits', 'CCD Preview-250-beam_damaged.emi')
#        load(fname2, verbose=verbose)
        # TIA new format 4Go file
#        fname2= os.path.join('/mnt/data/test/64bits', '11.22.31 CCD Preview-250-bean_damaged-bin2-0.2s.emi')
#        load(fname2, verbose=verbose)

    def test_load_image_content(self, verbose=True):
        # TEM image of the beam stop
        fname0 = os.path.join(self.dirpathold, '64x64_TEM_images_acquire.emi')
        s0 = load(fname0, verbose=verbose)
        data = np.load(fname0.replace('emi', 'npy'))
        np.testing.assert_array_equal(s0.data, data)

    def test_load_ser_reader_old_new_format(self, verbose=True):
        # test TIA old format
        fname0 = os.path.join(
            self.dirpathold, '64x64_TEM_images_acquire_1.ser')
        header0, data0 = load_ser_file(fname0, verbose=verbose)
        nt.assert_equal(header0['SeriesVersion'], 528)
        # test TIA new format
        fname1 = os.path.join(
            self.dirpathnew, '128x128_TEM_acquire-sum1_1.ser')
        header1, data1 = load_ser_file(fname1, verbose=verbose)
        nt.assert_equal(header1['SeriesVersion'], 544)

    def test_load_diffraction_point(self, verbose=True):
        fname0 = os.path.join(self.dirpathold, '64x64_diffraction_acquire.emi')
        s0 = load(fname0, verbose=verbose)
        nt.assert_equal(s0.data.shape, (64, 64))
        nt.assert_equal(s0.metadata.Signal.record_by, 'image')
        nt.assert_equal(
            s0.metadata.Acquisition_instrument.TEM.acquisition_mode, 'TEM')
        nt.assert_almost_equal(s0.axes_manager[0].scale, 0.10157, places=5)
        nt.assert_equal(s0.axes_manager[0].units, '1/nm')
        nt.assert_almost_equal(s0.axes_manager[1].scale, 0.10157, places=5)
        nt.assert_equal(s0.axes_manager[1].units, '1/nm')

    def test_load_diffraction_line_scan(self, verbose=True):
        fname0 = os.path.join(
            self.dirpathnew, '16x16-line_profile_horizontal_5x128x128_EDS.emi')
        s0 = load(fname0, verbose=verbose)
        # s0[0] contains EDS
        nt.assert_equal(s0[0].data.shape, (5, 4000))
        nt.assert_equal(s0[0].metadata.Signal.record_by, 'spectrum')
        nt.assert_equal(
            s0[0].metadata.Acquisition_instrument.TEM.acquisition_mode, 'STEM')
        nt.assert_almost_equal(s0[0].axes_manager[0].scale, 3.68864, places=5)
        nt.assert_equal(s0[0].axes_manager[0].units, 'nm')
        nt.assert_almost_equal(s0[0].axes_manager[1].scale, 5.0, places=5)
        nt.assert_equal(s0[0].axes_manager[1].units, 'eV')
        # s0[1] contains diffraction patterns
        nt.assert_equal(s0[1].data.shape, (5, 128, 128))
        nt.assert_equal(s0[1].metadata.Signal.record_by, 'image')
        nt.assert_equal(
            s0[1].metadata.Acquisition_instrument.TEM.acquisition_mode, 'STEM')
        nt.assert_almost_equal(s0[1].axes_manager[0].scale, 3.68864, places=5)
        nt.assert_equal(s0[1].axes_manager[0].units, 'nm')
        nt.assert_equal(s0[1].axes_manager[1].units, '1/nm')
        nt.assert_almost_equal(s0[1].axes_manager[1].scale, 0.17435, places=5)
        nt.assert_almost_equal(s0[1].axes_manager[2].scale, 0.17435, places=5)
        nt.assert_equal(s0[1].axes_manager[2].units, '1/nm')

    def test_load_diffraction_area_scan(self, verbose=True):
        fname0 = os.path.join(
            self.dirpathnew, '16x16-diffraction_imagel_5x5x256x256_EDS.emi')
        s0 = load(fname0, verbose=verbose)
        # s0[0] contains EDS
        nt.assert_equal(s0[0].data.shape, (5, 5, 4000))
        nt.assert_equal(s0[0].metadata.Signal.record_by, 'spectrum')
        nt.assert_equal(
            s0[0].metadata.Acquisition_instrument.TEM.acquisition_mode, 'STEM')
        nt.assert_almost_equal(s0[0].axes_manager[0].scale, 1.87390, places=5)
        nt.assert_equal(s0[0].axes_manager[0].units, 'nm')
        nt.assert_almost_equal(s0[0].axes_manager[1].scale, -1.87390, places=5)
        nt.assert_equal(s0[0].axes_manager[1].units, 'nm')
        nt.assert_almost_equal(s0[0].axes_manager[2].scale, 5.0, places=5)
        nt.assert_equal(s0[0].axes_manager[2].units, 'eV')
        # s0[1] contains diffraction patterns
        nt.assert_equal(s0[1].data.shape, (5, 5, 256, 256))
        nt.assert_equal(s0[1].metadata.Signal.record_by, 'image')
        nt.assert_equal(
            s0[1].metadata.Acquisition_instrument.TEM.acquisition_mode, 'STEM')
        nt.assert_almost_equal(s0[1].axes_manager[0].scale, -1.87390, places=5)
        nt.assert_equal(s0[1].axes_manager[0].units, 'nm')
        nt.assert_almost_equal(s0[1].axes_manager[2].scale, 0.17435, places=5)
        nt.assert_equal(s0[1].axes_manager[2].units, '1/nm')

    def test_load_spectrum_point(self, verbose=True):
        fname0 = os.path.join(
            self.dirpathold, '16x16-point_spectrum-1x1024.emi')
        s0 = load(fname0, verbose=verbose)
        nt.assert_equal(s0.data.shape, (1, 1024))
        nt.assert_equal(s0.metadata.Signal.record_by, 'spectrum')
        nt.assert_equal(
            s0.metadata.Acquisition_instrument.TEM.acquisition_mode, 'STEM')
        # single spectrum should be imported as 1D data, not 2D
        nt.assert_almost_equal(
            s0.axes_manager[0].scale, 1000000000.0, places=5)
        nt.assert_equal(s0.axes_manager[0].units, 'nm')
        nt.assert_almost_equal(s0.axes_manager[1].scale, 0.2, places=5)
        nt.assert_equal(s0.axes_manager[1].units, 'eV')

        fname1 = os.path.join(
            self.dirpathold, '16x16-2_point-spectra-2x1024.emi')
        s1 = load(fname1, verbose=verbose)
        nt.assert_equal(s1.data.shape, (2, 1024))
        nt.assert_equal(s1.metadata.Signal.record_by, 'spectrum')
        nt.assert_equal(
            s1.metadata.Acquisition_instrument.TEM.acquisition_mode, 'STEM')
        nt.assert_almost_equal(
            s0.axes_manager[0].scale, 1000000000.0, places=5)
        nt.assert_equal(s0.axes_manager[0].units, 'nm')
        nt.assert_almost_equal(s0.axes_manager[1].scale, 0.2, places=5)
        nt.assert_equal(s0.axes_manager[1].units, 'eV')

    def test_load_spectrum_line_scan(self, verbose=True):
        fname0 = os.path.join(
            self.dirpathold, '16x16-line_profile_horizontal_10x1024.emi')
        s0 = load(fname0, verbose=verbose)
        nt.assert_equal(s0.data.shape, (10, 1024))
        nt.assert_equal(s0.metadata.Signal.record_by, 'spectrum')
        nt.assert_equal(
            s0.metadata.Acquisition_instrument.TEM.acquisition_mode, 'STEM')
        nt.assert_almost_equal(s0.axes_manager[0].scale, 0.12303, places=5)
        nt.assert_equal(s0.axes_manager[0].units, 'nm')
        nt.assert_almost_equal(s0.axes_manager[1].scale, 0.2, places=5)
        nt.assert_equal(s0.axes_manager[1].units, 'eV')

        fname1 = os.path.join(
            self.dirpathold, '16x16-line_profile_diagonal_10x1024.emi')
        s1 = load(fname1, verbose=verbose)
        nt.assert_equal(s1.data.shape, (10, 1024))
        nt.assert_equal(s1.metadata.Signal.record_by, 'spectrum')
        nt.assert_equal(
            s1.metadata.Acquisition_instrument.TEM.acquisition_mode, 'STEM')
        nt.assert_almost_equal(s1.axes_manager[0].scale, 0.166318, places=5)
        nt.assert_equal(s1.axes_manager[0].units, 'nm')
        nt.assert_almost_equal(s1.axes_manager[1].scale, 0.2, places=5)
        nt.assert_equal(s1.axes_manager[1].units, 'eV')

    def test_load_spectrum_area_scan(self, verbose=True):
        fname0 = os.path.join(
            self.dirpathold, '16x16-spectrum_image-5x5x1024.emi')
        s0 = load(fname0, verbose=verbose)
        nt.assert_equal(s0.data.shape, (5, 5, 1024))
        nt.assert_equal(s0.metadata.Signal.record_by, 'spectrum')
        nt.assert_equal(
            s0.metadata.Acquisition_instrument.TEM.acquisition_mode, 'STEM')
        nt.assert_almost_equal(s0.axes_manager[0].scale, 0.120539, places=5)
        nt.assert_equal(s0.axes_manager[0].units, 'nm')
        nt.assert_almost_equal(s0.axes_manager[1].scale, -0.120539, places=5)
        nt.assert_equal(s0.axes_manager[1].units, 'nm')
        nt.assert_almost_equal(s0.axes_manager[2].scale, 0.2, places=5)
        nt.assert_equal(s0.axes_manager[2].units, 'eV')

    def test_load_spectrum_area_scan_not_square(self, verbose=True):
        fname0 = os.path.join(
            self.dirpathnew, '16x16-spectrum_image_5x5x4000-not_square.emi')
        s0 = load(fname0, verbose=verbose)
        nt.assert_equal(s0.data.shape, (5, 5, 4000))
        nt.assert_equal(s0.metadata.Signal.record_by, 'spectrum')
        nt.assert_equal(
            s0.metadata.Acquisition_instrument.TEM.acquisition_mode, 'STEM')
        nt.assert_almost_equal(s0.axes_manager[0].scale, 1.98591, places=5)
        nt.assert_equal(s0.axes_manager[0].units, 'nm')
        nt.assert_almost_equal(s0.axes_manager[1].scale, -4.25819, places=5)
        nt.assert_equal(s0.axes_manager[1].units, 'nm')
        nt.assert_almost_equal(s0.axes_manager[2].scale, 5.0, places=5)
        nt.assert_equal(s0.axes_manager[2].units, 'eV')

    def test_load_search(self, verbose=True):
        fname0 = os.path.join(self.dirpathnew, '128x128-TEM_search.emi')
        s0 = load(fname0, verbose=verbose)
        nt.assert_equal(s0.data.shape, (128, 128))
        nt.assert_almost_equal(s0.axes_manager[0].scale, 5.26121, places=5)
        nt.assert_equal(s0.axes_manager[0].units, 'nm')
        nt.assert_almost_equal(s0.axes_manager[1].scale, 5.26121, places=5)
        nt.assert_equal(s0.axes_manager[1].units, 'nm')

        fname1 = os.path.join(self.dirpathold, '16x16_STEM_BF_DF_search.emi')
        s1 = load(fname1, verbose=verbose)
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

    def test_load_stack_image_preview(self, verbose=True):
        fname0 = os.path.join(self.dirpathold, '64x64x5_TEM_preview.emi')
        s0 = load(fname0, verbose=verbose)
        nt.assert_equal(s0.data.shape, (5, 64, 64))
        nt.assert_equal(s0.metadata.Signal.record_by, 'image')
        nt.assert_equal(
            s0.metadata.Acquisition_instrument.TEM.acquisition_mode, 'TEM')
        nt.assert_almost_equal(s0.axes_manager[0].scale, 1.0, places=5)
        nt.assert_equal(s0.axes_manager[0].units, 'Unknown')
        nt.assert_almost_equal(s0.axes_manager[1].scale, 6.281833, places=5)
        nt.assert_equal(s0.axes_manager[1].units, 'nm')
        nt.assert_almost_equal(s0.axes_manager[2].scale, 6.281833, places=5)
        nt.assert_equal(s0.axes_manager[2].units, 'nm')

        fname2 = os.path.join(
            self.dirpathnew, '128x128x5-diffraction_preview.emi')
        s2 = load(fname2, verbose=verbose)
        nt.assert_equal(s2.data.shape, (5, 128, 128))
        nt.assert_almost_equal(s2.axes_manager[1].scale, 0.042464, places=5)
        nt.assert_equal(s0.axes_manager[0].units, 'Unknown')
        nt.assert_equal(s2.axes_manager[1].units, '1/nm')
        nt.assert_almost_equal(s2.axes_manager[2].scale, 0.042464, places=5)
        nt.assert_equal(s2.axes_manager[2].units, '1/nm')

        fname1 = os.path.join(
            self.dirpathold, '16x16x5_STEM_BF_DF_preview.emi')
        s1 = load(fname1, verbose=verbose)
        nt.assert_equal(len(s1), 2)
        for s in s1:
            nt.assert_equal(s.data.shape, (5, 16, 16))
            nt.assert_equal(
                s.metadata.Acquisition_instrument.TEM.acquisition_mode, 'STEM')
            nt.assert_almost_equal(s.axes_manager[0].scale, 1.0, places=5)
            nt.assert_equal(s.axes_manager[1].units, 'nm')
            nt.assert_almost_equal(
                s.axes_manager[1].scale, 21.510044, places=5)

    def test_load_acquire(self, verbose=True):
        fname0 = os.path.join(self.dirpathold, '64x64_TEM_images_acquire.emi')
        s0 = load(fname0, verbose=verbose)
        nt.assert_equal(s0.metadata.Signal.record_by, 'image')
        nt.assert_equal(
            s0.metadata.Acquisition_instrument.TEM.acquisition_mode, 'TEM')
        nt.assert_almost_equal(s0.axes_manager[0].scale, 6.281833, places=5)
        nt.assert_equal(s0.axes_manager[0].units, 'nm')
        nt.assert_almost_equal(s0.axes_manager[1].scale, 6.281833, places=5)
        nt.assert_equal(s0.axes_manager[1].units, 'nm')

        fname1 = os.path.join(self.dirpathold, '16x16_STEM_BF_DF_acquire.emi')
        s1 = load(fname1, verbose=verbose)
        nt.assert_equal(len(s1), 2)
        for s in s1:
            nt.assert_equal(s.data.shape, (16, 16))
            nt.assert_equal(
                s.metadata.Acquisition_instrument.TEM.acquisition_mode, 'STEM')
            nt.assert_almost_equal(
                s.axes_manager[0].scale, 21.510044, places=5)
            nt.assert_equal(s.axes_manager[0].units, 'nm')
            nt.assert_almost_equal(
                s.axes_manager[1].scale, 21.510044, places=5)
            nt.assert_equal(s.axes_manager[1].units, 'nm')

    def test_read_STEM_TEM_mode(self, verbose=True):
        # TEM image
        fname0 = os.path.join(self.dirpathold, '64x64_TEM_images_acquire.emi')
        s0 = load(fname0, verbose=verbose)
        nt.assert_equal(
            s0.metadata.Acquisition_instrument.TEM.acquisition_mode, "TEM")
        # TEM diffraction
        fname1 = os.path.join(self.dirpathold, '64x64_diffraction_acquire.emi')
        s1 = load(fname1, verbose=verbose)
        nt.assert_equal(
            s1.metadata.Acquisition_instrument.TEM.acquisition_mode, "TEM")
        fname2 = os.path.join(self.dirpathold, '16x16_STEM_BF_DF_acquire.emi')
        # STEM diffraction
        s2 = load(fname2, verbose=verbose)
        nt.assert_equal(
            s2[0].metadata.Acquisition_instrument.TEM.acquisition_mode, "STEM")
        nt.assert_equal(
            s2[1].metadata.Acquisition_instrument.TEM.acquisition_mode, "STEM")

    def test_load_units_scale(self, verbose=True):
        # TEM image
        fname0 = os.path.join(self.dirpathold, '64x64_TEM_images_acquire.emi')
        s0 = load(fname0, verbose=verbose)
        nt.assert_almost_equal(s0.axes_manager[0].scale, 6.28183, places=5)
        nt.assert_equal(s0.axes_manager[0].units, 'nm')
        # TEM diffraction
        fname1 = os.path.join(self.dirpathold, '64x64_diffraction_acquire.emi')
        s1 = load(fname1, verbose=verbose)
        nt.assert_almost_equal(s1.axes_manager[0].scale, 0.10157, places=4)
        nt.assert_equal(s1.axes_manager[0].units, '1/nm')
        # STEM diffraction
        fname2 = os.path.join(self.dirpathold, '16x16_STEM_BF_DF_acquire.emi')
        s2 = load(fname2, verbose=verbose)
        nt.assert_equal(s2[0].axes_manager[0].units, 'nm')
        nt.assert_almost_equal(s2[0].axes_manager[0].scale, 21.5100, places=4)

    def test_guess_units_from_mode(self, verbose=True):
        from hyperspy.io_plugins.fei import guess_units_from_mode, \
            convert_xml_to_dict, get_xml_info_from_emi
        fname0_emi = os.path.join(
            self.dirpathold, '64x64_TEM_images_acquire.emi')
        fname0_ser = os.path.join(
            self.dirpathold, '64x64_TEM_images_acquire_1.ser')
        objects = get_xml_info_from_emi(fname0_emi)
        header0, data0 = load_ser_file(fname0_ser, verbose=verbose)
        objects_dict = convert_xml_to_dict(objects[0])

        unit = guess_units_from_mode(objects_dict, header0)
        nt.assert_equal(unit, 'meters')

        # objects is empty dictionary
        unit = guess_units_from_mode({}, header0)
        nt.assert_equal(unit, 'meters')

        # objects is empty dictionary
        unit = guess_units_from_mode({}, header0)
        nt.assert_equal(unit, 'meters')
