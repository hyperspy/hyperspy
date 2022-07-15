# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
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
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import os

import numpy as np
import pytest
import traits.api as t

from hyperspy.io import load
from hyperspy.io_plugins.fei import load_ser_file


MY_PATH = os.path.dirname(__file__)


class TestFEIReader():

    def setup_method(self, method):
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
        assert header0['SeriesVersion'] == 528
        # test TIA new format
        fname1 = os.path.join(
            self.dirpathnew, '128x128_TEM_acquire-sum1_1.ser')
        header1, data1 = load_ser_file(fname1)
        assert header1['SeriesVersion'] == 544

    def test_load_no_acquire_date(self, caplog):
        fname = os.path.join(
            self.dirpathold, 'no_AcquireDate.emi')
        s = load(fname)
        assert not hasattr(s.metadata.General, 'date')
        assert not hasattr(s.metadata.General, 'time')
        assert 'AcquireDate not found in metadata' in caplog.text

    def test_load_more_ser_than_metadata(self, caplog):
        fname = os.path.join(
            self.dirpathold, 'more_ser_then_emi_metadata.emi')
        s0, s1 = load(fname, only_valid_data=True)
        assert hasattr(s0.original_metadata, 'ObjectInfo')
        assert not hasattr(s1.original_metadata, 'ObjectInfo')
        assert 'more_ser_then_emi_metadata.emi did not contain any metadata' \
               in caplog.text

    @pytest.fixture(scope="function")
    def prepare_non_zero_float(self):
        import tarfile
        tgz_fname = os.path.join(
            self.dirpathold, 'non_float_meta_value_zeroed.tar.gz')
        with tarfile.open(tgz_fname, 'r:gz') as tar:
            tar.extractall(path=os.path.dirname(tgz_fname))

        yield True

        # teardown code
        os.remove(os.path.join(
            self.dirpathold, 'non_float_meta_value_zeroed.emi'))
        os.remove(os.path.join(
            self.dirpathold, 'non_float_meta_value_zeroed_1.ser'))

    def test_load_non_zero_float(self, prepare_non_zero_float, caplog):
        fname = os.path.join(self.dirpathold, 'non_float_meta_value_zeroed.emi')
        s = load(fname)
        assert s.original_metadata.ObjectInfo.ExperimentalDescription\
            .as_dictionary()['OBJ Aperture_um'] == 'V'
        assert 'Expected decimal value for OBJ Aperture' in caplog.text

    def test_load_diffraction_point(self):
        fname0 = os.path.join(self.dirpathold, '64x64_diffraction_acquire.emi')
        s0 = load(fname0)
        assert s0.data.shape == (64, 64)
        assert s0.axes_manager.signal_dimension == 2
        assert (
            s0.metadata.Acquisition_instrument.TEM.acquisition_mode == 'TEM')
        np.testing.assert_allclose(s0.axes_manager[0].scale, 0.101571, rtol=1E-5)
        assert s0.axes_manager[0].units == '1 / nm'
        assert (s0.axes_manager[0].name == 'x')
        np.testing.assert_allclose(s0.axes_manager[1].scale, 0.101571, rtol=1E-5)
        assert s0.axes_manager[1].units == '1 / nm'
        assert (s0.axes_manager[1].name == 'y')

    def test_load_diffraction_line_scan(self):
        fname0 = os.path.join(
            self.dirpathnew, '16x16-line_profile_horizontal_5x128x128_EDS.emi')
        s0 = load(fname0)
        # s0[0] contains EDS
        assert s0[0].data.shape == (5, 4000)
        assert s0[0].axes_manager.signal_dimension == 1
        assert (
            s0[0].metadata.Acquisition_instrument.TEM.acquisition_mode == 'STEM')
        np.testing.assert_allclose(s0[0].axes_manager[0].scale, 3.68864, rtol=1E-5)
        assert s0[0].axes_manager[0].units == 'nm'
        np.testing.assert_allclose(s0[0].axes_manager[1].scale, 5.0, rtol=1E-5)
        assert s0[0].axes_manager[1].units == 'eV'
        assert (s0[0].axes_manager[0].name == 'x')
        assert (s0[0].axes_manager[1].name == 'Energy')
        # s0[1] contains diffraction patterns
        assert s0[1].data.shape == (5, 128, 128)
        assert s0[1].axes_manager.signal_dimension == 2
        assert (
            s0[1].metadata.Acquisition_instrument.TEM.acquisition_mode == 'STEM')
        np.testing.assert_allclose(s0[1].axes_manager[0].scale, 3.68864, rtol=1E-5)
        assert s0[1].axes_manager[0].units == 'nm'
        assert s0[1].axes_manager[1].units == '1 / nm'
        assert (s0[1].axes_manager[0].name == 'x')
        np.testing.assert_allclose(s0[1].axes_manager[1].scale, 0.174353, rtol=1E-5)
        np.testing.assert_allclose(s0[1].axes_manager[2].scale, 0.174353, rtol=1E-5)
        assert s0[1].axes_manager[2].units == '1 / nm'
        assert (s0[1].axes_manager[1].units == '1 / nm')
        assert (s0[1].axes_manager[1].name == 'x')
        assert (s0[1].axes_manager[2].name == 'y')

    def test_load_diffraction_area_scan(self):
        fname0 = os.path.join(
            self.dirpathnew, '16x16-diffraction_imagel_5x5x256x256_EDS.emi')
        s0 = load(fname0)
        # s0[0] contains EDS
        assert s0[0].data.shape == (5, 5, 4000)
        assert s0[0].axes_manager.signal_dimension == 1
        assert (
            s0[0].metadata.Acquisition_instrument.TEM.acquisition_mode == 'STEM')
        np.testing.assert_allclose(s0[0].axes_manager[0].scale, 1.87390, rtol=1E-5)
        assert s0[0].axes_manager[0].units == 'nm'
        np.testing.assert_allclose(s0[0].axes_manager[1].scale, -1.87390, rtol=1E-5)
        assert s0[0].axes_manager[1].units == 'nm'
        np.testing.assert_allclose(s0[0].axes_manager[2].scale, 5.0, rtol=1E-5)
        assert s0[0].axes_manager[2].units == 'eV'
        # s0[1] contains diffraction patterns
        assert s0[1].data.shape == (5, 5, 256, 256)
        assert s0[1].axes_manager.signal_dimension == 2
        assert (
            s0[1].metadata.Acquisition_instrument.TEM.acquisition_mode == 'STEM')
        np.testing.assert_allclose(s0[1].axes_manager[0].scale, 1.87390, rtol=1E-5)
        assert s0[1].axes_manager[0].units == 'nm'
        np.testing.assert_allclose(s0[1].axes_manager[2].scale, 0.174353, rtol=1E-5)
        assert s0[1].axes_manager[2].units == '1 / nm'
        assert (s0[0].axes_manager[0].name == 'x')
        assert (s0[0].axes_manager[1].name == 'y')
        assert (s0[0].axes_manager[2].name == 'Energy')
        np.testing.assert_allclose(s0[1].axes_manager[0].scale, 1.87390, rtol=1E-5)
        assert (s0[1].axes_manager[0].name == 'x')
        np.testing.assert_allclose(s0[1].axes_manager[1].scale, -1.87390, rtol=1E-5)
        assert (s0[1].axes_manager[1].units == 'nm')
        assert (s0[1].axes_manager[1].name == 'y')
        assert (s0[1].axes_manager[2].name == 'x')
        np.testing.assert_allclose(s0[1].axes_manager[3].scale, 0.174353, rtol=1E-5)
        assert (s0[1].axes_manager[3].units == '1 / nm')
        assert (s0[1].axes_manager[3].name == 'y')

    def test_load_spectrum_point(self):
        fname0 = os.path.join(
            self.dirpathold, '16x16-point_spectrum-1x1024.emi')
        s0 = load(fname0)
        assert s0.data.shape == (1, 1024)
        assert s0.axes_manager.signal_dimension == 1
        assert (
            s0.metadata.Acquisition_instrument.TEM.acquisition_mode == 'STEM')
        # single spectrum should be imported as 1D data, not 2D
        # TODO: the following calibration is wrong because it parse the
        # 'Dim-1_CalibrationDelta' from the ser header, which is not correct in
        # case of point spectra. However, the position seems to be saved in
        # 'PositionX' and 'PositionY' arrays of the ser header, so it should
        # be possible to workaround using the position arrays.
#        np.testing.assert_almost_equal(
#            s0.axes_manager[0].scale, 1.0, places=5)
#        np.testing.assert_equal(s0.axes_manager[0].units, '')
#        np.testing.assert_is(s0.axes_manager[0].name, 'Position index')
        np.testing.assert_allclose(s0.axes_manager[1].scale, 0.2, rtol=1E-5)
        assert s0.axes_manager[1].units == 'eV'
        assert (s0.axes_manager[1].name == 'Energy')

        fname1 = os.path.join(
            self.dirpathold, '16x16-2_point-spectra-2x1024.emi')
        s1 = load(fname1)
        assert s1.data.shape == (2, 1024)
        assert s1.axes_manager.signal_dimension == 1
        assert (
            s1.metadata.Acquisition_instrument.TEM.acquisition_mode == 'STEM')
        np.testing.assert_allclose(s0.axes_manager[1].scale, 0.2, rtol=1E-5)
        assert s0.axes_manager[1].units == 'eV'
        assert (s0.axes_manager[1].name == 'Energy')

    def test_load_spectrum_line_scan(self):
        fname0 = os.path.join(
            self.dirpathold, '16x16-line_profile_horizontal_10x1024.emi')
        s0 = load(fname0)
        assert s0.data.shape == (10, 1024)
        assert s0.axes_manager.signal_dimension == 1
        assert (
            s0.metadata.Acquisition_instrument.TEM.acquisition_mode == 'STEM')
        np.testing.assert_allclose(s0.axes_manager[0].scale, 0.123034, rtol=1E-5)
        assert s0.axes_manager[0].units == 'nm'
        np.testing.assert_allclose(s0.axes_manager[1].scale, 0.2, rtol=1E-5)
        assert s0.axes_manager[1].units == 'eV'
        assert (s0.axes_manager[0].name == 'x')
        assert (s0.axes_manager[1].name == 'Energy')

        fname1 = os.path.join(
            self.dirpathold, '16x16-line_profile_diagonal_10x1024.emi')
        s1 = load(fname1)
        assert s1.data.shape == (10, 1024)
        assert s1.axes_manager.signal_dimension == 1
        assert (
            s1.metadata.Acquisition_instrument.TEM.acquisition_mode == 'STEM')
        np.testing.assert_allclose(s1.axes_manager[0].scale, 0.166318, rtol=1E-5)
        assert s1.axes_manager[0].units == 'nm'
        np.testing.assert_allclose(s1.axes_manager[1].scale, 0.2, rtol=1E-5)
        assert s1.axes_manager[1].units == 'eV'
        assert (s0.axes_manager[0].name == 'x')

    def test_load_spectrum_area_scan(self):
        fname0 = os.path.join(
            self.dirpathold, '16x16-spectrum_image-5x5x1024.emi')
        s0 = load(fname0)
        assert s0.data.shape == (5, 5, 1024)
        assert s0.axes_manager.signal_dimension == 1
        assert (
            s0.metadata.Acquisition_instrument.TEM.acquisition_mode == 'STEM')
        np.testing.assert_allclose(s0.axes_manager[0].scale, 0.120539, rtol=1E-5)
        assert s0.axes_manager[0].units == 'nm'
        np.testing.assert_allclose(s0.axes_manager[1].scale, -0.120539, rtol=1E-5)
        assert s0.axes_manager[1].units == 'nm'
        np.testing.assert_allclose(s0.axes_manager[2].scale, 0.2, rtol=1E-5)
        assert s0.axes_manager[2].units == 'eV'
        assert (s0.axes_manager[2].name == 'Energy')
        assert (s0.axes_manager[1].name == 'y')
        assert (s0.axes_manager[2].name == 'Energy')

    def test_load_spectrum_area_scan_not_square(self):
        fname0 = os.path.join(
            self.dirpathnew, '16x16-spectrum_image_5x5x4000-not_square.emi')
        s0 = load(fname0)
        assert s0.data.shape == (5, 5, 4000)
        assert s0.axes_manager.signal_dimension == 1
        assert (
            s0.metadata.Acquisition_instrument.TEM.acquisition_mode == 'STEM')
        np.testing.assert_allclose(s0.axes_manager[0].scale, 1.98591, rtol=1E-5)
        assert s0.axes_manager[0].units == 'nm'
        np.testing.assert_allclose(s0.axes_manager[1].scale, -4.25819, rtol=1E-5)
        assert s0.axes_manager[1].units == 'nm'
        np.testing.assert_allclose(s0.axes_manager[2].scale, 5.0, rtol=1E-5)
        assert s0.axes_manager[2].units == 'eV'

    def test_load_search(self):
        fname0 = os.path.join(self.dirpathnew, '128x128-TEM_search.emi')
        s0 = load(fname0)
        assert s0.data.shape == (128, 128)
        np.testing.assert_allclose(s0.axes_manager[0].scale, 5.26121, rtol=1E-5)
        assert s0.axes_manager[0].units == 'nm'
        np.testing.assert_allclose(s0.axes_manager[1].scale, 5.26121, rtol=1E-5)
        assert s0.axes_manager[1].units == 'nm'

        fname1 = os.path.join(self.dirpathold, '16x16_STEM_BF_DF_search.emi')
        s1 = load(fname1)
        assert len(s1) == 2
        for s in s1:
            assert s.data.shape == (16, 16)
            assert (
                s.metadata.Acquisition_instrument.TEM.acquisition_mode == 'STEM')
            np.testing.assert_allclose(
                s.axes_manager[0].scale, 22.026285, rtol=1E-5)
            assert s.axes_manager[0].units == 'nm'
            np.testing.assert_allclose(
                s.axes_manager[1].scale, 22.026285, rtol=1E-5)
            assert s.axes_manager[1].units == 'nm'

    def test_load_stack_image_preview(self):
        fname0 = os.path.join(self.dirpathold, '64x64x5_TEM_preview.emi')
        s0 = load(fname0)
        assert s0.data.shape == (5, 64, 64)
        assert s0.axes_manager.signal_dimension == 2
        assert (
            s0.metadata.Acquisition_instrument.TEM.acquisition_mode == 'TEM')
        np.testing.assert_allclose(s0.axes_manager[0].scale, 1.0, rtol=1E-5)
        assert s0.axes_manager[0].units is t.Undefined
        np.testing.assert_allclose(s0.axes_manager[1].scale, 6.281833, rtol=1E-5)
        assert s0.axes_manager[1].units == 'nm'
        np.testing.assert_allclose(s0.axes_manager[2].scale, 6.281833, rtol=1E-5)
        assert s0.axes_manager[2].units == 'nm'
        assert s0.axes_manager[0].units is t.Undefined
        assert (s0.axes_manager[0].scale == 1.0)
        assert (s0.axes_manager[0].name is t.Undefined)
        assert (s0.axes_manager[1].name == 'x')
        assert (s0.axes_manager[2].name == 'y')

        fname2 = os.path.join(
            self.dirpathnew, '128x128x5-diffraction_preview.emi')
        s2 = load(fname2)
        assert s2.data.shape == (5, 128, 128)
        np.testing.assert_allclose(s2.axes_manager[1].scale, 0.042464, rtol=1E-5)
        assert s0.axes_manager[0].units is t.Undefined
        assert s2.axes_manager[1].units == '1 / nm'
        np.testing.assert_allclose(s2.axes_manager[2].scale, 0.042464, rtol=1E-5)
        assert s2.axes_manager[2].units == '1 / nm'

        fname1 = os.path.join(
            self.dirpathold, '16x16x5_STEM_BF_DF_preview.emi')
        s1 = load(fname1)
        assert len(s1) == 2
        for s in s1:
            assert s.data.shape == (5, 16, 16)
            assert (
                s.metadata.Acquisition_instrument.TEM.acquisition_mode == 'STEM')
            np.testing.assert_allclose(s.axes_manager[0].scale, 1.0, rtol=1E-5)
            assert s.axes_manager[1].units == 'nm'
            np.testing.assert_allclose(
                s.axes_manager[1].scale, 21.510044, rtol=1E-5)

    def test_load_acquire(self):
        fname0 = os.path.join(self.dirpathold, '64x64_TEM_images_acquire.emi')
        s0 = load(fname0)
        assert s0.axes_manager.signal_dimension == 2
        assert (
            s0.metadata.Acquisition_instrument.TEM.acquisition_mode == 'TEM')
        np.testing.assert_allclose(s0.axes_manager[0].scale, 6.281833, rtol=1E-5)
        assert s0.axes_manager[0].units == 'nm'
        np.testing.assert_allclose(s0.axes_manager[1].scale, 6.281833, rtol=1E-5)
        assert s0.axes_manager[1].units == 'nm'
        assert (s0.axes_manager[0].name == 'x')
        assert (s0.axes_manager[1].name == 'y')

        fname1 = os.path.join(self.dirpathold, '16x16_STEM_BF_DF_acquire.emi')
        s1 = load(fname1)
        assert len(s1) == 2
        for s in s1:
            assert s.data.shape == (16, 16)
            assert (
                s.metadata.Acquisition_instrument.TEM.acquisition_mode == 'STEM')
            np.testing.assert_allclose(
                s.axes_manager[0].scale, 21.510044, rtol=1E-5)
            assert s.axes_manager[0].units == 'nm'
            np.testing.assert_allclose(
                s.axes_manager[1].scale, 21.510044, rtol=1E-5)
            assert s.axes_manager[1].units == 'nm'
            assert (s.axes_manager[0].name == 'x')
            assert (s.axes_manager[1].name == 'y')

    @pytest.mark.parametrize("only_valid_data", (True, False))
    def test_load_TotalNumberElements_ne_ValidNumberElements(self,
                                                             only_valid_data):
        fname0 = os.path.join(self.dirpathold, 'X - Au NP EELS_2.ser')
        s0 = load(fname0, only_valid_data=only_valid_data)
        nav_shape = () if only_valid_data else (2, )
        assert s0.data.shape == nav_shape + (2048, )
        assert len(s0.axes_manager.navigation_axes) == len(nav_shape)
        np.testing.assert_allclose(s0.axes_manager[-1].offset, 2160, rtol=1E-5)
        np.testing.assert_allclose(s0.axes_manager[-1].scale, 0.2, rtol=1E-5)

        fname1 = os.path.join(self.dirpathold, '03_Scanning Preview.emi')
        s1 = load(fname1, only_valid_data=only_valid_data)
        nav_shape = (5, ) if only_valid_data else (200, )
        assert s1.data.shape == nav_shape + (128, 128)
        nav_axes = s1.axes_manager.navigation_axes
        sig_axes = s1.axes_manager.signal_axes
        assert len(nav_axes) == len(nav_shape)
        assert sig_axes[-1].size == sig_axes[1].size == 128
        np.testing.assert_allclose(sig_axes[0].scale, 0.38435, rtol=1E-5)
        np.testing.assert_allclose(sig_axes[1].scale, 0.38435, rtol=1E-5)

    def test_read_STEM_TEM_mode(self):
        # TEM image
        fname0 = os.path.join(self.dirpathold, '64x64_TEM_images_acquire.emi')
        s0 = load(fname0)
        assert (
            s0.metadata.Acquisition_instrument.TEM.acquisition_mode == "TEM")
        # TEM diffraction
        fname1 = os.path.join(self.dirpathold, '64x64_diffraction_acquire.emi')
        s1 = load(fname1)
        assert (
            s1.metadata.Acquisition_instrument.TEM.acquisition_mode == "TEM")
        fname2 = os.path.join(self.dirpathold, '16x16_STEM_BF_DF_acquire.emi')
        # STEM diffraction
        s2 = load(fname2)
        assert (
            s2[0].metadata.Acquisition_instrument.TEM.acquisition_mode == "STEM")
        assert (
            s2[1].metadata.Acquisition_instrument.TEM.acquisition_mode == "STEM")

    def test_load_units_scale(self):
        # TEM image
        fname0 = os.path.join(self.dirpathold, '64x64_TEM_images_acquire.emi')
        s0 = load(fname0)
        np.testing.assert_allclose(s0.axes_manager[0].scale, 6.28183, rtol=1E-5)
        assert s0.axes_manager[0].units == 'nm'
        np.testing.assert_allclose(s0.metadata.Acquisition_instrument.TEM.magnification,
                        19500.0, rtol=1E-5)
        # TEM diffraction
        fname1 = os.path.join(self.dirpathold, '64x64_diffraction_acquire.emi')
        s1 = load(fname1)
        np.testing.assert_allclose(s1.axes_manager[0].scale, 0.101571, rtol=1E-5)
        assert s1.axes_manager[0].units == '1 / nm'
        np.testing.assert_allclose(s1.metadata.Acquisition_instrument.TEM.camera_length,
                        490.0, rtol=1E-5)
        # STEM diffraction
        fname2 = os.path.join(self.dirpathold, '16x16_STEM_BF_DF_acquire.emi')
        s2 = load(fname2)
        assert s2[0].axes_manager[0].units == 'nm'
        np.testing.assert_allclose(s2[0].axes_manager[0].scale, 21.5100, rtol=1E-5)
        np.testing.assert_allclose(s2[0].metadata.Acquisition_instrument.TEM.magnification,
                        10000.0, rtol=1E-5)

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
        assert unit == 'meters'

        # objects is empty dictionary
        with pytest.warns(
                UserWarning,
                match="The navigation axes units could not be determined."):
            unit = _guess_units_from_mode({}, header0)
        assert unit == 'meters'

    @pytest.mark.parametrize('stack_metadata', [True, False, 0])
    def test_load_multisignal_stack(self, stack_metadata):
        fname0 = os.path.join(
            self.dirpathnew, '16x16-line_profile_horizontal_5x128x128_EDS.emi')
        s = load([fname0, fname0], stack=True, stack_metadata=stack_metadata)
        assert s[0].axes_manager.navigation_shape == (5, 2)
        assert s[0].axes_manager.signal_shape == (4000, )
        assert s[1].axes_manager.navigation_shape == (5, 2)
        assert s[1].axes_manager.signal_shape == (128, 128)

        om = s[0].original_metadata
        assert om.has_item('stack_elements') is (stack_metadata is True)

    def test_load_multisignal_stack_mismatch(self):
        fname0 = os.path.join(
            self.dirpathnew, '16x16-diffraction_imagel_5x5x256x256_EDS.emi')
        fname1 = os.path.join(
            self.dirpathnew,
            '16x16-diffraction_imagel_5x5x256x256_EDS_copy.emi')
        with pytest.raises(ValueError) as cm:
            load([fname0, fname1], stack=True)
            cm.match("The number of sub-signals per file does not match*")
        load([fname0, fname1])

    def test_date_time(self):
        fname0 = os.path.join(self.dirpathold, '64x64_TEM_images_acquire.emi')
        s = load(fname0)
        assert s.metadata.General.date == "2016-02-21"
        assert s.metadata.General.time == "17:50:18"
        fname1 = os.path.join(self.dirpathold,
                              '16x16-line_profile_horizontal_10x1024.emi')
        s = load(fname1)
        assert (s.metadata.General.date == "2016-02-22")
        assert (s.metadata.General.time == "11:50:36")

    def test_metadata_TEM(self):
        fname0 = os.path.join(self.dirpathold, '64x64_TEM_images_acquire.emi')
        s = load(fname0)
        assert (
            s.metadata.Acquisition_instrument.TEM.beam_energy == 200.0)
        assert (
            s.metadata.Acquisition_instrument.TEM.magnification ==
            19500.0)
        assert (
            s.metadata.Acquisition_instrument.TEM.microscope ==
            "Tecnai 200 kV D2267 SuperTwin")
        np.testing.assert_allclose(s.metadata.Acquisition_instrument.TEM.Stage.tilt_alpha,
                        0.0, rtol=1E-5)

    def test_metadata_STEM(self):
        fname0 = os.path.join(self.dirpathold, '16x16_STEM_BF_DF_acquire.emi')
        s = load(fname0)[0]
        assert (
            s.metadata.Acquisition_instrument.TEM.beam_energy == 200.0)
        assert (
            s.metadata.Acquisition_instrument.TEM.camera_length == 40.0)
        assert (
            s.metadata.Acquisition_instrument.TEM.magnification ==
            10000.0)
        assert (
            s.metadata.Acquisition_instrument.TEM.microscope ==
            "Tecnai 200 kV D2267 SuperTwin")
        np.testing.assert_allclose(s.metadata.Acquisition_instrument.TEM.Stage.tilt_alpha,
                        0.0, rtol=1E-6)
        np.testing.assert_allclose(s.metadata.Acquisition_instrument.TEM.Stage.tilt_beta,
                        0.0, rtol=1E-6)
        np.testing.assert_allclose(s.metadata.Acquisition_instrument.TEM.Stage.x,
                        -0.000158, rtol=1E-6)
        np.testing.assert_allclose(s.metadata.Acquisition_instrument.TEM.Stage.y,
                        1.9e-05, rtol=1E-6)
        np.testing.assert_allclose(s.metadata.Acquisition_instrument.TEM.Stage.z,
                        0.0, rtol=1E-6)

    def test_metadata_diffraction(self):
        fname0 = os.path.join(self.dirpathold, '64x64_diffraction_acquire.emi')
        s = load(fname0)
        assert (
            s.metadata.Acquisition_instrument.TEM.beam_energy == 200.0)
        assert (
            s.metadata.Acquisition_instrument.TEM.camera_length ==
            490.0)
        assert (
            s.metadata.Acquisition_instrument.TEM.microscope ==
            "Tecnai 200 kV D2267 SuperTwin")
