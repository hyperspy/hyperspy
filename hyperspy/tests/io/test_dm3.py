# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
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

import numpy as np
from numpy.testing import assert_allclose
import pytest

from hyperspy.tests.io.generate_dm_testing_files import (dm3_data_types,
                                                         dm4_data_types)
from hyperspy.io import load
from hyperspy.io_plugins.digital_micrograph import DigitalMicrographReader, ImageObject

MY_PATH = os.path.dirname(__file__)


class TestImageObject():

    def setup_method(self, method):
        self.imageobject = ImageObject({}, "")

    def _load_file(self, fname):
        with open(fname, "rb") as f:
            dm = DigitalMicrographReader(f)
            dm.parse_file()
            self.imdict = dm.get_image_dictionaries()
        return [ImageObject(imdict, fname) for imdict in self.imdict]

    def test_get_microscope_name(self):
        fname = os.path.join(MY_PATH, "dm3_2D_data",
                             "test_diffraction_pattern_tags_removed.dm3")
        images = self._load_file(fname)
        image = images[0]
        # Should return None because the tags are missing
        assert image._get_microscope_name(image.imdict.ImageTags) is None

        fname = os.path.join(MY_PATH, "dm3_2D_data",
                             "test_diffraction_pattern.dm3")
        images = self._load_file(fname)
        image = images[0]
        assert image._get_microscope_name(
            image.imdict.ImageTags) == "FEI Tecnai"

    def test_get_date(self):
        assert self.imageobject._get_date("11/13/2016") == "2016-11-13"

    def test_get_time(self):
        assert self.imageobject._get_time("6:56:37 pm") == "18:56:37"

    def test_parse_string(self):
        assert self.imageobject._parse_string("") is None
        assert self.imageobject._parse_string("string") is "string"


def test_missing_tag():
    fname = os.path.join(MY_PATH, "dm3_2D_data",
                         "test_diffraction_pattern_tags_removed.dm3")
    s = load(fname)
    md = s.metadata
    assert_allclose(md.Acquisition_instrument.TEM.beam_energy, 200.0)
    assert_allclose(md.Acquisition_instrument.TEM.exposure_time, 0.2)
    assert md.General.date == "2014-07-09"
    assert md.General.time == "18:56:37"
    assert md.General.title == "test_diffraction_pattern_tags_removed"


def test_read_TEM_metadata():
    fname = os.path.join(MY_PATH, "tiff_files", "test_dm_image_um_unit.dm3")
    s = load(fname)
    md = s.metadata
    assert md.Acquisition_instrument.TEM.acquisition_mode == "TEM"
    assert_allclose(md.Acquisition_instrument.TEM.beam_energy, 200.0)
    assert_allclose(md.Acquisition_instrument.TEM.exposure_time, 0.5)
    assert_allclose(md.Acquisition_instrument.TEM.magnification, 51.0)
    assert md.Acquisition_instrument.TEM.microscope == "FEI Tecnai"
    assert md.General.date == "2015-07-20"
    assert md.General.original_filename == "test_dm_image_um_unit.dm3"
    assert md.General.title == "test_dm_image_um_unit"
    assert md.General.time == "18:48:25"
    assert md.Signal.quantity == "Intensity"
    assert md.Signal.signal_type == ""


def test_read_Diffraction_metadata():
    fname = os.path.join(
        MY_PATH,
        "dm3_2D_data",
        "test_diffraction_pattern.dm3")
    s = load(fname)
    md = s.metadata
    assert md.Acquisition_instrument.TEM.acquisition_mode == "TEM"
    assert_allclose(md.Acquisition_instrument.TEM.beam_energy, 200.0)
    assert_allclose(md.Acquisition_instrument.TEM.exposure_time, 0.2)
    assert_allclose(md.Acquisition_instrument.TEM.camera_length, 320.0)
    assert md.Acquisition_instrument.TEM.microscope == "FEI Tecnai"
    assert md.General.date == "2014-07-09"
    assert (
        md.General.original_filename ==
        "test_diffraction_pattern.dm3")
    assert md.General.title == "test_diffraction_pattern"
    assert md.General.time == "18:56:37"
    assert md.Signal.quantity == "Intensity"
    assert md.Signal.signal_type == ""


def test_read_STEM_metadata():
    fname = os.path.join(MY_PATH, "dm3_2D_data", "test_STEM_image.dm3")
    s = load(fname)
    md = s.metadata
    assert md.Acquisition_instrument.TEM.acquisition_mode == "STEM"
    assert_allclose(md.Acquisition_instrument.TEM.beam_energy, 200.0)
    assert_allclose(md.Acquisition_instrument.TEM.dwell_time, 3.5E-6)
    assert_allclose(md.Acquisition_instrument.TEM.camera_length, 135.0)
    assert_allclose(
        md.Acquisition_instrument.TEM.magnification,
        225000.0)
    assert md.Acquisition_instrument.TEM.microscope == "FEI Titan"
    assert md.General.date == "2016-08-08"
    assert md.General.original_filename == "test_STEM_image.dm3"
    assert md.General.title == "test_STEM_image"
    assert md.General.time == "16:26:37"
    assert md.Signal.quantity == "Intensity"
    assert md.Signal.signal_type == ""


def test_read_EELS_metadata():
    fname = os.path.join(MY_PATH, "dm3_1D_data", "test-EELS_spectrum.dm3")
    s = load(fname)
    md = s.metadata
    assert md.Acquisition_instrument.TEM.acquisition_mode == "STEM"
    assert_allclose(md.Acquisition_instrument.TEM.beam_energy, 200.0)
    assert md.Acquisition_instrument.TEM.microscope == "FEI Titan"
    assert_allclose(md.Acquisition_instrument.TEM.camera_length, 135.0)
    assert_allclose(
        md.Acquisition_instrument.TEM.magnification,
        640000.0)
    assert_allclose(md.Acquisition_instrument.TEM.tilt_stage, 24.95,
                    atol=1E-2)
    assert_allclose(
        md.Acquisition_instrument.TEM.convergence_angle, 21.0)
    assert_allclose(
        md.Acquisition_instrument.TEM.Detector.EELS.collection_angle, 0.0)
    assert_allclose(
        md.Acquisition_instrument.TEM.Detector.EELS.exposure,
        0.0035)
    assert_allclose(
        md.Acquisition_instrument.TEM.Detector.EELS.frame_number, 50)
    assert (
        md.Acquisition_instrument.TEM.Detector.EELS.spectrometer ==
        'GIF Quantum ER')
    assert (
        md.Acquisition_instrument.TEM.Detector.EELS.aperture_size ==
        5.0)
    assert md.General.date == "2016-08-08"
    assert md.General.original_filename == "test-EELS_spectrum.dm3"
    assert md.General.title == "EELS Acquire"
    assert md.General.time == "19:35:17"
    assert md.Signal.quantity == "Electrons (Counts)"
    assert md.Signal.signal_type == "EELS"
    assert_allclose(
        md.Signal.Noise_properties.Variance_linear_model.gain_factor,
        0.1285347)
    assert_allclose(
        md.Signal.Noise_properties.Variance_linear_model.gain_offset,
        0.0)


def test_read_EDS_metadata():
    fname = os.path.join(MY_PATH, "dm3_1D_data", "test-EDS_spectrum.dm3")
    s = load(fname)
    md = s.metadata
    assert md.Acquisition_instrument.TEM.acquisition_mode == "STEM"
    assert_allclose(
        md.Acquisition_instrument.TEM.Detector.EDS.azimuth_angle, 45.0)
    assert_allclose(
        md.Acquisition_instrument.TEM.Detector.EDS.elevation_angle, 18.0)
    assert_allclose(
        md.Acquisition_instrument.TEM.Detector.EDS.energy_resolution_MnKa, 130.0)
    assert_allclose(
        md.Acquisition_instrument.TEM.Detector.EDS.live_time, 3.806)
    assert_allclose(
        md.Acquisition_instrument.TEM.Detector.EDS.real_time, 4.233)
    assert_allclose(md.Acquisition_instrument.TEM.tilt_stage, 24.95,
                    atol=1E-2)
    assert_allclose(md.Acquisition_instrument.TEM.beam_energy, 200.0)
    assert md.Acquisition_instrument.TEM.microscope == "FEI Titan"
    assert_allclose(md.Acquisition_instrument.TEM.camera_length, 135.0)
    assert_allclose(
        md.Acquisition_instrument.TEM.magnification,
        320000.0)
    assert md.General.date == "2016-08-08"
    assert md.General.original_filename == "test-EDS_spectrum.dm3"
    assert md.General.title == "EDS Spectrum"
    assert md.General.time == "21:46:19"
    assert md.Signal.quantity == "X-rays (Counts)"
    assert md.Signal.signal_type == "EDS_TEM"
    assert_allclose(
        md.Signal.Noise_properties.Variance_linear_model.gain_factor,
        1.0)
    assert_allclose(
        md.Signal.Noise_properties.Variance_linear_model.gain_offset,
        0.0)


def test_location():
    fname_list = ['Fei HAADF-DE_location.dm3', 'Fei HAADF-FR_location.dm3',
                  'Fei HAADF-MX_location.dm3', 'Fei HAADF-UK_location.dm3']
    s = load(os.path.join(MY_PATH, "dm3_locale", fname_list[0]))
    assert s.metadata.General.date == "2016-08-27"
    assert s.metadata.General.time == "20:54:33"
    s = load(os.path.join(MY_PATH, "dm3_locale", fname_list[1]))
    assert s.metadata.General.date == "2016-08-27"
    assert s.metadata.General.time == "20:55:20"
    s = load(os.path.join(MY_PATH, "dm3_locale", fname_list[2]))
    assert s.metadata.General.date == "2016-08-27"
#    assert_equal(s.metadata.General.time, "20:55:20") # MX not working
    s = load(os.path.join(MY_PATH, "dm3_locale", fname_list[3]))
    assert s.metadata.General.date == "2016-08-27"
    assert s.metadata.General.time == "20:52:30"


def generate_parameters():
    parameters = []
    for dim in range(1, 4):
        for key in dm3_data_types.keys():
            subfolder = 'dm3_%iD_data' % dim
            fname = "test-%s.dm3" % key
            filename = os.path.join(MY_PATH, subfolder, fname)
            parameters.append({
                "filename": filename,
                "subfolder": subfolder,
                "key": key, })
        for key in dm4_data_types.keys():
            subfolder = 'dm4_%iD_data' % dim
            fname = "test-%s.dm4" % key
            filename = os.path.join(MY_PATH, subfolder, fname)
            parameters.append({
                "filename": filename,
                "subfolder": subfolder,
                "key": key, })
    return parameters


@pytest.mark.parametrize("pdict", generate_parameters())
def test_data(pdict):
    s = load(pdict["filename"])
    key = pdict["key"]
    assert s.data.dtype == np.dtype(dm4_data_types[key])
    subfolder = pdict["subfolder"]
    print(pdict["subfolder"])
    if subfolder in ('dm3_1D_data', 'dm4_1D_data'):
        dat = np.arange(1, 3)
    elif subfolder in ('dm3_2D_data', 'dm4_2D_data'):
        dat = np.arange(1, 5).reshape(2, 2)
    elif subfolder in ('dm3_3D_data', 'dm4_3D_data'):
        dat = np.arange(1, 9).reshape(2, 2, 2)
    else:
        raise ValueError
    dat = dat.astype(dm4_data_types[key])
    if key in (8, 23):  # RGBA
        dat["A"][:] = 0
    np.testing.assert_array_equal(s.data, dat,
                                  err_msg='content %s type % i: '
                                  '\n%s not equal to \n%s' %
                                  (subfolder, key, str(s.data), str(dat)))
