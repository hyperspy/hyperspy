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


import json
import os

import numpy as np
import pytest

from hyperspy import __version__ as hs_version
from hyperspy.io import load
from hyperspy.io_plugins.digital_micrograph import (DigitalMicrographReader,
                                                    ImageObject)
from hyperspy.signals import Signal1D, Signal2D
from hyperspy.tests.io.generate_dm_testing_files import (dm3_data_types,
                                                         dm4_data_types)

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
        assert self.imageobject._parse_string("string") == "string"

    def test_parse_string_convert_float(self):
        assert self.imageobject._parse_string("5", False) == '5'
        assert self.imageobject._parse_string("5", True) == 5
        assert self.imageobject._parse_string("Imaging", True) == None


def test_missing_tag():
    fname = os.path.join(MY_PATH, "dm3_2D_data",
                         "test_diffraction_pattern_tags_removed.dm3")
    s = load(fname)
    md = s.metadata
    np.testing.assert_allclose(md.Acquisition_instrument.TEM.beam_energy, 200.0)
    np.testing.assert_allclose(md.Acquisition_instrument.TEM.Camera.exposure, 0.2)
    assert md.General.date == "2014-07-09"
    assert md.General.time == "18:56:37"
    assert md.General.title == "test_diffraction_pattern_tags_removed"


def test_read_TEM_metadata():
    fname = os.path.join(MY_PATH, "tiff_files", "test_dm_image_um_unit.dm3")
    s = load(fname)
    md = s.metadata
    assert md.Acquisition_instrument.TEM.acquisition_mode == "TEM"
    np.testing.assert_allclose(md.Acquisition_instrument.TEM.beam_energy, 200.0)
    np.testing.assert_allclose(md.Acquisition_instrument.TEM.Camera.exposure, 0.5)
    np.testing.assert_allclose(md.Acquisition_instrument.TEM.magnification, 51.0)
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
    np.testing.assert_allclose(md.Acquisition_instrument.TEM.beam_energy, 200.0)
    np.testing.assert_allclose(md.Acquisition_instrument.TEM.Camera.exposure, 0.2)
    np.testing.assert_allclose(md.Acquisition_instrument.TEM.camera_length, 320.0)
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
    np.testing.assert_allclose(md.Acquisition_instrument.TEM.beam_energy, 200.0)
    np.testing.assert_allclose(md.Acquisition_instrument.TEM.dwell_time, 3.5E-6)
    np.testing.assert_allclose(md.Acquisition_instrument.TEM.camera_length, 135.0)
    np.testing.assert_allclose(
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
    np.testing.assert_allclose(md.Acquisition_instrument.TEM.beam_energy, 200.0)
    assert md.Acquisition_instrument.TEM.microscope == "FEI Titan"
    np.testing.assert_allclose(md.Acquisition_instrument.TEM.camera_length, 135.0)
    np.testing.assert_allclose(
        md.Acquisition_instrument.TEM.magnification,
        640000.0)
    np.testing.assert_allclose(md.Acquisition_instrument.TEM.Stage.tilt_alpha, 24.95,
                    atol=1E-2)
    np.testing.assert_allclose(md.Acquisition_instrument.TEM.Stage.x, -0.478619,
                    atol=1E-2)
    np.testing.assert_allclose(md.Acquisition_instrument.TEM.Stage.y, 0.0554612,
                    atol=1E-2)
    np.testing.assert_allclose(md.Acquisition_instrument.TEM.Stage.z, 0.036348,
                    atol=1E-2)
    np.testing.assert_allclose(
        md.Acquisition_instrument.TEM.convergence_angle, 21.0)
    np.testing.assert_allclose(
        md.Acquisition_instrument.TEM.Detector.EELS.collection_angle, 0.0)
    np.testing.assert_allclose(
        md.Acquisition_instrument.TEM.Detector.EELS.exposure,
        0.0035)
    np.testing.assert_allclose(
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
    np.testing.assert_allclose(
        md.Signal.Noise_properties.Variance_linear_model.gain_factor,
        0.1285347)
    np.testing.assert_allclose(
        md.Signal.Noise_properties.Variance_linear_model.gain_offset,
        0.0)


def test_read_SI_metadata():
    fname = os.path.join(MY_PATH, "dm4_3D_data", "EELS_SI.dm4")
    s = load(fname)
    md = s.metadata
    assert md.Acquisition_instrument.TEM.acquisition_mode == "STEM"
    assert md.General.date == "2019-05-14"
    assert md.General.time == "20:50:13"
    np.testing.assert_allclose(
        md.Acquisition_instrument.TEM.Detector.EELS.aperture_size, 5.0)
    np.testing.assert_allclose(
        md.Acquisition_instrument.TEM.convergence_angle, 21.0)
    np.testing.assert_allclose(
        md.Acquisition_instrument.TEM.Detector.EELS.collection_angle, 62.0)
    np.testing.assert_allclose(
        md.Acquisition_instrument.TEM.Detector.EELS.frame_number, 1)
    np.testing.assert_allclose(
        md.Acquisition_instrument.TEM.Detector.EELS.dwell_time,
        1.9950125E-2)


def test_read_EDS_metadata():
    fname = os.path.join(MY_PATH, "dm3_1D_data", "test-EDS_spectrum.dm3")
    s = load(fname)
    md = s.metadata
    assert md.Acquisition_instrument.TEM.acquisition_mode == "STEM"
    np.testing.assert_allclose(
        md.Acquisition_instrument.TEM.Detector.EDS.azimuth_angle, 45.0)
    np.testing.assert_allclose(
        md.Acquisition_instrument.TEM.Detector.EDS.elevation_angle, 18.0)
    np.testing.assert_allclose(
        md.Acquisition_instrument.TEM.Detector.EDS.energy_resolution_MnKa, 130.0)
    np.testing.assert_allclose(
        md.Acquisition_instrument.TEM.Detector.EDS.live_time, 3.806)
    np.testing.assert_allclose(
        md.Acquisition_instrument.TEM.Detector.EDS.real_time, 4.233)
    np.testing.assert_allclose(md.Acquisition_instrument.TEM.Stage.tilt_alpha, 24.95,
                    atol=1E-2)
    np.testing.assert_allclose(md.Acquisition_instrument.TEM.beam_energy, 200.0)
    assert md.Acquisition_instrument.TEM.microscope == "FEI Titan"
    np.testing.assert_allclose(md.Acquisition_instrument.TEM.camera_length, 135.0)
    np.testing.assert_allclose(
        md.Acquisition_instrument.TEM.magnification,
        320000.0)
    assert md.General.date == "2016-08-08"
    assert md.General.original_filename == "test-EDS_spectrum.dm3"
    assert md.General.title == "EDS Spectrum"
    assert md.General.time == "21:46:19"
    assert md.Signal.quantity == "X-rays (Counts)"
    assert md.Signal.signal_type == "EDS_TEM"
    np.testing.assert_allclose(
        md.Signal.Noise_properties.Variance_linear_model.gain_factor,
        1.0)
    np.testing.assert_allclose(
        md.Signal.Noise_properties.Variance_linear_model.gain_offset,
        0.0)
    assert s.axes_manager[-1].units == "keV"

def test_read_MonoCL_pmt_metadata():
    fname = os.path.join(MY_PATH, "dm4_1D_data", "test-MonoCL_spectrum-pmt.dm4")
    s = load(fname)
    md = s.metadata
    assert md.Signal.signal_type == "CL"
    assert md.Signal.format == "Spectrum"
    assert md.Signal.quantity == "Intensity (Counts)"
    assert md.General.date == "2020-10-27"
    assert md.General.original_filename == "test-MonoCL_spectrum-pmt.dm4"
    assert md.General.title == "test-CL_spectrum-pmt"
    assert md.Acquisition_instrument.Spectrometer.acquisition_mode == "Serial dispersive"
    assert md.Acquisition_instrument.Detector.detector_type == "PMT"
    assert md.Acquisition_instrument.Spectrometer.Grating.groove_density == 1200
    assert md.Acquisition_instrument.Detector.integration_time == 1.0
    assert md.Acquisition_instrument.Spectrometer.step_size == 0.5
    np.testing.assert_allclose(
        md.Acquisition_instrument.Spectrometer.start_wavelength, 166.233642)

def test_read_MonarcCL_pmt_metadata():
    fname = os.path.join(MY_PATH, "dm4_1D_data", "test-MonarcCL_spectrum-pmt.dm4")
    s = load(fname)
    md = s.metadata
    assert md.Signal.signal_type == "CL"
    assert md.Signal.format == "Spectrum"
    assert md.Signal.quantity == "Intensity (Counts)"
    assert md.General.date == "2022-01-17"
    assert md.General.original_filename == "test-MonarcCL_spectrum-pmt.dm4"
    assert md.General.title == "CL_15kX_3-60_pmt2s"
    assert md.Acquisition_instrument.Spectrometer.acquisition_mode == "Serial dispersive"
    assert md.Acquisition_instrument.Detector.detector_type == "PMT"
    assert md.Acquisition_instrument.Spectrometer.Grating.groove_density == 300
    assert md.Acquisition_instrument.Detector.integration_time == 2.0
    assert md.Acquisition_instrument.Spectrometer.step_size == 1.0
    np.testing.assert_allclose(
        md.Acquisition_instrument.Spectrometer.start_wavelength, 160.067505)

def test_read_MonoCL_ccd_metadata():
    fname = os.path.join(MY_PATH, "dm4_1D_data", "test-MonoCL_spectrum-ccd.dm4")
    s = load(fname)
    md = s.metadata
    assert md.Signal.signal_type == "CL"
    assert md.Signal.format == "Spectrum"
    assert md.Signal.quantity == "Intensity (Counts)"
    assert md.General.date == "2020-09-11"
    assert md.General.time == "17:04:19"
    assert md.General.original_filename == "test-MonoCL_spectrum-ccd.dm4"
    assert md.General.title == "test-CL_spectrum-ccd"
    assert md.Acquisition_instrument.Detector.detector_type == "CCD"
    assert md.Acquisition_instrument.SEM.acquisition_mode == "SEM"
    assert md.Acquisition_instrument.SEM.microscope == "Ultra55"
    assert md.Acquisition_instrument.SEM.beam_energy == 5.0
    assert md.Acquisition_instrument.SEM.magnification == 10104.515625
    assert md.Acquisition_instrument.Spectrometer.acquisition_mode == "Parallel dispersive"
    assert md.Acquisition_instrument.Spectrometer.Grating.groove_density == 300.0
    assert md.Acquisition_instrument.Detector.exposure_per_frame == 30.0
    assert md.Acquisition_instrument.Detector.frames == 1.0
    assert md.Acquisition_instrument.Detector.integration_time == 30.0
    np.testing.assert_allclose(
        md.Acquisition_instrument.Spectrometer.central_wavelength, 949.974182)
    np.testing.assert_allclose(
        md.Acquisition_instrument.Detector.saturation_fraction, 0.01861909)
    assert md.Acquisition_instrument.Detector.binning == (1, 100)
    assert md.Acquisition_instrument.Detector.processing == "Dark Subtracted"
    assert md.Acquisition_instrument.Detector.sensor_roi == (0, 0, 100, 1336)
    assert md.Acquisition_instrument.Detector.pixel_size == 20.0

def test_read_MonarcCL_ccd_metadata():
    fname = os.path.join(MY_PATH, "dm4_1D_data", "test-MonarcCL_spectrum-ccd.dm4")
    s = load(fname)
    md = s.metadata
    assert md.Signal.signal_type == "CL"
    assert md.Signal.format == "Spectrum"
    assert md.Signal.quantity == "Intensity (Counts)"
    assert md.General.date == "2022-01-17"
    assert md.General.time == "16:09:21"
    assert md.General.original_filename == "test-MonarcCL_spectrum-ccd.dm4"
    assert md.General.title == "CL_15kX_3-60_CCD300s_bin2"
    assert md.Acquisition_instrument.Detector.detector_type == "CCD"
    assert md.Acquisition_instrument.SEM.acquisition_mode == "SEM"
    assert md.Acquisition_instrument.SEM.microscope == "Ultra55"
    assert md.Acquisition_instrument.SEM.beam_energy == 3.0
    assert md.Acquisition_instrument.SEM.magnification == 15000.0
    assert md.Acquisition_instrument.Spectrometer.acquisition_mode == "Parallel dispersive"
    np.testing.assert_allclose(
        md.Acquisition_instrument.Spectrometer.central_wavelength, 320.049683)
    assert md.Acquisition_instrument.Spectrometer.Grating.groove_density == 300.0
    assert md.Acquisition_instrument.Detector.exposure_per_frame == 300.0
    assert md.Acquisition_instrument.Detector.frames == 1.0
    assert md.Acquisition_instrument.Detector.integration_time == 300.0
    np.testing.assert_allclose(
        md.Acquisition_instrument.Detector.saturation_fraction, 0.08890307)
    assert md.Acquisition_instrument.Detector.binning == (2, 100)
    assert md.Acquisition_instrument.Detector.processing == "Dark Subtracted"
    assert md.Acquisition_instrument.Detector.sensor_roi == (0, 0, 100, 1336)
    assert md.Acquisition_instrument.Detector.pixel_size == 20.0
    #assert md.Acquisition_instrument.Spectrometer.entrance_slit_width == 1

def test_read_MonoCL_SI_metadata():
    fname = os.path.join(MY_PATH, "dm4_2D_data", "test-MonoCL_spectrum-SI.dm4")
    s = load(fname)
    md = s.metadata
    assert md.Signal.signal_type == "CL"
    assert md.Signal.format == "Spectrum image"
    assert md.Signal.quantity == "Intensity (Counts)"
    assert md.General.date == "2020-04-11"
    assert md.General.time == "14:41:01"
    assert md.General.original_filename == "test-MonoCL_spectrum-SI.dm4"
    assert md.General.title == "test-CL_spectrum-SI"
    assert md.Acquisition_instrument.Detector.detector_type == "CCD"
    assert md.Acquisition_instrument.SEM.acquisition_mode == "SEM"
    assert md.Acquisition_instrument.SEM.microscope == "Ultra55"
    assert md.Acquisition_instrument.SEM.beam_energy == 5.0
    np.testing.assert_allclose(
        md.Acquisition_instrument.SEM.magnification, 31661.427734)
    assert md.Acquisition_instrument.Spectrometer.acquisition_mode == "Parallel dispersive"
    assert md.Acquisition_instrument.Spectrometer.Grating.groove_density == 600.0
    np.testing.assert_allclose(
        md.Acquisition_instrument.Detector.exposure_per_frame, 0.05)
    assert md.Acquisition_instrument.Detector.frames == 1
    np.testing.assert_allclose(
        md.Acquisition_instrument.Detector.integration_time, 0.05)
    assert md.Acquisition_instrument.Detector.pixel_size == 20.0
    np.testing.assert_allclose(
        md.Acquisition_instrument.Spectrometer.central_wavelength, 869.983825)
    np.testing.assert_allclose(
        md.Acquisition_instrument.Detector.saturation_fraction[0], 0.09676377)
    assert md.Acquisition_instrument.Detector.binning == (1, 100)
    assert md.Acquisition_instrument.Detector.processing == "Dark Subtracted"
    assert md.Acquisition_instrument.Detector.sensor_roi == (0, 0, 100, 1336)
    assert md.Acquisition_instrument.Spectrum_image.drift_correction_periodicity == 1
    assert md.Acquisition_instrument.Spectrum_image.drift_correction_units == "second(s)"
    assert md.Acquisition_instrument.Spectrum_image.mode == "LineScan"

def test_read_MonarcCL_SI_metadata():
    fname = os.path.join(MY_PATH, "dm4_2D_data", "test-MonarcCL_spectrum-SI.dm4")
    s = load(fname)
    md = s.metadata
    assert md.Signal.signal_type == "CL"
    assert md.Signal.format == "Spectrum image"
    assert md.Signal.quantity == "Intensity (Counts)"
    assert md.General.date == "2021-09-16"
    assert md.General.time == "12:06:16"
    assert md.General.original_filename == "test-MonarcCL_spectrum-SI.dm4"
    assert md.General.title == "Monarc_SI_9pix"
    assert md.Acquisition_instrument.Detector.detector_type == "CCD"
    assert md.Acquisition_instrument.SEM.acquisition_mode == "SEM"
    assert md.Acquisition_instrument.SEM.microscope == "Zeiss SEM COM"
    assert md.Acquisition_instrument.SEM.beam_energy == 5.0
    np.testing.assert_allclose(
        md.Acquisition_instrument.SEM.magnification, 5884.540039)
    assert md.Acquisition_instrument.Spectrometer.acquisition_mode == "Parallel dispersive"
    assert md.Acquisition_instrument.Spectrometer.Grating.groove_density == 1200.0
    assert md.Acquisition_instrument.Spectrometer.entrance_slit_width == 0.256
    assert md.Acquisition_instrument.Spectrometer.bandpass == 0.9984
    np.testing.assert_allclose(
        md.Acquisition_instrument.Detector.exposure_per_frame, 0.05)
    assert md.Acquisition_instrument.Detector.frames == 1
    np.testing.assert_allclose(
        md.Acquisition_instrument.Detector.integration_time, 0.05)
    assert md.Acquisition_instrument.Detector.pixel_size == 20.0
    #np.testing.assert_allclose(
    #    md.Acquisition_instrument.Spectrometer.central_wavelength, 869.9838)
    np.testing.assert_allclose(
        md.Acquisition_instrument.Detector.saturation_fraction[0], 0.004867628)
    assert md.Acquisition_instrument.Detector.binning == (2, 400)
    assert md.Acquisition_instrument.Detector.processing == "Dark Subtracted"
    assert md.Acquisition_instrument.Detector.sensor_roi == (0, 0, 400, 1340)
    #assert md.Acquisition_instrument.Spectrum_image.drift_correction_periodicity == 1
    #assert md.Acquisition_instrument.Spectrum_image.drift_correction_units == "second(s)"
    assert md.Acquisition_instrument.Spectrum_image.mode == "2D Array"

def test_read_MonarcCL_image_metadata():
    fname = os.path.join(MY_PATH, "dm4_2D_data", "test-MonarcCL_mono-image.dm4")
    s = load(fname)
    md = s.metadata
    assert md.Signal.signal_type == "CL"
    assert md.Signal.quantity == "Intensity (counts)"
    assert md.General.date == "2021-05-14"
    assert md.General.time == "11:41:07"
    assert md.General.original_filename == "test-MonarcCL_mono-image.dm4"
    assert md.General.title == "MonoCL-image-rebin"
    assert md.Acquisition_instrument.SEM.acquisition_mode == "SEM"
    assert md.Acquisition_instrument.SEM.microscope == "Zeiss SEM COM"
    assert md.Acquisition_instrument.SEM.beam_energy == 3.0
    assert md.Acquisition_instrument.SEM.magnification == 2500.
    assert md.Acquisition_instrument.SEM.dwell_time == 1e-5
    assert md.Acquisition_instrument.Spectrometer.Grating.groove_density == 300.0
    assert md.Acquisition_instrument.Spectrometer.entrance_slit_width == 0.961
    np.testing.assert_allclose(
        md.Acquisition_instrument.Spectrometer.bandpass, 14.9915999)
    assert md.Acquisition_instrument.Detector.pmt_voltage == 1000

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

def test_multi_signal():
    fname = os.path.join(MY_PATH, "dm3_2D_data", "multi_signal.dm3")
    s = load(fname)

    # Make sure file is read as a list, and exactly two signals are found
    assert isinstance(s, list)
    assert len(s) == 2

    s1, s2 = s

    # First signal is an image, second is a plot
    assert isinstance(s1, Signal2D)
    assert isinstance(s2, Signal1D)

    s1_md_truth = {
        '_HyperSpy': {
            'Folding': {
                'unfolded': False,
                'signal_unfolded': False,
                'original_shape': None,
                'original_axes_manager': None}},
        'General': {'title': 'HAADF',
                    'original_filename': 'multi_signal.dm3',
                    'date': '2019-12-10',
                    'time': '15:32:41',
                    'authors': 'JohnDoe',
                    'FileIO': {
                        '0': {
                            'operation': 'load',
                            'hyperspy_version': hs_version,
                            'io_plugin':
                                'hyperspy.io_plugins.digital_micrograph'
                        }
                    }
        },
        'Signal': {'signal_type': '',
                   'quantity': 'Intensity',
                   'Noise_properties': {
                       'Variance_linear_model': {
                           'gain_factor': 1.0,
                           'gain_offset': 0.0}}},
        'Acquisition_instrument': {
            'TEM': {
                'beam_energy': 300.0,
                'Stage': {
                    'tilt_alpha': 0.001951998453075299,
                    'x': 0.07872150000000001,
                    'y': 0.100896,
                    'z': -0.0895279},
                'acquisition_mode': 'STEM',
                'beam_current': 0.0,
                'camera_length': 77.0,
                'magnification': 10000000.0,
                'microscope': 'Example Microscope',
                'dwell_time': 3.2400001525878905e-05}},
        'Sample': {
            'description': 'PrecipitateA'}
    }

    s2_md_truth = {
        '_HyperSpy': {
            'Folding': {
                'unfolded': False,
                'signal_unfolded': False,
                'original_shape': None,
                'original_axes_manager': None}},
        'General': {
            'title': 'Plot',
            'original_filename': 'multi_signal.dm3',
            'FileIO': {
                '0': {
                    'operation': 'load',
                    'hyperspy_version': hs_version,
                    'io_plugin': 'hyperspy.io_plugins.digital_micrograph'
                }
            }},
        'Signal': {
            'signal_type': '',
            'quantity': 'Intensity',
            'Noise_properties': {
                'Variance_linear_model': {
                    'gain_factor': 1.0,
                    'gain_offset': 0.0}}}
    }
    # remove timestamps from metadata since these are runtime dependent
    del s1.metadata.General.FileIO.Number_0.timestamp
    del s2.metadata.General.FileIO.Number_0.timestamp

    # make sure the metadata dictionaries are as we expect
    assert s1.metadata.as_dictionary() == s1_md_truth
    assert s2.metadata.as_dictionary() == s2_md_truth

    # rather than testing all of original metadata (huge), use length as a proxy
    assert len(json.dumps(s1.original_metadata.as_dictionary())) == 17779
    assert len(json.dumps(s2.original_metadata.as_dictionary())) == 15024
    
    # test axes
    assert s1.axes_manager[-1].is_binned == False
    assert s2.axes_manager[-1].is_binned == False

    # simple tests on the data itself:
    assert s1.data.sum() == 949490255
    assert s1.data.shape == (512, 512)
    assert s2.data.sum() == pytest.approx(28.085794, 0.01)
    assert s2.data.shape == (512,)


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
@pytest.mark.parametrize("lazy", (True, False))
def test_data(pdict, lazy):
    s = load(pdict["filename"], lazy=lazy)
    if lazy:
        s.compute(close_file=True)
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

def test_axes_bug_for_image():
    fname = os.path.join(MY_PATH, "dm3_2D_data", "test_STEM_image.dm3")
    s = load(fname)
    assert s.axes_manager[1].name == 'y'
