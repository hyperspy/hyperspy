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
# along with HyperSpy. If not, see <http://www.gnu.org/licenses/>.


import json
import os
import gc

import numpy as np
import pytest
import tempfile

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

def test_read_CL_pmt_metadata():
    fname = os.path.join(MY_PATH, "dm4_1D_data", "test-CL_spectrum-pmt.dm4")
    s = load(fname)
    md = s.metadata
    assert md.Signal.signal_type == "CL"
    assert md.Signal.format == "Spectrum"
    assert md.Signal.quantity == "Intensity (Counts)"
    assert md.General.date == "2020-10-27"
    assert md.General.original_filename == "test-CL_spectrum-pmt.dm4"
    assert md.General.title == "test-CL_spectrum-pmt"
    assert md.Acquisition_instrument.CL.acquisition_mode == "Serial dispersive"
    assert md.Acquisition_instrument.CL.detector_type == "linear"
    np.testing.assert_allclose(
        md.Acquisition_instrument.CL.dispersion_grating, 1200)
    np.testing.assert_allclose(
        md.Acquisition_instrument.CL.dwell_time, 1.0)
    np.testing.assert_allclose(
        md.Acquisition_instrument.CL.step_size, 0.5)
    np.testing.assert_allclose(
        md.Acquisition_instrument.CL.start_wavelength, 166.233642578125)

def test_read_CL_ccd_metadata():
    fname = os.path.join(MY_PATH, "dm4_1D_data", "test-CL_spectrum-ccd.dm4")
    s = load(fname)
    md = s.metadata
    assert md.Signal.signal_type == "CL"
    assert md.Signal.format == "Spectrum"
    assert md.Signal.quantity == "Intensity (Counts)"
    assert md.General.date == "2020-09-11"
    assert md.General.time == "17:04:19"
    assert md.General.original_filename == "test-CL_spectrum-ccd.dm4"
    assert md.General.title == "test-CL_spectrum-ccd"
    assert md.Acquisition_instrument.TEM.acquisition_mode == "SEM"
    assert md.Acquisition_instrument.TEM.microscope == "Ultra55"
    np.testing.assert_allclose(md.Acquisition_instrument.TEM.beam_energy, 5.0)
    np.testing.assert_allclose(
        md.Acquisition_instrument.TEM.magnification, 10104.515625)
    assert md.Acquisition_instrument.CL.acquisition_mode == "Parallel dispersive"
    np.testing.assert_allclose(
        md.Acquisition_instrument.CL.dispersion_grating, 300.0)
    np.testing.assert_allclose(
        md.Acquisition_instrument.CL.exposure, 30.0)
    np.testing.assert_allclose(
        md.Acquisition_instrument.CL.frame_number, 1.0)
    np.testing.assert_allclose(
        md.Acquisition_instrument.CL.integration_time, 30.0)
    np.testing.assert_allclose(
        md.Acquisition_instrument.CL.central_wavelength, 949.9741821289062)
    np.testing.assert_allclose(
        md.Acquisition_instrument.CL.saturation_fraction, 0.01861908845603466)
    assert md.Acquisition_instrument.CL.CCD.binning == (1, 100)
    assert md.Acquisition_instrument.CL.CCD.processing == "Dark Subtracted"
    assert md.Acquisition_instrument.CL.CCD.read_area == (0, 0, 100, 1336)

def test_read_CL_SI_metadata():
    fname = os.path.join(MY_PATH, "dm4_2D_data", "test-CL_spectrum-SI.dm4")
    s = load(fname)
    md = s.metadata
    assert md.Signal.signal_type == "CL"
    assert md.Signal.format == "Spectrum image"
    assert md.Signal.quantity == "Intensity (Counts)"
    assert md.General.date == "2020-04-11"
    assert md.General.time == "14:41:01"
    assert md.General.original_filename == "test-CL_spectrum-SI.dm4"
    assert md.General.title == "test-CL_spectrum-SI"
    assert md.Acquisition_instrument.TEM.acquisition_mode == "SEM"
    assert md.Acquisition_instrument.TEM.microscope == "Ultra55"
    np.testing.assert_allclose(md.Acquisition_instrument.TEM.beam_energy, 5.0)
    np.testing.assert_allclose(
        md.Acquisition_instrument.TEM.magnification, 31661.427734375)
    assert md.Acquisition_instrument.CL.acquisition_mode == "Parallel dispersive"
    np.testing.assert_allclose(
        md.Acquisition_instrument.CL.dispersion_grating, 600.0)
    np.testing.assert_allclose(
        md.Acquisition_instrument.CL.exposure, 0.05000000074505806)
    np.testing.assert_allclose(
        md.Acquisition_instrument.CL.frame_number, 1)
    np.testing.assert_allclose(
        md.Acquisition_instrument.CL.central_wavelength, 869.9838256835938)
    np.testing.assert_allclose(
        md.Acquisition_instrument.CL.saturation_fraction[0], 0.09676377475261688)
    assert md.Acquisition_instrument.CL.CCD.binning == (1, 100)
    assert md.Acquisition_instrument.CL.CCD.processing == "Dark Subtracted"
    assert md.Acquisition_instrument.CL.CCD.read_area == (0, 0, 100, 1336)
    assert md.Acquisition_instrument.CL.SI.drift_correction_periodicity == 1
    assert md.Acquisition_instrument.CL.SI.drift_correction_units == "second(s)"
    assert md.Acquisition_instrument.CL.SI.mode == "LineScan"


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
                    'authors': 'JohnDoe'},
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
            'original_filename': 'multi_signal.dm3'},
        'Signal': {
            'signal_type': '',
            'quantity': 'Intensity',
            'Noise_properties': {
                'Variance_linear_model': {
                    'gain_factor': 1.0,
                    'gain_offset': 0.0}}}
    }

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


@pytest.fixture(scope='module')
def tmpdir():
    import zipfile
    zipf = os.path.join(MY_PATH, "dm3_marker_files.zip")
    with zipfile.ZipFile(zipf, 'r') as zipped:
        with tempfile.TemporaryDirectory() as tmp:
            zipped.extractall(tmp)
            yield tmp
            gc.collect()


class TestAnnotationsToMarkers:

    def test_marker_roi_line_white(self, tmpdir):
        fname = os.path.join(
            tmpdir,
            "roi_line_marker_0_9_9_0_red_white_bottom_left_to_top_right.dm3")
        s = load(fname)
        marker_name = s.metadata.Markers.keys()[0]
        marker = s.metadata.Markers[marker_name]
        assert marker.get_data_position('x1') == 0.0
        assert marker.get_data_position('y1') == 9.0
        assert marker.get_data_position('x2') == 9.0
        assert marker.get_data_position('y2') == 0.0
        # The white color in the filename refers to the face color of the
        # drawing this is currently not implemented in HyperSpy's markers
        assert marker.marker_properties['color'] == (1, 0, 0)
        assert marker.marker_properties['linestyle'] == '--'

    def test_marker_roi_rectangle_white(self, tmpdir):
        fname = os.path.join(
            tmpdir,
            "roi_rectangle_marker_6_6_8_8_bottom_right_red_white.dm3")
        s = load(fname)
        marker_name = s.metadata.Markers.keys()[0]
        marker = s.metadata.Markers[marker_name]
        # Note: the coordinates are shifted by 1 in both the x and y direction
        # compared to how the rectangle is visualized in DM.
        assert marker.get_data_position('x1') == 7.0
        assert marker.get_data_position('y1') == 7.0
        assert marker.get_data_position('x2') == 9.0
        assert marker.get_data_position('y2') == 9.0
        # The white color refers to the face color of the drawing
        # this is currently not implemented in HyperSpy's markers
        assert marker.marker_properties['color'] == (1, 0, 0)
        assert marker.marker_properties['linestyle'] == '--'

    def test_marker_line_colors(self, tmpdir):
        s_black = load(os.path.join(tmpdir, "line_marker_black.dm3"))
        s_blue = load(os.path.join(tmpdir, "line_marker_blue.dm3"))
        s_green = load(os.path.join(tmpdir, "line_marker_green.dm3"))
        s_red = load(os.path.join(tmpdir, "line_marker_red.dm3"))
        s_white = load(os.path.join(tmpdir, "line_marker_white.dm3"))
        marker_name_black = s_black.metadata.Markers.keys()[0]
        marker_black = s_black.metadata.Markers[marker_name_black]
        marker_name_blue = s_blue.metadata.Markers.keys()[0]
        marker_blue = s_blue.metadata.Markers[marker_name_blue]
        marker_name_green = s_green.metadata.Markers.keys()[0]
        marker_green = s_green.metadata.Markers[marker_name_green]
        marker_name_red = s_red.metadata.Markers.keys()[0]
        marker_red = s_red.metadata.Markers[marker_name_red]
        marker_name_white = s_white.metadata.Markers.keys()[0]
        marker_white = s_white.metadata.Markers[marker_name_white]
        assert marker_black.marker_properties['color'] == (0, 0, 0)
        assert marker_blue.marker_properties['color'] == (0, 0, 1)
        assert marker_green.marker_properties['color'] == (0, 1, 0)
        assert marker_red.marker_properties['color'] == (1, 0, 0)
        assert marker_white.marker_properties['color'] == (1, 1, 1)

    def test_line_with_text(self, tmpdir):
        s = load(os.path.join(tmpdir, "line_with_text_white.dm3"))
        marker_line = s.metadata.Markers.LineSegment16
        marker_text = s.metadata.Markers.Text17
        assert marker_line.marker_properties['color'] == (1, 1, 1)
        assert marker_text.get_data_position('text') == '2.93 '

    def test_point_marker(self, tmpdir):
        s = load(os.path.join(tmpdir, "point_marker_top_left_red_0_0.dm3"))
        marker = s.metadata.Markers[s.metadata.Markers.keys()[0]]
        assert marker.get_data_position('x1') == 0.0
        assert marker.get_data_position('y1') == 0.0
        assert marker.marker_properties['color'] == (1, 0, 0)

    def test_rectangle_marker_white(self, tmpdir):
        s = load(os.path.join(tmpdir, "rectangle_marker_white_top_left.dm3"))
        marker = s.metadata.Markers[s.metadata.Markers.keys()[0]]
        assert marker.marker_properties['color'] == (1, 1, 1)

    def test_text_marker(self, tmpdir):
        s = load(os.path.join(tmpdir, "text_marker_white_top_right.dm3"))
        marker = s.metadata.Markers[s.metadata.Markers.keys()[0]]
        assert marker.get_data_position('text') == 'A text'
        assert marker.marker_properties['color'] == (1, 1, 1)

    def test_si_survey_line_roi(self, tmpdir):
        s = load(os.path.join(tmpdir,
                    "SI_survey_image_active_beam_15_1_"
                    "line_roi_spectrum_image_0_3_2_6_"
                    "roi_box_spatial_drift_4_0_14_10.dm3"))
        scale = s.axes_manager[0].scale
        marker_beam = s.metadata.Markers.Point11
        marker_beam_text = s.metadata.Markers.Text11
        assert marker_beam.get_data_position('x1')/scale == 15.
        assert marker_beam.get_data_position('y1')/scale == 1.
        assert marker_beam_text.get_data_position('text') == 'Beam'

        marker_line = s.metadata.Markers.LineSegment14
        marker_line_text = s.metadata.Markers.Text14
        assert marker_line.get_data_position('x1')/scale == 0.
        assert marker_line.get_data_position('y1')/scale == 3.
        assert marker_line.get_data_position('x2')/scale == 2.
        assert marker_line.get_data_position('y2')/scale == 6.
        assert marker_line_text.get_data_position('text') == 'Spectrum Image'

        marker_rect = s.metadata.Markers.Rectangle15
        marker_rect_text = s.metadata.Markers.Text15
        assert marker_rect.get_data_position('x1')/scale == 4.
        assert marker_rect.get_data_position('y1')/scale == 0.
        assert marker_rect.get_data_position('x2')/scale == 14.
        assert marker_rect.get_data_position('y2')/scale == 10.
        assert marker_rect_text.get_data_position('text') == 'Spatial Drift'


def test_axes_bug_for_image():
    fname = os.path.join(MY_PATH, "dm3_2D_data", "test_STEM_image.dm3")
    s = load(fname)
    assert s.axes_manager[1].name == 'y'
