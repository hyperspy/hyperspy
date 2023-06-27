import os
from packaging.version import Version
from pathlib import Path
import tempfile
import warnings
import zipfile


import numpy as np
import pytest
import traits.api as t
import tifffile

import hyperspy.api as hs
from hyperspy.misc.test_utils import assert_deep_almost_equal
from hyperspy import __version__ as hs_version
import hyperspy.io_plugins.tiff


MY_PATH = os.path.dirname(__file__)
MY_PATH2 = os.path.join(MY_PATH, "tiff_files")
TMP_DIR = tempfile.TemporaryDirectory()


def teardown_module():
    TMP_DIR.cleanup()


def test_rgba16():
    path = Path(TMP_DIR.name)
    zipf = os.path.join(MY_PATH2, "test_rgba16.zip")
    with zipfile.ZipFile(zipf, 'r') as zipped:
        zipped.extractall(path)
    
    s = hs.load(path / 'test_rgba16.tif')
    data = np.load(Path(MY_PATH) / "npz_files" / "test_rgba16.npz")['a']
    assert (s.data == data).all()
    assert s.axes_manager[0].units == t.Undefined
    assert s.axes_manager[1].units == t.Undefined
    assert s.axes_manager[2].units == t.Undefined
    np.testing.assert_allclose(s.axes_manager[0].scale, 1.0, atol=1E-5)
    np.testing.assert_allclose(s.axes_manager[1].scale, 1.0, atol=1E-5)
    np.testing.assert_allclose(s.axes_manager[2].scale, 1.0, atol=1E-5)
    assert s.metadata.General.date == '2014-03-31'
    assert s.metadata.General.time == '16:35:46'


def _compare_signal_shape_data(s0, s1):
    assert s0.data.shape == s1.data.shape
    np.testing.assert_equal(s0.data, s1.data)


def test_read_unit_um():
    # Load DM file and save it as tif
    s = hs.load(os.path.join(MY_PATH2, 'test_dm_image_um_unit.dm3'))
    assert s.axes_manager[0].units == 'µm'
    assert s.axes_manager[1].units == 'µm'
    np.testing.assert_allclose(s.axes_manager[0].scale, 0.16867, atol=1E-5)
    np.testing.assert_allclose(s.axes_manager[1].scale, 0.16867, atol=1E-5)
    assert s.metadata.General.date == '2015-07-20'
    assert s.metadata.General.time == '18:48:25'
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'tiff_files', 'test_export_um_unit.tif')
        s.save(fname, overwrite=True, export_scale=True)
        # load tif file
        s2 = hs.load(fname)
        assert s.axes_manager[0].units == 'µm'
        assert s.axes_manager[1].units == 'µm'
        np.testing.assert_allclose(s2.axes_manager[0].scale, 0.16867, atol=1E-5)
        np.testing.assert_allclose(s2.axes_manager[1].scale, 0.16867, atol=1E-5)
        assert s2.metadata.General.date == s.metadata.General.date
        assert s2.metadata.General.time == s.metadata.General.time


def test_write_read_intensity_axes_DM():
    s = hs.load(os.path.join(MY_PATH2, 'test_dm_image_um_unit.dm3'))
    s.metadata.Signal.set_item('quantity', 'Electrons (Counts)')
    d = {'gain_factor': 5.0,
         'gain_offset': 2.0}
    s.metadata.Signal.set_item('Noise_properties.Variance_linear_model', d)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'tiff_files', 'test_export_um_unit2.tif')
        s.save(fname, overwrite=True, export_scale=True)
        s2 = hs.load(fname)
        assert_deep_almost_equal(s.metadata.Signal.as_dictionary(),
                                 s2.metadata.Signal.as_dictionary())


def test_read_unit_from_imagej():
    fname = os.path.join(MY_PATH, 'tiff_files',
                         'test_loading_image_saved_with_imageJ.tif')
    s = hs.load(fname)
    assert s.axes_manager[0].units == 'µm'
    assert s.axes_manager[1].units == 'µm'
    np.testing.assert_allclose(s.axes_manager[0].scale, 0.16867, atol=1E-5)
    np.testing.assert_allclose(s.axes_manager[1].scale, 0.16867, atol=1E-5)


def test_read_unit_from_imagej_stack():
    fname = os.path.join(MY_PATH, 'tiff_files',
                         'test_loading_image_saved_with_imageJ_stack.tif')
    s = hs.load(fname)
    assert s.data.shape == (2, 68, 68)
    assert s.axes_manager[0].units == t.Undefined
    assert s.axes_manager[1].units == 'µm'
    assert s.axes_manager[2].units == 'µm'
    np.testing.assert_allclose(s.axes_manager[0].scale, 2.5, atol=1E-5)
    np.testing.assert_allclose(s.axes_manager[1].scale, 0.16867, atol=1E-5)
    np.testing.assert_allclose(s.axes_manager[2].scale, 0.16867, atol=1E-5)


@pytest.mark.parametrize("lazy", [True, False])
def test_read_unit_from_DM_stack(lazy):
    fname = os.path.join(MY_PATH, 'tiff_files',
                         'test_loading_image_saved_with_DM_stack.tif')
    s = hs.load(fname, lazy=lazy)
    assert s.data.shape == (2, 68, 68)
    assert s.axes_manager[0].units == 's'
    assert s.axes_manager[1].units == 'µm'
    assert s.axes_manager[2].units == 'µm'
    np.testing.assert_allclose(s.axes_manager[0].scale, 2.5, atol=1E-5)
    np.testing.assert_allclose(s.axes_manager[1].scale, 0.16867, atol=1E-5)
    np.testing.assert_allclose(s.axes_manager[2].scale, 1.68674, atol=1E-5)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname2 = os.path.join(
            tmpdir, 'test_loading_image_saved_with_DM_stack2.tif')
        s.save(fname2, overwrite=True)
        s2 = hs.load(fname2)
        _compare_signal_shape_data(s, s2)
        assert s2.axes_manager[0].units == s.axes_manager[0].units
        assert s2.axes_manager[1].units == 'µm'
        assert s2.axes_manager[2].units == 'µm'
        np.testing.assert_allclose(
            s2.axes_manager[0].scale, s.axes_manager[0].scale, atol=1E-5)
        np.testing.assert_allclose(
            s2.axes_manager[1].scale, s.axes_manager[1].scale, atol=1E-5)
        np.testing.assert_allclose(
            s2.axes_manager[2].scale, s.axes_manager[2].scale, atol=1E-5)
        np.testing.assert_allclose(
            s2.axes_manager[0].offset, s.axes_manager[0].offset, atol=1E-5)
        np.testing.assert_allclose(
            s2.axes_manager[1].offset, s.axes_manager[1].offset, atol=1E-5)
        np.testing.assert_allclose(
            s2.axes_manager[2].offset, s.axes_manager[2].offset, atol=1E-5)


def test_read_unit_from_imagej_stack_no_scale():
    fname = os.path.join(MY_PATH, 'tiff_files',
                         'test_loading_image_saved_with_imageJ_stack_no_scale.tif')
    s = hs.load(fname)
    assert s.data.shape == (2, 68, 68)
    assert s.axes_manager[0].units == t.Undefined
    assert s.axes_manager[1].units == t.Undefined
    assert s.axes_manager[2].units == t.Undefined
    np.testing.assert_allclose(s.axes_manager[0].scale, 1.0, atol=1E-5)
    np.testing.assert_allclose(s.axes_manager[1].scale, 1.0, atol=1E-5)
    np.testing.assert_allclose(s.axes_manager[2].scale, 1.0, atol=1E-5)


def test_read_unit_from_imagej_no_scale():
    fname = os.path.join(MY_PATH, 'tiff_files',
                         'test_loading_image_saved_with_imageJ_no_scale.tif')
    s = hs.load(fname)
    assert s.axes_manager[0].units == t.Undefined
    assert s.axes_manager[1].units == t.Undefined
    np.testing.assert_allclose(s.axes_manager[0].scale, 1.0, atol=1E-5)
    np.testing.assert_allclose(s.axes_manager[1].scale, 1.0, atol=1E-5)


def test_write_read_unit_imagej():
    fname = os.path.join(MY_PATH, 'tiff_files',
                         'test_loading_image_saved_with_imageJ.tif')
    s = hs.load(fname, convert_units=True)
    s.axes_manager[0].units = 'µm'
    s.axes_manager[1].units = 'µm'
    with tempfile.TemporaryDirectory() as tmpdir:
        fname2 = os.path.join(
            tmpdir, 'test_loading_image_saved_with_imageJ2.tif')
        s.save(fname2, export_scale=True, overwrite=True)
        s2 = hs.load(fname2)
        assert s2.axes_manager[0].units == 'µm'
        assert s2.axes_manager[1].units == 'µm'
        assert s.data.shape == s.data.shape


def test_write_read_unit_imagej_with_description():
    fname = os.path.join(MY_PATH, 'tiff_files',
                         'test_loading_image_saved_with_imageJ.tif')
    s = hs.load(fname)
    s.axes_manager[0].units = 'µm'
    s.axes_manager[1].units = 'µm'
    np.testing.assert_allclose(s.axes_manager[0].scale, 0.16867, atol=1E-5)
    np.testing.assert_allclose(s.axes_manager[1].scale, 0.16867, atol=1E-5)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname2 = os.path.join(tmpdir, 'description.tif')
        s.save(fname2, export_scale=False, overwrite=True, description='test')
        s2 = hs.load(fname2)
        assert s2.axes_manager[0].units == t.Undefined
        assert s2.axes_manager[1].units == t.Undefined
        np.testing.assert_allclose(s2.axes_manager[0].scale, 1.0, atol=1E-5)
        np.testing.assert_allclose(s2.axes_manager[1].scale, 1.0, atol=1E-5)

        fname3 = os.path.join(tmpdir, 'description2.tif')
        s.save(fname3, export_scale=True, overwrite=True, description='test')
        s3 = hs.load(fname3, convert_units=True)
        assert s3.axes_manager[0].units == 'µm'
        assert s3.axes_manager[1].units == 'µm'
        np.testing.assert_allclose(s3.axes_manager[0].scale, 0.16867, atol=1E-5)
        np.testing.assert_allclose(s3.axes_manager[1].scale, 0.16867, atol=1E-5)


def test_saving_with_custom_tag():
    s = hs.signals.Signal2D(
        np.arange(
            10 * 15,
            dtype=np.uint8).reshape(
            (10,
             15)))
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test_saving_with_custom_tag.tif')
        extratag = [(65000, 's', 1, "Random metadata", False)]
        s.save(fname, extratags=extratag, overwrite=True)
        s2 = hs.load(fname)
        assert (s2.original_metadata['Number_65000'] ==
                "Random metadata")


def _test_read_unit_from_dm():
    fname = os.path.join(MY_PATH2, 'test_loading_image_saved_with_DM.tif')
    s = hs.load(fname)
    assert s.axes_manager[0].units == 'µm'
    assert s.axes_manager[1].units == 'µm'
    np.testing.assert_allclose(s.axes_manager[0].scale, 0.16867, atol=1E-5)
    np.testing.assert_allclose(s.axes_manager[1].scale, 0.16867, atol=1E-5)
    np.testing.assert_allclose(s.axes_manager[0].offset, 139.66264, atol=1E-5)
    np.testing.assert_allclose(s.axes_manager[1].offset, 128.19276, atol=1E-5)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname2 = os.path.join(tmpdir, "DM2.tif")
        s.save(fname2, overwrite=True)
        s2 = hs.load(fname2)
        _compare_signal_shape_data(s, s2)
        assert s2.axes_manager[0].units == 'micron'
        assert s2.axes_manager[1].units == 'micron'
        np.testing.assert_allclose(s2.axes_manager[0].scale, s.axes_manager[0].scale,
                        atol=1E-5)
        np.testing.assert_allclose(s2.axes_manager[1].scale, s.axes_manager[1].scale,
                        atol=1E-5)
        np.testing.assert_allclose(s2.axes_manager[0].offset, s.axes_manager[0].offset,
                        atol=1E-5)
        np.testing.assert_allclose(s2.axes_manager[1].offset, s.axes_manager[1].offset,
                        atol=1E-5)


def test_write_scale_unit():
    _test_write_scale_unit(export_scale=True)


def test_write_scale_unit_no_export_scale():
    _test_write_scale_unit(export_scale=False)


def _test_write_scale_unit(export_scale=True):
    """ Lazy test, still need to open the files in ImageJ or DM to check if the
        scale and unit are correct """
    s = hs.signals.Signal2D(
        np.arange(
            10 * 15,
            dtype=np.uint8).reshape(
            (10,
             15)))
    s.axes_manager[0].name = 'x'
    s.axes_manager[1].name = 'y'
    s.axes_manager['x'].scale = 0.25
    s.axes_manager['y'].scale = 0.25
    s.axes_manager['x'].units = 'nm'
    s.axes_manager['y'].units = 'nm'
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(
            tmpdir, 'test_export_scale_unit_%s.tif' % export_scale)
        s.save(fname, overwrite=True, export_scale=export_scale)


def test_write_scale_unit_not_square_pixel():
    """ Lazy test, still need to open the files in ImageJ or DM to check if the
        scale and unit are correct """
    s = hs.signals.Signal2D(
        np.arange(
            10 * 15,
            dtype=np.uint8).reshape(
            (10,
             15)))
    s.change_dtype(np.uint8)
    s.axes_manager[0].name = 'x'
    s.axes_manager[1].name = 'y'
    s.axes_manager['x'].scale = 0.25
    s.axes_manager['y'].scale = 0.5
    s.axes_manager['x'].units = 'nm'
    s.axes_manager['y'].units = 'µm'
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(
            tmpdir, 'test_export_scale_unit_not_square_pixel.tif')
        s.save(fname, overwrite=True, export_scale=True)


def test_write_scale_with_undefined_unit():
    """ Lazy test, still need to open the files in ImageJ or DM to check if the
        scale and unit are correct """
    s = hs.signals.Signal2D(
        np.arange(
            10 * 15,
            dtype=np.uint8).reshape(
            (10,
             15)))
    s.axes_manager[0].name = 'x'
    s.axes_manager[1].name = 'y'
    s.axes_manager['x'].scale = 0.25
    s.axes_manager['y'].scale = 0.25
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(
            tmpdir, 'test_export_scale_undefined_unit.tif')
        s.save(fname, overwrite=True, export_scale=True)


def test_write_scale_with_undefined_scale():
    """ Lazy test, still need to open the files in ImageJ or DM to check if the
        scale and unit are correct """
    s = hs.signals.Signal2D(
        np.arange(
            10 * 15,
            dtype=np.uint8).reshape(
            (10,
             15)))
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(
            tmpdir, 'test_export_scale_undefined_scale.tif')
        s.save(fname, overwrite=True, export_scale=True)
        s1 = hs.load(fname)
        _compare_signal_shape_data(s, s1)


def test_write_scale_with_um_unit():
    """ Lazy test, still need to open the files in ImageJ or DM to check if the
        scale and unit are correct """
    s = hs.load(os.path.join(MY_PATH, 'tiff_files',
                             'test_dm_image_um_unit.dm3'))
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test_export_um_unit.tif')
        s.save(fname, overwrite=True, export_scale=True)
        s1 = hs.load(fname)
        _compare_signal_shape_data(s, s1)


def test_write_scale_unit_image_stack():
    """ Lazy test, still need to open the files in ImageJ or DM to check if the
        scale and unit are correct """
    s = hs.signals.Signal2D(
        np.arange(
            5 * 10 * 15,
            dtype=np.uint8).reshape(
            (5,
             10,
             15)))
    s.axes_manager[0].scale = 0.25
    s.axes_manager[1].scale = 0.5
    s.axes_manager[2].scale = 1.5
    s.axes_manager[0].units = 'nm'
    s.axes_manager[1].units = 'µm'
    s.axes_manager[2].units = 'µm'
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test_export_scale_unit_stack2.tif')
        s.save(fname, overwrite=True, export_scale=True)
        s1 = hs.load(fname, convert_units=True)
        _compare_signal_shape_data(s, s1)
        assert s1.axes_manager[0].units == 'pm'
        # only one unit can be read
        assert s1.axes_manager[1].units == 'µm'
        assert s1.axes_manager[2].units == 'µm'
        np.testing.assert_allclose(s1.axes_manager[0].scale, 250.0)
        np.testing.assert_allclose(s1.axes_manager[1].scale, s.axes_manager[1].scale)
        np.testing.assert_allclose(s1.axes_manager[2].scale, s.axes_manager[2].scale)


def test_saving_loading_stack_no_scale():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test_export_scale_unit_stack2.tif')
        s0 = hs.signals.Signal2D(np.zeros((10, 20, 30)))
        s0.save(fname, overwrite=True)
        s1 = hs.load(fname)
        _compare_signal_shape_data(s0, s1)


FEI_Helios_metadata = {'Acquisition_instrument': {'SEM': {'Stage': {'rotation': -2.3611,
                                                                    'tilt': 6.54498e-06,
                                                                    'x': 2.576e-05,
                                                                    'y': -0.000194177,
                                                                    'z': 0.007965},
                                                          'beam_current': 0.00625,
                                                          'beam_energy': 5.0,
                                                          'dwell_time': 1e-05,
                                                          'microscope': 'Helios NanoLab" 660',
                                                          'working_distance': 4.03466}},
                       'General': {'original_filename': 'FEI-Helios-Ebeam-8bits.tif',
                                   'title': '',
                                   'authors': 'supervisor',
                                   'date': '2016-06-13',
                                   'time': '17:06:40',
                                   'FileIO': {'0': {
                                       'operation': 'load',
                                       'hyperspy_version': hs_version,
                                       'io_plugin': 'hyperspy.io_plugins.tiff'
                                   }}
                                   },
                       'Signal': {'signal_type': ''},
                       '_HyperSpy': {'Folding': {'original_axes_manager': None,
                                                 'original_shape': None,
                                                 'signal_unfolded': False,
                                                 'unfolded': False}}}

FEI_navcam_metadata = {
    '_HyperSpy': {'Folding': {'unfolded': False, 'signal_unfolded': False,
                              'original_shape': None, 'original_axes_manager': None}},
    'General': {'original_filename': 'FEI-Helios-navcam-with-no-IRBeam.tif',
                'title': '', 'date': '2022-05-17', 'time': '09:07:08', 'authors': 'user',
                'FileIO': {'0': {'operation': 'load', 'io_plugin': 'hyperspy.io_plugins.tiff'}}},
    'Signal': {'signal_type': ''},
    'Acquisition_instrument': {'SEM': {'Stage': {'x': 0.0699197, 'y': 0.000811186, 'z': 0,
                                                 'rotation': 1.07874, 'tilt': 6.54498e-06},
                                       'working_distance': -0.012, 'microscope': 'Helios NanoLab" 660'}}}


class TestReadFEIHelios:
    
    path = Path(TMP_DIR.name)
    
    @classmethod
    def setup_class(cls):
        zipf = os.path.join(MY_PATH2, "tiff_FEI_Helios.zip")
        with zipfile.ZipFile(zipf, 'r') as zipped:
            zipped.extractall(cls.path)

    def test_read_FEI_SEM_scale_metadata_8bits(self):
        fname = self.path / 'FEI-Helios-Ebeam-8bits.tif'
        s = hs.load(fname, convert_units=True)
        assert s.axes_manager[0].units == 'µm'
        assert s.axes_manager[1].units == 'µm'
        np.testing.assert_allclose(s.axes_manager[0].scale, 3.3724, rtol=1E-5)
        np.testing.assert_allclose(s.axes_manager[1].scale, 3.3724, rtol=1E-5)
        assert s.data.dtype == 'uint8'
        # delete timestamp from metadata since it's runtime dependent
        del s.metadata.General.FileIO.Number_0.timestamp
        FEI_Helios_metadata['General'][
            'original_filename'] = 'FEI-Helios-Ebeam-8bits.tif'
        assert_deep_almost_equal(s.metadata.as_dictionary(), FEI_Helios_metadata)
    
    def test_read_FEI_SEM_scale_metadata_16bits(self):
        fname = self.path / 'FEI-Helios-Ebeam-16bits.tif'
        s = hs.load(fname, convert_units=True)
        assert s.axes_manager[0].units == 'µm'
        assert s.axes_manager[1].units == 'µm'
        np.testing.assert_allclose(s.axes_manager[0].scale, 3.3724, rtol=1E-5)
        np.testing.assert_allclose(s.axes_manager[1].scale, 3.3724, rtol=1E-5)
        assert s.data.dtype == 'uint16'
        # delete timestamp from metadata since it's runtime dependent
        del s.metadata.General.FileIO.Number_0.timestamp
        FEI_Helios_metadata['General'][
            'original_filename'] = 'FEI-Helios-Ebeam-16bits.tif'
        assert_deep_almost_equal(s.metadata.as_dictionary(), FEI_Helios_metadata)

    def test_read_FEI_navcam_metadata(self):
        fname = self.path / 'FEI-Helios-navcam.tif'
        s = hs.load(fname, convert_units=True)
        assert s.axes_manager[0].units == 'mm'
        assert s.axes_manager[1].units == 'mm'
        np.testing.assert_allclose(s.axes_manager[0].scale, 0.2640, rtol=.0001)
        np.testing.assert_allclose(s.axes_manager[1].scale, 0.2640, rtol=.0001)
        assert s.data.dtype == 'uint8'
        # delete timestamp and version from metadata since it's runtime dependent
        del s.metadata.General.FileIO.Number_0.timestamp
        del s.metadata.General.FileIO.Number_0.hyperspy_version
        FEI_navcam_metadata['General']['original_filename'] = 'FEI-Helios-navcam.tif'
        assert_deep_almost_equal(s.metadata.as_dictionary(), FEI_navcam_metadata)


    def test_read_FEI_navcam_no_IRBeam_metadata(self):
        fname = self.path / 'FEI-Helios-navcam-with-no-IRBeam.tif'
        s = hs.load(fname, convert_units=True)
        assert s.axes_manager[0].units == t.Undefined
        assert s.axes_manager[1].units == t.Undefined
        np.testing.assert_allclose(s.axes_manager[0].scale, 1, rtol=0)
        np.testing.assert_allclose(s.axes_manager[1].scale, 1, rtol=0)
        assert s.data.dtype == 'uint8'
        # delete timestamp and version from metadata since it's runtime dependent
        del s.metadata.General.FileIO.Number_0.timestamp
        del s.metadata.General.FileIO.Number_0.hyperspy_version
        FEI_navcam_metadata['General']['original_filename'] = 'FEI-Helios-navcam-with-no-IRBeam.tif'
        assert_deep_almost_equal(s.metadata.as_dictionary(), FEI_navcam_metadata)


    def test_read_FEI_navcam_no_IRBeam_bad_floats_metadata(self):
        fname = self.path / 'FEI-Helios-navcam-with-no-IRBeam-bad-floats.tif'
        s = hs.load(fname, convert_units=True)

        # delete timestamp and version from metadata since it's runtime dependent
        del s.metadata.General.FileIO.Number_0.timestamp
        del s.metadata.General.FileIO.Number_0.hyperspy_version
        FEI_navcam_metadata['General']['original_filename'] = 'FEI-Helios-navcam-with-no-IRBeam-bad-floats.tif'

        # working distance in the file was a bogus value, so it shouldn't be in the resulting metadata
        del FEI_navcam_metadata['Acquisition_instrument']['SEM']['working_distance']
        assert_deep_almost_equal(s.metadata.as_dictionary(), FEI_navcam_metadata)


class TestReadZeissSEM:
    
    path = Path(TMP_DIR.name)
    
    @classmethod
    def setup_class(cls):
        zipf = os.path.join(MY_PATH2, "tiff_Zeiss_SEM.zip")
        with zipfile.ZipFile(zipf, 'r') as zipped:
            zipped.extractall(cls.path)

    def test_read_Zeiss_SEM_scale_metadata_1k_image(self):
        md = {'Acquisition_instrument': {'SEM': {'Stage': {'rotation': 10.2,
                                                           'tilt': -0.0,
                                                           'x': 75.6442,
                                                           'y': 60.4901,
                                                           'z': 25.193},
                                                 'Detector':{'detector_type':
                                                     'HE-SE2'},
                                                 'beam_current': 1.0,
                                                 'beam_energy': 25.0,
                                                 'dwell_time': 5e-08,
                                                 'magnification': 105.0,
                                                 'microscope': 'Merlin-61-08',
                                                 'working_distance': 14.808}},
              'General': {'authors': 'LIM',
                          'date': '2015-12-23',
                          'original_filename': 'test_tiff_Zeiss_SEM_1k.tif',
                          'time': '09:40:32',
                          'title': '',
                          'FileIO': {'0': {
                              'operation': 'load',
                              'hyperspy_version': hs_version,
                              'io_plugin': 'hyperspy.io_plugins.tiff'
                          }}
                          },
              'Signal': {'signal_type': ''},
              '_HyperSpy': {'Folding': {'original_axes_manager': None,
                                        'original_shape': None,
                                        'signal_unfolded': False,
                                        'unfolded': False}}}
        
        fname = self.path / 'test_tiff_Zeiss_SEM_1k.tif'
        s = hs.load(fname, convert_units=True)
    
        assert s.axes_manager[0].units == 'µm'
        assert s.axes_manager[1].units == 'µm'
        np.testing.assert_allclose(s.axes_manager[0].scale, 2.614514, rtol=1E-6)
        np.testing.assert_allclose(s.axes_manager[1].scale, 2.614514, rtol=1E-6)
        assert s.data.dtype == 'uint8'
        # delete timestamp from metadata since it's runtime dependent
        del s.metadata.General.FileIO.Number_0.timestamp
        assert_deep_almost_equal(s.metadata.as_dictionary(), md)

    def test_read_Zeiss_SEM_scale_metadata_512_image(self):
        md = {'Acquisition_instrument': {'SEM': {'Stage': {'rotation': 245.8,
                                                           'tilt': 0.0,
                                                           'x': 62.9961,
                                                           'y': 65.3168,
                                                           'z': 44.678},
                                                 'beam_energy': 5.0,
                                                 'magnification': '50.00 K X',
                                                 'microscope': 'ULTRA 55-36-06',
                                                 'working_distance': 3.9}},
              'General': {'authors': 'LIBERATO',
                          'date': '2018-09-25',
                          'original_filename': 'test_tiff_Zeiss_SEM_512pix.tif',
                          'time': '08:20:42',
                          'title': '',
                          'FileIO': {'0': {
                              'operation': 'load',
                              'hyperspy_version': hs_version,
                              'io_plugin': 'hyperspy.io_plugins.tiff'
                          }}
                          },
              'Signal': {'signal_type': ''},
              '_HyperSpy': {'Folding': {'original_axes_manager': None,
                                        'original_shape': None,
                                        'signal_unfolded': False,
                                        'unfolded': False}}}
    
        fname = self.path / 'test_tiff_Zeiss_SEM_512pix.tif'
        s = hs.load(fname, convert_units=True)
        assert s.axes_manager[0].units == 'µm'
        assert s.axes_manager[1].units == 'µm'
        np.testing.assert_allclose(s.axes_manager[0].scale, 0.011649976, rtol=1E-6)
        np.testing.assert_allclose(s.axes_manager[1].scale, 0.011649976, rtol=1E-6)
        assert s.data.dtype == 'uint8'
        # delete timestamp from metadata since it's runtime dependent
        del s.metadata.General.FileIO.Number_0.timestamp
        assert_deep_almost_equal(s.metadata.as_dictionary(), md)


def test_read_RGB_Zeiss_optical_scale_metadata():
    fname = os.path.join(MY_PATH2, 'optical_Zeiss_AxioVision_RGB.tif')
    s = hs.load(fname, )
    dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
    assert s.data.dtype == dtype
    assert s.data.shape == (10, 13)
    assert s.axes_manager[0].units == t.Undefined
    assert s.axes_manager[1].units == t.Undefined
    np.testing.assert_allclose(s.axes_manager[0].scale, 1.0, rtol=1E-5)
    np.testing.assert_allclose(s.axes_manager[1].scale, 1.0, rtol=1E-5)
    assert s.metadata.General.date == '2016-06-13'
    assert s.metadata.General.time == '15:59:52'
    assert s.metadata.General.FileIO.Number_0.hyperspy_version == hs_version
    assert s.metadata.General.FileIO.Number_0.io_plugin == 'hyperspy.io_plugins.tiff'


def test_read_BW_Zeiss_optical_scale_metadata():
    fname = os.path.join(MY_PATH2, 'optical_Zeiss_AxioVision_BW.tif')
    s = hs.load(fname, force_read_resolution=True, convert_units=True)
    assert s.data.dtype == np.uint8
    assert s.data.shape == (10, 13)
    assert s.axes_manager[0].units == 'µm'
    assert s.axes_manager[1].units == 'µm'
    np.testing.assert_allclose(s.axes_manager[0].scale, 169.333, rtol=1E-5)
    np.testing.assert_allclose(s.axes_manager[1].scale, 169.333, rtol=1E-5)
    assert s.metadata.General.date == '2016-06-13'
    assert s.metadata.General.time == '16:08:49'


def test_read_BW_Zeiss_optical_scale_metadata_convert_units_false():
    fname = os.path.join(MY_PATH2, 'optical_Zeiss_AxioVision_BW.tif')
    s = hs.load(fname, force_read_resolution=True, convert_units=False)
    assert s.data.dtype == np.uint8
    assert s.data.shape == (10, 13)
    assert s.axes_manager[0].units == 'µm'
    assert s.axes_manager[1].units == 'µm'
    np.testing.assert_allclose(s.axes_manager[0].scale, 169.333, rtol=1E-5)
    np.testing.assert_allclose(s.axes_manager[1].scale, 169.333, rtol=1E-5)


def test_read_BW_Zeiss_optical_scale_metadata2():
    fname = os.path.join(MY_PATH2, 'optical_Zeiss_AxioVision_BW.tif')
    s = hs.load(fname, force_read_resolution=True, convert_units=True)
    assert s.data.dtype == np.uint8
    assert s.data.shape == (10, 13)
    assert s.axes_manager[0].units == 'µm'
    assert s.axes_manager[1].units == 'µm'
    np.testing.assert_allclose(s.axes_manager[0].scale, 169.333, rtol=1E-5)
    np.testing.assert_allclose(s.axes_manager[1].scale, 169.333, rtol=1E-5)
    assert s.metadata.General.date == '2016-06-13'
    assert s.metadata.General.time == '16:08:49'


def test_read_BW_Zeiss_optical_scale_metadata3():
    fname = os.path.join(MY_PATH2, 'optical_Zeiss_AxioVision_BW.tif')
    s = hs.load(fname, force_read_resolution=False)
    assert s.data.dtype == np.uint8
    assert s.data.shape == (10, 13)
    assert s.axes_manager[0].units == t.Undefined
    assert s.axes_manager[1].units == t.Undefined
    np.testing.assert_allclose(s.axes_manager[0].scale, 1.0, rtol=1E-5)
    np.testing.assert_allclose(s.axes_manager[1].scale, 1.0, rtol=1E-5)
    assert s.metadata.General.date == '2016-06-13'
    assert s.metadata.General.time == '16:08:49'


def test_read_TVIPS_metadata():
    md = {'Acquisition_instrument': {'TEM': {'Detector': {'Camera': {'exposure': 0.4,
                                                                     'name': 'F416'}},
                                             'Stage': {'tilt_alpha': -0.0070000002,
                                                       'tilt_beta': -0.055,
                                                       'x': 0.0,
                                                       'y': -9.2000000506686774e-05,
                                                       'z': 7.0000001350933871e-06},
                                             'beam_energy': 99.0,
                                             'magnification': 32000.0}},
          'General': {'original_filename': 'TVIPS_bin4.tif',
                      'time': '9:01:17',
                      'title': '',
                      'FileIO': {'0': {
                          'operation': 'load',
                          'hyperspy_version': hs_version,
                          'io_plugin': 'hyperspy.io_plugins.tiff'
                      }}
                      },
          'Signal': {'signal_type': ''},
          '_HyperSpy': {'Folding': {'original_axes_manager': None,
                                    'original_shape': None,
                                    'signal_unfolded': False,
                                    'unfolded': False}}}
    
    zipf = os.path.join(MY_PATH2, "TVIPS_bin4.zip")
    path = Path(TMP_DIR.name)
    with zipfile.ZipFile(zipf, 'r') as zipped:
        zipped.extractall(path)
        s = hs.load(path / 'TVIPS_bin4.tif', convert_units=True)
        
    assert s.data.dtype == np.uint8
    assert s.data.shape == (1024, 1024)
    assert s.axes_manager[0].units == 'nm'
    assert s.axes_manager[1].units == 'nm'
    np.testing.assert_allclose(s.axes_manager[0].scale, 1.42080, rtol=1E-5)
    np.testing.assert_allclose(s.axes_manager[1].scale, 1.42080, rtol=1E-5)
    # delete timestamp from metadata since it's runtime dependent
    del s.metadata.General.FileIO.Number_0.timestamp
    assert_deep_almost_equal(s.metadata.as_dictionary(), md)


def test_axes_metadata():
    data = np.arange(2*5*10).reshape((2, 5, 10))
    s = hs.signals.Signal2D(data)
    nav_unit = 's'
    s.axes_manager.navigation_axes[0].units = nav_unit
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'axes_metadata_default.tif')
        s.save(fname)
        s2 = hs.load(fname)
        assert s2.axes_manager.navigation_axes[0].name == 'image series'
        assert s2.axes_manager.navigation_axes[0].units == nav_unit
        assert s2.axes_manager.navigation_axes[0].is_binned == False

        fname2 = os.path.join(tmpdir, 'axes_metadata_IYX.tif')
        s.save(fname2, metadata={'axes':'IYX'})
        s3 = hs.load(fname2)
        assert s3.axes_manager.navigation_axes[0].name == 'image series'
        assert s3.axes_manager.navigation_axes[0].units == nav_unit
        assert s3.axes_manager.navigation_axes[0].is_binned == False

        fname2 = os.path.join(tmpdir, 'axes_metadata_ZYX.tif')
        s.save(fname2, metadata={'axes':'ZYX'})
        s3 = hs.load(fname2)
        assert s3.axes_manager.navigation_axes[0].units == nav_unit
        assert s3.axes_manager.navigation_axes[0].is_binned == False


def test_olympus_SIS():
    pytest.importorskip("imagecodecs", reason="imagecodecs is required")
    fname = os.path.join(MY_PATH2, 'olympus_SIS.tif')
    s = hs.load(fname)
    # This olympus SIS contains two images:
    # - the first one is a RGB 8-bits (used for preview purposes)
    # - the second one is the raw data
    # only the second one is calibrated.
    assert len(s) == 2
    am = s[1].axes_manager
    for axis in am._axes:
        assert axis.units == 'm'
        np.testing.assert_allclose(axis.scale, 2.3928e-11)
        np.testing.assert_allclose(axis.offset, 0.0)

    for ima in s:
        assert ima.data.shape == (101, 112)

    assert s[1].data.dtype is np.dtype('uint16')


def test_save_angstrom_units():
    s = hs.signals.Signal2D(np.arange(200*200, dtype='float32').reshape((200, 200)))
    for axis in s.axes_manager.signal_axes:
        axis.units = 'Å'
        axis.scale = 0.1
        axis.offset = 10

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'save_angstrom_units.tif')
        s.save(fname)
        s2 = hs.load(fname)
        if Version(tifffile.__version__) >= Version("2020.7.17"):
            assert s2.axes_manager[0].units == s.axes_manager[0].units
        assert s2.axes_manager[0].scale == s.axes_manager[0].scale
        assert s2.axes_manager[0].offset == s.axes_manager[0].offset
        assert s2.axes_manager[0].is_binned == s.axes_manager[0].is_binned


def test_JEOL_SightX():
    files = [ ("JEOL-SightX-Ronchigram-dummy.tif.gz", 1.0, t.Undefined),
              ("JEOL-SightX-SAED-dummy.tif.gz", 0.2723, "1 / nm"),
              ("JEOL-SightX-TEM-mag-dummy.tif.gz", 1.8208, "nm") ]
    for file in files:
        fname = file[0]
        if fname[-3:]==".gz":
            import tempfile
            import gzip
            with tempfile.TemporaryDirectory() as tmp_dir:
                with gzip.open(os.path.join(MY_PATH2,fname),'rb') as f:
                    content = f.read()
                fname = os.path.join(tmp_dir, fname[:-3])
                with open(fname,"wb") as f2:
                    f2.write(content)
                s = hs.load(fname)
        else:
            fname = os.path.join(MY_PATH2, file[0])
            s = hs.load(fname)
        for i in range(2): # x, y
            assert s.axes_manager[i].size == 556
            np.testing.assert_allclose(s.axes_manager[i].scale, file[1], rtol=1E-3)
            assert s.axes_manager[i].units == file[2]


class TestReadHamamatsu:
    
    path = Path(TMP_DIR.name)
    
    @classmethod
    def setup_class(cls):
        zipf = os.path.join(MY_PATH2, "tiff_hamamatsu.zip")
        with zipfile.ZipFile(zipf, 'r') as zipped:
            zipped.extractall(cls.path)


    def test_hamamatsu_streak_loadwarnings(self):
        file = 'test_hamamatsu_streak_SCAN.tif'
        fname = os.path.join(self.path, file)
    
        # No hamamatsu_streak_axis_type argument should :
        # - raise warning
        # - Initialise uniform data axis
        with pytest.warns(UserWarning):
            s = hs.load(fname)
            assert s.axes_manager.all_uniform
    
        #Invalid hamamatsu_streak_axis_type argument should:
        # - raise warning
        # - Initialise uniform data axis
        with pytest.warns(UserWarning):
            s = hs.load(fname,hamamatsu_streak_axis_type='xxx')
            assert s.axes_manager.all_uniform
    
        #Explicitly calling hamamatsu_streak_axis_type='uniform'
        # should NOT raise a warning
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            s = hs.load(fname, hamamatsu_streak_axis_type='uniform')
            assert s.axes_manager.all_uniform

    def test_hamamatsu_streak_scanfile(self):
        file = 'test_hamamatsu_streak_SCAN.tif'
        fname = os.path.join(self.path, file)
    
        with pytest.warns(UserWarning):
            s = hs.load(fname)
    
        assert s.data.shape == (508, 672)
        assert s.axes_manager[1].units == 'ps'
        np.testing.assert_allclose(s.axes_manager[1].scale, 2.3081, rtol=1E-3)
        np.testing.assert_allclose(s.axes_manager[1].offset, 652.3756, rtol=1E-3)
        np.testing.assert_allclose(s.axes_manager[0].scale, 0.01714, rtol=1E-3)
        np.testing.assert_allclose(s.axes_manager[0].offset, 231.0909, rtol=1E-3)
    
    def test_hamamatsu_streak_focusfile(self):
        file = 'test_hamamatsu_streak_FOCUS.tif'
        fname = os.path.join(self.path, file)
    
        with pytest.warns(UserWarning):
            s = hs.load(fname)
    
        assert s.data.shape == (508, 672)
        assert s.axes_manager[1].units == ''
        np.testing.assert_allclose(s.axes_manager[1].scale, 1.0, rtol=1E-3)
        np.testing.assert_allclose(s.axes_manager[1].offset, 0.0, rtol=1E-3, atol=1e-5)
        np.testing.assert_allclose(s.axes_manager[0].scale, 0.01714, rtol=1E-3)
        np.testing.assert_allclose(s.axes_manager[0].offset, 231.0909, rtol=1E-3)
    
    def test_hamamatsu_streak_non_uniform_load(self):
        file = 'test_hamamatsu_streak_SCAN.tif'
        fname = os.path.join(self.path, file)
    
        s = hs.load(fname, hamamatsu_streak_axis_type='data')
    
        np.testing.assert_allclose(
            s.original_metadata.ImageDescriptionParsed.Scaling.ScalingYaxis,
            s.axes_manager[1].axis, rtol=1e-5
            )
    
        s = hs.load(fname, hamamatsu_streak_axis_type='functional')
    
        np.testing.assert_allclose(
            s.original_metadata.ImageDescriptionParsed.Scaling.ScalingYaxis,
            s.axes_manager[1].axis, rtol=1e-5
            )
    
    def test_is_hamamatsu_streak(self):
        file = 'test_hamamatsu_streak_SCAN.tif'
        fname = os.path.join(self.path, file)
    
        with pytest.warns(UserWarning):
            s = hs.load(fname)
    
        omd = s.original_metadata.as_dictionary()
    
        omd['Artist'] = "TAPTAP"
    
        assert not hyperspy.io_plugins.tiff._is_streak_hamamatsu(omd)
    
        _ = omd.pop('Artist')
    
        assert not hyperspy.io_plugins.tiff._is_streak_hamamatsu(omd)
    
        omd.update({'Artist': "Copyright Hamamatsu GmbH, 2018"})
    
        omd['Software'] = "TAPTAPTAP"
    
        assert not hyperspy.io_plugins.tiff._is_streak_hamamatsu(omd)
    
        _ = omd.pop('Software')
    
        assert not hyperspy.io_plugins.tiff._is_streak_hamamatsu(omd)
    
        omd.update({'Software': "HPD-TA 9.5 pf4"})
    
        assert hyperspy.io_plugins.tiff._is_streak_hamamatsu(omd)
