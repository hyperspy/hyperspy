import os
import tempfile

import numpy as np
from numpy.testing import assert_allclose
import pytest
import traits.api as t

import hyperspy.api as hs
from hyperspy.misc.test_utils import assert_deep_almost_equal

MY_PATH = os.path.dirname(__file__)
MY_PATH2 = os.path.join(MY_PATH, "tiff_files")


def test_rgba16():
    s = hs.load(os.path.join(MY_PATH2, "test_rgba16.tif"))
    data = np.load(os.path.join(MY_PATH, "npy_files", "test_rgba16.npy"))
    assert (s.data == data).all()
    assert s.axes_manager[0].units == t.Undefined
    assert s.axes_manager[1].units == t.Undefined
    assert s.axes_manager[2].units == t.Undefined
    assert_allclose(s.axes_manager[0].scale, 1.0, atol=1E-5)
    assert_allclose(s.axes_manager[1].scale, 1.0, atol=1E-5)
    assert_allclose(s.axes_manager[2].scale, 1.0, atol=1E-5)
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
    assert_allclose(s.axes_manager[0].scale, 0.16867, atol=1E-5)
    assert_allclose(s.axes_manager[1].scale, 0.16867, atol=1E-5)
    assert s.metadata.General.date == '2015-07-20'
    assert s.metadata.General.time == '18:48:25'
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'tiff_files', 'test_export_um_unit.tif')
        s.save(fname, overwrite=True, export_scale=True)
        # load tif file
        s2 = hs.load(fname)
        assert s.axes_manager[0].units == 'µm'
        assert s.axes_manager[1].units == 'µm'
        assert_allclose(s2.axes_manager[0].scale, 0.16867, atol=1E-5)
        assert_allclose(s2.axes_manager[1].scale, 0.16867, atol=1E-5)
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
    assert_allclose(s.axes_manager[0].scale, 0.16867, atol=1E-5)
    assert_allclose(s.axes_manager[1].scale, 0.16867, atol=1E-5)


def test_read_unit_from_imagej_stack():
    fname = os.path.join(MY_PATH, 'tiff_files',
                         'test_loading_image_saved_with_imageJ_stack.tif')
    s = hs.load(fname)
    assert s.data.shape == (2, 68, 68)
    assert s.axes_manager[0].units == t.Undefined
    assert s.axes_manager[1].units == 'µm'
    assert s.axes_manager[2].units == 'µm'
    assert_allclose(s.axes_manager[0].scale, 2.5, atol=1E-5)
    assert_allclose(s.axes_manager[1].scale, 0.16867, atol=1E-5)
    assert_allclose(s.axes_manager[2].scale, 0.16867, atol=1E-5)


@pytest.mark.parametrize("lazy", [True, False])
def test_read_unit_from_DM_stack(lazy):
    fname = os.path.join(MY_PATH, 'tiff_files',
                         'test_loading_image_saved_with_DM_stack.tif')
    s = hs.load(fname, lazy=lazy)
    assert s.data.shape == (2, 68, 68)
    assert s.axes_manager[0].units == 's'
    assert s.axes_manager[1].units == 'µm'
    assert s.axes_manager[2].units == 'µm'
    assert_allclose(s.axes_manager[0].scale, 2.5, atol=1E-5)
    assert_allclose(s.axes_manager[1].scale, 0.16867, atol=1E-5)
    assert_allclose(s.axes_manager[2].scale, 1.68674, atol=1E-5)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname2 = os.path.join(
            tmpdir, 'test_loading_image_saved_with_DM_stack2.tif')
        s.save(fname2, overwrite=True)
        s2 = hs.load(fname2)
        _compare_signal_shape_data(s, s2)
        assert s2.axes_manager[0].units == s.axes_manager[0].units
        assert s2.axes_manager[1].units == 'µm'
        assert s2.axes_manager[2].units == 'µm'
        assert_allclose(
            s2.axes_manager[0].scale, s.axes_manager[0].scale, atol=1E-5)
        assert_allclose(
            s2.axes_manager[1].scale, s.axes_manager[1].scale, atol=1E-5)
        assert_allclose(
            s2.axes_manager[2].scale, s.axes_manager[2].scale, atol=1E-5)
        assert_allclose(
            s2.axes_manager[0].offset, s.axes_manager[0].offset, atol=1E-5)
        assert_allclose(
            s2.axes_manager[1].offset, s.axes_manager[1].offset, atol=1E-5)
        assert_allclose(
            s2.axes_manager[2].offset, s.axes_manager[2].offset, atol=1E-5)


def test_read_unit_from_imagej_stack_no_scale():
    fname = os.path.join(MY_PATH, 'tiff_files',
                         'test_loading_image_saved_with_imageJ_stack_no_scale.tif')
    s = hs.load(fname)
    assert s.data.shape == (2, 68, 68)
    assert s.axes_manager[0].units == t.Undefined
    assert s.axes_manager[1].units == t.Undefined
    assert s.axes_manager[2].units == t.Undefined
    assert_allclose(s.axes_manager[0].scale, 1.0, atol=1E-5)
    assert_allclose(s.axes_manager[1].scale, 1.0, atol=1E-5)
    assert_allclose(s.axes_manager[2].scale, 1.0, atol=1E-5)


def test_read_unit_from_imagej_no_scale():
    fname = os.path.join(MY_PATH, 'tiff_files',
                         'test_loading_image_saved_with_imageJ_no_scale.tif')
    s = hs.load(fname)
    assert s.axes_manager[0].units == t.Undefined
    assert s.axes_manager[1].units == t.Undefined
    assert_allclose(s.axes_manager[0].scale, 1.0, atol=1E-5)
    assert_allclose(s.axes_manager[1].scale, 1.0, atol=1E-5)


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
    assert_allclose(s.axes_manager[0].scale, 0.16867, atol=1E-5)
    assert_allclose(s.axes_manager[1].scale, 0.16867, atol=1E-5)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname2 = os.path.join(tmpdir, 'description.tif')
        s.save(fname2, export_scale=False, overwrite=True, description='test')
        s2 = hs.load(fname2)
        assert s2.axes_manager[0].units == t.Undefined
        assert s2.axes_manager[1].units == t.Undefined
        assert_allclose(s2.axes_manager[0].scale, 1.0, atol=1E-5)
        assert_allclose(s2.axes_manager[1].scale, 1.0, atol=1E-5)

        fname3 = os.path.join(tmpdir, 'description2.tif')
        s.save(fname3, export_scale=True, overwrite=True, description='test')
        s3 = hs.load(fname3, convert_units=True)
        assert s3.axes_manager[0].units == 'um'
        assert s3.axes_manager[1].units == 'um'
        assert_allclose(s3.axes_manager[0].scale, 0.16867, atol=1E-5)
        assert_allclose(s3.axes_manager[1].scale, 0.16867, atol=1E-5)


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
    assert_allclose(s.axes_manager[0].scale, 0.16867, atol=1E-5)
    assert_allclose(s.axes_manager[1].scale, 0.16867, atol=1E-5)
    assert_allclose(s.axes_manager[0].offset, 139.66264, atol=1E-5)
    assert_allclose(s.axes_manager[1].offset, 128.19276, atol=1E-5)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname2 = os.path.join(tmpdir, "DM2.tif")
        s.save(fname2, overwrite=True)
        s2 = hs.load(fname2)
        _compare_signal_shape_data(s, s2)
        assert s2.axes_manager[0].units == 'micron'
        assert s2.axes_manager[1].units == 'micron'
        assert_allclose(s2.axes_manager[0].scale, s.axes_manager[0].scale,
                        atol=1E-5)
        assert_allclose(s2.axes_manager[1].scale, s.axes_manager[1].scale,
                        atol=1E-5)
        assert_allclose(s2.axes_manager[0].offset, s.axes_manager[0].offset,
                        atol=1E-5)
        assert_allclose(s2.axes_manager[1].offset, s.axes_manager[1].offset,
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
    s.axes_manager[1].units = 'um'
    s.axes_manager[2].units = 'um'
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test_export_scale_unit_stack2.tif')
        s.save(fname, overwrite=True, export_scale=True)
        s1 = hs.load(fname, convert_units=True)
        _compare_signal_shape_data(s, s1)
        assert s1.axes_manager[0].units == 'pm'
        # only one unit can be read
        assert s1.axes_manager[1].units == 'um'
        assert s1.axes_manager[2].units == 'um'
        assert_allclose(s1.axes_manager[0].scale, 250.0)
        assert_allclose(s1.axes_manager[1].scale, s.axes_manager[1].scale)
        assert_allclose(s1.axes_manager[2].scale, s.axes_manager[2].scale)


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
                                   'time': '17:06:40'},
                       'Signal': {'binned': False,
                                  'signal_type': ''},
                       '_HyperSpy': {'Folding': {'original_axes_manager': None,
                                                 'original_shape': None,
                                                 'signal_unfolded': False,
                                                 'unfolded': False}}}


def test_read_FEI_SEM_scale_metadata_8bits():
    fname = os.path.join(MY_PATH2, 'FEI-Helios-Ebeam-8bits.tif')
    s = hs.load(fname, convert_units=True)
    assert s.axes_manager[0].units == 'um'
    assert s.axes_manager[1].units == 'um'
    assert_allclose(s.axes_manager[0].scale, 3.3724, rtol=1E-5)
    assert_allclose(s.axes_manager[1].scale, 3.3724, rtol=1E-5)
    assert s.data.dtype == 'uint8'
    FEI_Helios_metadata['General'][
        'original_filename'] = 'FEI-Helios-Ebeam-8bits.tif'
    assert_deep_almost_equal(s.metadata.as_dictionary(), FEI_Helios_metadata)


def test_read_FEI_SEM_scale_metadata_16bits():
    fname = os.path.join(MY_PATH2, 'FEI-Helios-Ebeam-16bits.tif')
    s = hs.load(fname, convert_units=True)
    assert s.axes_manager[0].units == 'um'
    assert s.axes_manager[1].units == 'um'
    assert_allclose(s.axes_manager[0].scale, 3.3724, rtol=1E-5)
    assert_allclose(s.axes_manager[1].scale, 3.3724, rtol=1E-5)
    assert s.data.dtype == 'uint16'
    FEI_Helios_metadata['General'][
        'original_filename'] = 'FEI-Helios-Ebeam-16bits.tif'
    assert_deep_almost_equal(s.metadata.as_dictionary(), FEI_Helios_metadata)


def test_read_Zeiss_SEM_scale_metadata_1k_image():
    md = {'Acquisition_instrument': {'SEM': {'Stage': {'rotation': 10.2,
                                                       'tilt': -0.0,
                                                       'x': 75.6442,
                                                       'y': 60.4901,
                                                       'z': 25.193},
                                             'beam_current': 80000.0,
                                             'beam_energy': 25.0,
                                             'dwell_time': 5e-08,
                                             'magnification': 105.0,
                                             'microscope': 'Merlin-61-08',
                                             'working_distance': 14.81}},
          'General': {'authors': 'LIM',
                      'date': '2015-12-23',
                      'original_filename': 'test_tiff_Zeiss_SEM_1k.tif',
                      'time': '09:40:32',
                      'title': ''},
          'Signal': {'binned': False, 'signal_type': ''},
          '_HyperSpy': {'Folding': {'original_axes_manager': None,
                                    'original_shape': None,
                                    'signal_unfolded': False,
                                    'unfolded': False}}}

    fname = os.path.join(MY_PATH2, 'test_tiff_Zeiss_SEM_1k.tif')
    s = hs.load(fname, convert_units=True)
    assert s.axes_manager[0].units == 'um'
    assert s.axes_manager[1].units == 'um'
    assert_allclose(s.axes_manager[0].scale, 2.614514, rtol=1E-6)
    assert_allclose(s.axes_manager[1].scale, 2.614514, rtol=1E-6)
    assert s.data.dtype == 'uint8'
    assert_deep_almost_equal(s.metadata.as_dictionary(), md)


def test_read_Zeiss_SEM_scale_metadata_512_image():
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
                      'title': ''},
          'Signal': {'binned': False, 'signal_type': ''},
          '_HyperSpy': {'Folding': {'original_axes_manager': None,
                                    'original_shape': None,
                                    'signal_unfolded': False,
                                    'unfolded': False}}}

    fname = os.path.join(MY_PATH2, 'test_tiff_Zeiss_SEM_512pix.tif')
    s = hs.load(fname, convert_units=True)
    assert s.axes_manager[0].units == 'um'
    assert s.axes_manager[1].units == 'um'
    assert_allclose(s.axes_manager[0].scale, 0.011649976, rtol=1E-6)
    assert_allclose(s.axes_manager[1].scale, 0.011649976, rtol=1E-6)
    assert s.data.dtype == 'uint8'
    assert_deep_almost_equal(s.metadata.as_dictionary(), md)


def test_read_RGB_Zeiss_optical_scale_metadata():
    fname = os.path.join(MY_PATH2, 'optical_Zeiss_AxioVision_RGB.tif')
    s = hs.load(fname, )
    dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
    assert s.data.dtype == dtype
    assert s.data.shape == (10, 13)
    assert s.axes_manager[0].units == t.Undefined
    assert s.axes_manager[1].units == t.Undefined
    assert_allclose(s.axes_manager[0].scale, 1.0, rtol=1E-5)
    assert_allclose(s.axes_manager[1].scale, 1.0, rtol=1E-5)
    assert s.metadata.General.date == '2016-06-13'
    assert s.metadata.General.time == '15:59:52'


def test_read_BW_Zeiss_optical_scale_metadata():
    fname = os.path.join(MY_PATH2, 'optical_Zeiss_AxioVision_BW.tif')
    s = hs.load(fname, force_read_resolution=True, convert_units=True)
    assert s.data.dtype == np.uint8
    assert s.data.shape == (10, 13)
    assert s.axes_manager[0].units == 'um'
    assert s.axes_manager[1].units == 'um'
    assert_allclose(s.axes_manager[0].scale, 169.333, rtol=1E-5)
    assert_allclose(s.axes_manager[1].scale, 169.333, rtol=1E-5)
    assert s.metadata.General.date == '2016-06-13'
    assert s.metadata.General.time == '16:08:49'


def test_read_BW_Zeiss_optical_scale_metadata_convert_units_false():
    fname = os.path.join(MY_PATH2, 'optical_Zeiss_AxioVision_BW.tif')
    s = hs.load(fname, force_read_resolution=True, convert_units=False)
    assert s.data.dtype == np.uint8
    assert s.data.shape == (10, 13)
    assert s.axes_manager[0].units == 'µm'
    assert s.axes_manager[1].units == 'µm'
    assert_allclose(s.axes_manager[0].scale, 169.333, rtol=1E-5)
    assert_allclose(s.axes_manager[1].scale, 169.333, rtol=1E-5)


def test_read_BW_Zeiss_optical_scale_metadata2():
    fname = os.path.join(MY_PATH2, 'optical_Zeiss_AxioVision_BW.tif')
    s = hs.load(fname, force_read_resolution=True, convert_units=True)
    assert s.data.dtype == np.uint8
    assert s.data.shape == (10, 13)
    assert s.axes_manager[0].units == 'um'
    assert s.axes_manager[1].units == 'um'
    assert_allclose(s.axes_manager[0].scale, 169.333, rtol=1E-5)
    assert_allclose(s.axes_manager[1].scale, 169.333, rtol=1E-5)
    assert s.metadata.General.date == '2016-06-13'
    assert s.metadata.General.time == '16:08:49'


def test_read_BW_Zeiss_optical_scale_metadata3():
    fname = os.path.join(MY_PATH2, 'optical_Zeiss_AxioVision_BW.tif')
    s = hs.load(fname, force_read_resolution=False)
    assert s.data.dtype == np.uint8
    assert s.data.shape == (10, 13)
    assert s.axes_manager[0].units == t.Undefined
    assert s.axes_manager[1].units == t.Undefined
    assert_allclose(s.axes_manager[0].scale, 1.0, rtol=1E-5)
    assert_allclose(s.axes_manager[1].scale, 1.0, rtol=1E-5)
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
                      'title': ''},
          'Signal': {'binned': False, 'signal_type': ''},
          '_HyperSpy': {'Folding': {'original_axes_manager': None,
                                    'original_shape': None,
                                    'signal_unfolded': False,
                                    'unfolded': False}}}
    fname = os.path.join(MY_PATH2, 'TVIPS_bin4.tif')
    s = hs.load(fname, convert_units=True)
    assert s.data.dtype == np.uint8
    assert s.data.shape == (1024, 1024)
    assert s.axes_manager[0].units == 'nm'
    assert s.axes_manager[1].units == 'nm'
    assert_allclose(s.axes_manager[0].scale, 1.42080, rtol=1E-5)
    assert_allclose(s.axes_manager[1].scale, 1.42080, rtol=1E-5)
    assert_deep_almost_equal(s.metadata.as_dictionary(), md)
