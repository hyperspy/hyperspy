import os

import numpy as np
import nose.tools as nt
import traits.api as t

import hyperspy.api as hs

my_path = os.path.dirname(__file__)
my_path2 = os.path.join(my_path, "tiff_files")

# set to False, if you want to keep the files to check them with ImageJ or DM
remove_files = True


def test_rgba16():
    """ Use skimage tifffile.py library """
    _test_rgba16(import_local_tifffile=False)


def test_rgba16_local_tifffile():
    """ Use local tifffile.py library """
    _test_rgba16(import_local_tifffile=True)


def _test_rgba16(import_local_tifffile=False):
    s = hs.load(os.path.join(my_path2, "test_rgba16.tif"),
                import_local_tifffile=import_local_tifffile)
    data = np.load(os.path.join(my_path, "npy_files", "test_rgba16.npy"))
    nt.assert_true((s.data == data).all())
    nt.assert_equal(s.axes_manager[0].units, t.Undefined)
    nt.assert_equal(s.axes_manager[1].units, t.Undefined)
    nt.assert_equal(s.axes_manager[2].units, t.Undefined)
    nt.assert_almost_equal(s.axes_manager[0].scale, 1.0, places=5)
    nt.assert_almost_equal(s.axes_manager[1].scale, 1.0, places=5)
    nt.assert_almost_equal(s.axes_manager[2].scale, 1.0, places=5)


def test_read_unit_um():
    # Load DM file and save it as tif
    s = hs.load(os.path.join(my_path2, 'test_dm_image_um_unit.dm3'))
    nt.assert_equal(s.axes_manager[0].units, 'µm')
    nt.assert_equal(s.axes_manager[1].units, 'µm')
    nt.assert_almost_equal(s.axes_manager[0].scale, 0.16867, places=5)
    nt.assert_almost_equal(s.axes_manager[1].scale, 0.16867, places=5)

    fname = os.path.join(my_path, 'tiff_files', 'test_export_um_unit.tif')
    s.save(fname, overwrite=True, export_scale=True)

    # load tif file
    s2 = hs.load(fname, import_local_tifffile=True)
    nt.assert_equal(s.axes_manager[0].units, 'µm')
    nt.assert_equal(s.axes_manager[1].units, 'µm')
    nt.assert_almost_equal(s2.axes_manager[0].scale, 0.16867, places=5)
    nt.assert_almost_equal(s2.axes_manager[1].scale, 0.16867, places=5)
    if remove_files:
        os.remove(fname)


def test_read_unit_from_imagej():
    """ Use skimage tifffile.py library """
    _test_read_unit_from_imagej(import_local_tifffile=False)


def test_read_unit_from_imagej_local_tifffile():
    """ Use local tifffile.py library """
    _test_read_unit_from_imagej(import_local_tifffile=True)


def _test_read_unit_from_imagej(import_local_tifffile=False):
    fname = os.path.join(my_path, 'tiff_files',
                         'test_loading_image_saved_with_imageJ.tif')
    s = hs.load(fname, import_local_tifffile=import_local_tifffile)
    nt.assert_equal(s.axes_manager[0].units, 'micron')
    nt.assert_equal(s.axes_manager[1].units, 'micron')
    nt.assert_almost_equal(s.axes_manager[0].scale, 0.16867, places=5)
    nt.assert_almost_equal(s.axes_manager[1].scale, 0.16867, places=5)


def test_write_read_unit_imagej(import_local_tifffile=True):
    fname = os.path.join(my_path, 'tiff_files',
                         'test_loading_image_saved_with_imageJ.tif')
    s = hs.load(fname, import_local_tifffile=import_local_tifffile)
    s.axes_manager[0].units = 'µm'
    s.axes_manager[1].units = 'µm'
    fname2 = fname.replace('.tif', '2.tif')
    s.save(fname2, export_scale=True, overwrite=True)
    s2 = hs.load(fname2, import_local_tifffile=import_local_tifffile)
    nt.assert_equal(s2.axes_manager[0].units, 'µm')
    nt.assert_equal(s2.axes_manager[1].units, 'µm')
    if remove_files:
        os.remove(fname2)


def test_write_read_unit_imagej_with_description(import_local_tifffile=True):
    fname = os.path.join(my_path, 'tiff_files',
                         'test_loading_image_saved_with_imageJ.tif')
    s = hs.load(fname, import_local_tifffile=import_local_tifffile)
    s.axes_manager[0].units = 'µm'
    s.axes_manager[1].units = 'µm'
    nt.assert_almost_equal(s.axes_manager[0].scale, 0.16867, places=5)
    nt.assert_almost_equal(s.axes_manager[1].scale, 0.16867, places=5)
    fname2 = fname.replace('.tif', '_description.tif')
    s.save(fname2, export_scale=False, overwrite=True, description='test')
    s2 = hs.load(fname2, import_local_tifffile=import_local_tifffile)
    nt.assert_equal(s2.axes_manager[0].units, t.Undefined)
    nt.assert_equal(s2.axes_manager[1].units, t.Undefined)
    nt.assert_almost_equal(s2.axes_manager[0].scale, 1.0, places=5)
    nt.assert_almost_equal(s2.axes_manager[1].scale, 1.0, places=5)

    fname3 = fname.replace('.tif', '_description2.tif')
    s.save(fname3, export_scale=True, overwrite=True, description='test')
    s3 = hs.load(fname3, import_local_tifffile=import_local_tifffile)
    nt.assert_equal(s3.axes_manager[0].units, 'µm')
    nt.assert_equal(s3.axes_manager[1].units, 'µm')
    nt.assert_almost_equal(s3.axes_manager[0].scale, 0.16867, places=5)
    nt.assert_almost_equal(s3.axes_manager[1].scale, 0.16867, places=5)

    if remove_files:
        os.remove(fname2)
        os.remove(fname3)


def test_saving_with_custom_tag():
    s = hs.signals.Signal2D(
        np.arange(
            10 * 15,
            dtype=np.uint8).reshape(
            (10,
             15)))
    fname = os.path.join(my_path, 'tiff_files',
                         'test_saving_with_custom_tag.tif')
    extratag = [(65000, 's', 1, "Random metadata", False)]
    s.save(fname, extratags=extratag, overwrite=True)
    s2 = hs.load(fname)
    nt.assert_equal(s2.original_metadata['Number_65000'], b"Random metadata")
    if remove_files:
        os.remove(fname)


def test_read_unit_from_dm():
    """ Use skimage tifffile.py library """
    _test_read_unit_from_dm(import_local_tifffile=False)


def test_read_unit_from_dm_local_tifffile():
    """ Use local tifffile.py library """
    _test_read_unit_from_dm(import_local_tifffile=True)


def _test_read_unit_from_dm(import_local_tifffile=False):
    fname = os.path.join(my_path2, 'test_loading_image_saved_with_DM.tif')
    s = hs.load(fname, import_local_tifffile=import_local_tifffile)
    nt.assert_equal(s.axes_manager[0].units, 'µm')
    nt.assert_equal(s.axes_manager[1].units, 'µm')
    nt.assert_almost_equal(s.axes_manager[0].scale, 0.16867, places=5)
    nt.assert_almost_equal(s.axes_manager[1].scale, 0.16867, places=5)


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
    fname = os.path.join(my_path, 'tiff_files',
                         'test_export_scale_unit_%s.tif' % export_scale)
    s.save(fname, overwrite=True, export_scale=export_scale)
    if remove_files:
        os.remove(fname)


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
    fname = os.path.join(my_path, 'tiff_files',
                         'test_export_scale_unit_not_square_pixel.tif')
    s.save(fname, overwrite=True, export_scale=True)
    if remove_files:
        os.remove(fname)


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
    fname = os.path.join(my_path, 'tiff_files',
                         'test_export_scale_undefined_unit.tif')
    s.save(fname, overwrite=True, export_scale=True)
    if remove_files:
        os.remove(fname)


def test_write_scale_with_undefined_scale():
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
    fname = os.path.join(my_path, 'tiff_files',
                         'test_export_scale_undefined_scale.tif')
    s.save(fname, overwrite=True, export_scale=True)
    if remove_files:
        os.remove(fname)


def test_write_scale_with_um_unit():
    """ Lazy test, still need to open the files in ImageJ or DM to check if the
        scale and unit are correct """
    s = hs.load(os.path.join(my_path, 'tiff_files',
                             'test_dm_image_um_unit.dm3'))
    fname = os.path.join(my_path, 'tiff_files', 'test_export_um_unit.tif')
    s.save(fname, overwrite=True, export_scale=True)
    if remove_files:
        os.remove(fname)


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
    s.axes_manager[0].name = 'x'
    s.axes_manager[1].name = 'y'
    s.axes_manager[2].name = 'z'
    s.axes_manager['x'].scale = 0.25
    s.axes_manager['y'].scale = 0.5
    s.axes_manager['z'].scale = 1.5
    s.axes_manager['x'].units = 'nm'
    s.axes_manager['y'].units = 'um'
    s.axes_manager['z'].units = 'mm'
    fname = os.path.join(my_path, 'tiff_files',
                         'test_export_scale_unit_stack.tif')
    s.save(fname, overwrite=True, export_scale=True)
    if remove_files:
        os.remove(fname)


def test_read_FEI_SEM_scale_metadata_8bits():
    fname = os.path.join(my_path2, 'FEI-Helios-Ebeam-8bits.tif')
    s = hs.load(fname)
    nt.assert_equal(s.axes_manager[0].units, 'm')
    nt.assert_equal(s.axes_manager[1].units, 'm')
    nt.assert_almost_equal(s.axes_manager[0].scale, 3.3724e-06, places=12)
    nt.assert_almost_equal(s.axes_manager[1].scale, 3.3724e-06, places=12)
    nt.assert_equal(s.data.dtype, 'uint8')


def test_read_FEI_SEM_scale_metadata_16bits():
    fname = os.path.join(my_path2, 'FEI-Helios-Ebeam-16bits.tif')
    s = hs.load(fname)
    nt.assert_equal(s.axes_manager[0].units, 'm')
    nt.assert_equal(s.axes_manager[1].units, 'm')
    nt.assert_almost_equal(s.axes_manager[0].scale, 3.3724e-06, places=12)
    nt.assert_almost_equal(s.axes_manager[1].scale, 3.3724e-06, places=12)
    nt.assert_equal(s.data.dtype, 'uint16')


def test_read_Zeiss_SEM_scale_metadata_1k_image():
    fname = os.path.join(my_path2, 'test_tiff_Zeiss_SEM_1k.tif')
    s = hs.load(fname)
    nt.assert_equal(s.axes_manager[0].units, 'm')
    nt.assert_equal(s.axes_manager[1].units, 'm')
    nt.assert_almost_equal(s.axes_manager[0].scale, 2.614514e-06, places=12)
    nt.assert_almost_equal(s.axes_manager[1].scale, 2.614514e-06, places=12)
    nt.assert_equal(s.data.dtype, 'uint16')


def test_read_Zeiss_SEM_scale_metadata_512_image():
    fname = os.path.join(my_path2, 'test_tiff_Zeiss_SEM_512.tif')
    s = hs.load(fname)
    nt.assert_equal(s.axes_manager[0].units, 'm')
    nt.assert_equal(s.axes_manager[1].units, 'm')
    nt.assert_almost_equal(s.axes_manager[0].scale, 7.4240e-08, places=12)
    nt.assert_almost_equal(s.axes_manager[1].scale, 7.4240e-08, places=12)
    nt.assert_equal(s.data.dtype, 'uint16')


def test_read_RGB_Zeiss_optical_scale_metadata():
    fname = os.path.join(my_path2, 'optical_Zeiss_AxioVision_RGB.tif')
    s = hs.load(fname, import_local_tifffile=True)
    dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
    nt.assert_equal(s.data.dtype, dtype)
    nt.assert_equal(s.data.shape, (10, 13))
    nt.assert_equal(s.axes_manager[0].units, t.Undefined)
    nt.assert_equal(s.axes_manager[1].units, t.Undefined)
    nt.assert_almost_equal(s.axes_manager[0].scale, 1.0, places=3)
    nt.assert_almost_equal(s.axes_manager[1].scale, 1.0, places=3)


def test_read_BW_Zeiss_optical_scale_metadata():
    fname = os.path.join(my_path2, 'optical_Zeiss_AxioVision_BW.tif')
    s = hs.load(fname, force_read_resolution=True, import_local_tifffile=True)
    nt.assert_equal(s.data.dtype, np.uint16)
    nt.assert_equal(s.data.shape, (10, 13))
    nt.assert_equal(s.axes_manager[0].units, 'µm')
    nt.assert_equal(s.axes_manager[1].units, 'µm')
    nt.assert_almost_equal(s.axes_manager[0].scale, 169.3333, places=3)
    nt.assert_almost_equal(s.axes_manager[1].scale, 169.3333, places=3)


def test_read_BW_Zeiss_optical_scale_metadata_old():
    fname = os.path.join(my_path2, 'optical_Zeiss_AxioVision_BW.tif')
    s = hs.load(fname, force_read_resolution=True)
    nt.assert_equal(s.data.dtype, np.uint16)
    nt.assert_equal(s.data.shape, (10, 13))
    nt.assert_equal(s.axes_manager[0].units, 'µm')
    nt.assert_equal(s.axes_manager[1].units, 'µm')
    nt.assert_almost_equal(s.axes_manager[0].scale, 169.3333, places=3)
    nt.assert_almost_equal(s.axes_manager[1].scale, 169.3333, places=3)


def test_read_BW_Zeiss_optical_scale_metadata_old2():
    fname = os.path.join(my_path2, 'optical_Zeiss_AxioVision_BW.tif')
    s = hs.load(fname, force_read_resolution=False)
    nt.assert_equal(s.data.dtype, np.uint16)
    nt.assert_equal(s.data.shape, (10, 13))
    nt.assert_equal(s.axes_manager[0].units, t.Undefined)
    nt.assert_equal(s.axes_manager[1].units, t.Undefined)
    nt.assert_almost_equal(s.axes_manager[0].scale, 1.0, places=3)
    nt.assert_almost_equal(s.axes_manager[1].scale, 1.0, places=3)
