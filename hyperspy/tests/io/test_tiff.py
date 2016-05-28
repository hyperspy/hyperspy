import os

import numpy as np
from nose.tools import assert_true

import hyperspy.api as hs

my_path = os.path.dirname(__file__)

# set to False, if you want to keep the files to open them with ImageJ or DM
remove_files = True

def test_rgba16():
    s = hs.load(os.path.join(
        my_path,
        "tiff_files",
        "test_rgba16.tif"))
    data = np.load(os.path.join(
        my_path,
        "npy_files",
        "test_rgba16.npy"))
    assert_true((s.data == data).all())
    
def test_write_scale_unit():
    """ Lazy test, still need to open the files in ImageJ or DM to check if the
        scale and unit are correct """
    s = hs.signals.Image(np.arange(10*15, dtype=np.uint8).reshape((10, 15)))
    s.axes_manager[0].name = 'x'
    s.axes_manager[1].name = 'y'
    s.axes_manager['x'].scale = 0.25
    s.axes_manager['y'].scale = 0.25
    s.axes_manager['x'].units = 'nm'
    s.axes_manager['y'].units = 'nm'
    fname = os.path.join('tiff_files', 'test_export_scale_unit.tif')
    s.save(fname, overwrite=True, export_scale=True)
    if remove_files:
        os.remove(fname)

def test_write_scale_unit_not_square_pixel():
    """ Lazy test, still need to open the files in ImageJ or DM to check if the
        scale and unit are correct """
    s = hs.signals.Image(np.arange(10*15, dtype=np.uint8).reshape((10, 15)))
    s.change_dtype(np.uint8)
    s.axes_manager[0].name = 'x'
    s.axes_manager[1].name = 'y'
    s.axes_manager['x'].scale = 0.25
    s.axes_manager['y'].scale = 0.5
    s.axes_manager['x'].units = 'nm'
    s.axes_manager['y'].units = 'um'
    fname = os.path.join('tiff_files', 'test_export_scale_unit_not_square_pixel.tif')
    s.save(fname, overwrite=True, export_scale=True)
    if remove_files:
        os.remove(fname)
        
def test_write_scale_with_undefined_unit():
    """ Lazy test, still need to open the files in ImageJ or DM to check if the
        scale and unit are correct """
    s = hs.signals.Image(np.arange(10*15, dtype=np.uint8).reshape((10, 15)))
    s.axes_manager[0].name = 'x'
    s.axes_manager[1].name = 'y'
    s.axes_manager['x'].scale = 0.25
    s.axes_manager['y'].scale = 0.25
    fname = os.path.join('tiff_files', 'test_export_scale_undefined_unit.tif')
    s.save(fname, overwrite=True, export_scale=True)
    if remove_files:
        os.remove(fname)
        
def test_write_scale_with_undefined_scale():
    """ Lazy test, still need to open the files in ImageJ or DM to check if the
        scale and unit are correct """
    s = hs.signals.Image(np.arange(10*15, dtype=np.uint8).reshape((10, 15)))
    s.axes_manager[0].name = 'x'
    s.axes_manager[1].name = 'y'
    fname = os.path.join('tiff_files', 'test_export_scale_undefined_scale.tif')
    s.save(fname, overwrite=True, export_scale=True)
    if remove_files:
        os.remove(fname)
    
def test_write_scale_with_um_unit():
    """ Lazy test, still need to open the files in ImageJ or DM to check if the
        scale and unit are correct """
    s = hs.load(os.path.join('tiff_files', 'test_dm_image_um_unit.dm3'))
    fname = os.path.join('tiff_files', 'test_export_um_unit.tif')
    s.save(fname, overwrite=True, export_scale=True)
    if remove_files:
        os.remove(fname)
    
def test_write_scale_unit_image_stack():
    """ Lazy test, still need to open the files in ImageJ or DM to check if the
        scale and unit are correct """
    s = hs.signals.Image(np.arange(5*10*15, dtype=np.uint8).reshape((5, 10, 15)))
    s.axes_manager[0].name = 'x'
    s.axes_manager[1].name = 'y'
    s.axes_manager[2].name = 'z'
    s.axes_manager['x'].scale = 0.25
    s.axes_manager['y'].scale = 0.5
    s.axes_manager['z'].scale = 1.5
    s.axes_manager['x'].units = 'nm'
    s.axes_manager['y'].units = 'um'
    s.axes_manager['z'].units = 'mm'
    fname = os.path.join('tiff_files', 'test_export_scale_unit_stack.tif')
    s.save(fname, overwrite=True, export_scale=True)
    if remove_files:
        os.remove(fname)