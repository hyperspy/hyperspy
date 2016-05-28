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
import warnings
from distutils.version import LooseVersion

import traits.api as t
from hyperspy.misc import rgb_tools
#try:
#    from skimage.external.tifffile import imsave, TiffFile
#except ImportError:
#    with warnings.catch_warnings():
#        warnings.simplefilter("ignore")
#        from hyperspy.external.tifffile import imsave, TiffFile
#    warnings.warn(
#        "Failed to import the optional scikit image package. "
#        "Loading of some compressed images will be slow.\n")

# Plugin characteristics
# ----------------------
format_name = 'TIFF'
description = ('Import/Export standard image formats Christoph Gohlke\'s '
               'tifffile library')
full_support = False
file_extensions = ['tif', 'tiff']
default_extension = 0  # tif


# Writing features
writes = [(2, 0), (2, 1)]
# ----------------------

axes_label_codes = {
    'X': "width",
    'Y': "height",
    'S': "sample",
    'P': "plane",
    'I': "image series",
    'Z': "depth",
    'C': "color|em-wavelength|channel",
    'E': "ex-wavelength|lambda",
    'T': "time",
    'R': "region|tile",
    'A': "angle",
    'F': "phase",
    'H': "lifetime",
    'L': "exposure",
    'V': "event",
    'Q': t.Undefined,
    '_': t.Undefined}


def import_tifffile_library(import_local_tifffile_if_necessary=False):
    import skimage
    skimage_version = LooseVersion(skimage.__version__)
    if import_local_tifffile_if_necessary and skimage_version <= LooseVersion('0.12.2'):
        from hyperspy.external.tifffile import imsave, TiffFile
        return imsave, TiffFile
    try:
        from skimage.external.tifffile import imsave, TiffFile
    except ImportError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from hyperspy.external.tifffile import imsave, TiffFile
        warnings.warn(
            "Failed to import the optional scikit image package. "
            "Loading of some compressed images will be slow.\n")
    return imsave, TiffFile

def file_writer(filename, signal, export_scale=True, **kwds):
    """Writes data to tif using Christoph Gohlke's tifffile library

    Parameters
    ----------
    filename: str
    signal: a Signal instance
    export_scale: {True}
        If the scikit-image version is too old, use the hyperspy embedded
        tifffile library to allow exporting the scale and the unit.
    """
    imsave, TiffFile = import_tifffile_library(export_scale)
    data = signal.data
    if signal.is_rgbx is True:
        data = rgb_tools.rgbx2regular_array(data)
        photometric = "rgb"
    else:
        photometric = "minisblack"
    if description not in kwds:
        if signal.metadata.General.title:
            kwds['description'] = signal.metadata.General.title
    kwds.update(get_tags_dict(signal))

    imsave(filename, data,
           software="hyperspy",
           photometric=photometric,
           **kwds)


def file_reader(filename, record_by='image', **kwds):
    """Read data from tif files using Christoph Gohlke's tifffile
    library

    Parameters
    ----------
    filename: str
    record_by: {'image'}
        Has no effect because this format only supports recording by
        image.
    """
    imsave, TiffFile = import_tifffile_library()
    with TiffFile(filename, **kwds) as tiff:
        dc = tiff.asarray()
        axes = tiff.series[0]['axes']
        if tiff.is_rgb:
            dc = rgb_tools.regular_array2rgbx(dc)
            axes = axes[:-1]
        op = {}
        names = [axes_label_codes[axis] for axis in axes]
        axes = [{'size': size,
                 'name': str(name),
                 #'scale': scales[i],
                 #'offset' : origins[i],
                 #'units' : unicode(units[i]),
                 }
                for size, name in zip(dc.shape, names)]
        op = {}
        for key, tag in tiff[0].tags.items():
            op[key] = tag.value
    return [
        {
            'data': dc,
            'original_metadata': op,
            'metadata': {
                'General': {
                    'original_filename': os.path.split(filename)[1]},
                "Signal": {
                    'signal_type': "",
                    'record_by': "image",
                },
            },
        }]

def get_tags_dict(signal, factor=int(1E8)):
    """ Get the tags to export the scale and the unit to be used in
        Digital Micrograph and ImageJ.
    """
    scales, units = _get_scale_unit(signal)
    tags_dict = _get_imagej_kwargs(signal, scales, units, factor=factor)
    tags_dict["extratags"].extend(_get_dm_kwargs_extratag(signal, scales, units))
    return tags_dict
        
def _get_imagej_kwargs(signal, scales, units, factor=int(1E8)):
    resolution = ((factor, int(scales[0]*factor)), (factor, int(scales[1]*factor)))
    description_string = imagej_description(kwargs={"unit":units[0], "scale":scales[0]})
    description_string = imagej_description(kwargs={"unit":units[0]})
    extratag = [(270, 's', 1, description_string, False)]
    return {"resolution":resolution, "extratags":extratag}

def _get_dm_kwargs_extratag(signal, scales, units):
    extratags = [(65003, 's', 3, units[0], False), # x unit
                 (65004, 's', 3, units[1], False), # y unit
                 (65006, 'd', 1, 0.0, False), # x origin in pixel
                 (65007, 'd', 1, 2.0, False), # y origin in pixel
                 (65009, 'd', 1, float(scales[0]), False), # x scale
                 (65010, 'd', 1, float(scales[1]), False), # y scale
                 (65012, 's', 3, units[0], False), # x unit
                 (65013, 's', 3, units[1], False), # y unit
                 (65015, 'i', 1, 1, False),
                 (65016, 'i', 1, 1, False),
                 (65024, 'd', 1, 0.0, False),
                 (65025, 'd', 1, 1.0, False),
                 (65026, 'i', 1, 1, False)]
    if signal.axes_manager.navigation_dimension > 0:
        extratags.extend([(65005, 's', 3, units[2], False), # z unit
                          (65008, 'd', 1, 3.0, False), # z origin in pixel
                          (65011, 'd', 1, float(scales[2]), False), # z scale
                          (65014, 's', 3, units[2], False),  # z unit
                          (65017, 'i', 1, 1, False)])
    return extratags

def _get_scale_unit(signal):
    """ Return a list of scales and units, the length of the list is egal to 
        the signal dimension """
    signal_axes = signal.axes_manager.navigation_axes + signal.axes_manager.signal_axes 
    scales = [signal_axis.scale for signal_axis in signal_axes]
    units = [signal_axis.units for signal_axis in signal_axes]
    for i, unit in enumerate(units):
        if unit == '\xb5m':
            units[i] = 'um'
        if unit == t.Undefined:
            units[i] = ''
    return scales, units
    
def imagej_description(version='1.11a', kwargs={}):
    """ Return a string that will be used by ImageJ to read the unit when
        appropriate arguments are provided """
    result = ['ImageJ=%s' % version]
    append = []
    for key, value in list(kwargs.items()):
        append.append('%s=%s' % (key.lower(), value))

    return '\n'.join(result + append + [''])