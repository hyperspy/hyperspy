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

import traits.api as t
from hyperspy.misc import rgb_tools
try:
    from skimage.external.tifffile import imsave, TiffFile
except ImportError:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from hyperspy.external.tifffile import imsave, TiffFile
    warnings.warn(
        "Failed to import the optional scikit image package. "
        "Loading of some compressed images will be slow.\n")


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


def file_writer(filename, signal, **kwds):
    """Writes data to tif using Christoph Gohlke's tifffile library

        Parameters
        ----------
        filename: str
        signal: a Signal instance

    """
    data = signal.data
    if signal.is_rgbx is True:
        data = rgb_tools.rgbx2regular_array(data)
        photometric = "rgb"
    else:
        photometric = "minisblack"
    if description not in kwds:
        if signal.metadata.General.title:
            kwds['description'] = signal.metadata.General.title

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
