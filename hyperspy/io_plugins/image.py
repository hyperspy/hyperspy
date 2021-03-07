# -*- coding: utf-8 -*-
# Copyright 2007-2021 The HyperSpy developers
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

from imageio import imread, imwrite


from hyperspy.misc import rgb_tools

# Plugin characteristics
# ----------------------
format_name = 'Signal2D'
description = 'Import/Export standard image formats using PIL or freeimage'
full_support = False
file_extensions = ['png', 'bmp', 'dib', 'gif', 'jpeg', 'jpe', 'jpg',
                   'msp', 'pcx', 'ppm', "pbm", "pgm", 'xbm', 'spi', ]
default_extension = 0  # png
# Writing features
writes = [(2, 0), ]
# ----------------------


# TODO Extend it to support SI
def file_writer(filename, signal, **kwds):
    """Writes data to any format supported by imageio (PIL/pillow).
    For a list of formats see https://imageio.readthedocs.io/en/stable/formats.html

    Parameters
    ----------
    filename: {str, pathlib.Path, bytes, file}
        The resource to write the image to, e.g. a filename, pathlib.Path or
        file object, see the docs for more info. The file format is defined by 
        the file extension that is any one supported by imageio.
    signal: a Signal instance
    format: str, optional
        The format to use to read the file. By default imageio selects the
        appropriate for you based on the filename and its contents.
    **kwds: keyword arguments
        Allows to pass keyword arguments supported by the individual file
        writers as documented at https://imageio.readthedocs.io/en/stable/formats.html
        
    """
    data = signal.data
    if rgb_tools.is_rgbx(data):
        data = rgb_tools.rgbx2regular_array(data)
    imwrite(filename, data, **kwds)


def file_reader(filename, **kwds):
    """Read data from any format supported by imageio (PIL/pillow).
    For a list of formats see https://imageio.readthedocs.io/en/stable/formats.html

    Parameters
    ----------
    filename: {str, pathlib.Path, bytes, file}
        The resource to load the image from, e.g. a filename, pathlib.Path,
        http address or file object, see the docs for more info. The file format
        is defined by the file extension that is any one supported by imageio.
    format: str, optional
        The format to use to read the file. By default imageio selects the
        appropriate for you based on the filename and its contents.
    **kwds: keyword arguments
        Allows to pass keyword arguments supported by the individual file
        readers as documented at https://imageio.readthedocs.io/en/stable/formats.html

    """
    dc = _read_data(filename, **kwds)
    lazy = kwds.pop('lazy', False)
    if lazy:
        # load the image fully to check the dtype and shape, should be cheap.
        # Then store this info for later re-loading when required
        from dask.array import from_delayed
        from dask import delayed
        val = delayed(_read_data, pure=True)(filename)
        dc = from_delayed(val, shape=dc.shape, dtype=dc.dtype)
    return [{'data': dc,
             'metadata':
             {
                 'General': {'original_filename': os.path.split(filename)[1]},
                 "Signal": {'signal_type': "",
                            'record_by': 'image', },
             }
             }]


def _read_data(filename, **kwds):
    dc = imread(filename)
    if len(dc.shape) > 2:
        # It may be a grayscale image that was saved in the RGB or RGBA
        # format
        if (dc[:, :, 1] == dc[:, :, 2]).all() and \
                (dc[:, :, 1] == dc[:, :, 2]).all():
            dc = dc[:, :, 0]
        else:
            dc = rgb_tools.regular_array2rgbx(dc)
    return dc
