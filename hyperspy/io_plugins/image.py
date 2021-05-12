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
import logging

from imageio import imread, imwrite
from matplotlib.figure import Figure
import traits.api as t
import pint

from hyperspy.misc import rgb_tools

# Plugin characteristics
# ----------------------
format_name = 'Signal2D'
description = 'Import/Export standard image formats using pillow, freeimage or matplotlib (with scalebar)'
full_support = False
file_extensions = ['png', 'bmp', 'dib', 'gif', 'jpeg', 'jpe', 'jpg',
                   'msp', 'pcx', 'ppm', "pbm", "pgm", 'xbm', 'spi', ]
default_extension = 0  # png
# Writing capabilities
writes = [(2, 0), ]
non_uniform_axis = False
# ----------------------

_ureg = pint.UnitRegistry()
_logger = logging.getLogger(__name__)


def file_writer(filename, signal, scalebar=False,
                scalebar_kwds={'box_alpha':0.75, 'location':'lower left'},
                **kwds):
    """Writes data to any format supported by PIL

    Parameters
    ----------
    filename: {str, pathlib.Path, bytes, file}
        The resource to write the image to, e.g. a filename, pathlib.Path or
        file object, see the docs for more info. The file format is defined by 
        the file extension that is any one supported by imageio.
    signal: a Signal instance
    scalebar : bool, optional
        Export the image with a scalebar.
    scalebar_kwds : dict
        Dictionary of keyword arguments for the scalebar. Useful to set
        formattiong, location, etc. of the scalebar. See the documentation of
        the 'matplotlib-scalebar' library for more information.
    **kwds: keyword arguments
        Allows to pass keyword arguments supported by the individual file
        writers as documented at https://imageio.readthedocs.io/en/stable/formats.html

    """
    data = signal.data
    if rgb_tools.is_rgbx(data):
        data = rgb_tools.rgbx2regular_array(data)
    if scalebar:
        try:
            from matplotlib_scalebar.scalebar import ScaleBar
            export_scalebar = True
        except ImportError:  # pragma: no cover
            export_scalebar = False
            _logger.warning("Exporting image with scalebar requires the "
                            "matplotlib-scalebar library.")
        dpi = 100
        fig = Figure(figsize=[v/dpi for v in signal.axes_manager.signal_shape],
                     dpi=dpi)

        try:
            # List of format supported by matplotlib
            supported_format = sorted(fig.canvas.get_supported_filetypes())
            if os.path.splitext(filename)[1].replace('.', '') not in supported_format:
                export_scalebar = False
                _logger.warning("Exporting image with scalebar is supported "
                                f"only with {', '.join(supported_format)}.")
        except AttributeError:  # pragma: no cover
            export_scalebar = False
            _logger.warning("Exporting image with scalebar requires the "
                            "matplotlib 3.1 or newer.")

    if scalebar and export_scalebar:
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.imshow(data, cmap='gray')

        # Add scalebar
        axis = signal.axes_manager.signal_axes[0]
        if axis.units == t.Undefined:
            axis.units = "px"
            scalebar_kwds['dimension'] = "pixel-length"
        if _ureg.Quantity(axis.units).check('1/[length]'):
            scalebar_kwds['dimension'] = "si-length-reciprocal"
        scalebar = ScaleBar(axis.scale, axis.units, **scalebar_kwds)
        ax.add_artist(scalebar)
        fig.savefig(filename, dpi=dpi, pil_kwargs=kwds)
    else:
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
