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

# The details of the format were taken from
# http://www.biochem.mpg.de/doc_tom/TOM_Release_2008/IOfun/tom_mrcread.html
# and http://ami.scripps.edu/software/mrctools/mrc_specification.php

import os
import logging

import numpy as np
from traits.api import Undefined

from hyperspy.misc.array_tools import sarray2dict

_logger = logging.getLogger(__name__)


# Plugin characteristics
# ----------------------
format_name = 'MRC'
description = ''
full_support = False
# Recognised file extension
file_extensions = ['mrc', 'MRC', 'ALI', 'ali']
default_extension = 0

# Writing capabilities
writes = False


def get_std_dtype_list(endianess='<'):
    end = endianess
    dtype_list = \
        [
            ('NX', end + 'u4'),
            ('NY', end + 'u4'),
            ('NZ', end + 'u4'),
            ('MODE', end + 'u4'),
            ('NXSTART', end + 'u4'),
            ('NYSTART', end + 'u4'),
            ('NZSTART', end + 'u4'),
            ('MX', end + 'u4'),
            ('MY', end + 'u4'),
            ('MZ', end + 'u4'),
            ('Xlen', end + 'f4'),
            ('Ylen', end + 'f4'),
            ('Zlen', end + 'f4'),
            ('ALPHA', end + 'f4'),
            ('BETA', end + 'f4'),
            ('GAMMA', end + 'f4'),
            ('MAPC', end + 'u4'),
            ('MAPR', end + 'u4'),
            ('MAPS', end + 'u4'),
            ('AMIN', end + 'f4'),
            ('AMAX', end + 'f4'),
            ('AMEAN', end + 'f4'),
            ('ISPG', end + 'u2'),
            ('NSYMBT', end + 'u2'),
            ('NEXT', end + 'u4'),
            ('CREATID', end + 'u2'),
            ('EXTRA', (np.void, 30)),
            ('NINT', end + 'u2'),
            ('NREAL', end + 'u2'),
            ('EXTRA2', (np.void, 28)),
            ('IDTYPE', end + 'u2'),
            ('LENS', end + 'u2'),
            ('ND1', end + 'u2'),
            ('ND2', end + 'u2'),
            ('VD1', end + 'u2'),
            ('VD2', end + 'u2'),
            ('TILTANGLES', (np.float32, 6)),
            ('XORIGIN', end + 'f4'),
            ('YORIGIN', end + 'f4'),
            ('ZORIGIN', end + 'f4'),
            ('CMAP', (bytes, 4)),
            ('STAMP', (bytes, 4)),
            ('RMS', end + 'f4'),
            ('NLABL', end + 'u4'),
            ('LABELS', (bytes, 800)),
        ]

    return dtype_list


def get_fei_dtype_list(endianess='<'):
    end = endianess
    dtype_list = [
        ('a_tilt', end + 'f4'),  # Alpha tilt (deg)
        ('b_tilt', end + 'f4'),  # Beta tilt (deg)
        # Stage x position (Unit=m. But if value>1, unit=???m)
        ('x_stage', end + 'f4'),
        # Stage y position (Unit=m. But if value>1, unit=???m)
        ('y_stage', end + 'f4'),
        # Stage z position (Unit=m. But if value>1, unit=???m)
        ('z_stage', end + 'f4'),
        # Signal2D shift x (Unit=m. But if value>1, unit=???m)
        ('x_shift', end + 'f4'),
        # Signal2D shift y (Unit=m. But if value>1, unit=???m)
        ('y_shift', end + 'f4'),
        ('defocus', end + 'f4'),  # Defocus Unit=m. But if value>1, unit=???m)
        ('exp_time', end + 'f4'),  # Exposure time (s)
        ('mean_int', end + 'f4'),  # Mean value of image
        ('tilt_axis', end + 'f4'),  # Tilt axis (deg)
        ('pixel_size', end + 'f4'),  # Pixel size of image (m)
        ('magnification', end + 'f4'),  # Magnification used
        # Not used (filling up to 128 bytes)
        ('empty', (np.void, 128 - 13 * 4)),
    ]
    return dtype_list


def get_data_type(index, endianess='<'):
    end = endianess
    data_type = [
        end + 'u2',         # 0 = Signal2D     unsigned bytes
        end + 'i2',         # 1 = Signal2D     signed short integer (16 bits)
        end + 'f4',         # 2 = Signal2D     float
        (end + 'i2', 2),    # 3 = Complex   short*2
        end + 'c8',         # 4 = Complex   float*2
    ]
    return data_type[index]


def file_reader(filename, endianess='<', **kwds):
    metadata = {}
    f = open(filename, 'rb')
    std_header = np.fromfile(f, dtype=get_std_dtype_list(endianess),
                             count=1)
    fei_header = None
    if std_header['NEXT'] / 1024 == 128:
        _logger.info("%s seems to contain an extended FEI header", filename)
        fei_header = np.fromfile(f, dtype=get_fei_dtype_list(endianess),
                                 count=1024)
    if f.tell() == 1024 + std_header['NEXT']:
        _logger.debug("The FEI header was correctly loaded")
    else:
        _logger.warn("There was a problem reading the extended header")
        f.seek(1024 + std_header['NEXT'])
        fei_header = None
    NX, NY, NZ = std_header['NX'], std_header['NY'], std_header['NZ']
    mmap_mode = kwds.pop('mmap_mode', 'c')
    lazy = kwds.pop('lazy', False)
    if lazy:
        mmap_mode = 'r'
    data = np.memmap(f, mode=mmap_mode, offset=f.tell(),
                     dtype=get_data_type(std_header['MODE'], endianess)
                     ).squeeze().reshape((NX, NY, NZ), order='F').T

    original_metadata = {'std_header': sarray2dict(std_header)}
    # Convert bytes to unicode
    for key in ["CMAP", "STAMP", "LABELS"]:
        original_metadata["std_header"][key] = \
            original_metadata["std_header"][key].decode()
    if fei_header is not None:
        fei_dict = sarray2dict(fei_header,)
        del fei_dict['empty']
        original_metadata['fei_header'] = fei_dict

    dim = len(data.shape)
    if fei_header is None:
        # The scale is in Amstrongs, we convert it to nm
        scales = [10 * float(std_header['Zlen'] / std_header['MZ'])
                  if float(std_header['MZ']) != 0 else 1,
                  10 * float(std_header['Ylen'] / std_header['MY'])
                  if float(std_header['MY']) != 0 else 1,
                  10 * float(std_header['Xlen'] / std_header['MX'])
                  if float(std_header['MX']) != 0 else 1, ]
        offsets = [10 * float(std_header['ZORIGIN']),
                   10 * float(std_header['YORIGIN']),
                   10 * float(std_header['XORIGIN']), ]

    else:
        # FEI does not use the standard header to store the scale
        # It does store the spatial scale in pixel_size, one per angle in
        # meters
        scales = [1, ] + [fei_header['pixel_size'][0] * 10 ** 9, ] * 2
        offsets = [0, ] * 3

    units = [Undefined, 'nm', 'nm']
    names = ['z', 'y', 'x']
    metadata = {'General': {'original_filename': os.path.split(filename)[1]},
                "Signal": {'signal_type': "",
                           'record_by': 'image', },
                }
    # create the axis objects for each axis
    axes = [
        {
            'size': data.shape[i],
            'index_in_array': i,
            'name': names[i + 3 - dim],
            'scale': scales[i + 3 - dim],
            'offset': offsets[i + 3 - dim],
            'units': units[i + 3 - dim], }
        for i in range(dim)]

    dictionary = {'data': data,
                  'axes': axes,
                  'metadata': metadata,
                  'original_metadata': original_metadata, }

    return [dictionary, ]
