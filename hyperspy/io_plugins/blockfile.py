# -*- coding: utf-8 -*-
# Copyright 2007-2015 The HyperSpy developers
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

import numpy as np

from hyperspy.misc.array_tools import sarray2dict


# Plugin characteristics
# ----------------------
format_name = 'Blockfile'
description = ''
full_support = False
# Recognised file extension
file_extensions = ['blo', 'BLO']
default_extension = 0

# Writing capabilities:
writes = False
# Limited write abilities should be possible if shape hasn't changed, but
# currently disabled


def get_header_dtype_list(endianess='<'):
    end = endianess
    dtype_list = \
        [
            ('ID', (str, 6)),
            ('MAGIC', end + 'u2'),
            ('DATA_OFFSET_1', end + 'u4'),
            ('DATA_OFFSET_2', end + 'u4'),
            ('UNKNOWN1', end + 'u4'),
            ('DP_SZ', end + 'u2'),
            ('UNKNOWN2', end + 'u2'),
            ('NX', end + 'u2'),
            ('NY', end + 'u2'),
            ('UNKNOWN3', end + 'u2'),
            ('SX', end + 'f8'),
            ('SY', end + 'f8'),
            ('UNKNOWN4', end + 'u4'),
            ('UNKNOWN5', end + 'u2'),
            ('UNKNOWN6', end + 'u4'),
            ('UNKNOWN7', end + 'f8'),
            # There are also two more unknown doubles (f8) in the header.
            # Possibly, they specify scale in diffraction image
        ]

    return dtype_list


def file_reader(filename, endianess='<', **kwds):
    metadata = {}
    f = open(filename, 'rb')
    header = np.fromfile(f, dtype=get_header_dtype_list(endianess), count=1)
    NX, NY = int(header['NX']), int(header['NY'])
    NDP = int(header['DP_SZ'])
    original_metadata = {'header': sarray2dict(header)}

    # A Virtual BF/DF is stored first
#    offset1 = int(header['DATA_OFFSET_1'][0])
#    f.seek(offset1)
#    data_pre = np.array(f.read(offset2 - offset1), dtype=endianess+'u1'
#        ).squeeze().reshape((NX, NY), order='C').T
#    print len(data_pre)

    # Then comes actual blockfile
    offset2 = int(header['DATA_OFFSET_2'])
    f.seek(offset2)
    data = np.memmap(f, mode='c', offset=offset2,
                     dtype=endianess+'u1'
                     )

    # Every frame is preceeded by a 6 byte sequence (AA 55, and then a 4 byte
    # integer specifying frame number)
    data = data.squeeze().reshape((NY, NX, NDP*NDP + 6), order='C')
    data = data[:, :, 6:]
    data = data.reshape((NY, NX, NDP, NDP), order='C')

    units = ['nm', '1/nm', '1/nm', 'nm']
    names = ['x', 'dy', 'dx', 'y']
    scales = [float(header['SX']), 1.0, 1.0, float(header['SY'])]
    metadata = {'General': {'original_filename': os.path.split(filename)[1]},
                "Signal": {'signal_type': "",
                           'record_by': 'image', },
                }
    # create the axis objects for each axis
    dim = 4
    axes = [
        {
            'size': data.shape[i],
            'index_in_array': i,
            'name': names[i + 3 - dim],
            'scale': scales[i + 3 - dim],
            'offset': 0.0,
            'units': units[i + 3 - dim], }
        for i in xrange(dim)]

    dictionary = {'data': data,
                  'axes': axes,
                  'metadata': metadata,
                  'original_metadata': original_metadata, }

    return [dictionary, ]
