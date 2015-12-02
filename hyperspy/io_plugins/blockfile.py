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

from hyperspy.misc.array_tools import sarray2dict, dict2sarray


# Plugin characteristics
# ----------------------
format_name = 'Blockfile'
description = 'Read support for ASTAR blockfiles'
full_support = False
# Recognised file extension
file_extensions = ['blo', 'BLO']
default_extension = 0

# Writing capabilities:
# writes = False
writes = [(2, 2)]


mapping = {
    'blockfile_header.Beam_energy':
    ("Acquisition_instrument.TEM.beam_energy", lambda x: x / 1e3),
}


def get_header_dtype_list(endianess='<'):
    end = endianess
    dtype_list = \
        [
            ('ID', (bytes, 6)),
            ('MAGIC', end + 'u2'),
            ('Data_offset_1', end + 'u4'),
            ('Data_offset_2', end + 'u4'),
            ('UNKNOWN1', end + 'u4'),
            ('DP_SZ', end + 'u2'),
            ('UNKNOWN2', end + 'u2'),
            ('NX', end + 'u2'),
            ('NY', end + 'u2'),
            ('UNKNOWN3', end + 'u2'),   # Seems like NX*NY + ???
            ('SX', end + 'f8'),
            ('SY', end + 'f8'),
            ('Beam_energy', end + 'u4'),   # kV
            ('UNKNOWN5', end + 'u2'),
            ('UNKNOWN6', end + 'u4'),
            ('UNKNOWN7', end + 'f8'),
        ] + [
            ('DISTROTION%d' % i, 'f8') for i in xrange(22)
        ]

    return dtype_list


def get_default_header(endianess='<'):
    dt = np.dtype(get_header_dtype_list())
    header = np.zeros((1,), dtype=dt)
    header['ID'][0] = bytes('IMGBLO')
    header['MAGIC'][0] = 0x0102
    header['Data_offset_1'][0] = 0x1000
    header['UNKNOWN1'][0] = 131141
    header['UNKNOWN2'][0] = 0
    header['UNKNOWN3'][0] = 0
    header['BEAM_ENERGY'][0] = 200000
    header['UNKNOWN5'][0] = 6226
    header['UNKNOWN6'][0] = 2000
    header['UNKNOWN7'][0] = 42200.76572
    return header


def get_header_from_signal(signal, endianess='<'):
    if 'blockfile_header' in signal.original_metadata:
        header = dict2sarray(signal.original_metadata['blockfile_header'],
                             dtype=get_header_dtype_list(endianess))
        note = signal.original_metadata['blockfile_header']['Note']
    else:
        header = get_default_header(endianess)
        note = ''
    NX, NY = signal.axes_manager.navigation_shape
    SX = signal.axes_manager.navigation_axes[0].scale
    SY = signal.axes_manager.navigation_axes[0].scale
    DP_SZ = signal.axes_manager.signal_shape
    if DP_SZ[0] != DP_SZ[1]:
        raise ValueError('Blockfiles require signal shape to be square!')
    DP_SZ = DP_SZ[0]

    offset2 = NX*NY + header['Data_offset_1']
    # Based on inspected files, the DPs are stored at 16-bit boundary...
    # Normally, you'd expect word alignment (32-bits) ¯\_(°_o)_/¯
    offset2 += offset2 % 16

    header = dict2sarray({
        'NX': NX, 'NY': NY,
        'DP_SZ': DP_SZ,
        'SX': SX, 'SY': SY,
        'Data_offset_2': offset2,
        }, sarray=header)
    return header, note


def file_reader(filename, endianess='<', **kwds):
    metadata = {}
    f = open(filename, 'rb')
    header = np.fromfile(f, dtype=get_header_dtype_list(endianess), count=1)
    header = sarray2dict(header)
    note = str(f.read(header['Data_offset_1'] - f.tell()))
    header['Note'] = note
    NX, NY = header['NX'], header['NY']
    NDP = header['DP_SZ']
    original_metadata = {'blockfile_header': header}

    # A Virtual BF/DF is stored first
#    offset1 = int(header['DATA_OFFSET_1'][0])
#    f.seek(offset1)
#    data_pre = np.array(f.read(offset2 - offset1), dtype=endianess+'u1'
#        ).squeeze().reshape((NX, NY), order='C').T
#    print len(data_pre)

    # Then comes actual blockfile
    offset2 = header['Data_offset_2']
    f.seek(offset2)
    data = np.memmap(f, mode='c', offset=offset2,
                     dtype=endianess+'u1', shape=(NY, NX, NDP*NDP + 6)
                     )

    # Every frame is preceeded by a 6 byte sequence (AA 55, and then a 4 byte
    # integer specifying frame number)
    data = data[:, :, 6:]
    data = data.reshape((NY, NX, NDP, NDP), order='C')

    units = ['nm', '1/nm', '1/nm', 'nm']
    names = ['x', 'dy', 'dx', 'y']
    scales = [header['SX'], 1.0, 1.0, header['SY']]
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
                  'original_metadata': original_metadata,
                  'mapping': mapping, }

    return [dictionary, ]


def file_writer(filename, signal, **kwds):
    endianess = kwds.pop('endianess', '<')
    header, note = get_header_from_signal(signal, endianess=endianess)
    with open(filename, 'wb') as f:
        # TODO. Use memmap
        # Write header
        header.tofile(f)
        # Write header note field:
        if len(note) > int(header['Data_offset_1']) - f.tell():
            note = note[:int(header['Data_offset_1']) - f.tell() - len(note)]
        f.write(note)
        # Zero pad until next data block
        zero_pad = int(header['Data_offset_1']) - f.tell()
        np.zeros((zero_pad,), np.byte).tofile(f)
        # Write virtual bright field
        vbf = signal.mean(2j).mean(2j).data.astype(endianess+'u1')
        vbf.tofile(f)
        # Zero pad until next data block
        zero_pad = int(header['Data_offset_2']) - f.tell()
        np.zeros((zero_pad,), np.byte).tofile(f)

        # Write full data stack:
        # We need to pad each image with magic 'AA55', then a u32 serial
        dp_head = np.zeros((1,), dtype=[('MAGIC', endianess+'u2'),
                           ('ID', endianess+'u4')])
        dp_head['MAGIC'] = 0x55AA
        # Write by loop:
        for img in signal._iterate_signal():
            dp_head.tofile(f)
            img.astype(endianess+'u1').tofile(f)
            dp_head['ID'] += 1
            if dp_head['ID'] > header['NX'] * header['NY']:
                raise ValueError('Unexpected navigation size.')
