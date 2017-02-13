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
from traits.api import Undefined
import numpy as np
import logging
import warnings
import datetime
import dateutil

from hyperspy.misc.array_tools import sarray2dict, dict2sarray
from hyperspy.misc.date_time_tools import serial_date_to_ISO_format, datetime_to_serial_date

_logger = logging.getLogger(__name__)


# Plugin characteristics
# ----------------------
format_name = 'Blockfile'
description = 'Read/write support for ASTAR blockfiles'
full_support = False
# Recognised file extension
file_extensions = ['blo', 'BLO']
default_extension = 0

# Writing capabilities:
writes = [(2, 2), (2, 1), (2, 0)]
magics = [0x0102]


mapping = {
    'blockfile_header.Beam_energy':
    ("Acquisition_instrument.TEM.beam_energy", lambda x: x * 1e-3),
    'blockfile_header.Camera_length':
    ("Acquisition_instrument.TEM.camera_length", lambda x: x * 1e-4),
    'blockfile_header.Scan_rotation':
    ("Acquisition_instrument.TEM.scan_rotation", lambda x: x * 1e-2),
}


def get_header_dtype_list(endianess='<'):
    end = endianess
    dtype_list = \
        [
            ('ID', (bytes, 6)),
            ('MAGIC', end + 'u2'),
            ('Data_offset_1', end + 'u4'),      # Offset VBF
            ('Data_offset_2', end + 'u4'),      # Offset DPs
            ('UNKNOWN1', end + 'u4'),           # Flags for ASTAR software?
            ('DP_SZ', end + 'u2'),              # Pixel dim DPs
            ('DP_rotation', end + 'u2'),        # [degrees ( * 100 ?)]
            ('NX', end + 'u2'),                 # Scan dim 1
            ('NY', end + 'u2'),                 # Scan dim 2
            ('Scan_rotation', end + 'u2'),      # [100 * degrees]
            ('SX', end + 'f8'),                 # Pixel size [nm]
            ('SY', end + 'f8'),                 # Pixel size [nm]
            ('Beam_energy', end + 'u4'),        # [V]
            ('SDP', end + 'u2'),                # Pixel size [100 * ppcm]
            ('Camera_length', end + 'u4'),      # [10 * mm]
            ('Acquisition_time', end + 'f8'),   # [Serial date]
        ] + [
            ('Centering_N%d' % i, 'f8') for i in range(8)
        ] + [
            ('Distortion_N%02d' % i, 'f8') for i in range(14)
        ]

    return dtype_list


def get_default_header(endianess='<'):
    """Returns a header pre-populated with default values.
    """
    dt = np.dtype(get_header_dtype_list())
    header = np.zeros((1,), dtype=dt)
    header['ID'][0] = 'IMGBLO'.encode()
    header['MAGIC'][0] = magics[0]
    header['Data_offset_1'][0] = 0x1000     # Always this value observed
    header['UNKNOWN1'][0] = 131141          # Very typical value (always?)
    header['Acquisition_time'][0] = datetime_to_serial_date(
        datetime.datetime.fromtimestamp(86400, dateutil.tz.tzutc()))
    # Default to UNIX epoch + 1 day
    # Have to add 1 day, as dateutil's timezones dont work before epoch
    return header


def get_header_from_signal(signal, endianess='<'):
    header = get_default_header(endianess)
    if 'blockfile_header' in signal.original_metadata:
        header = dict2sarray(signal.original_metadata['blockfile_header'],
                             sarray=header)
        note = signal.original_metadata['blockfile_header']['Note']
    else:
        note = ''
    if signal.axes_manager.navigation_dimension == 2:
        NX, NY = signal.axes_manager.navigation_shape
        SX = signal.axes_manager.navigation_axes[0].scale
        SY = signal.axes_manager.navigation_axes[1].scale
    elif signal.axes_manager.navigation_dimension == 1:
        NX = signal.axes_manager.navigation_shape[0]
        NY = 1
        SX = signal.axes_manager.navigation_axes[0].scale
        SY = SX
    elif signal.axes_manager.navigation_dimension == 0:
        NX = NY = SX = SY = 1

    DP_SZ = signal.axes_manager.signal_shape
    if DP_SZ[0] != DP_SZ[1]:
        raise ValueError('Blockfiles require signal shape to be square!')
    DP_SZ = DP_SZ[0]
    SDP = 100. / signal.axes_manager.signal_axes[0].scale

    offset2 = NX * NY + header['Data_offset_1']
    # Based on inspected files, the DPs are stored at 16-bit boundary...
    # Normally, you'd expect word alignment (32-bits) ¯\_(°_o)_/¯
    offset2 += offset2 % 16

    header = dict2sarray({
        'NX': NX, 'NY': NY,
        'DP_SZ': DP_SZ,
        'SX': SX, 'SY': SY,
        'SDP': SDP,
        'Data_offset_2': offset2,
    }, sarray=header)
    return header, note


def file_reader(filename, endianess='<',  mmap_mode='c',
                lazy=False, **kwds):
    _logger.debug("Reading blockfile: %s" % filename)
    metadata = {}
    # Makes sure we open in right mode:
    if '+' in mmap_mode or ('write' in mmap_mode and
                            'copyonwrite' != mmap_mode):
        if lazy:
            raise ValueError("Lazy loading does not support in-place writing")
        f = open(filename, 'r+b')
    else:
        f = open(filename, 'rb')
    _logger.debug("File opened")

    # Get header
    header = np.fromfile(f, dtype=get_header_dtype_list(endianess), count=1)
    if header['MAGIC'][0] not in magics:
        warnings.warn("Blockfile has unrecognized header signature. "
                      "Will attempt to read, but correcteness not guaranteed!")
    header = sarray2dict(header)
    note = f.read(header['Data_offset_1'] - f.tell())
    note = note.strip(b'\x00')
    header['Note'] = note.decode()
    _logger.debug("File header: " + str(header))
    NX, NY = int(header['NX']), int(header['NY'])
    DP_SZ = int(header['DP_SZ'])
    if header['SDP']:
        SDP = 100. / header['SDP']
    else:
        SDP = Undefined
    original_metadata = {'blockfile_header': header}

    # Get data:

    # A Virtual BF/DF is stored first
#    offset1 = header['Data_offset_1']
#    f.seek(offset1)
#    data_pre = np.array(f.read(NX*NY), dtype=endianess+'u1'
#        ).squeeze().reshape((NX, NY), order='C').T

    # Then comes actual blockfile
    offset2 = header['Data_offset_2']
    if not lazy:
        f.seek(offset2)
        data = np.fromfile(f, dtype=endianess + 'u1')
    else:
        data = np.memmap(f, mode=mmap_mode, offset=offset2,
                         dtype=endianess + 'u1')
    try:
        data = data.reshape((NY, NX, DP_SZ * DP_SZ + 6))
    except ValueError:
        warnings.warn(
            'Blockfile header dimensions larger than file size! '
            'Will attempt to load by zero padding incomplete frames.')
        # Data is stored DP by DP:
        pw = [(0, NX * NY * (DP_SZ * DP_SZ + 6) - data.size)]
        data = np.pad(data, pw, mode='constant')
        data = data.reshape((NY, NX, DP_SZ * DP_SZ + 6))

    # Every frame is preceeded by a 6 byte sequence (AA 55, and then a 4 byte
    # integer specifying frame number)
    data = data[:, :, 6:]
    data = data.reshape((NY, NX, DP_SZ, DP_SZ), order='C').squeeze()

    units = ['nm', 'nm', 'cm', 'cm']
    names = ['y', 'x', 'dy', 'dx']
    scales = [header['SY'], header['SX'], SDP, SDP]
    date, time, time_zone = serial_date_to_ISO_format(header['Acquisition_time'])
    metadata = {'General': {'original_filename': os.path.split(filename)[1],
                            'date': date,
                            'time': time,
                            'time_zone': time_zone,
                            'notes': header['Note']},
                "Signal": {'signal_type': "diffraction",
                           'record_by': 'image', },
                }
    # Create the axis objects for each axis
    dim = data.ndim
    axes = [
        {
            'size': data.shape[i],
            'index_in_array': i,
            'name': names[i],
            'scale': scales[i],
            'offset': 0.0,
            'units': units[i], }
        for i in range(dim)]

    dictionary = {'data': data,
                  'axes': axes,
                  'metadata': metadata,
                  'original_metadata': original_metadata,
                  'mapping': mapping, }

    f.close()
    return [dictionary, ]


def file_writer(filename, signal, **kwds):
    endianess = kwds.pop('endianess', '<')
    header, note = get_header_from_signal(signal, endianess=endianess)
    with open(filename, 'wb') as f:
        # Write header
        header.tofile(f)
        # Write header note field:
        if len(note) > int(header['Data_offset_1']) - f.tell():
            note = note[:int(header['Data_offset_1']) - f.tell() - len(note)]
        f.write(note.encode())
        # Zero pad until next data block
        zero_pad = int(header['Data_offset_1']) - f.tell()
        np.zeros((zero_pad,), np.byte).tofile(f)
        # Write virtual bright field
        vbf = signal.mean(
            signal.axes_manager.signal_axes[
                :2]).data.astype(
            endianess +
            'u1')
        vbf.tofile(f)
        # Zero pad until next data block
        if f.tell() > int(header['Data_offset_2']):
            raise ValueError("Signal navigation size does not match "
                             "data dimensions.")
        zero_pad = int(header['Data_offset_2']) - f.tell()
        np.zeros((zero_pad,), np.byte).tofile(f)

        # Write full data stack:
        # We need to pad each image with magic 'AA55', then a u32 serial
        dp_head = np.zeros((1,), dtype=[('MAGIC', endianess + 'u2'),
                                        ('ID', endianess + 'u4')])
        dp_head['MAGIC'] = 0x55AA
        # Write by loop:
        for img in signal._iterate_signal():
            dp_head.tofile(f)
            img.astype(endianess + 'u1').tofile(f)
            dp_head['ID'] += 1
