#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2010 Stefano Mazzucco
#
# This file is part of dm3_data_plugin.
#
# dm3_data_plugin is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# dm3_data_plugin is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301
# USA

# general functions for reading data from files

import struct
import logging

from hyperspy.exceptions import ByteOrderError

_logger = logging.getLogger(__name__)

# Declare simple TagDataType structures for faster execution.
# The variables are named as following:
# Endianness_type
# Endianness = B (big) or L (little)
# type = (u)short, (u)long, float, double, bool (unsigned char),
# byte (signed char), char

B_short = struct.Struct('>h')
L_short = struct.Struct('<h')

B_ushort = struct.Struct('>H')
L_ushort = struct.Struct('<H')

B_long = struct.Struct('>l')
L_long = struct.Struct('<l')

B_long_long = struct.Struct('>q')
L_long_long = struct.Struct('<q')

B_ulong = struct.Struct('>L')
L_ulong = struct.Struct('<L')

B_ulong_long = struct.Struct('>Q')
L_ulong_long = struct.Struct('<Q')

B_float = struct.Struct('>f')
L_float = struct.Struct('<f')

B_double = struct.Struct('>d')
L_double = struct.Struct('<d')

B_bool = struct.Struct('>B')    # use unsigned char
L_bool = struct.Struct('<B')

B_byte = struct.Struct('>b')    # use signed char
L_byte = struct.Struct('<b')

B_char = struct.Struct('>c')
L_char = struct.Struct('<c')


def read_short(f, endian):
    """Read a 2-Byte integer from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != 'little') and (endian != 'big'):
        _logger.debug('File address:', f.tell())
        raise ByteOrderError(endian)
    else:
        data = f.read(2)      # hexadecimal representation
        if endian == 'big':
            s = B_short
        elif endian == 'little':
            s = L_short
        return s.unpack(data)[0]  # struct.unpack returns a tuple


def read_ushort(f, endian):
    """Read a 2-Byte integer from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != 'little') and (endian != 'big'):
        _logger.debug('File address:', f.tell())
        raise ByteOrderError(endian)
    else:
        data = f.read(2)
        if endian == 'big':
            s = B_ushort
        elif endian == 'little':
            s = L_ushort
        return s.unpack(data)[0]


def read_long(f, endian):
    """Read a 4-Byte integer from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != 'little') and (endian != 'big'):
        _logger.debug('File address:', f.tell())
        raise ByteOrderError(endian)
    else:
        data = f.read(4)
        if endian == 'big':
            s = B_long
        elif endian == 'little':
            s = L_long
        return s.unpack(data)[0]


def read_long_long(f, endian):
    """Read a 8-Byte integer from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != 'little') and (endian != 'big'):
        _logger.debug('File address:', f.tell())
        raise ByteOrderError(endian)
    else:
        data = f.read(8)
        if endian == 'big':
            s = B_long_long
        elif endian == 'little':
            s = L_long_long
        return s.unpack(data)[0]


def read_ulong(f, endian):
    """Read a 4-Byte integer from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != 'little') and (endian != 'big'):
        _logger.debug('File address:', f.tell())
        raise ByteOrderError(endian)
    else:
        data = f.read(4)
        if endian == 'big':
            s = B_ulong
        elif endian == 'little':
            s = L_ulong
        return s.unpack(data)[0]


def read_ulong_long(f, endian):
    """Read a 8-Byte integer from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != 'little') and (endian != 'big'):
        _logger.debug('File address:', f.tell())
        raise ByteOrderError(endian)
    else:
        data = f.read(8)
        if endian == 'big':
            s = B_ulong_long
        elif endian == 'little':
            s = L_ulong_long
        return s.unpack(data)[0]


def read_float(f, endian):
    """Read a 4-Byte floating point from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != 'little') and (endian != 'big'):
        _logger.debug('File address:', f.tell())
        raise ByteOrderError(endian)
    else:
        data = f.read(4)
        if endian == 'big':
            s = B_float
        elif endian == 'little':
            s = L_float
        return s.unpack(data)[0]


def read_double(f, endian):
    """Read a 8-Byte floating point from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != 'little') and (endian != 'big'):
        _logger.debug('File address:', f.tell())
        raise ByteOrderError(endian)
    else:
        data = f.read(8)
        if endian == 'big':
            s = B_double
        elif endian == 'little':
            s = L_double
        return s.unpack(data)[0]


def read_boolean(f, endian):
    """Read a 1-Byte charater from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != 'little') and (endian != 'big'):
        _logger.debug('File address:', f.tell())
        raise ByteOrderError(endian)
    else:
        data = f.read(1)
        if endian == 'big':
            s = B_bool
        elif endian == 'little':
            s = L_bool
        return s.unpack(data)[0]


def read_byte(f, endian):
    """Read a 1-Byte charater from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != 'little') and (endian != 'big'):
        _logger.debug('File address:', f.tell())
        raise ByteOrderError(endian)
    else:
        data = f.read(1)
        if endian == 'big':
            s = B_byte
        elif endian == 'little':
            s = L_byte
        return s.unpack(data)[0]


def read_char(f, endian):
    """Read a 1-Byte charater from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != 'little') and (endian != 'big'):
        _logger.debug('File address:', f.tell())
        raise ByteOrderError(endian)
    else:
        data = f.read(1)
        if endian == 'big':
            s = B_char
        elif endian == 'little':
            s = L_char
        return s.unpack(data)[0]
