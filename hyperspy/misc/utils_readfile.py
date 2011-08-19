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
# along with Hyperspy; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  
# USA

# general functions for reading data from files

import struct
import os
from exceptions import *

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

B_ulong = struct.Struct('>L')
L_ulong = struct.Struct('<L')

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
        print('File address:', f.tell())
        raise ByteOrderError(endian)
    else:
        data = f.read(2)      # hexadecimal representation
        if endian == 'big':
           s = B_short
        elif endian == 'little':
            s = L_short
        return s.unpack(data)[0] # struct.unpack returns a tuple

def read_ushort(f, endian):
    """Read a 2-Byte integer from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != 'little') and (endian != 'big'):
        print('File address:', f.tell())
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
        print('File address:', f.tell())
        raise ByteOrderError(endian)
    else:
        data = f.read(4)
        if endian == 'big':
            s = B_long
        elif endian == 'little':
            s = L_long
        return s.unpack(data)[0]

def read_ulong(f, endian):
    """Read a 4-Byte integer from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != 'little') and (endian != 'big'):
        print('File address:', f.tell())
        raise ByteOrderError(endian)
    else:
        data = f.read(4)
        if endian == 'big':
            s = B_ulong
        elif endian == 'little':
            s = L_ulong
        return s.unpack(data)[0]
    
def read_float(f, endian):
    """Read a 4-Byte floating point from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != 'little') and (endian != 'big'):
        print('File address:', f.tell())
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
        print('File address:', f.tell())
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
        print('File address:', f.tell())
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
        print('File address:', f.tell())
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
        print('File address:', f.tell())
        raise ByteOrderError(endian)
    else:
        data = f.read(1)
        if endian == 'big':
            s = B_char
        elif endian == 'little':
            s = L_char
        return s.unpack(data)[0]

def read_data_array(filename, byte_size=0, byte_address=0,
                    data_type='uint8', write=True):
    # inspired by numpy's memmap
    """Return a 1-D numpy ndarray from data contained in a binary file.

    Parameters:
    ----------
    filename : str or file-like object.
        The file name or file object to be used as the array data buffer.

    byte_size : int, optional.
        The size in Bytes of the data. If not specified, the
        whole file will be read.

    byte_address : int, optional
        In the file, array data starts at this offset. Since it is measured
        in bytes, it should be a multiple of the byte-size of
        'data_type'. The default is 0 (beginning of the file).

    data_type : data-type, optional
        The data-type used to interpret the file contents.
        Default is 'uint8'.
    write : bool, optional
        Whether the output array should be writeable
    """
    # import here to minimize import overhead
    import mmap
    import numpy as np
    if hasattr(filename,'read'):
        fobj = filename
    else:
        fobj = file(filename, 'r+b')
    if byte_size == 0:
        byte_size = os.fstat(fobj.fileno())[6]
    # To be fast, we want to map just the bytes of the file
    # where the image data is stored. However, the module mmap
    # only allows one to offset at multiples of ALLOCATIONGRANULARITY,
    # so we must set up a little trick to map as little bytes
    # as possible (see also mmap documentation).
    byte_remainder = byte_address % mmap.ALLOCATIONGRANULARITY
    byte_offset = byte_address - byte_remainder
    map_bytes = byte_remainder + byte_size
    dt = np.dtype(data_type).itemsize
    size = byte_size // dt
    shape = (size,)
    bytes = byte_address + byte_size
    acc = mmap.ACCESS_READ
    fmap = mmap.mmap(fobj.fileno(), map_bytes, access=acc, offset=byte_offset)
    fmap.seek(byte_remainder)
    data =  np.ndarray(shape, data_type, fmap.read(byte_size))
    data.setflags(write)
    fmap.flush()
    fmap.close()
    fobj.close()
    return data

# def read_chunk(file_obj, chunk_size=1024):
#     """To be used in a loop
#     chunk_size in Bytes (defaults to 1 kByte)... or whatever
#     """
#     # see also
#     # http://stackoverflow.com/questions/519633/lazy-method-for-reading-big-file-in-python
#     while True:
#         data = file_obj.read(chunk_size)
#         if not data:
#             break
#         yield data

