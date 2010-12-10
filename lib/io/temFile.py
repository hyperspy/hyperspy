#!/usr/bin/env python
# -*- coding: latin-1 -*-

# general functions for reading TEM Image files
#
# Copyright (c) 2010 Stefano Mazzucco.
# All rights reserved.
#
# This program is still at an early stage to be released, so the use of this
# code must be explicitly authorized by its author and cannot be shared for any reason.
#
# Once the program will be mature, it will be released under a GNU GPL license

import platform, sys, re, mmap, struct
import numpy as np
from temExceptions import *



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

def readShort(f, endian):
    """Read a 2-Byte integer from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != 'little') and (endian != 'big'):
        print 'File address:', f.tell()
        raise ByteOrderError, endian
    else:
        data = f.read(2)      # hexadecimal representation
        if endian == 'big':
           s = B_short
        elif endian == 'little':
            s = L_short
        return s.unpack(data)[0] # struct.unpack returns a tuple

def readUShort(f, endian):
    """Read a 2-Byte integer from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != 'little') and (endian != 'big'):
        print 'File address:', f.tell()
        raise ByteOrderError, endian
    else:
        data = f.read(2)
        if endian == 'big':
            s = B_ushort
        elif endian == 'little':
            s = L_ushort
        return s.unpack(data)[0]

def readLong(f, endian):
    """Read a 4-Byte integer from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != 'little') and (endian != 'big'):
        print 'File address:', f.tell()
        raise ByteOrderError, endian
    else:
        data = f.read(4)
        if endian == 'big':
            s = B_long
        elif endian == 'little':
            s = L_long
        return s.unpack(data)[0]

def readULong(f, endian):
    """Read a 4-Byte integer from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != 'little') and (endian != 'big'):
        print 'File address:', f.tell()
        raise ByteOrderError, endian
    else:
        data = f.read(4)
        if endian == 'big':
            s = B_ulong
        elif endian == 'little':
            s = L_ulong
        return s.unpack(data)[0]
    
def readFloat(f, endian):
    """Read a 4-Byte floating point from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != 'little') and (endian != 'big'):
        print 'File address:', f.tell()
        raise ByteOrderError, endian
    else:
        data = f.read(4)
        if endian == 'big':
            s = B_float
        elif endian == 'little':
            s = L_float
        return s.unpack(data)[0]    

def readDouble(f, endian):
    """Read a 8-Byte floating point from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != 'little') and (endian != 'big'):
        print 'File address:', f.tell()
        raise ByteOrderError, endian
    else:
        data = f.read(8)
        if endian == 'big':
            s = B_double
        elif endian == 'little':
            s = L_double
        return s.unpack(data)[0]            

def readBoolean(f, endian):
    """Read a 1-Byte charater from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != 'little') and (endian != 'big'):
        print 'File address:', f.tell()
        raise ByteOrderError, endian
    else:
        data = f.read(1)
        if endian == 'big':
            s = B_bool
        elif endian == 'little':
            s = L_bool
        return s.unpack(data)[0]    

def readByte(f, endian):
    """Read a 1-Byte charater from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != 'little') and (endian != 'big'):
        print 'File address:', f.tell()
        raise ByteOrderError, endian
    else:
        data = f.read(1)
        if endian == 'big':
            s = B_byte
        elif endian == 'little':
            s = L_byte
        return s.unpack(data)[0]

def readChar(f, endian):
    """Read a 1-Byte charater from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != 'little') and (endian != 'big'):
        print 'File address:', f.tell()
        raise ByteOrderError, endian
    else:
        data = f.read(1)
        if endian == 'big':
            s = B_char
        elif endian == 'little':
            s = L_char
        return s.unpack(data)[0]
