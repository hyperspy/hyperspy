# -*- coding: utf-8 -*-
# Copyright 2010 Stefano Mazzucco
# Copyright 2011 The Hyperspy developers
#
# This file is part of  Hyperspy. It is a fork of the original PIL dm3 plugin 
# written by Stefano Mazzucco.
#
#  Hyperspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  Hyperspy.  If not, see <http://www.gnu.org/licenses/>.

# Plugin to read the Gatan Digital Micrograph(TM) file format



# Tags structure
#DM3
# |---ApplicationBounds*
# |---Version* {custom}
# |---isLittleEndian* {custom}
# |---FileSize* {custom}
# |---DocumentObjectList
#     |---DocumentTags
#          |---HasWindowPosition*
#          |---Image Behavior
#          |   |---UnscaledTransform
#          |   |   |---[...]
#          |   |
#          |   |---IsZoomedToWindow*
#          |   |---ImageDisplayBounds*
#          |   |---DoIntegralZoom*
#          |   |---ImageList
#          |       |---ImageSourceList
#          |       |   |---[...]
#          |       |   |---[...]
#          |       |
#          |       |---Group[ID]
#          |           |---ImageData
#          |               |---ImageTags
#          |               |   |---Camera {ASU - JEOL2010 - EELS}
#          |               |   |   |---Device
#          |               |   |   |   |---Camera Number*
#          |               |   |   |   |---Active Size (pixels)*
#          |               |   |   |   |---CCD
#          |               |   |   |   |   |---Pixel Size (um)*
#          |               |   |   |   |   |---Configuration
#          |               |   |   |   |       |---Source*
#          |               |   |   |   |       |---Name*
#          |               |   |   |   |       |---Transpose
#          |               |   |   |   |           |---[...]
#          |               |   |   |   |---Frame
#          |               |   |   |       |---Parameters
#          |               |   |   |       |   |---Acquisition Write Flags*
#          |               |   |   |       |   |---Base Detector
#          |               |   |   |       |       |---[...]
#          |               |   |   |       | 
#          |               |   |   |       |---Area
#          |               |   |   |           |---Transform
#          |               |   |   |           |   |---[...]
#          |               |   |   |           | 
#          |               |   |   |           |---CCD
#          |               |   |   |               |---Intensity
#          |               |   |   |               |---Pixel Size (um)
#          |               |   |   |---EELS
#          |               |   |       |---Meta Data
#          |               |   |       |   |---Signal*
#          |               |   |       |   |---Format*
#          |               |   |       |   |---Acquisition Mode*
#          |               |   |       |   |---Microscope Info
#          |               |   |       |       |---Imaging Mode*
#          |               |   |       |       |---Emission Current (uA)*
#          |               |   |       |       |---Illumination Mode*
#          |               |   |       |       |---Cs(mm)*
#          |               |   |       |       |---Items
#          |               |   |       |           |---Voltage*
#          |               |   |       |           |---Specimen*
#          |               |   |       |           |---Microscope*
#          |               |   |       |           |---Operation Mode*
#          |               |   |       |           |---Probe Current (nA)*
#          |               |   |       |           |---Operator*
#          |               |   |       |           |---Probe Size (nm)*
#          |               |   |       |           |---Group[X]
#          |               |   |       |               |---[...]
#          |               |   |       |---Acquisition
#          |               |   |           |---Number of frames
#          |               |   |           |---Integration time (s)
#          |               |   |           |---End time
#          |               |   |           |---Experimental Conditions
#          |               |   |           |---Saturation fraction
#          |               |   |           |---Exposure (s)
#          |               |   |           |---Date
#          |               |   |           |---Spectrometer
#          |               |   |   
#          |               |   |---Microscope Info {ORSAY}
#          |               |   |   |---Microscope*
#          |               |   |   |---Private
#          |               |   |       |---DataBar
#          |               |   |       |   |---Applied*
#          |               |   |       |
#          |               |   |       |---Processing
#          |               |   |           |---3 windows
#          |               |   |           |   |---dispaly bgd*
#          |               |   |           |   |---fit*
#          |               |   |           |
#          |               |   |           |---spim
#          |               |   |               |---detectors
#          |               |   |                   |---1*
#          |               |   |                   |---eels
#          |               |   |                       |---gain ID*
#          |               |   |                       |---energy shift*
#          |               |   |                       |---dwell time*
#          |               |   |                       |---version*
#          |               |   |                       |---data type*
#          |               |   |                       |---nb spectra per pixel*
#          |               |   |                       |---label x*
#          |               |   |---Private
#          |               |   |   |---DataBar
#          |               |   |       |---Applied*
#          |               |   |
#          |               |   |---Name*
#          |               |   |---UniqueID
#          |               |       |---Data0*
#          |               |       |---Data1*
#          |               |       |---Data2*
#          |               | 
#          |               |---Calibrations
#          |                   |---DataType*
#          |                   |---Data*
#          |                   |---Dimensions
#          |                   |   |---PixelDepth
#          |                   |   |---Data[X]*
#          |                   |
#          |                   |---Brightness
#          |                       |---Units*
#          |                       |---Origin*
#          |                       |---Scale*
#          |                       |---Dimension
#          |                           |---DisplayCalibratedUnits*
#          |                           |---Group[X]
#          |                               |---Origin*
#          |                               |---Units*
#          |                               |---Scale*
#          |---Group[ID]
#              |---AnnotationGroupList
#                  |---AnnotationType*
#                  |---ForegroundColor*
#                  |---HasBackground*
#                  |---FillMode*
#                  |---BackgroundMode*
#                  |---BackgroundColor*
#                  |---ImageDisplayInfo
#                      |---CaptionOn*
#                      |---Contrast*
#                      |---ImageDisplayType*
#                      |---ComplexRange*
#                      |---Brightness*
#                      |---ContrastMode*
#                      |---CLUTName*
#                      |---CLUT*
#                      |---ObjectTags*
#                      |---IsTranslatable*
#                      |---IsVisible*
#                      |---IsSelectable*
#                      |---BrightColor*
#                      |---ImageSource*
#                      |---IsResizable*
#                      |---IsMoveable*
#                      |---ComplexMode*
#                      |---DimensionLabels
#                          |---[...]

from __future__ import with_statement #for Python versions < 2.6
from __future__ import division

import os
import mmap
import re
from types import StringType
import numpy as np

from hyperspy.axes import DataAxis
from hyperspy.misc.utils_readfile import *
from hyperspy.exceptions import *
import hyperspy.misc.utils
from hyperspy.misc.utils_varia import overwrite, swapelem
from hyperspy.misc.utils_varia import DictBrowser, fsdict
from hyperspy.misc.dm3reader import parseDM3


# Plugin characteristics
# ----------------------
format_name = 'Digital Micrograph dm3'
description = 'Read data from Gatan Digital Micrograph (TM) files'
full_suport = False
# Recognised file extension
file_extensions = ('dm3', 'DM3')
default_extension = 0

# Writing features
writes = False
# ----------------------

## used in crawl_dm3 ##
tag_group_pattern = re.compile('\.Group[0-9]{1,}$')
tag_data_pattern = re.compile('\.Data[0-9]{1,}$')
image_data_pattern = re.compile('\.Calibrations\.Data$')
micinfo_pattern = re.compile('\.Microscope Info$')
orsay_pattern = re.compile('\.spim$')
# root_pattern = re.compile('^\w{1,}\.')
image_tags_pattern = re.compile('.*ImageTags\.')
document_tags_pattern = re.compile('.*DocumentTags\.')

####

read_char = read_byte # dm3 uses chars for 1-Byte signed integers

def node_valve(nodes, value, dictionary):
    node = nodes.pop(0)
    if node not in dictionary:
        dictionary[node] = {}
    if len(nodes) != 0:
        node_valve(nodes,value, dictionary[node])
    else:
        dictionary[node] = value
        return
    
def read_infoarray(f):
    """Read the infoarray from file f and return it.
    """
    infoarray_size = read_long(f, 'big')
    infoarray = [read_long(f, 'big') for index in xrange(infoarray_size)]
    infoarray = tuple(infoarray)
    return infoarray

def _infoarray_databytes(iarray):
    """Read the info array iarray and return the number of bytes
    of the corresponding TagData.
    """
    if iarray[0] in _complex_type:
        if iarray[0] == 18: # it's a string
            nbytes = iarray[1]
        elif iarray[0] == 15:   # it's a struct
            field_type =  [iarray[i] for i in xrange(4, len(iarray), 2)]
            field_bytes = [_data_type[i][1] for i in field_type]
            nbytes = reduce(lambda x, y: x +y, field_bytes)
        elif iarray[0] == 20:   # it's an array            
            if iarray[1] != 15:
                nbytes = iarray[-1] * _data_type[iarray[1]][1]
            else:  # it's an array of structs
                subiarray = iarray[1:-1]
                nbytes = _infoarray_databytes(subiarray) * iarray[-1]
    elif iarray[0] in _simple_type:
        nbytes = _data_type[iarray[0]][1]
    else:
        raise DM3DataTypeError(iarray[0])
    return nbytes
 
def read_string(f, iarray, endian):
    """Read a string defined by the infoArray iarray from
     file f with a given endianness (byte order).
    endian can be either 'big' or 'little'.

    If it's a tag name, each char is 1-Byte;
    if it's a tag data, each char is 2-Bytes Unicode,
    """    
    if (endian != 'little') and (endian != 'big'):
        print('File address:', f.tell())
        raise ByteOrderError(endian)
    else:
        if iarray[0] != 18:
            print('File address:', f.tell())
            raise DM3DataTypeError(iarray[0])
        data = ''
        if endian == 'little':
            s = L_char
        elif endian == 'big':
            s = B_char
        for char in xrange(iarray[1]):
            data += s.unpack(f.read(1))[0]
        #~ if '\x00' in data:      # it's a Unicode string (TagData)
            #~ uenc = 'utf_16_'+endian[0]+'e'
            #~ data = unicode(data, uenc, 'replace')
        try:
            data = data.decode('utf8')
        except:
            # Sometimes the dm3 file strings are encoded in latin-1
            # instead of utf8
            data = data.decode('latin-1', errors = 'ignore')
        return data

def read_struct(f, iarray, endian):
    """Read a struct, defined by iarray, from file f
    with a given endianness (byte order).
    Returns a list of 2-tuples in the form
    (fieldAddress, fieldValue).
    endian can be either 'big' or 'little'.
    """
    if (endian != 'little') and (endian != 'big'):
        print('File address:', f.tell())
        raise ByteOrderError(endian)
    else:    
        if iarray[0] != 15:
            print('File address:', f.tell())
            raise DM3DataTypeError(iarray[0])
        # name_length = iarray[1]
        # name_length always 0?
        # n_fields = iarray[2]
        # field_name_length = [iarray[i] for i in xrange(3, len(iarray), 2)]
        # field_name_length always 0?
        field_type =  [iarray[i] for i in xrange(4, len(iarray), 2)]
        # field_ctype = [_data_type[iarray[i]][2] for i in xrange(4, len(iarray), 2)]
        field_addr = []
        # field_bytes = [_data_type[i][1] for i in field_type]
        field_value = []
        for dtype in field_type:
            if dtype in _simple_type:
                field_addr.append(f.tell())
                read_data = _data_type[dtype][0]
                data = read_data(f, endian)
                field_value.append(data)
            else:
                raise DM3DataTypeError(dtype)    
        return zip(field_addr, field_value)
    
def read_array(f, iarray, endian):
    """Read an array, defined by iarray, from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != 'little') and (endian != 'big'):
        print('File address:', f.tell())
        raise ByteOrderError(endian)
    else:        
        if iarray[0] != 20:
            print('File address:', f.tell())
            raise DM3DataTypeError(iarray[0])
        arraysize = iarray[-1]
        if arraysize == 0:
            return None
        eltype = _data_type[iarray[1]][0] # same for all elements
        if len(iarray) > 3:  # complex type
            subiarray = iarray[1:-1]
            data = [eltype(f, subiarray, endian)
                    for element in xrange(arraysize)]
        else: # simple type
            data = [eltype(f, endian) for element in xrange(arraysize)]
            if iarray[1] == 4: # it's actually a string
                # disregard values that are not characters:
                data = [chr(i) for i in data if i in xrange(256)]
                data = reduce(lambda x, y: x + y, data)
        return data
    
# _data_type dictionary.
# The first element of the InfoArray in the TagType
# will always be one of _data_type keys.
# the tuple reads: ('read bytes function', 'number of bytes', 'type')
_data_type = {
    2 : (read_short, 2, 'h'),
    3 : (read_long, 4, 'l'),
    4 : (read_ushort, 2, 'H'), # dm3 uses ushorts for unicode chars
    5 : (read_ulong, 4, 'L'),
    6 : (read_float, 4, 'f'),
    7 : (read_double, 8, 'd'),
    8 : (read_boolean, 1, 'B'),
    9 : (read_char, 1, 'b'), # dm3 uses chars for 1-Byte signed integers
    10 : (read_byte, 1, 'b'),   # 0x0a
    15 : (read_struct, None, 'struct',), # 0x0f
    18 : (read_string, None, 'c'), # 0x12
    20 : (read_array, None, 'array'),  # 0x14
    }
                          
_complex_type = (15, 18, 20)
_simple_type =  (2, 3, 4, 5, 6, 7, 8, 9, 10)

def parse_tag_group(f, endian='big'):
    """Parse the root TagGroup of the given DM3 file f.
    Returns the tuple (is_sorted, is_open, n_tags).
    endian can be either 'big' or 'little'.
    """
    is_sorted = read_byte(f, endian)
    is_open = read_byte(f, endian)
    n_tags = read_long(f, endian)
    return bool(is_sorted), bool(is_open), n_tags

def parse_tag_entry(f, endian='big'):
    """Parse a tag entry of the given DM3 file f.
    Returns the tuple (tag_id, tag_name_length, tag_name).
    endian can be either 'big' or 'little'.
    """
    tag_id = read_byte(f, endian)
    tag_name_length = read_short(f, endian)
    str_infoarray = (18, tag_name_length)
    tag_name_arr = read_string(f, str_infoarray, endian)
    if len(tag_name_arr):
        tag_name = reduce(lambda x, y: x + y, tag_name_arr)
    else:
        tag_name = ''
    return tag_id, tag_name_length, tag_name

def parse_tag_type(f):
    """Parse a tag type of the given DM3 file f.
    Returns the tuple infoArray.
    """
    str_infoarray = (18, 4)
    delim = read_string(f, str_infoarray, 'big')
    delimiter = reduce(lambda x, y: x + y, delim)
    if delimiter != '%%%%':
        print('Wrong delimiter: "%s".' % str(delimiter))
        print('File address:', f.tell())
        raise DM3TagTypeError(delimiter)
    else:        
        return read_infoarray(f)
       
def parse_tag_data(f, iarray, endian, skip=0):
    """Parse the data of the given DM3 file f
    with the given endianness (byte order).
    The infoArray iarray specifies how to read the data.
    Returns the tuple (file address, data).
    The tag data is stored in the platform's byte order:
    'little' endian for Intel, PC; 'big' endian for Mac, Motorola.
    If skip != 0 the data is actually skipped.
    """
    faddress = f.tell()
    # nbytes = _infoarray_databytes(iarray)
    if not skip:
        read_data = _data_type[iarray[0]][0]
        if iarray[0] in _complex_type:
            data = read_data(f, iarray, endian)
        elif iarray[0] in _simple_type:
            data = read_data(f, endian)
        else:
            raise DM3DataTypeError(iarray[0])
        if isinstance(data, str):
            data = hyperspy.misc.utils.ensure_unicode(data)
    else:
        data = '__skipped__'        
        # print('Skipping', nbytes, 'Bytes.')
        nbytes = _infoarray_databytes(iarray)
        f.seek(nbytes, 1)
    # return faddress, nbytes, data
    return faddress, data

def parse_image_data(f, iarray):
    """Returns a tuple with the file offset and the number
    of bytes corresponding to the image data:
    (offset, bytes)
    """
    faddress = f.tell()
    nbytes = _infoarray_databytes(iarray)
    f.seek(nbytes, 1)        
    return faddress, nbytes

def parse_header(f, data_dict, endian='big', debug=0):
    """Parse the header (first 12 Bytes) of the given DM3 file f.
    The relevant information is saved in the dictionary data_dict.
    Optionally, a debug level !=0 may be specified.
    endian can be either 'little' or 'big'.
    Returns the boolean is_little_endian (byte order) of the DM3 file.
    """
    iarray = (3, ) # long
    dm_version = parse_tag_data(f, iarray, endian)
    if dm_version[1] != 3:
        print('File address:', dm_version[1])
        raise DM3FileVersionError(dm_version[1])
    data_dict['DM3.Version'] = dm_version

    filesizeB = parse_tag_data(f, iarray, endian)
    filesizeB = list(filesizeB)
    filesizeB[1] = filesizeB[1] + 16
    filesizeB = tuple(filesizeB)
    data_dict['DM3.FileSize'] = filesizeB

    is_little_endian = parse_tag_data(f, iarray, endian)
    data_dict['DM3.isLittleEndian'] = is_little_endian

    if debug > 0:
        filesizeKB = filesizeB[1] / 2.**10
        # filesizeMB = filesizeB[3] / 2.**20
        print('DM version:', dm_version[1])
        print('size %i B (%.2f KB)' % (filesizeB[1], filesizeKB))
        # print 'size: {0} B ({1:.2f} KB)'.format(filesizeB[1] , filesizeKB)
        print('Is file Little endian?', bool(is_little_endian[1]))
    return bool(is_little_endian[1])

def crawl_dm3(f, data_dict, endian, ntags, group_name='root',
             skip=0, debug=0,depth=1):
    """Recursively scan the ntags TagEntrys in DM3 file f
    with a given endianness (byte order) looking for
    TagTypes (data) or TagGroups (groups).
    endian can be either 'little' or 'big'.
    The dictionary data_dict is filled with tags and data.
    Each key is generated in a file system fashion using
    '.' as separator.
    e.g. key = 'root.dir0.dir1.dir2.value0'
    If skip != 0 the data reading is actually skipped.
    If debug > 0, 3, 5, 10 useful debug information is printed on screen.
    """
    depth+=1
    group_name='.'.join(group_name.split('.')[:depth])
    for tag in xrange(ntags):
        if debug > 3 and debug < 10:
            print('Crawling at address:', f.tell())

        tag_id, tag_name_length, tag_name = parse_tag_entry(f)

        if debug > 5 and debug < 10:
            print('Tag name:', tag_name)
            print('Tag ID:', tag_id)
            
        if tag_id == 21: # it's a TagType (DATA)
            if not tag_name:
                tag_name = 'Data0'

            data_key = group_name + '.' + tag_name

            if debug > 3 and debug < 10:
                print('Crawling at address:', f.tell())

            infoarray = parse_tag_type(f)

            if debug > 5 and debug < 10:
                print('Infoarray:', infoarray)

            # Don't overwrite duplicate keys, just rename them
            while data_dict.has_key(data_key):
                data_search = tag_data_pattern.search(data_key)
                try:
                    tag_name = data_search.group()
                    j = int(tag_name.strip('.Data'))
                    data_key = tag_data_pattern.sub('', data_key)
                    if debug > 5 and debug < 10:
                        print('key exists... renaming')
                    tag_name = '.Data' + str(j+1)
                    data_key = data_key + tag_name
                except:
                    if debug >3:
                        print "duplicate key found: %s"%data_key
                        print "renaming failed."
                    break
                
            if image_data_pattern.search(data_key):
                # don't read the data now          
                data_dict[data_key] = parse_image_data(f, infoarray)
            else:
                data_dict[data_key] = parse_tag_data(f, infoarray,
                                                       endian, skip)

            if debug > 10:
                try:
                    if not raw_input('(debug) Press "Enter" to continue\n'):
                        print '######################################'
                        print 'TAG:\n', data_key, '\n'
                        print 'ADDRESS:\n', data_dict[data_key][0], '\n'
                        try:
                            if len(data_dict[data_key][1]) > 10:
                                print 'VALUE:'
                                print data_dict[data_key][1][:10], '[...]\n'
                            else:
                                print 'VALUE:\n', data_dict[data_key][1], '\n'
                        except:
                            print 'VALUE:\n', data_dict[data_key][1], '\n'
                        print '######################################\n'
                except KeyboardInterrupt:
                    print '\n\n###################'
                    print 'Operation canceled.'
                    print '###################\n\n'
                    raise       # exit violently

        elif tag_id == 20: # it's a TagGroup (GROUP)
            if not tag_name:
                tag_name = '.Group0'
                # don't duplicate subgroups, just rename them
                group_search = tag_group_pattern.search(group_name)
                if group_search:
                    tag_name = group_search.group()
                    j = int(tag_name.strip('.Group')) 
                    group_name = tag_group_pattern.sub('', group_name)
                    tag_name = '.Group' + str(j+1)
                    group_name = group_name + tag_name
                else:
                    group_name = group_name + tag_name
            else:
                orsay_search = orsay_pattern.search(group_name)
                if orsay_search: # move orsay-specific dir in the ImageTags dir
                    o = 'Orsay' + orsay_search.group()
                    r = document_tags_pattern.search(group_name).group()
                    group_name = r + o
                
                micinfo_search = micinfo_pattern.search(group_name)
                if micinfo_search: # move Microscope Info in the ImageTags dir
                    m = micinfo_search.group()[1:]
                    r = image_tags_pattern.search(group_name)
                    if r is not None:
                        r=r.group()
                    else:
                        r='DM3.DocumentObjectList.DocumentTags.Group1.ImageData.ImageTags'
                    group_name = r + m
                
                group_name += '.' + tag_name
            if debug > 3 and debug < 10:
                print('Crawling at address:', f.tell())
            ntags = parse_tag_group(f)[2]
            crawl_dm3(f, data_dict, endian, ntags, group_name,
                      skip, debug, depth) # recursion
        else:
            print('File address:', f.tell())
            raise DM3TagIDError(tag_id)

def open_dm3(fname, skip=0, debug=0, log=''):
    """Open a DM3 file given its name and return the dictionary data_dict
    containint the parsed information.
    If skip != 0 the data is actually skipped.
    Optionally, a debug value debug > 0 may be given.
    If log='filename' is specified, the keys, file address and
    (part of) the data parsed in data_dict are written in the log file.

    NOTE:
    All fields, except the TagData are stored using the Big-endian
    byte order. The TagData are stored in the platform's
    byte order (e.g. 'big' for Mac, 'little' for PC).
    """
    with open(fname, 'r+b') as dm3file:
        fmap = mmap.mmap(dm3file.fileno(), 0, access=mmap.ACCESS_READ)
        data_dict = {}
        if parse_header(fmap, data_dict, debug=debug):
            fendian = 'little'
        else:
            fendian = 'big'
        rntags = parse_tag_group(fmap)[2]
        if debug > 3:
            print('Total tags in root group:', rntags)
        rname = 'DM3'
        crawl_dm3(fmap, data_dict, fendian, rntags, group_name=rname,
                  skip=skip, debug=debug)
#         if platform.system() in ('Linux', 'Unix'):
#             try:
#                 fmap.flush()
#             except:
#                 print("Error. Could not write to file", fname)
#         if platform.system() in ('Windows', 'Microsoft'):
#             if fmap.flush() == 0:
#                 print("Error. Could not write to file", fname)
        fmap.close()

        if log:
            exists = overwrite(log)
            if exists:
                with open(log, 'w') as logfile:
                    for key in data_dict:
                        try:
                            line = '%s    %s    %s' % (key, data_dict[key][0],
                                                       data_dict[key][1][:10])
                        except:
                            try:
                                line = '%s    %s    %s' % (key,
                                                           data_dict[key][0],
                                                           data_dict[key][1])
                            except:
                                line = '%s    %s    %s' % (key,
                                                           data_dict[key][0],
                                                           data_dict[key][1])
                        print >> logfile, line, '\n'
                print('Logfile %s saved in current directory' & log)
                
        # Convert data_dict into a file system-like dictionary, datadict_fs
        datadict_fs = {}
        for nodes in data_dict.items():
            fsdict(nodes[0].split('.'), nodes[1], datadict_fs)
        fsbrowser =  DictBrowser(datadict_fs) # browsable dictionary'
        return fsbrowser
   
class DM3ImageFile(object):
    """ Class to handle Gatan Digital Micrograph (TM) files.
    """

    format = 'dm3'
    format_description = 'Gatan Digital Micrograph (TM) Version 3'

    # Image data types (Image Object chapter on DM help)#
    # key = DM data type code
    # value = numpy data type
    imdtype_dict = {
        0 : 'not_implemented', # null
        1 : 'int16',
        2 : 'float32',
        3 : 'complex64',
        4 : 'not_implemented', # obsolete
        5 : 'complex64_packed', # not numpy: 8-Byte packed complex (FFT data)
        6 : 'uint8',
        7 : 'int32',
        8 : 'argb', # not numpy: 4-Byte RGB (alpha, R, G, B)
        9 : 'int8',
        10 : 'uint16',
        11 : 'uint32',
        12 : 'float64',
        13 : 'complex128',
        14 : 'bool',
        23 : 'rgb', # not numpy: 4-Byte RGB (0, R, G, B)
        }
    
    rootdir = ['DM3']
    endian = rootdir + ['isLittleEndian',]
    version = rootdir + ['Version',]
    micinfodir = rootdir + ['Microscope Info',]
    rootdir = rootdir + ['DocumentObjectList',] # DocumentTags, Group0..
    imlistdir = rootdir + ['DocumentTags']
    # imlistdir contains ImageSourceList, Group0, Group1, ... Group[N]
    # "GroupX" dirs contain the useful info in subdirs
    # Group0 is always THUMBNAIL (?)
    # imdisplaydir = ['AnnotationGroupList', 'ImageDisplayInfo']
    # clutname = imdisplaydir + ['CLUTName',] # Greyscale, Rainbow or Temperature
    imdatadir = imlistdir+['Group1','ImageData',]
    imtagsdir = imdatadir + ['ImageTags',]
    imname = imtagsdir + ['Name',]
    Gatan_SI_dir = imtagsdir + ['Acquisition','DataBar','DigiScan',]
    Gatan_EELS_SI_dir=Gatan_SI_dir + ['EELS']
    # TODO: EFTEM not tested! MCS May 2011
    Gatan_EFTEM_SI_dir=Gatan_SI_dir + ['EFTEM']
    orsaydir = imtagsdir + ['Orsay', 'spim', 'detectors', 'eels']
    vsm = orsaydir + ['vsm',]
    dwelltime = orsaydir + ['dwell time',]
    orsaymicdir = orsaydir + ['microscope',]
    calibdir = imdatadir + ['Calibrations',] # ['DataType', 'Data',
                                             # 'Dimensions', 'Brightness']
    im = calibdir + ['Data',]     # file addres and size of image
    imdtype = calibdir + ['DataType',] # data type to compare with imdtype_dict
    brightdir = calibdir + ['Brightness',]

    pixdepth = calibdir + ['Dimensions','PixelDepth', ]

    units = ['Units',]          # in brightdir + 'Group[X]
    origin = ['Origin',]        # in brightdir + 'Group[X]
    scale = ['Scale',]          # in brightdir + 'Group[X]

    def __init__(self, fname, data_id=1, order = None, SI = None, 
                 record_by = None, output_level=1):
        self.filename = fname
        self.info = '' # should be a dictionary with the microscope info
        self.mode = ''
        self.record_by = record_by
        self.output_level=output_level
        self.order = order
        self.SI = SI
        if data_id < 0:
            raise ImageIDError(data_id)
        else:
            self.data_id = data_id
        self.open()

    def __repr__(self):
        message = 'Instance of ' + repr(self.__class__)
        message += '\n' + self.mode + ' ' + str(self.imsize)
        return message

    def open(self):        
        self.data_dict = open_dm3(self.filename)
        byte_order = self.data_dict.ls(DM3ImageFile.endian)[1][1]
        if byte_order == 1:
            self.byte_order = 'little'
        elif byte_order == 0:
            self.byte_order = 'big'
        else:
            raise ByteOrderError, byte_order
        self.endian = self.byte_order
                    
        self.data_dict.cd(DM3ImageFile.imlistdir) # enter ImageList

        image_id = [im for im in self.data_dict.ls() if ('Group' in im
                                                         and im != 'Group0')]
        #Group0 is THUMBNAIL and GroupX (X !=0) is IMAGE
        image_id.sort()

        if len(image_id) > 1 or self.data_id == 0 and self.output_level>1:
            print 'File "%s" contains %i images:' % (self.filename,
                                                     len(image_id))
            print
            print 'ID  | Name'
            print '    |     '
            for i in image_id:
                imid, imname = (image_id.index(i) + 1,
                        self.data_dict.ls([i,] + DM3ImageFile.imname)[1][1])
                print ' ', imid, '|', imname
            print '_____________'

            if self.data_id == 0:
                print 'Image ID "%i" is not valid.' % self.data_id
                print 'Please specify a valid image ID'
                return None
        try:
            im = image_id[self.data_id - 1]
            name = self.data_dict.ls([im,] + DM3ImageFile.imname)
            if name:
                self.name = self.data_dict.ls([im,] + DM3ImageFile.imname)[1][1]
            else:
                self.name = ""
            #~ if self.output_level>1:
                #~ print 'Loading image "%s" (ID: %i) from file %s'% (self.name,
                                                               #~ self.data_id,
                                                               #~ self.filename)
        except IndexError:
            raise ImageIDError(self.data_id)

        self.data_dict.cd(image_id[self.data_id - 1]) # enter Group[ID]

        try:
            self.exposure =  self.data_dict.ls(DM3ImageFile.dwelltime)[1][1]
        except:
            self.exposure = None
        try:
            self.vsm =  self.data_dict.ls(DM3ImageFile.vsm)[1][1]
        except:
            self.vsm = None
            
        self.old_code_tags = parseDM3(self.filename)
        self.SI_format = None
        self.signal = ""
        for tag, value in self.old_code_tags.iteritems():
            if 'Format' in tag and 'Spectrum image' in str(value):
                self.SI_format = value
            if 'Signal' in tag and 'EELS' in str(value):
                self.signal = value
                
#        try:
#            self.SI_format = self.data_dict.ls(DM3ImageFile.Gatan_EELS_SI_dir+['Meta Data','Format'])[1][1]
#        except:
#            try:
#                self.SI_format = self.data_dict.ls(DM3ImageFile.Gatan_EFTEM_SI_dir+['Meta Data','Format'])[1][1]
#            except:
#                try:
#                    # Fall-back for images that have been manually converted to SIs
#                    self.SI_format = self.data_dict.ls(DM3ImageFile.imtagsdir+['Meta Data','Format'])[1][1]
#                except:
#                    self.SI_format = None

#        try:
#            self.signal = self.data_dict.ls(DM3ImageFile.Gatan_EELS_SI_dir+['Meta Data','Signal'])[1][1]
#        except:
#            try:
#                self.signal=self.data_dict.ls(DM3ImageFile.Gatan_EFTEM_SI_dir+['Meta Data','Format'])[1][1]
#            except:
#                try:
#                    # Fall-back for images that have been manually converted to SIs
#                    self.signal = self.data_dict.ls(DM3ImageFile.imtagsdir+['Meta Data','Signal'])[1]
#                except:
#                    self.signal = None

        self.data_dict.cd()
        imdtype =  self.data_dict.ls(DM3ImageFile.imdtype)[1][1]
        self.imdtype = DM3ImageFile.imdtype_dict[imdtype]

        self.byte_offset = self.data_dict.ls(DM3ImageFile.im)[1][0]

        self.imbytes = self.data_dict.ls(DM3ImageFile.im)[1][1]

        self.pixel_depth =  self.data_dict.ls(DM3ImageFile.pixdepth)[1][1]

        sizes = []
        for i in self.data_dict.ls(DM3ImageFile.calibdir):
            if 'Data' in i:
                try:
                    int(i[-1])
                    sizes.append((i,
                              self.data_dict.ls(DM3ImageFile.calibdir
                                                + [i,])))
                except:
                    pass
        sizes.sort()
#        swapelem(sizes, 0, 1)

        origins = []
        for i in self.data_dict.ls(DM3ImageFile.brightdir):
            if 'Group' in i:
                origins.append((i,
                                self.data_dict.ls(DM3ImageFile.brightdir
                                                  + [i,]
                                                  + DM3ImageFile.origin)))
        origins.sort()
#        swapelem(origins, 0, 1)
        
        scales = []
        for i in self.data_dict.ls(DM3ImageFile.brightdir):
            if 'Group' in i:
                scales.append((i,
                               self.data_dict.ls(DM3ImageFile.brightdir
                                                    + [i,]
                                                    + DM3ImageFile.scale)))
        scales.sort()
#        swapelem(scales, 0, 1)

        units = []
        for i in self.data_dict.ls(DM3ImageFile.brightdir):
            if 'Group' in i:
                units.append((i,
                              self.data_dict.ls(DM3ImageFile.brightdir
                                                + [i,]
                                                + DM3ImageFile.units)))
        units.sort()
#        swapelem(units, 0, 1)
        
    
        # Determine the dimensions of the data
        self.dim = 0
        for i in xrange(len(sizes)):
            if sizes[i][1][1][1] > 1:
                self.dim += 1
            if units[i][1][1][1] in ('eV', 'keV'):
                eV_in = True
            else:
                eV_in = False
        
        # Try to guess the order if not given
        # (there must be a better way!!)
        if self.order is None:
            if self.SI_format == 'Spectrum image':
                self.order = 'F'
            else:
                self.order = 'C'
        # Try to guess the record_by if not given
        # (there must be a better way!!)        
        if self.record_by is None:
            if (self.dim > 1 and eV_in) or self.dim == 1 or \
            self.signal == 'EELS' or self.SI_format == 'Spectrum image':
                self.record_by = 'spectrum'
            else:
                self.record_by = 'image'
                
        names = ['X', 'Y', 'Z'] if self.record_by == 'image' \
        else ['X', 'Y', 'Energy']
        to_swap = [sizes, origins, scales, units, names]       
        for l in to_swap:
            if self.record_by == 'spectrum':
                swapelem(l,0,1)
            elif self.record_by == 'image':
                l.reverse()
            
        dimensions = [(
                sizes[i][1][1][1],
                origins[i][1][1][1],
                scales[i][1][1][1],
                units[i][1][1][1],
                names[i])
               for i in xrange(len(sizes))]
        # create a structured array:
        # self.dimensions['sizes'] -> integer
        # self.dimensions['origins'] -> float
        # self.dimensions['scales'] -> float
        # self.dimensions['units'] -> string
        self.dimensions = np.asarray(dimensions,
                                     dtype={'names':['sizes',
                                                     'origins',
                                                     'scales',
                                                     'units',
                                                     'names',],
                                            'formats':['i8',
                                                       'f4',
                                                       'f4',
                                                       'U8',
                                                       'U8',]})
        self.imsize = self.dimensions['sizes']
        self.units = self.dimensions['units']
       
        br_orig = self.data_dict.ls(DM3ImageFile.brightdir
                                    + DM3ImageFile.origin)[1][1]
        br_scale = self.data_dict.ls(DM3ImageFile.brightdir
                                     + DM3ImageFile.scale)[1][1]
        br_units = self.data_dict.ls(DM3ImageFile.brightdir
                                     + DM3ImageFile.units)[1][1]
        self.brightness = np.array((br_orig, br_scale, br_units))

        # self.data = self.read_image_data()
#        try:
        self.data = self.read_image_data()

#        except AttributeError:
#            print('Error. Could not read data.')
#            self.data = 'UNAVAILABLE'
#            return None
        
        # remove axes whose dimension is 1, they are useless:
        while 1 in self.data.shape:
            i = self.data.shape.index(1)
            self.dimensions = np.delete(self.dimensions, i)
            self.imsize = np.delete(self.imsize, i)
            self.data = self.data.squeeze()

        d = len(self.dimensions)
        if d == 0: # could also implement a 'mode' dictionary...
            raise ImageModeError(d)
        else:
            self.mode += str(d) + 'D'

    def read_image_data(self):
        if self.imdtype == 'not_implemented':
            raise AttributeError, "image data type: %s" % self.imdtype
        if ('packed' in self.imdtype):
            return  self.read_packed_complex()
        elif ('rgb' in self.imdtype):
            return self.read_rgb()
        else:
            data = read_data_array(self.filename, self.imbytes,
                                   self.byte_offset, self.imdtype)
            imsize = self.imsize.tolist()
            if self.order == 'F':
                if self.record_by == 'spectrum':
                    swapelem(imsize, 0, 1)
                    data = data.reshape(imsize, order = self.order)
                    data = np.swapaxes(data, 0, 1).copy()
                elif self.record_by == 'image':
                    data = data.reshape(imsize, order = 'C')
            elif self.order == 'C':
                if self.record_by == 'spectrum':
                    data = data.reshape(np.roll(self.imsize,1), order = self.order)
                    data = np.rollaxis(data, 0, self.dim).copy()
                elif self.record_by == 'image':
                    data = data.reshape(self.imsize, order = self.order)                    
            return data
            
    def read_rgb(self):
        self.imsize = list(self.imsize)
        self.imsize.append(4)
        self.imsize = tuple(self.imsize)
        data = read_data_array(self.filename, self.imbytes,
                               self.byte_offset)
        data = data.reshape(self.imsize, order='C') # (B, G, R, A)
        if self.imdtype == 'rgb':
            data = data[:, :, -2::-1] # (R, G, B)
            self.mode += 'rgb_'
            self.imsize = list(self.imsize)
            self.imsize[-1] = self.imsize[-1] - 1
            self.imsize = tuple(self.imsize)
        elif self.imdtype == 'argb':
            data = np.concatenate((data[:, :, -2::-1], data[:, :, -1:]),
                                  axis=2) # (R, G, B, A)
            self.mode += 'rgba_'
        return data

    def read_packed_complex(self):
        if (self.imsize[0] != self.imsize[1]) or (len(self.imsize)>2):
            msg = "Packed complex format works only for a 2Nx2N image"
            msg += " -> width == height"
            print msg
            raise ImageModeError('FFT')
        # print "This image is likely a FFT and each pixel is a complex number"
        # print "You might want to display its complex norm"
        # print "with a logarithmic intensity scale: log(abs(IMAGE))"
        self.mode += 'FFT_'
        N = int(self.imsize[0] / 2)      # think about a 2Nx2N matrix
        # read all the bytes as 1D array of 4-Byte float
        tmpdata = read_data_array(self.filename, self.imbytes,
                                   self.byte_offset, 'float32')
        
        # create an empty 2Nx2N ndarray of complex
        data = np.zeros(self.imsize, 'complex64', 'C')
        
        # fill in the real values:
        data[N, 0] = tmpdata[0]
        data[0, 0] = tmpdata[1]
        data[N, N] = tmpdata[2*N**2] # Nyquist frequency
        data[0, N] = tmpdata[2*N**2+1] # Nyquist frequency
                
        # fill in the non-redundant complex values:
        # top right quarter, except 1st column
        for i in xrange(N): # this could be optimized
            start = 2 * i * N + 2
            stop = start + 2 * (N - 1) - 1
            step = 2
            realpart = tmpdata[start:stop:step]
            imagpart = tmpdata[start+1:stop+1:step]
            data[i, N+1:2*N] = realpart + imagpart * 1j
        # 1st column, bottom left quarter
        start = 2 * N
        stop = start + 2 * N * (N - 1) - 1
        step = 2 * N
        realpart = tmpdata[start:stop:step]
        imagpart = tmpdata[start+1:stop+1:step]
        data[N+1:2*N, 0] = realpart + imagpart * 1j
        # 1st row, bottom right quarter
        start = 2 * N**2 + 2
        stop = start + 2 * (N - 1) - 1
        step = 2
        realpart = tmpdata[start:stop:step]
        imagpart = tmpdata[start+1:stop+1:step]
        data[N, N+1:2*N] = realpart + imagpart * 1j
        # bottom right quarter, except 1st row
        start = stop + 1
        stop = start + 2 * N * (N - 1) - 1
        step = 2
        realpart = tmpdata[start:stop:step]
        imagpart = tmpdata[start+1:stop+1:step]
        complexdata = realpart + imagpart * 1j
        data[N+1:2*N, N:2*N] = complexdata.reshape(N-1, N)

        # fill in the empty pixels: A(i)(j) = A(2N-i)(2N-j)*
        # 1st row, top left quarter, except 1st element
        data[0, 1:N] = np.conjugate(data[0, -1:-N:-1])
        # 1st row, bottom left quarter, except 1st element
        data[N, 1:N] = np.conjugate(data[N, -1:-N:-1])
        # 1st column, top left quarter, except 1st element
        data[1:N, 0] = np.conjugate(data[-1:-N:-1, 0])
        # 1st column, top right quarter, except 1st element
        data[1:N, N] = np.conjugate(data[-1:-N:-1, N])
        # top left quarter, except 1st row and 1st column
        data[1:N, 1:N] = np.conjugate(data[-1:-N:-1, -1:-N:-1])
        # bottom left quarter, except 1st row and 1st column
        data[N+1:2*N, 1:N] = np.conjugate(data[-N-1:-2*N:-1, -1:-N:-1])

        return data

def file_reader(filename, record_by=None, order = None, data_id=1, 
                dump = False, output_level=1):
    """Reads a DM3 file and loads the data into the appropriate class.
    data_id can be specified to load a given image within a DM3 file that
    contains more than one dataset.
    
    Parameters
    ----------
    record_by: Str
        One of: SI, Image
    order: Str
        One of 'C' or 'F'
    dump: Bool
        If True it dumps the tags into a txt file
    """
         
    dm3 = DM3ImageFile(filename, data_id, order = order, record_by = record_by,
                       output_level=output_level)
    
    if dump is True:
        import codecs
        f = codecs.open(filename.replace('.dm3', '_tags_dumped.txt'), 'w')
        hyperspy.misc.utils.dump_dictionary(f, dm3.data_dict.dic)
        f.close()
    mapped_parameters={}

    if 'rgb' in dm3.mode:
        # create a structured array (see issue #25)
        buf = np.zeros(dm3.data.shape[:-1],
                       dtype={'names' : ['R', 'G', 'B'],
                              'formats' : ['u4', 'u4', 'u4']})
        buf['R'] = dm3.data[..., 0]
        buf['G'] = dm3.data[..., 1]
        buf['B'] = dm3.data[..., 2]
        dm3.data = buf

    if dm3.name:
        mapped_parameters['title'] = dm3.name
    else:
        mapped_parameters['title'] = ''

    data = dm3.data
    # set dm3 object's data attribute to None so it doesn't
    #   get passed in the original_parameters dict (to save memory)
    dm3.data = None

    # Determine the dimensions
    names = list(dm3.dimensions['names'])
    units = list(dm3.dimensions['units'])
    origins = np.asarray([dm3.dimensions[i][1]
                          for i in xrange(len(dm3.dimensions))],
                         dtype=np.float)
    scales =np.asarray([dm3.dimensions[i][2]
                        for i in xrange(len(dm3.dimensions))],
                       dtype=np.float)
    # The units must be strings
    while None in units: 
        units[units.index(None)] = ''
    # Scale the origins
    origins = origins * scales
    if dm3.record_by == 'spectrum': 
        if dm3.exposure:
            mapped_parameters['exposure'] = dm3.exposure            
        if dm3.vsm:
            if 'EELS' not in mapped_parameters:
                mapped_parameters['EELS'] = {}
            mapped_parameters['EELS']['vsm'] = dm3.vsm
            mapped_parameters['vsm'] = float(dm3.vsm)

        if 'eV' in units: # could use reexp or match to a 'energy units' dict?
            energy_index = units.index('eV')
        elif 'keV' in units:
            energy_index = units.index('keV')
        else:
            energy_index = -1

        # In DM the origin is negative. Change it to positive
        origins[energy_index] *= -1

        # Store the calibration in the calibration dict

    dim = len(data.shape)
    axes=[{'size' : int(data.shape[i]), 
           'index_in_array' : i ,
           'name' : unicode(names[i]),
           'scale': scales[i],
           'offset' : origins[i],
           'units' : unicode(units[i]),} \
           for i in xrange(dim)]

    mapped_parameters['original_filename'] = os.path.split(filename)[1]
    mapped_parameters['record_by'] = dm3.record_by
    mapped_parameters['signal_type'] = dm3.signal
    original_parameters = {}
    for tag in dm3.old_code_tags.items():
        node_valve(tag[0].split('.'), tag[1], original_parameters)
    dictionary = {
        'data' : data,
        'axes' : axes,
        'mapped_parameters': mapped_parameters,
        'original_parameters':original_parameters,
        }
    
    return [dictionary, ]
