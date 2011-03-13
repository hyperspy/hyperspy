# -*- coding: utf-8 -*-
# Copyright © 2007 Francisco Javier de la Peña
#
# This file is part of EELSLab.
#
# EELSLab is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# EELSLab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EELSLab; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  
# USA


#  for more information on the RPL/RAW format, see
#  http://www.nist.gov/lispix/
#  and
#  http://www.nist.gov/lispix/doc/image-file-formats/raw-file-format.htm

import numpy as np
import os
from ..utils_readfile import *

# Plugin characteristics
# ----------------------
format_name = 'Ripple'
description = 'RPL file contains the information on how to read the RAW Bytes'
full_suport = False             #  but maybe True
# Recognised file extension
file_extensions = ['rpl','RPL']
default_extension = 0
# Reading capabilities
reads_images = True
reads_spectrum = False          # but maybe True
reads_spectrum_image = True
# Writing capabilities
writes_images = True            # but maybe True
writes_spectrum = False
writes_spectrum_image = True
# ----------------------

newline = ('\n', '\r\n')
comment = ';'
sep = '\t'

rpl_keys = {
    # spectrum/image information:
    'width' : int,
    'height' : int,
    'depth' : int,
    'offset': int ,
    'data-length' : ['1', '2', '4', '8'],
    'data-type' : ['signed', 'unsigned', 'float'],
    'byte-order' : ['little-endian', 'big-endian', 'dont-care'],
    'record-by' : ['image', 'vector', 'dont-care'],
    # X-ray information:
    'ev-per-chan' : float,    # usually 5 or 10 eV
    'detector-peak-width-ev' : float, # usually 150 eV
    }

def parse_ripple(fp):
    """Parse information from ripple (.rpl) file.
    Accepts file object 'fp. Returns dictionary rpl_info.
    """
    rpl_info = {}
    first_non_comment = False
    for line in fp.readlines():
        line = line.replace(' ', '')
        if line[:2] not in newline and line[0] != comment:
            line = line.strip('\r\n')
            line = line.lower()
            if comment in line:
                line = line[:line.find(comment)]
            if not sep in line:
                err = 'Separator in line "%s" is wrong, ' % line
                err += 'it should be a <TAB> ("\\t")'
                raise IOError, err
            line = line.split(sep) # now it's a list
            if not rpl_keys.has_key(line[0]):
                first_non_comment = True
            else:
                if not first_non_comment:
                    err = 'The first non-comment line MUST have two column'
                    err += 'names of type "name1<TAB>name2" '
                    err += '(any name would do, e.g. key<TAB>value).'
                    raise IOError, err
                try:
                    line[1] = rpl_keys[line[0]](line[1])
                except TypeError:
                    if not line[1] in rpl_keys[line[0]]:
                        err = 'Wrong value for key %s.\n' % line[0]
                        err += 'Value read is %s' % line[1]
                        err += ' but it should be one of', rpl_keys[line[0]]
                        raise IOError, err
                rpl_info[line[0]] = line[1]

    if rpl_info['depth'] == 1 and rpl_info['record-by'] != 'dont-care':
        err = '"depth" and "record-by" keys mismatch.\n'
        err += '"depth" cannot be "1" if "record-by" is "dont-care" '
        err += 'and vice versa.'
        err += 'Check %s' % fp.name
        raise IOError, err
    if rpl_info['data-type'] == 'float' and int(rpl_info['data-length']) < 4:
        err = '"data-length" for float "data-type" must be "4" or "8".\n'
        err += 'Check %s' % fp.name
        raise IOError, err
    if rpl_info['data-length'] == '1' and rpl_info['byte-order'] != 'dont-care':
        err = '"data-length" and "byte-order" mismatch.\n'
        err += '"data-length" cannot be "1" if "byte-order" is "dont-care" '
        err += 'and vice versa.'
        err += 'Check %s' % fp.name
        raise IOError, err
    return rpl_info
        
def read_raw(rpl_info, fp):
    """Read the raw file object 'fp' based on the information given in the
    'rpl_info' dictionary.
    """
    width = rpl_info['width']
    height = rpl_info['height']
    depth = rpl_info['depth']
    offset = rpl_info['offset']
    data_length = rpl_info['data-length']
    data_type = rpl_info['data-type']
    endian = rpl_info['byte-order']
    record_by = rpl_info['record-by']
    
    # if rpl_info.has_key('ev-per-chan'):
    #     ev_per_chan = rpl_info['ev-per-chan']
    # if rpl_info.has_key('detector-peak-width-ev'):
    #     det_fwhm = rpl_info['detector-peak-width-ev']

    if data_type == 'signed':
        data_type = 'int'
    elif data_type == 'unsigned':
        data_type = 'uint'
    elif data_type == 'float':
        pass
    else:
        raise TypeError, 'Unknown "data-type" string.'

    if endian == 'big-endian':
        endian = '>'
    elif endian == 'little-endian':
        endian = '<'
    else:
        endian = '='

    data_type = data_type + str(int(data_length) * 8)
    data_type = np.dtype(data_type)
    data_type = data_type.newbyteorder(endian)

    data = read_data_array(fp,
                           byte_address=offset,
                           data_type=data_type)

    if record_by == 'vector':   # spectral image
        size = (width, height, depth)
        # old EELSLab; energy first
        data = data.reshape(size)
        data = np.rollaxis(data, 2, 0)
    elif record_by == 'image':  # stack of images
        size = (depth, width, height)
        data = data.reshape(size)

    return data

def file_reader(filename, rpl_info=None, *args, **kwds):
    """Parses a Lispix (http://www.nist.gov/lispix/) ripple (.rpl) file
    and reads the data from the corresponding raw (.raw) file;    
    or, read a raw file if the dictionary rpl_info is provided.
    
    Images and spectral images or data cubes that are written in the
    (Lispix) raw file format are just a continuous string of numbers.

    Data cubes can be stored image by image, or spectrum by spectrum.
    Single images are stored row by row, vector cubes are stored row by row
    (each row spectrum by spectrum), image cubes are stored image by image.

    All of the numbers are in the same format, such as 16 bit signed integer,
    IEEE 8-byte real, 8-bit unsigned byte, etc.
    
    The "raw" file should be accompanied by text file with the same name and
    ".rpl" extension. This file lists the characteristics of the raw file so
    that it can be loaded without human intervention.

    Alternatively, dictionary 'rpl_info' containing the information can
    be given.
    
    RPL stands for "Raw Parameter List", an ASCII text, tab delimited file in
    which Lispix records or reads the image parameters for a raw file.

                    PARAMETERS
        key             type     description            
      ----------   ------------ ---------------------- 
      width        int           # pixels per row 
      height       int           # number of rows
      depth        int           # number of images or spectral pts
      offset       int           # bytes to skip
      data-type    str           # 'signed', 'unsigned', or 'float' 
      data-length  str           # bytes per pixel  '1', '2', '4', or '8'
      byte-order   str           # 'big-endian', 'little-endian', or 'dont-care'
      record-by    str           # 'image', 'vector', or 'dont-care'
      ev-per-chan  int           # optional, eV per channel
      detector-peak-width-ev int # optional, FWHM for the Mn K-alpha line

    NOTES

    When 'data-length' is 1, the 'byte order' is not relevant as there is only
    one byte per datum, and 'byte-order' should be 'dont-care'.
    
    When 'depth' is 1, the file has one image, 'record-by' is not relevant and
    should be 'dont-care'. For spectral images, 'record-by' is 'vector'.
    For stacks of images, 'record-by' is 'image'.
    
    Floating point numbers can be IEEE 4-byte, or IEEE 8-byte. Therefore if
    data-type is float, data-length MUST be 4 or 8.
    
    The rpl file is read in a case-insensitive manner. However, when providing
    a dictionary as input, the keys MUST be lowercase.
    
    Comment lines, beginning with a semi-colon ';' are allowed anywhere.

    The first non-comment in the rpl file line MUST have two column names:
    'name_1'<TAB>'name_2'; any name would do e.g. 'key'<TAB>'value'.

    Parameters can be in ANY order.
    
    In the rpl file, the parameter name is followed by ONE tab (spaces are
    ignored) e.g.: 'data-length'<TAB>'2'
    
    In the rpl file, other data and more tabs can follow the two items on
    each row, and are ignored.
    
    Other keys and values can be included and are ignored.

    Any number of spaces can go along with each tab.
    """
    if not rpl_info:
        with open(filename) as f:
            rpl_info = parse_ripple(f)
    rawfname = ''
    for ext in ['raw', 'RAW']:
        rawfname = filename[:-3] + ext
        print rawfname
        if os.path.exists(rawfname):
            print 'EXISTS'
            break
        else:
            rawfname = ''
    if not rawfname:
        raise IOError, 'RAW file does not exists'
    else:
        data_cube = read_raw(rpl_info, rawfname)

    if rpl_info['record-by'] in ['vector', 'image']: # CHECK THIS
        data_type = 'SI'
    else:
        data_type = 'Image'

    calibration_dict = {
        'title' : filename,
        'data_cube' : data_cube,
        'energyorigin' : 0,
        'energyscale' : 1,
        'energyunits' : 'eV',    # arbitrary
        'xorigin' : 0,
        'xscale' : 1,
        'yorigin' : 0,
        'yscale' : 1,
        }

    acquisition_dict = {}

    dictionary = {
        'data_type' : data_type, 
        'calibration' : calibration_dict, 
        'acquisition' : acquisition_dict,
        'imported_parameters' : calibration_dict}
    
    return [dictionary, ]

    
        




