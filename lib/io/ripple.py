# -*- coding: utf-8 -*-
# Copyright © 2007 Francisco Javier de la Peña
# Copyright © 2010 Francisco Javier de la Peña & Stefano Mazzucco
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
import os.path
from ..utils_readfile import *
from .. import Release

# Plugin characteristics
# ----------------------
format_name = 'Ripple'
description = 'RPL file contains the information on how to read\n'
description += 'the RAW file with the same name.'
description += '\nThis format may not provide information on the calibration.'
description += '\nIf so, you should add that after loading the file.'
full_suport = False             #  but maybe True
# Recognised file extension
file_extensions = ['rpl','RPL']
default_extension = 0
# Reading capabilities
reads_images = True
reads_spectrum = True          # but maybe True
reads_spectrum_image = True
# Writing capabilities
writes_images = True           # but maybe True
writes_spectrum = True
writes_spectrum_image = True
# ----------------------

# The format only support the followng data types
newline = ('\n', '\r\n')
comment = ';'
sep = '\t'

dtype2keys = {
                'float64' : ('float', 8),
                'float32' : ('float', 4),
                'uint8' : ('unsigned', 1),
                'uint16' : ('unsigned', 2),
                'int32' : ('signed', 4),
                'int64' : ('signed', 8),}

endianess2rpl = {
                    '=' : 'dont-care',
                    '<' : 'little-endian',
                    '>' : 'big-endian'}

rpl_keys = {
    # spectrum/image keys
    'width' : int,
    'height' : int,
    'depth' : int,
    'offset': int ,
    'data-length' : ['1', '2', '4', '8'],
    'data-type' : ['signed', 'unsigned', 'float'],
    'byte-order' : ['little-endian', 'big-endian', 'dont-care'],
    'record-by' : ['image', 'vector', 'dont-care'],
    # X-ray keys
    'ev-per-chan' : float,    # usually 5 or 10 eV
    'detector-peak-width-ev' : float, # usually 150 eV
    # EELSLab-specific keys
    'energy-origin' : int,
    'energy-scale' : float,
    'energy-units' : str,
    'x-origin' : int,
    'x-scale' : float,
    'x-units' : str,
    'y-origin' : int,
    'y-scale' : float,
    'y-units' : str,
    }

def parse_ripple(fp):
    """Parse information from ripple (.rpl) file.
    Accepts file object 'fp. Returns dictionary rpl_info.
    """
    rpl_info = {}
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
            if rpl_keys.has_key(line[0]) is True:
                if hasattr(rpl_keys[line[0]],'__iter__'):
                    if line[1] not in rpl_keys[line[0]]:
                        err = \
                        'Wrong value for key %s.\n' \
                        'Value read is %s'  \
                        ' but it should be one of %s' % \
                        (line[0], line[1], str(rpl_keys[line[0]]))
                        raise IOError, err
                else:
                    # rpl_keys[line[0]] must then be a type
                    line[1] = rpl_keys[line[0]](line[1])
            
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

    if 'signed' in data_type:
        # data cubes of int give misleading results
        convert2float = True
    else:
        convert2float = False

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
    if convert2float:
        data = data.astype(np.float32)

    if record_by == 'vector':   # spectral image
        size = (height, width, depth)
        data = data.reshape(size)
        # old EELSLab; energy first (this will hopefully be changed)
        data = data.T
    elif record_by == 'image':  # stack of images
        size = (depth, height, width)
        data = data.reshape(size).T

    return data

def file_reader(filename, rpl_info=None, *args, **kwds):
    """Parses a Lispix (http://www.nist.gov/lispix/) ripple (.rpl) file
    and reads the data from the corresponding raw (.raw) file;    
    or, read a raw file if the dictionary rpl_info is provided.

    This format is often uses in EDS/EDX experiments.
    
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

    Some keys are specific to EELSLab and will be ignored by other software.
    
    RPL stands for "Raw Parameter List", an ASCII text, tab delimited file in
    which EELSLab reads the image parameters for a raw file.

                    TABLE OF RPL PARAMETERS
        key             type     description            
      ----------   ------------ --------------------
      # Mandatory      keys:
      width            int      # pixels per row 
      height           int      # number of rows
      depth            int      # number of images or spectral pts
      offset           int      # bytes to skip
      data-type        str      # 'signed', 'unsigned', or 'float' 
      data-length      str      # bytes per pixel  '1', '2', '4', or '8'
      byte-order       str      # 'big-endian', 'little-endian', or 'dont-care'
      record-by        str      # 'image', 'vector', or 'dont-care'
      # X-ray keys:
      ev-per-chan      int      # optional, eV per channel
      detector-peak-width-ev  int   # optional, FWHM for the Mn K-alpha line
      # EELSLab-specific keys
      energy-origin    int      # energy offset in pixels          
      energy-scale     float    # energy scaling (units per pixel) 
      energy-units     str      # energy units, usually eV
      x-origin         int      # column offset in pixels
      x-scale          float    # column scaling (units per pixel)
      x-units          str      # column units, usually nm         
      y-origin         int      # row offset in pixels          
      y-scale          float    # row scaling (units per pixel) 
      y-units          str      # row units, usually nm
     
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
        if filename[-3:] in file_extensions:
            with open(filename) as f:
                rpl_info = parse_ripple(f)
        else:
            raise IOError, 'File has wrong extension: "%s"' % filename[-3:]
    for ext in ['raw', 'RAW']:
        rawfname = filename[:-3] + ext
        if os.path.exists(rawfname):
            break
        else:
            rawfname = ''
    if not rawfname:
        raise IOError, 'RAW file "%s" does not exists' % rawfname
    else:
        data_cube = read_raw(rpl_info, rawfname)

    if rpl_info['record-by'] in ['vector', 'image']: # CHECK THIS
        print 'Loading as SI'
        data_type = 'SI'
    else:
        print 'Loading as Image'
        data_type = 'Image'

    if rpl_info.has_key('ev-per-chan'):
        energyscale = rpl_info['ev-per-chan']
    elif rpl_info.has_key('energy-scale'):
        # superseed previous key
        energyscale = rpl_info['energy-scale']
    else:
        energyscale = 1.

    if rpl_info.has_key('detector-peak-width-ev'):
        det_fwhm = rpl_info['detector-peak-width-ev']
    else:
        det_fwhm = None

    if rpl_info.has_key('energy-origin'):
        energyorigin = rpl_info['energy-origin']
    else:
        energyorigin = 0
        
    if rpl_info.has_key('energy-units'):
        energyunits = rpl_info['energy-units']
    else:
        energyunits = ''

    if rpl_info.has_key('x-origin'):
        xorigin = rpl_info['x-origin']
    else:
        xorigin = 0

    if rpl_info.has_key('x-scale'):
        xscale = rpl_info['x-scale']
    else:
        xscale = 1.
        
    if rpl_info.has_key('x-units'):
        xunits = rpl_info['x-units']
    else:
        xunits = ''

    if rpl_info.has_key('y-origin'):
        yorigin = rpl_info['y-origin']
    else:
        yorigin = 0

    if rpl_info.has_key('y-scale'):
        yscale = rpl_info['y-scale']
    else:
        yscale = 1.
        
    if rpl_info.has_key('y-units'):
        yunits = rpl_info['y-units']
    else:
        yunits = ''
        
    calibration_dict = {
        'title' : filename,
        'data_cube' : data_cube,
        'energyorigin' : energyorigin,
        'energyscale' : energyscale,
        'energyunits' : energyunits, 
        'xorigin' : xorigin,
        'xscale' : xscale,
        'xunits' : xunits,
        'yorigin' : yorigin,
        'yscale' : yscale,
        'yunits' : yunits,
        'detector_fwhm' : det_fwhm,
        }

    acquisition_dict = {}

    dictionary = {
        'data_type' : data_type, 
        'calibration' : calibration_dict, 
        'acquisition' : acquisition_dict,
        'imported_parameters' : calibration_dict}
    
    return [dictionary, ]




def file_writer(filename, object2save, *args, **kwds):
    from .. import spectrum
    from .. import image
    
    # Set the optional keys to None    
    ev_per_chan = None
    
    # Check if the dtype is supported
    dc = object2save.data_cube
    dtype_name = object2save.data_cube.dtype.name
    if  dtype_name not in dtype2keys.keys():
        err = 'The ripple format does not support writting data of %s type' % (
        dtype_name)
        raise IOError, err
    # Check if the dimensions are supported
    dim = len(object2save.data_cube.shape)
    if  dim > 3:
        err = 'This file format does not support %i dimension data' % (
        dim)
        raise IOError, err
        
    # Gather the information to write the rpl        
    data_type, data_length = dtype2keys[dc.dtype.name]
    byte_order = endianess2rpl[dc.dtype.byteorder]
    offset = 0
    if isinstance(object2save,spectrum.Spectrum) is True:
        record_by = 'vector'
        ev_per_chan = int(round(object2save.energyscale))
        if dim == 3:
            depth, width, height = dc.shape
        elif dim == 2:
            depth, width, height = list(dc.shape) + [1,]
        elif dim == 1:
            record_by == 'dont-care'
            depth, width, height = list(dc.shape) + [1,1]

    elif isinstance(object2save, image.Image) is True: 
        if dim == 3:
            record_by = 'image'
            width, height, depth = dc.shape
        elif dim == 2:
            record_by == 'dont-care'
            width, height, depth = list(dc.shape) + [1,]
        elif dim == 1:
            record_by == 'dont-care'
            depth, width, height = list(dc.shape) + [1,1]
    else:
        print("Only Spectrum and Image objects can be saved")
        return
            
    # Fill the keys dictionary
    keys_dictionary = {
                         'width' : width,
                         'height' : height,
                         'depth' : depth,
                         'offset' : offset,
                         'data-type' : data_type,
                         'data-length' : data_length,
                         'byte-order' : byte_order,
                         'record-by' : record_by,
                         }
    if ev_per_chan is not None:
        keys_dictionary['ev-per-chan'] = ev_per_chan
        
    write_rpl(filename, keys_dictionary)
    write_raw(filename, dc, record_by)
        
def write_rpl(filename, keys_dictionary):
    f = open(filename, 'w')
    f.write(';File created by EELSLab version %s\n' % Release.version)
    f.write('key\tvalue\n')
    for key, value in keys_dictionary.iteritems():
        f.write(key + '\t' + str(value) + '\n')
    f.close()
    
def write_raw(filename, data_cube, record_by):
    """Writes the raw file object

    Parameters:
    -----------
    filename : string
        the filename, either with the extension or without it
    record_by : string
     'vector' or 'image'
         
        """
    filename = os.path.splitext(filename)[0] + '.raw'
    dshape = data_cube.shape
    if len(dshape) == 3:
        if record_by == 'vector':
            data_cube.T.ravel().tofile(filename)
        elif record_by == 'image':
            data_cube.swapaxes(1,2).ravel().tofile(filename)
    elif len(dshape) == 2:
        if record_by == 'vector':
            data_cube.ravel().tofile(filename)
        elif record_by in ('image' or 'dont-care'):
            data_cube.T.ravel().tofile(filename)
    elif len(dshape) == 1:
        data_cube.ravel().tofile(filename)
        