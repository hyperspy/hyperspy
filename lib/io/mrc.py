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

# The details of the format were taken from http://www.biochem.mpg.de/tom

import numpy as np
from silib.axes import DataAxis

# Plugin characteristics
# ----------------------
format_name = 'MRC'
description = ''
full_suport = False
# Recognised file extension
file_extensions = ['mrc', 'MRC', 'ALI', 'ali']
default_extension = 0
# Reading capabilities
reads_images = True
reads_spectrum = False
reads_spectrum_image = False
# Writing capabilities
writes_images = False
writes_spectrum = False
writes_spectrum_image = False



def get_std_dtype_list(endianess = '<'):
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
    ('CMAP', (str, 4)),
    ('STAMP', (str, 4)),
    ('RMS', end + 'f4'),
    ('NLABL', end + 'u4'),
    ('LABELS', (str, (800))),
    ]
    
    return dtype_list

def get_fei_dtype_list(endianess = '<'):
    end = endianess
    dtype_list = [
    ('a_tilt', end + 'f4'), #  Alpha tilt (deg)
    ('b_tilt', end + 'f4'), #  Beta tilt (deg)
    ('x_stage', end + 'f4'), #  Stage x position (Unit=m. But if value>1, unit=???m)
    ('y_stage', end + 'f4'), #  Stage y position (Unit=m. But if value>1, unit=???m)
    ('z_stage', end + 'f4'), #  Stage z position (Unit=m. But if value>1, unit=???m)
    ('x_shift', end + 'f4'), #  Image shift x (Unit=m. But if value>1, unit=???m)
    ('y_shift', end + 'f4'), #  Image shift y (Unit=m. But if value>1, unit=???m)
    ('defocus', end + 'f4'), #  Defocus Unit=m. But if value>1, unit=???m)
    ('exp_time', end + 'f4'), # Exposure time (s)
    ('mean_int', end + 'f4'), # Mean value of image
    ('tilt_axis', end + 'f4'), #   Tilt axis (deg)
    ('pixel_size', end + 'f4'), # Pixel size of image (m)
    ('magnification', end + 'f4'), #   Magnification used
    ('empty', (np.void, 128 - 13*4)), #   Not used (filling up to 128 bytes)
    ]
    return dtype_list

def get_data_type(index, endianess = '<'):
    end = endianess
    data_type = [
    end + 'u2',         # 0 = Image     unsigned bytes
    end + 'i2',         # 1 = Image     signed short integer (16 bits)
    end + 'f4',         # 2 = Image     float
    (end + 'i2', 2),    # 3 = Complex   short*2
    end + 'c8',         # 4 = Complex   float*2
    ]
    return data_type[index]
                        
def file_reader(filename, endianess = '<', **kwds):
    mapped_parameters={}
    dtype_list = get_std_dtype_list(endianess) + get_fei_dtype_list(endianess)
    f = open(filename, 'rb')
    std_header = np.fromfile(f, dtype = get_std_dtype_list(endianess), 
    count = 1)
    
    scales=np.ones(3)
    units_list=np.array(['undefined']*3)
    names=['x','y','z']

    if std_header['NEXT'] / 1024 == 128:
        print "It seems to contain an extended FEI header"
        fei_header = np.fromfile(f, dtype = get_fei_dtype_list(endianess), 
                                 count = 1024)
        scale[0:2]=fei_header['pixel_size']
        units_list[0:2]='m'
    NX, NY, NZ = std_header['NX'], std_header['NY'], std_header['NZ']
    if f.tell() == 1024 + std_header['NEXT']:
        print "The FEI header was correctly loaded"
    else:
        print "There was a problem reading the extended header"
        f.seek(1024 + std_header['NEXT'])
        
    data = np.memmap(f, mode = 'c', offset = f.tell(), 
                     dtype = get_data_type(std_header['MODE'], endianess)).squeeze().reshape(
        (NX, NY, NZ), order = 'F')
    
                     
    original_parameters = {
        'std_header' : std_header, 
        'fei_header' : fei_header,
        }
    #create the axis objects for each axis
    axes=[DataAxis(data.shape[i],index_in_array=i,name=names[i],scale=scales[i],
                   offset=0, units=units_list[i],slice_bool=) for i in xrange(3)]
    # define the third axis as the slicing axis.
    axes[2].slice_bool=True
    dictionary = {'data_type' : 'Image', 
                  'data':data,
                  'axes':axes,
                  'mapped_parameters' : mapped_parameters,
                  'original_parameters' : original_parameters,
                  }
    
    return [dictionary,]
