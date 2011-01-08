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

import os.path

import numpy as np

from .. import messages

# Plugin characteristics
# ----------------------
format_name = 'Buker raw'
description = 'This format consists of two files with the extensions .raw ' \
'and .rpl. It is the .rpl file (that contains the shape and data type of the ' \
'data) the one that can be opened by eelslab' 

full_suport = False
# Recognised file extension
file_extensions = ['rpl', 'RPL']
default_extension = 0
# Reading capabilities
reads_images = True
reads_spectrum = True
reads_spectrum_image = True
# Writing capabilities
writes_images = False
writes_spectrum = False
writes_spectrum_image = False
# ----------------------

data_types = {
'unsigned1' : 'u1',	
'unsigned2' : 'u2',
'unsigned4' : 'u4',	
'signed1' : 'i1',	# Not tested
'signed2' : 'i2',	# Not tested
'signed4' : 'i4',	# Not tested
'float4' : 'f4',	# Not tested
'float8' : 'f8',      # Not tested
'complex8' : 'c8',    # Not tested	
'complex16' : 'c16',  # Not tested
}

def file_reader(filename, data_type, **kwds):
    # TODO: write the reader...
    print(data_type)
    basename = os.path.splitext(filename)[0]
    raw_file = basename + '.raw'
    if os.path.isfile(raw_file) is False:
        messages.warning_exit('The file %s could not be found' % raw_file)
    
    # Parse the rpl file
    f = open(filename, 'r')
    ints = ['width', 'depth', 'height', 'offset', 'data-length'] 
    parameters = dict()
    for line in f:
        if line:
            key, value = line.split()
            if key in ints:
                parameters[key] = int(value)
            else:
                parameters[key] = value
        if 'key' in parameters:
            del parameters['key']
    print("\nkey:\tValue")
    print("-"*12)
    for item in parameters.items():
        print("%s:\t%s" % item)
    print("\n")
    f.close()
    
    # Read the raw file
    if 'data-length' in parameters and 'data-type' in parameters:
        rdtype = '%s%i' % (parameters['data-type'], parameters['data-length'])
        if rdtype not in data_types:
            messages.warning_exit('The data type was not understood')
    else:
        messages.warning_exit('There is no data information in the rpl file')
    f = open(raw_file, 'r')
    if parameters['offset'] != 0:
        f.seek(parameters['offset'] )
        
        
    dc = np.fromfile(f, dtype = data_types[rdtype],)
    f.close()
    shape = parameters['height'], parameters['width'], parameters['depth']
    dc = dc.reshape(shape).T
    calibration_dict = {'data_cube' : dc}
    acquisition_dict = {}

    toreturn = {
    'data_type' : data_type, 
    'calibration' : calibration_dict, 
    'acquisition' : acquisition_dict, 
    'imported_parameters' : parameters
    }
    return [toreturn,]  
    

   
