# -*- coding: utf-8 -*-
# Copyright © 2007 Francisco Javier de la Peña
#
# This file is part of Hyperspy.
#
# Hyperspy is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Hyperspy; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  
# USA

import os.path

import numpy as np

from dm3reader_hyperspy import parseDM3
from hyperspy import messages

# Plugin characteristics
# ----------------------
format_name = 'Digital Micrograph dm3'
description = ''
full_suport = False
# Recognised file extension
file_extensions = ('dm3', 'DM3')
default_extension = 0
# Reading features
reads_images = True
reads_spectrum = True
reads_spectrum_image = True
# Writing features
writes_images = False
writes_spectrum = False
writes_spectrum_image = False
# ----------------------

data_types = {
    '1' :  '<u2', # 2 byte integer signed ("short")
    '2' :  '<f4', # 4 byte real (IEEE 754)
    '3' :  '<c8', # 8 byte complex (real, imaginary)
    '4' :  '',    # ?
    # 4 byte packed complex (see below)
    '5' :  (np.int16, {'real':(np.int8,0), 'imaginary':(np.int8,1)}),
    '6' :  '<u1', # 1 byte integer unsigned ("byte")
    '7' :  '<i4', # 4 byte integer signed ("long")
    # I do not have any dm3 file with this format to test it.
    '8' :  '',    # rgb view, 4 bytes/pixel, unused, red, green, blue?
    '9' :  '<i1', # byte integer signed
    '10' : '<u2', # 2 byte integer unsigned
    '11' : '<u4', # 4 byte integer unsigned
    '12' : '<f8', # 8 byte real
    '13' : '<c16', # byte complex
    '14' : 'bool', # 1 byte binary (ie 0 or 1)
     # Packed RGB. It must be a recent addition to the format because it does  
     # not appear in http://www.microscopy.cen.dtu.dk/~cbb/info/dmformat/
    '23' :  (np.float32, 
    {'R':('<u1',0), 'G':('<u1',1), 'B':('<u1',2), 'A':('<u1',3)}),
}
unknown_dm3_message = \
'Unknown dm3 file format. '
'If you are sure that this is a valid dm3 file, you can help us improving '
'the support of DM3 files in Hyperspy by sending us the file. '
'http://www.hyperspy.org'


def node_valve(nodes, value, calibration_dict):
    node = nodes.pop(0)
    if node not in calibration_dict:
        calibration_dict[node] = {}
    if len(nodes) != 0:
        node_valve(nodes,value, calibration_dict[node])
    else:
        calibration_dict[node] = value
        return
    
def list2dict(list_of_keys):
    dict_keys_string = ''
    for key in list_of_keys:
        dict_keys_string += '[\'%s\']' % key
    return dict_keys_string
              
def file_reader(filename, record_by = None):
    dimensions_pth = ['ImageData', 'Calibrations', 'Dimension']
    calibration_dict = {}
    acquisition_dict = {}
    tags_dict = {}
    # parse DM3 file
    tags = parseDM3(filename, dump=False)
    
    # Convert the list into a dictionary, tags_dict 
    for nodes in tags.items():
        node_valve(nodes[0].split('.'), nodes[1], tags_dict)
    
    # Determine the number of images
    # If it doesn't find the ImageList node it exits.
    # What if it only contain structs?
    try:
        number_of_images = len(tags_dict['root']['ImageList'].keys())
    except:
        messages.warning_exit("The given dm3 file doesn't contain valid data")
   
    if number_of_images > 1:
        image_id = [str(i) for i in xrange(0, number_of_images)] 
        # Find the index of the thumbnail if any and set image_id to the other one
        if 'Thumbnails' in tags_dict['root']:
            for thumbnail in  tags_dict['root']['Thumbnails']:
                image_id.remove(str(tags_dict['root']['Thumbnails'][thumbnail]['ImageIndex']))
        if len(image_id) > 1:
            messages.information(
            "The dm3 file contains %i images. Extracting just the last one" % len(image_id))
        image_id = image_id[-1]
    else:
        messages.warning_exit("The given dm3 file doesn't contain valid data")
    image_dict = tags_dict['root']['ImageList'][image_id]

    # Get the name (normally the name of the file)
    if 'Name' in image_dict:
        calibration_dict['title'] = image_dict['Name']
    else:
        calibration_dict['title'] = os.path.splitext(filename)[0]
    # Determine the dimensions
    dimensions = len(image_dict['ImageData']['Dimensions'])
    shape = np.ones((dimensions), dtype = np.int)
    units = ['' for i in xrange(dimensions)]
    origins = np.zeros((dimensions), dtype = np.float)
    scales =  np.ones((dimensions), dtype = np.float)

    for dimension in image_dict['ImageData']['Dimensions']:
        shape[int(dimension)] = image_dict['ImageData']['Dimensions'][dimension]

    for dimension in image_dict['ImageData']['Calibrations']['Dimension']:
        dim = image_dict['ImageData']['Calibrations']['Dimension'][dimension]
        origins[int(dimension)] = dim['Origin']
        scales[int(dimension)] = dim['Scale']
        units[int(dimension)] = dim['Units']
    brightness_dict = image_dict['ImageData']['Calibrations']['Brightness']
    brightness_origin = brightness_dict['Origin']
    brightness_scale = brightness_dict['Scale']
    brightness_units = brightness_dict['Units']
        
    # Convert the origins to their units scales
    origins *= scales

    # Find the address of the cube
    dc_size = image_dict['ImageData']['Data']['Size']
    dc_offset = image_dict['ImageData']['Data']['Offset']

    # access DM3 file
    dm3 = open(filename, 'rb')
    # Read the cube data
    dm3.seek(dc_offset)
    dt = data_types[str(image_dict['ImageData']['DataType'])]
    if dt == '':
        messages.warning_exit('The datatype is not supported')
    data_cube = np.fromfile(dm3, dtype = dt, count =np.cumprod(shape)[-1]).reshape(shape, order = 'F')
    dm3.close()
    print "Data cube correctly loaded"
    
    # Determine wether this is an image of SI and extract parameters
    if record_by is None:
        if 'eV' in units or 'keV' in units:
            record_by = 'spectrum'
        else:
            record_by = 'image'
    if record_by == 'spectrum': 
        print "Treating the data as an SI"

        # Try to determine the format (if known) to extract some parameters
        root = ['root', 'ImageList', image_id]
        eels = root + ['ImageTags', 'EELS', 'Acquisition']
        spim = root + ['ImageTags', 'spim']
        antwerp = root + ['ImageTags', 'SIm']
        if eval('tags_dict%s.has_key(\'%s\')' % (list2dict(root), 
        'ImageTags')):
            if eval('tags_dict%s.has_key(\'%s\')' % (list2dict(eels[0:-2]), 
            eels[-2])):
                print "Detected format: Standard dm3 SI file"
                print "Importing EELS parameters..."
                try:
                    eels_dict = eval('tags_dict%s' % list2dict(eels))
                    if eels_dict.has_key('Exposure (s)'):
                        acquisition_dict['exposure'] = float(eels_dict['Exposure (s)'])
                    else:
                        print "Warning: No exposure information could be found"
                except:
                    print "Warning: No exposure information could be found"
            elif eval('tags_dict%s.has_key(\'%s\')' % (list2dict(spim[:-1]), 
            spim[-1])):
                print "Detected format: Orsay SI format"
                print "Importing EELS parameters..."
                spim_dict = eval('tags_dict%s' % list2dict(spim))
                acquisition_dict['exposure'] = \
                float(spim_dict['eels']['dwell time'])
                calibration_dict['vsm'] = float(spim_dict['eels']['vsm'])
                    
            elif eval('tags_dict%s.has_key(\'%s\')' % (list2dict(antwerp[:-1]), 
            antwerp[-1])):
                    print "Detected format: Antwerp SI format"
                    print "Importing EELS parameters..."
                    antwerp_dict = eval('tags_dict%s' % list2dict(antwerp))
                    acquisition_dict['exposure'] = \
                float(antwerp_dict['Acquisition']['Spectrometer']
                ['Exposure Time'])       
            else:
                print "unknown dm3 format"
                print 
                print "Some of the information couldn't be retrieved"
                
        else:
            print "unknown dm3 format"
            print 
            print "Some of the information couldn't be retrieved"
        
        # In Hyperspy1 the first index must be the energy (this changes in Hyperspy2)
        # Rearrange the data_cube and parameters to have the energy first
        if 'eV' in units:
            energy_index = units.index('eV')
        elif 'keV' in units:
           energy_index = units.index('keV')
        else:
            energy_index = len(data_cube.squeeze().shape) - 1
        # In DM the origin is negative. Change it to positive
        origins[energy_index] *= -1

        if energy_index != 0:
            data_cube = np.rollaxis(data_cube, energy_index, 0)
            origins = np.roll(origins, 3 - energy_index)
            scales = np.roll(scales, 3 - energy_index)
            units = np.roll(units, 3 - energy_index)

        # Store the calibration in the calibration dict
        origins_keys = ['energyorigin', 'xorigin', 'yorigin']
        scales_keys = ['energyscale', 'xscale', 'yscale']
        units_keys = ['energyunits', 'xunits', 'yunits']

        for value in origins:
            calibration_dict.__setitem__(origins_keys.pop(0), value)

        for value in scales:
            calibration_dict.__setitem__(scales_keys.pop(0), value)

        for value in units:
            calibration_dict.__setitem__(units_keys.pop(0), value)

    else:
        print "Treating the data as an image"
        origins_keys = ['xorigin', 'yorigin', 'zorigin']
        scales_keys = ['xscale', 'yscale', 'zscale']
        units_keys = ['xunits', 'yunits', 'zunits']

        for value in origins:
            calibration_dict.__setitem__(origins_keys.pop(0), value)

        for value in scales:
            calibration_dict.__setitem__(scales_keys.pop(0), value)

        for value in units:
            calibration_dict.__setitem__(units_keys.pop(0), value)

       
    calibration_dict['data_cube'] = data_cube.squeeze()
    dm3.close()
    dictionary = {
        'record_by' : record_by, 
        'calibration' : calibration_dict, 
        'acquisition' : acquisition_dict,
        'imported_parameters' : calibration_dict}
    return [dictionary,]
