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

import struct
import os
import numpy as np

ser_extensions = ('ser', 'SER')
emi_extensions = ('emi', 'EMI')
# Plugin characteristics
# ----------------------
format_name = 'FEI TIA'
description = ''
full_suport = False
# Recognised file extension
file_extensions = ser_extensions + emi_extensions
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
'1' : '<u1',	
'2' : '<u2',
'3' : '<u4',	
'4' : '<i1',	
'5' : '<i2',	
'6' : '<i4',	
'7' : '<f4',	
'8' : '<f8',
'9' : '<c8',	
'10' : '<c16',
}

def readLELong(file):
    '''Read 4 bytes as *little endian* integer in file'''
    read_bytes = file.read(4)
    return struct.unpack('<L', read_bytes)[0]

def readLEShort(file):
    '''Read 2 bytes as *little endian* integer in file'''
    read_bytes = file.read(2)
    return struct.unpack('<H', read_bytes)[0]

def dimension_array_dtype(n, DescriptionLength, UnitsLength):
    dt_list = [
    ("Dim-%s_DimensionSize" % n, ("<u4")),   
    ("Dim-%s_CalibrationOffset" % n, "<f8"),    
    ("Dim-%s_CalibrationDelta" % n, "<f8"),        
    ("Dim-%s_CalibrationElement" % n, "<u4"),
    ("Dim-%s_DescriptionLength" % n, "<u4"),
    ("Dim-%s_Description" % n, (str, DescriptionLength)),     
    ("Dim-%s_UnitsLength" % n, "<u4"),  
    ("Dim-%s_Units" % n, (str, UnitsLength)),
    ]
    return dt_list

def get_number_of_dimensions(file):
    file.seek(26)
    number_of_dimensions = readLELong(file)
    return number_of_dimensions
    
def get_total_number_of_elements(file):
    file.seek(14)
    number_of_number_of_elements = readLELong(file)
    return number_of_number_of_elements

def get_lengths(file):
    file.seek(24,1)
    description_length = readLELong(file)
    file.seek(description_length,1)
    unit_length = readLELong(file)
    file.seek(unit_length,1)
    return description_length, unit_length
    
def get_header_dtype_list(file):
    header_list = [
    ("ByteOrder", ("<u2")),   
    ("SeriesID", "<u2"),    
    ("SeriesVersion", "<u2"),        
    ("DataTypeID", "<u4"),
    ("TagTypeID", "<u4"),
    ("TotalNumberElements", "<u4"),     
    ("ValidNumberElements", "<u4"),  
    ("OffsetArrayOffset", "<u4"),
    ("NumberDimensions", "<u4"),
    ]
    number_of_dimensions = get_number_of_dimensions(file)
    total_number_of_elements = get_total_number_of_elements(file)
    # Go to the beginning of the dimension array section
    file.seek(30) 
    for n in range(1, number_of_dimensions + 1):
        description_length, unit_length = get_lengths(file)
        header_list += dimension_array_dtype(
        n, description_length, unit_length)
    # Here we can check if the OffsetArrayOffset == file.tell()
    
    # Read the data offset
    header_list += [("Data_Offsets", ("<u4", total_number_of_elements)),]
    header_list += [("Tag_Offsets", ("<u4", total_number_of_elements)),]
    file.seek(0)
    return header_list
    
    
def get_data_dtype_list(file, offset, data_type):
    if data_type == 'SI':
        file.seek(offset + 20)
        data_type = readLEShort(file)
        array_size = readLELong(file)
        header = [
        ("CalibrationOffset", ("<f8")),   
        ("CalibrationDelta", "<f8"),    
        ("CalibrationElement", "<u4"),        
        ("DataType", "<u2"),
        ("ArrayLength", "<u4"),
        ("Array", (data_types[str(data_type)], array_size)),
        ]
    elif data_type == 'Image':  # Untested
        file.seek(offset + 40)
        data_type = readLEShort(file)
        array_size_x = readLELong(file)
        array_size_y = readLELong(file)
        array_size = array_size_x * array_size_y
        header = [
        ("CalibrationOffsetX", ("<f8")),   
        ("CalibrationDeltaX", "<f8"),    
        ("CalibrationElementX", "<u4"),
        ("CalibrationOffsetY", ("<f8")),   
        ("CalibrationDeltaY", "<f8"),    
        ("CalibrationElementY", "<u4"),        
        ("DataType", "<u2"),
        ("ArraySizeX", "<u4"),
        ("ArraySizeY", "<u4"),
        ("Array", (data_types[str(data_type)], (array_size_x, array_size_y))), 
        ]
    return header
    
def get_data_tag_dtype_list(data_type_id):
    # "TagTypeID" = 16706
    if data_type_id == 16706:
        header = [
        ("TagTypeID", ("<u2")),
        ("Unknown", ("<u2")),  # Not in Boothroyd description. = 0  
        ("Time", "<u4"),   # The precision is one second...   
        ("PositionX", "<f8"),        
        ("PositionY", "<f8"),
        ]
    else: # elif data_type_id == ?????
        header = [
        ("TagTypeID", ("<u2")),
        ("Unknown", ("<u2")),  # Not in Boothroyd description. = 0. Not tested.  
        ("Time", "<u4"),   # The precision is one second...   
        ]
    return header

def print_struct_array_values(struct_array):
    for key in struct_array.dtype.names:
        if type(struct_array[key]) is not np.ndarray or \
            np.array(struct_array[key].shape).sum() == 1:
            print "%s : %s" % (key, struct_array[key])
        else:
            print "%s : Array" % key
            
def guess_data_type(data_type_id):
    if data_type_id == 16672:
        return 'SI'
    else:
        return 'Image'
        
def emi_reader(filename, dump_xml = False, **kwds):
    # TODO: recover the tags from the emi file. It is easy: just look for 
    # <ObjectInfo> and </ObjectInfo>. It is standard xml :)
    objects = get_xml_info_from_emi(filename)
    filename = os.path.splitext(filename)[0]
    if dump_xml is True:
        i = 1
        for obj in objects:
            f = open(filename + '-object-%s.xml' % i, 'w')
            f.write(obj)
            f.close()
            i += 1
    from glob import glob
    ser_files = glob(filename + '_*.ser')
    
    sers = []
    for f in ser_files:
        print "Opening ", f
        sers.append(ser_reader(f))
    return sers
    
def file_reader(filename, *args, **kwds):
    ext = os.path.splitext(filename)[1][1:]
    if ext in ser_extensions:
        return [ser_reader(filename, *args, **kwds),]
    elif ext in emi_extensions:
        return emi_reader(filename, *args, **kwds)
            
def load_ser_file(filename, print_info = False):
    print "Opening the file: ", filename
    file = open(filename,'rb')
    header = np.fromfile(file, dtype = np.dtype(get_header_dtype_list(file)), 
    count = 1)
    if print_info is True:
        print "Extracting the information"
        print "\n"
        print "Header info:"
        print "------------"
        print_struct_array_values(header[0])
    data_dtype_list = get_data_dtype_list(file, header['Data_Offsets'][0][0], 
    guess_data_type(header['DataTypeID']))
    tag_dtype_list =  get_data_tag_dtype_list(header['TagTypeID'])
    file.seek(header['Data_Offsets'][0][0])
    data = np.fromfile(file, dtype=np.dtype(data_dtype_list + tag_dtype_list), 
    count = header["TotalNumberElements"])
    if print_info is True:
        print "\n"
        print "Data info:"
        print "----------"
        print_struct_array_values(data[0])
    file.close()
    return header, data
    
def get_xml_info_from_emi(emi_file):
    f = open(emi_file, 'rb')
    tx = f.read()
    f.close()
    objects = []
    i_start = 0
    while i_start != -1:
        i_start += 1
        i_start = tx.find('<ObjectInfo>', i_start)
        i_end =  tx.find('</ObjectInfo>', i_start)
        objects.append(tx[i_start:i_end + 13]) 
    return objects[:-1]
    
def ser_reader(filename, *args, **kwds):
    '''Reads the information from the file and returns it in the EELSLab 
    required format'''
    # Determine if it is an emi or a ser file.
    
    header, data = load_ser_file(filename)
    data_type = guess_data_type(header['DataTypeID'])
    array_shape = []
    coordinates = []
    ndim = header['NumberDimensions']
    
    
    if data_type == 'SI':
        
        # Extra dimensions
        for i in range(1, ndim + 1):
            if i == ndim:
                name = 'x'
            elif i == ndim - 1:
                name = 'y'
            else:
                name = 'undefined%' % i
            coordinates.append({
            'name' : name,
            'offset' : header['Dim-%i_CalibrationOffset' % i][0],
            'scale' : header['Dim-%i_CalibrationDelta' % i][0],
            'units' : header['Dim-%i_Units' % i][0],
            'size' : header['Dim-%i_DimensionSize' % i][0],
            'index_in_array' : i - 1
            })
            array_shape.append(header['Dim-%i_DimensionSize' % i][0])
        
        # Spectral dimension    
        coordinates.append({
            'name' : 'undefined',
            'offset' : data['CalibrationOffset'][0],
            'scale' : data['CalibrationDelta'][0],
            'units' : 'undefined',
            'size' : data['ArrayLength'][0],
            'index_in_array' : header['NumberDimensions'][0]
            })
        
        if len(data['PositionY']) > 1 and \
        (data['PositionY'][0] == data['PositionY'][1]):
            # The spatial dimensions are stored in the reversed order
            # We reverse the shape
            array_shape.reverse()
        array_shape.append(data['ArrayLength'][0])
        
    elif data_type == 'Image':
        
        # Y Coordinate
        coordinates.append({
            'name' : 'y',
            'offset' : data['CalibrationOffsetY'][0] - \
            data['CalibrationElementY'][0] * data['CalibrationDeltaY'][0],
            'scale' : data['CalibrationDeltaY'][0],
            'units' : 'Unknown',
            'size' : data['ArraySizeY'][0],
            'index_in_array' : 0
            })
        array_shape.append(data['ArraySizeY'][0])
        
        # X Coordinate
        coordinates.append({
            'name' : 'x',
            'offset' : data['CalibrationOffsetX'][0] - \
            data['CalibrationElementX'][0] * data['CalibrationDeltaX'][0],
            'scale' : data['CalibrationDeltaX'][0],
            'units' : 'undefined',
            'size' : data['ArraySizeX'][0],
            'index_in_array' : 1
            })
        array_shape.append(data['ArraySizeX'][0])
        
        # Extra dimensions
        for i in range(1, header['NumberDimensions'] + 1):
            coordinates.append({
            'name' : 'undefined%s' % i,
            'offset' : header['Dim-%i_CalibrationOffset' % i][0],
            'scale' : header['Dim-%i_CalibrationDelta' % i][0],
            'units' : header['Dim-%i_Units' % i][0],
            'size' : header['Dim-%i_DimensionSize' % i][0],
            'index_in_array' : 1 + i
            })
            array_shape.append(header['Dim-%i_DimensionSize' % i][0])

    # If the acquisition stops before finishing the job, the stored file will 
    # report the requested size even though no values are recorded. Therefore if
    # the shapes of the retrieved array does not match that of the data 
    # dimensions we must fill the rest with zeros
    if np.cumprod(array_shape)[-1] != np.cumprod(data['Array'].shape)[-1]:
        dc = np.zeros((array_shape[0] * array_shape[1], array_shape[2]), 
                      dtype = data['Array'].dtype)
        dc[:data['Array'].shape[0],...] = data['Array']
    else:
        dc = data['Array']
    
    dc = dc.reshape(array_shape)
    if data_type == 'Image':
        dc = dc[::-1]
      
    dictionary = {
    'data_type' : 'Signal',
    'filename' : filename,
    'data' : dc,
    'parameters' : {},
    'coordinates' : coordinates,
    'extra_parameters' : {'header' : header, 'data' : data}}
    return dictionary