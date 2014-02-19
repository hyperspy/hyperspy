# -*- coding: utf-8 -*-
# Copyright 2007-2011 The Hyperspy developers
#
# This file is part of  Hyperspy.
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

import struct
from glob import glob
import os
import xml.etree.ElementTree as ET
try:
    from collections import OrderedDict
    ordict = True
except ImportError:
    ordict = False

import numpy as np
import traits.api as t

from hyperspy.misc.array_tools import sarray2dict
from hyperspy.misc.utils import DictionaryBrowser

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

# Writing capabilities
writes = False
# ----------------------

data_types = {
    '1': '<u1',
    '2': '<u2',
    '3': '<u4',
    '4': '<i1',
    '5': '<i2',
    '6': '<i4',
    '7': '<f4',
    '8': '<f8',
    '9': '<c8',
    '10': '<c16',
}


def readLELong(file):
    """Read 4 bytes as *little endian* integer in file"""
    read_bytes = file.read(4)
    return struct.unpack('<L', read_bytes)[0]


def readLEShort(file):
    """Read 2 bytes as *little endian* integer in file"""
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
    file.seek(24, 1)
    description_length = readLELong(file)
    file.seek(description_length, 1)
    unit_length = readLELong(file)
    file.seek(unit_length, 1)
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
    for n in xrange(1, number_of_dimensions + 1):
        description_length, unit_length = get_lengths(file)
        header_list += dimension_array_dtype(
            n, description_length, unit_length)
    # Here we can check if the OffsetArrayOffset == file.tell()

    # Read the data offset
    header_list += [("Data_Offsets", ("<u4", total_number_of_elements)), ]
    header_list += [("Tag_Offsets", ("<u4", total_number_of_elements)), ]
    file.seek(0)
    return header_list


def get_data_dtype_list(file, offset, record_by):
    if record_by == 'spectrum':
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
    elif record_by == 'image':  # Untested
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
            ("Array",
             (data_types[str(data_type)], (array_size_x, array_size_y))),
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
    else:  # elif data_type_id == ?????
        header = [
            ("TagTypeID", ("<u2")),
            # Not in Boothroyd description. = 0. Not tested.
            ("Unknown", ("<u2")),
            ("Time", "<u4"),   # The precision is one second...
        ]
    return header


def print_struct_array_values(struct_array):
    for key in struct_array.dtype.names:
        if not isinstance(struct_array[key], np.ndarray) or \
                np.array(struct_array[key].shape).sum() == 1:
            print "%s : %s" % (key, struct_array[key])
        else:
            print "%s : Array" % key


def guess_record_by(record_by_id):
    if record_by_id == 16672:
        return 'spectrum'
    else:
        return 'image'


def parse_ExperimentalDescription(et, dictree):
    dictree.add_node(et.tag)
    dictree = dictree[et.tag]
    for data in et.find("Root").findall("Data"):
        label = data.find("Label").text
        value = data.find("Value").text
        units = data.find("Unit").text
        item = label if not units else label + "_%s" % units
        value = float(value) if units else value
        dictree[item] = value


def parse_TrueImageHeaderInfo(et, dictree):
    dictree.add_node(et.tag)
    dictree = dictree[et.tag]
    et = ET.fromstring(et.text)
    for data in et.findall("Data"):
        dictree[data.find("Index").text] = float(data.find("Value").text)


def emixml2dtb(et, dictree):
    if et.tag == "ExperimentalDescription":
        parse_ExperimentalDescription(et, dictree)
        return
    elif et.tag == "TrueImageHeaderInfo":
        parse_TrueImageHeaderInfo(et, dictree)
        return
    if et.text:
        dictree[et.tag] = et.text
        return
    else:
        dictree.add_node(et.tag)
        for child in et:
            emixml2dtb(child, dictree[et.tag])


def emi_reader(filename, dump_xml=False, verbose=False, **kwds):
    # TODO: recover the tags from the emi file. It is easy: just look for
    # <ObjectInfo> and </ObjectInfo>. It is standard xml :)
    objects = get_xml_info_from_emi(filename)
    filename = os.path.splitext(filename)[0]
    if dump_xml is True:
        for i, obj in enumerate(objects):
            with open(filename + '-object-%s.xml' % i, 'w') as f:
                f.write(obj)

    ser_files = glob(filename + '_[0-9].ser')
    sers = []
    for f in ser_files:
        if verbose is True:
            print "Opening ", f
        try:
            sers.append(ser_reader(f, objects))
        except IOError:  # Probably a single spectrum that we don't support
            continue

        index = int(os.path.splitext(f)[0].split("_")[-1]) - 1
        op = DictionaryBrowser(sers[-1]['original_parameters'])
        emixml2dtb(ET.fromstring(objects[index]), op)
        sers[-1]['original_parameters'] = op.as_dictionary()
    return sers


def file_reader(filename, *args, **kwds):
    ext = os.path.splitext(filename)[1][1:]
    if ext in ser_extensions:
        return [ser_reader(filename, *args, **kwds), ]
    elif ext in emi_extensions:
        return emi_reader(filename, *args, **kwds)


def load_ser_file(filename, verbose=False):
    if verbose:
        print "Opening the file: ", filename
    with open(filename, 'rb') as f:
        header = np.fromfile(f,
                             dtype=np.dtype(get_header_dtype_list(f)),
                             count=1)
        if verbose is True:
            print "Extracting the information"
            print "\n"
            print "Header info:"
            print "------------"
            print_struct_array_values(header[0])

        if header['ValidNumberElements'] == 0:
            raise IOError(
                "The file does not contains valid data"
                "If it is a single spectrum, the data is contained in the  "
                ".emi file but Hyperspy cannot currently extract this information.")

        data_offsets = header['Data_Offsets'][0]
        try:
            data_offsets = data_offsets[0]
        except:
            pass
        data_dtype_list = get_data_dtype_list(
            f,
            data_offsets,
            guess_record_by(header['DataTypeID']))
        tag_dtype_list = get_data_tag_dtype_list(header['TagTypeID'])
        f.seek(data_offsets)
        data = np.fromfile(f,
                           dtype=np.dtype(data_dtype_list + tag_dtype_list),
                           count=header["TotalNumberElements"])
        if verbose is True:
            print "\n"
            print "Data info:"
            print "----------"
            print_struct_array_values(data[0])
    return header, data


def get_xml_info_from_emi(emi_file):
    with open(emi_file, 'rb') as f:
        tx = f.read()
    objects = []
    i_start = 0
    while i_start != -1:
        i_start += 1
        i_start = tx.find('<ObjectInfo>', i_start)
        i_end = tx.find('</ObjectInfo>', i_start)
        objects.append(tx[i_start:i_end + 13])
    return objects[:-1]


def ser_reader(filename, objects=None, verbose=False, *args, **kwds):
    """Reads the information from the file and returns it in the Hyperspy
    required format.

    """

    header, data = load_ser_file(filename, verbose=verbose)
    record_by = guess_record_by(header['DataTypeID'])
    axes = []
    ndim = int(header['NumberDimensions'])
    if record_by == 'spectrum':
        array_shape = [None, ] * int(ndim)
        i_array = range(ndim)
        if len(data['PositionY']) > 1 and \
                (data['PositionY'][0] == data['PositionY'][1]):
            # The spatial dimensions are stored in the reversed order
            # We reverse the shape
            i_array.reverse()
        # Extra dimensions
        for i in xrange(ndim):
            if i_array[i] == ndim - 1:
                name = 'x'
            elif i_array[i] == ndim - 2:
                name = 'y'
            else:
                name = t.Undefined
            axes.append({
                'name': name,
                'offset': header['Dim-%i_CalibrationOffset' % (i + 1)][0],
                'scale': header['Dim-%i_CalibrationDelta' % (i + 1)][0],
                'units': header['Dim-%i_Units' % (i + 1)][0],
                'size': header['Dim-%i_DimensionSize' % (i + 1)][0],
                'index_in_array': i_array[i]
            })
            array_shape[i_array[i]] = \
                header['Dim-%i_DimensionSize' % (i + 1)][0]
        # FEI seems to use the international system of units (SI) for the
        # spatial scale. However, we prefer to work in nm
        for axis in axes:
            if axis['units'] == 'meters':
                axis['units'] = 'nm'
                axis['scale'] *= 10 ** 9

        # Spectral dimension
        axes.append({
            'offset': data['CalibrationOffset'][0],
            'scale': data['CalibrationDelta'][0],
            'size': data['ArrayLength'][0],
            'index_in_array': header['NumberDimensions'][0]
        })

        array_shape.append(data['ArrayLength'][0])

    elif record_by == 'image':
        array_shape = []
        # Extra dimensions
        for i in xrange(ndim):
            if header['Dim-%i_DimensionSize' % (i + 1)][0] != 1:
                axes.append({
                    'offset': header['Dim-%i_CalibrationOffset' % (i + 1)][0],
                    'scale': header['Dim-%i_CalibrationDelta' % (i + 1)][0],
                    'units': header['Dim-%i_Units' % (i + 1)][0],
                    'size': header['Dim-%i_DimensionSize' % (i + 1)][0],
                })
            array_shape.append(header['Dim-%i_DimensionSize' % (i + 1)][0])
        # Y axis
        axes.append({
            'name': 'y',
            'offset': data['CalibrationOffsetY'][0] -
            data['CalibrationElementY'][0] * data['CalibrationDeltaY'][0],
            'scale': data['CalibrationDeltaY'][0],
            'units': 'Unknown',
            'size': data['ArraySizeY'][0],
        })
        array_shape.append(data['ArraySizeY'][0])

        # X axis
        axes.append({
            'name': 'x',
            'offset': data['CalibrationOffsetX'][0] -
            data['CalibrationElementX'][0] * data['CalibrationDeltaX'][0],
            'scale': data['CalibrationDeltaX'][0],
            'size': data['ArraySizeX'][0],
        })
        array_shape.append(data['ArraySizeX'][0])

    # If the acquisition stops before finishing the job, the stored file will
    # report the requested size even though no values are recorded. Therefore if
    # the shapes of the retrieved array does not match that of the data
    # dimensions we must fill the rest with zeros or (better) nans if the
    # dtype is float
    if np.cumprod(array_shape)[-1] != np.cumprod(data['Array'].shape)[-1]:
        dc = np.zeros(np.cumprod(array_shape)[-1],
                      dtype=data['Array'].dtype)
        if dc.dtype is np.dtype('f') or dc.dtype is np.dtype('f8'):
            dc[:] = np.nan
        dc[:data['Array'].ravel().shape[0]] = data['Array'].ravel()
    else:
        dc = data['Array']

    dc = dc.reshape(array_shape)
    if record_by == 'image':
        dc = dc[::-1]
    if ordict:
        original_parameters = OrderedDict()
    else:
        original_parameters = {}
    header_parameters = sarray2dict(header)
    sarray2dict(data, header_parameters)

    # We remove the Array key to save memory avoiding duplication
    del header_parameters['Array']
    original_parameters['ser_header_parameters'] = header_parameters
    dictionary = {
        'data': dc,
        'mapped_parameters': {'original_filename': os.path.split(filename)[1],
                              'record_by': record_by,
                              'signal_type': "", },
        'axes': axes,
        'original_parameters': original_parameters,
        'mapping': mapping}
    return dictionary


def get_mode(mode):
    if "STEM" in mode:
        return "STEM"
    else:
        return "TEM"


mapping = {
    "ObjectInfo.ExperimentalDescription.High_tension_kV": ("TEM.beam_voltage", None),
    "ObjectInfo.ExperimentalDescription.Emission_uA": ("TEM.beam_intensity", None),
    "ObjectInfo.ExperimentalDescription.Microscope": ("TEM.microscope", None),
    "ObjectInfo.ExperimentalDescription.Mode": ("TEM.mode", get_mode),
}
