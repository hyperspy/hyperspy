# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

import struct
import warnings
from glob import glob
import os
from dateutil import parser
import logging
import xml.etree.ElementTree as ET
from collections import OrderedDict

import numpy as np
import traits.api as t

from hyperspy.misc.array_tools import sarray2dict
from hyperspy.misc.utils import DictionaryTreeBrowser, multiply

_logger = logging.getLogger(__name__)


ser_extensions = ('ser', 'SER')
emi_extensions = ('emi', 'EMI')
# Plugin characteristics
# ----------------------
format_name = 'FEI TIA'
description = ''
full_support = False
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

XY_TAG_ID = 16706  # header contains XY calibration


def readLELongLong(file):
    """Read 8 bytes as *little endian* integer in file"""
    read_bytes = file.read(8)
    return struct.unpack('<Q', read_bytes)[0]


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
        ("Dim-%s_Description" % n, (bytes, DescriptionLength)),
        ("Dim-%s_UnitsLength" % n, "<u4"),
        ("Dim-%s_Units" % n, (bytes, UnitsLength)),
    ]
    return dt_list


def get_lengths(file):
    file.seek(24, 1)
    description_length = readLELong(file)
    file.seek(description_length, 1)
    unit_length = readLELong(file)
    file.seek(unit_length, 1)
    return description_length, unit_length


def get_header_dtype_list(file):
    # Read the first part of the header
    header_list1 = [
        ("ByteOrder", "<u2"),
        ("SeriesID", "<u2"),
        ("SeriesVersion", "<u2"),
        ("DataTypeID", "<u4"),
        ("TagTypeID", "<u4"),
        ("TotalNumberElements", "<u4"),
        ("ValidNumberElements", "<u4")]
    header1 = np.fromfile(file,
                          dtype=np.dtype(header_list1),
                          count=1)
    # Depending on the SeriesVersion, the OffsetArrayOffset is 4 or 8 bytes
    if header1["SeriesVersion"] <= 528:
        OffsetArrayOffset_dtype = "<u4"
        beginning_dimension_array_section = 30
    else:
        OffsetArrayOffset_dtype = "<u8"
        beginning_dimension_array_section = 34

    # Once we know the type of the OffsetArrayOffset, we can continue reading
    # the 2nd part of the header
    file.seek(22)
    header_list2 = [("OffsetArrayOffset", OffsetArrayOffset_dtype),
                    ("NumberDimensions", "<u4")]
    header2 = np.fromfile(file,
                          dtype=np.dtype(header_list2),
                          count=1)
    header_list = header_list1 + header_list2
    # Go to the beginning of the dimension array section
    file.seek(beginning_dimension_array_section)
    for n in range(1, header2["NumberDimensions"][0] + 1):
        description_length, unit_length = get_lengths(file)
        header_list += dimension_array_dtype(
            n, description_length, unit_length)
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
        shape = (array_size)
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
        shape = (array_size_x, array_size_y)
    return header, shape


def get_data_tag_dtype_list(data_type_id):
    if data_type_id == XY_TAG_ID:
        header = [
            ("TagTypeID", ("<u2")),
            ("Unknown", ("<u2")),  # Not in Boothroyd description. = 0
            ("Time", "<u4"),   # The precision is one second...
            ("PositionX", "<f8"),
            ("PositionY", "<f8"),
        ]
    else:  # elif data_type_id == ?????, 16722?
        header = [
            ("TagTypeID", ("<u2")),
            # Not in Boothroyd description. = 0. Not tested.
            ("Unknown", ("<u2")),
            ("Time", "<u4"),   # The precision is one second...
        ]
    return header


def log_struct_array_values(struct_array):
    for key in struct_array.dtype.names:
        if not isinstance(struct_array[key], np.ndarray) or \
                np.array(struct_array[key].shape).sum() == 1:
            _logger.info("%s : %s", key, struct_array[key])
        else:
            _logger.info("%s : Array", key)


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
    for data in et.findall(b"Data"):
        dictree[data.find(b"Index").text] = float(data.find(b"Value").text)


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


def emi_reader(filename, dump_xml=False, **kwds):
    # TODO: recover the tags from the emi file. It is easy: just look for
    # <ObjectInfo> and </ObjectInfo>. It is standard xml :)
    # xml chunks are identified using UUID, if we can find how these UUID are
    # generated then, it will possible to match to the corresponding ser file
    # and add the detector information in the metadata
    objects = get_xml_info_from_emi(filename)
    filename = os.path.splitext(filename)[0]
    if dump_xml is True:
        for i, obj in enumerate(objects):
            with open(filename + '-object-%s.xml' % i, 'w') as f:
                f.write(obj)

    ser_files = sorted(glob(filename + '_[0-9].ser'))
    sers = []
    for f in ser_files:
        _logger.info("Opening %s", f)
        try:
            sers.append(ser_reader(f, objects, **kwds))
        except IOError:  # Probably a single spectrum that we don't support
            continue

        index = int(os.path.splitext(f)[0].split("_")[-1]) - 1
        op = DictionaryTreeBrowser(sers[-1]['original_metadata'])
        emixml2dtb(ET.fromstring(objects[index]), op)
        sers[-1]['original_metadata'] = op.as_dictionary()
    return sers


def file_reader(filename, *args, **kwds):
    ext = os.path.splitext(filename)[1][1:]
    if ext in ser_extensions:
        return [ser_reader(filename, *args, **kwds), ]
    elif ext in emi_extensions:
        return emi_reader(filename, *args, **kwds)


def load_ser_file(filename):
    _logger.info("Opening the file: %s", filename)
    with open(filename, 'rb') as f:
        header = np.fromfile(f,
                             dtype=np.dtype(get_header_dtype_list(f)),
                             count=1)
        _logger.info("Header info:")
        log_struct_array_values(header[0])

        if header['ValidNumberElements'] == 0:
            raise IOError(
                "The file does not contains valid data. "
                "If it is a single spectrum, the data is contained in the  "
                ".emi file but HyperSpy cannot currently extract this "
                "information.")

        # Read the first element of data offsets
        f.seek(header["OffsetArrayOffset"][0])
        # OffsetArrayOffset can contain 4 or 8 bytes integer depending if the
        # data have been acquired using a 32 or 64 bits platform.
        if header['SeriesVersion'] <= 528:
            data_offset = readLELong(f)
            data_offset_array = np.fromfile(f,
                                            dtype="<u4",
                                            count=header["TotalNumberElements"][0])
        else:
            data_offset = readLELongLong(f)
            data_offset_array = np.fromfile(f,
                                            dtype="<u8",
                                            count=header["TotalNumberElements"][0])
        data_dtype_list, shape = get_data_dtype_list(
            f,
            data_offset,
            guess_record_by(header['DataTypeID']))
        tag_dtype_list = get_data_tag_dtype_list(header['TagTypeID'])
        f.seek(data_offset)
        data = np.empty(header["TotalNumberElements"][0],
                        dtype=np.dtype(data_dtype_list + tag_dtype_list))
        for i, offset in enumerate(data_offset_array):
            data[i] = np.fromfile(f,
                                  dtype=np.dtype(
                                      data_dtype_list + tag_dtype_list),
                                  count=1)
            f.seek(offset)
        _logger.info("Data info:")
        log_struct_array_values(data[0])
    return header, data


def get_xml_info_from_emi(emi_file):
    with open(emi_file, 'rb') as f:
        tx = f.read()
    objects = []
    i_start = 0
    while i_start != -1:
        i_start += 1
        i_start = tx.find(b'<ObjectInfo>', i_start)
        i_end = tx.find(b'</ObjectInfo>', i_start)
        objects.append(tx[i_start:i_end + 13].decode('utf-8'))
    return objects[:-1]


def get_calibration_from_position(position):
    """Compute the size, scale and offset of a linear axis from coordinates.

    This function assumes rastering on a regular grid for the full size of
    each dimension before rastering over another one. Fox example: a11, a12,
    a13, a21, a22, a23 for a 2x3 grid.

    Parameters
    ----------
    position: numpy array.
        Position coordinates of the axis. Normally as in PositionX/Y of the
        ser file.

    Returns
    -------
    axis_attr: dictionary with `size`, `scale`, `offeset` keys.

    """
    offset = position[0]
    for i, x in enumerate(position):
        if x != position[0]:
            break
    if i == len(position) - 1:
        # No scanning over this axis
        scale = 0
        size = 0
    else:
        scale = x - position[0]
        if i == 1:  # Rastering over this dimension first
            for j, x in enumerate(position[1:]):
                if x == position[0]:
                    break
            size = j + 1
        else:  # Second rastering dimension
            size = len(position) / i
    axis_attr = {"size": size, "scale": scale, "offset": offset}
    return axis_attr


def get_axes_from_position(header, data):
    array_shape = []
    axes = []
    array_size = int(header["ValidNumberElements"])
    if data[b"TagTypeID"][0] == XY_TAG_ID:
        xcal = get_calibration_from_position(data[b"PositionX"])
        ycal = get_calibration_from_position(data[b"PositionY"])
        if xcal[b"size"] == 0 and ycal[b"size"] != 0:
            # Vertical line scan
            axes.append({
                'name': "x",
                'units': "meters",
                'index_in_array': 0,
            })
            axes[-1].update(xcal)
            array_shape.append(axes[-1]["size"])

        elif xcal[b"size"] != 0 and ycal[b"size"] == 0:
            # Horizontal line scan
            axes.append({
                'name': "y",
                'units': "meters",
                'index_in_array': 0,
            })
            axes[-1].update(ycal)
            array_shape.append(axes[-1]["size"])

        elif xcal[b"size"] * ycal[b"size"] == array_size:
            # Signal2D
            axes.append({
                'name': "y",
                'units': "meters",
                'index_in_array': 0,
            })
            axes[-1].update(ycal)
            array_shape.append(axes[-1]["size"])
            axes.append({
                'name': "x",
                'units': "meters",
                'index_in_array': 1,
            })
            axes[-1].update(xcal)
            array_shape.append(axes[-1]["size"])
        elif xcal[b"size"] == ycal[b"size"] == array_size:
            # Oblique line scan
            scale = np.sqrt(xcal["scale"] ** 2 + ycal["scale"] ** 2)
            axes.append({
                'name': "x",
                'units': "meters",
                'index_in_array': 0,
                "offset": 0,
                "scale": scale,
                "size": xcal["size"]
            })
            array_shape.append(axes[-1]["size"])
        else:
            raise IOError
    else:
        array_shape = [header["ValidNumberElements"]]
        axes.append({
            'name': "Unknown dimension",
            'offset': 0,
            'scale': 1,
            'units': "",
            'size': header["ValidNumberElements"],
            'index_in_array': 0
        })
    return array_shape, axes


def convert_xml_to_dict(xml_object):
    op = DictionaryTreeBrowser()
    emixml2dtb(ET.fromstring(xml_object), op)
    return op


def ser_reader(filename, objects=None, *args, **kwds):
    """Reads the information from the file and returns it in the HyperSpy
    required format.

    """
    header, data = load_ser_file(filename)
    record_by = guess_record_by(header['DataTypeID'])
    ndim = int(header['NumberDimensions'])
    date, time = None, None
    if objects is not None:
        objects_dict = convert_xml_to_dict(objects[0])
        date, time = _get_date_time(objects_dict.ObjectInfo.AcquireDate)
    if "PositionY" in data.dtype.names and len(data['PositionY']) > 1 and \
            (data['PositionY'][0] == data['PositionY'][1]):
        # The spatial dimensions are stored in F order i.e. X, Y, ...
        order = "F"
    else:
        # The spatial dimensions are stored in C order i.e. ..., Y, X
        order = "C"
    if ndim == 0 and header["ValidNumberElements"] != 0:
        # The calibration of the axes are not stored in the header.
        # We try to guess from the position coordinates.
        array_shape, axes = get_axes_from_position(header=header,
                                                   data=data)
    else:
        axes = []
        array_shape = [None, ] * int(ndim)
        spatial_axes = ["x", "y"][:ndim]
        for i in range(ndim):
            idim = 1 + i if order == "C" else ndim - i
            if (record_by == "spectrum" or
                    header['Dim-%i_DimensionSize' % (i + 1)][0] != 1):
                units = (header['Dim-%i_Units' % (idim)][0].decode('utf-8')
                         if header['Dim-%i_UnitsLength' % (idim)] > 0
                         else t.Undefined)
                if units == "meters":
                    name = (spatial_axes.pop() if order == "F"
                            else spatial_axes.pop(-1))
                else:
                    name = t.Undefined
                axes.append({
                    'offset': header['Dim-%i_CalibrationOffset' % idim][0],
                    'scale': header['Dim-%i_CalibrationDelta' % idim][0],
                    'units': units,
                    'size': header['Dim-%i_DimensionSize' % idim][0],
                    'name': name,
                })
                array_shape[i] = \
                    header['Dim-%i_DimensionSize' % idim][0]
    # Spectral dimension
    if record_by == "spectrum":
        axes.append({
            'offset': data['CalibrationOffset'][0],
            'scale': data['CalibrationDelta'][0],
            'size': data['ArrayLength'][0],
            'index_in_array': header['NumberDimensions'][0]
        })

        # FEI seems to use the international system of units (SI) for the
        # energy scale (eV).
        axes[-1]['units'] = 'eV'
        axes[-1]['name'] = 'Energy'

        array_shape.append(data['ArrayLength'][0])

    elif record_by == 'image':
        if objects is not None:
            units = _guess_units_from_mode(objects_dict, header)
        else:
            units = "meters"
        # Y axis
        axes.append({
            'name': 'y',
            'offset': data['CalibrationOffsetY'][0] -
            data['CalibrationElementY'][0] * data['CalibrationDeltaY'][0],
            'scale': data['CalibrationDeltaY'][0],
            'units': units,
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
            'units': units,
        })
        array_shape.append(data['ArraySizeX'][0])

    # FEI seems to use the international system of units (SI) for the
    # spatial scale. However, we prefer to work in nm
    for axis in axes:
        if axis['units'] == 'meters':
            axis['units'] = 'nm'
            axis['scale'] *= 10 ** 9
        elif axis['units'] == '1/meters':
            axis['units'] = '1 / nm'
            axis['scale'] /= 10 ** 9

    # Remove Nones from array_shape caused by squeezing size 1 dimensions
    array_shape = [dim for dim in array_shape if dim is not None]
    lazy = kwds.pop('lazy', False)
    if lazy:
        from dask import delayed
        from dask.array import from_delayed
        val = delayed(load_only_data, pure=True)(filename, array_shape,
                                                 record_by, len(axes))
        dc = from_delayed(val, shape=array_shape,
                          dtype=data['Array'].dtype)
    else:
        dc = load_only_data(filename, array_shape, record_by, len(axes),
                            data=data)

    original_metadata = OrderedDict()
    header_parameters = sarray2dict(header)
    sarray2dict(data, header_parameters)
    # We remove the Array key to save memory avoiding duplication
    del header_parameters['Array']
    original_metadata['ser_header_parameters'] = header_parameters
    metadata = {'General': {
        'original_filename': os.path.split(filename)[1],
    },
        "Signal": {
        'signal_type': "",
        'record_by': record_by,
    },
    }
    if date is not None and time is not None:
        metadata['General']['date'] = date
        metadata['General']['time'] = time
    dictionary = {
        'data': dc,
        'metadata': metadata,
        'axes': axes,
        'original_metadata': original_metadata,
        'mapping': mapping}
    return dictionary


def load_only_data(filename, array_shape, record_by, num_axes, data=None):
    if data is None:
        _, data = load_ser_file(filename)
    # If the acquisition stops before finishing the job, the stored file will
    # report the requested size even though no values are recorded. Therefore
    # if the shapes of the retrieved array does not match that of the data
    # dimensions we must fill the rest with zeros or (better) nans if the
    # dtype is float
    if multiply(array_shape) != multiply(data['Array'].shape):
        dc = np.zeros(multiply(array_shape),
                      dtype=data['Array'].dtype)
        if dc.dtype is np.dtype('f') or dc.dtype is np.dtype('f8'):
            dc[:] = np.nan
        dc[:data['Array'].ravel().shape[0]] = data['Array'].ravel()
    else:
        dc = data['Array']

    dc = dc.reshape(array_shape)
    if record_by == 'image':
        dc = dc[..., ::-1, :]
    if num_axes != len(dc.shape):
        dc = dc.squeeze()
    if num_axes != len(dc.shape):
        raise IOError("Please report this issue to the HyperSpy developers.")
    return dc


def _guess_units_from_mode(objects_dict, header):
    # in case the xml file doesn't contain the "Mode" or the header doesn't
    # contain 'Dim-1_UnitsLength', return "meters" as default, which will be
    # OK most of the time
    warn_str = "The navigation axes units could not be determined. " \
               "Setting them to `nm`, but this may be wrong."
    try:
        mode = objects_dict.ObjectInfo.ExperimentalDescription.Mode
        isCamera = (
            "CameraNamePath" in objects_dict.ObjectInfo.AcquireInfo.keys())
    except AttributeError:  # in case the xml chunk doesn't contain the Mode
        warnings.warn(warn_str)
        return 'meters'  # Most of the time, the unit will be meters!
    if 'Dim-1_UnitsLength' in header.dtype.fields:
        # assuming that for an image stack, the UnitsLength of the "3rd"
        # dimension is 0
        isImageStack = (header['Dim-1_UnitsLength'][0] == 0)
        # Workaround: if this is not an image stack and not a STEM image, then
        # we assume that it should be a diffraction
        isDiffractionScan = (header['Dim-1_DimensionSize'][0] > 1 and not
                             isImageStack)
    else:
        warnings.warn(warn_str)
        return 'meters'  # Most of the time, the unit will be meters!

    _logger.info(objects_dict.ObjectInfo.AcquireInfo)
    _logger.info("mode: %s", mode)
    _logger.info("isCamera: %s", isCamera)
    _logger.info("isImageStack: %s", isImageStack)
    _logger.info("isImageStack: %s", isDiffractionScan)
    if 'STEM' in mode:
        # data recorded in STEM with a camera, so we assume, it's a diffraction
        # in case we can't make use the detector is a camera, use a workaround
        if isCamera or isDiffractionScan:
            return "1/meters"
        else:
            return "meters"
    elif 'Diffraction' in mode:
        return "1/meters"
    else:
        return 'meters'


def _get_simplified_mode(mode):
    if "STEM" in mode:
        return "STEM"
    else:
        return "TEM"


def _get_date_time(value):
    dt = parser.parse(value)
    return dt.date().isoformat(), dt.time().isoformat()


def _get_microscope_name(value):
    return value.replace('Microscope ', '')


mapping = {
    "ObjectInfo.ExperimentalDescription.High_tension_kV": (
        "Acquisition_instrument.TEM.beam_energy",
        None),
    "ObjectInfo.ExperimentalDescription.Microscope": (
        "Acquisition_instrument.TEM.microscope",
        _get_microscope_name),
    "ObjectInfo.ExperimentalDescription.Mode": (
        "Acquisition_instrument.TEM.acquisition_mode",
        _get_simplified_mode),
    "ObjectInfo.ExperimentalDescription.Camera length_m": (
        "Acquisition_instrument.TEM.camera_length",
        lambda x: x * 1e3),
    "ObjectInfo.ExperimentalDescription.Magnification_x": (
        "Acquisition_instrument.TEM.magnification",
        None),
    "ObjectInfo.AcquireInfo.CameraNamePath": (
        "Acquisition_instrument.TEM.Detector.Camera.Name",
        None),
    "ObjectInfo.AcquireInfo.DwellTimePath": (
        "Acquisition_instrument.TEM.Detector.Camera.exposure",
        None),
    "ObjectInfo.ExperimentalDescription.Stage_A_deg": (
        "Acquisition_instrument.TEM.Stage.tilt_alpha",
        None),
    "ObjectInfo.ExperimentalDescription.Stage_B_deg": (
        "Acquisition_instrument.TEM.Stage.tilt_beta",
        None),
    "ObjectInfo.ExperimentalDescription.Stage_X_um": (
        "Acquisition_instrument.TEM.Stage.x",
        lambda x: x * 1e-3),
    "ObjectInfo.ExperimentalDescription.Stage_Y_um": (
        "Acquisition_instrument.TEM.Stage.y",
        lambda x: x * 1e-3),
    "ObjectInfo.ExperimentalDescription.Stage_Z_um": (
        "Acquisition_instrument.TEM.Stage.z",
        lambda x: x * 1e-3),
    "ObjectInfo.ExperimentalDescription.User": (
        "General.authors",
        None),
}
