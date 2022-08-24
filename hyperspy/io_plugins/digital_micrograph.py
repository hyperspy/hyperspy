# -*- coding: utf-8 -*-
# Copyright 2010 Stefano Mazzucco
# Copyright 2011-2020 The HyperSpy developers
#
# This file is part of HyperSpy. It is a fork of the original PIL dm3 plugin
# written by Stefano Mazzucco.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

# Plugin to read the Gatan Digital Micrograph(TM) file format


import os
import logging
import dateutil.parser

import numpy as np
import traits.api as t
from copy import deepcopy

import hyperspy.misc.io.utils_readfile as iou
from hyperspy.exceptions import DM3TagIDError, DM3DataTypeError, DM3TagTypeError
import hyperspy.misc.io.tools
from hyperspy.misc.utils import DictionaryTreeBrowser
from hyperspy.docstrings.signal import OPTIMIZE_ARG


_logger = logging.getLogger(__name__)


# Plugin characteristics
# ----------------------
format_name = 'Digital Micrograph'
description = 'Read data from Gatan Digital Micrograph (TM) files'
full_support = False
# Recognised file extension
file_extensions = ('dm3', 'DM3', 'dm4', 'DM4')
default_extension = 0
# Writing capabilities
writes = False
non_uniform_axis = False
# ----------------------


class DigitalMicrographReader(object):

    """ Class to read Gatan Digital Micrograph (TM) files.

    Currently it supports versions 3 and 4.

    Attributes
    ----------
    dm_version, endian, tags_dict

    Methods
    -------
    parse_file, parse_header, get_image_dictionaries

    """

    _complex_type = (15, 18, 20)
    simple_type = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

    def __init__(self, f):
        self.dm_version = None
        self.endian = None
        self.tags_dict = None
        self.f = f

    def parse_file(self):
        self.f.seek(0)
        self.parse_header()
        self.tags_dict = {"root": {}}
        number_of_root_tags = self.parse_tag_group()[2]
        _logger.info('Total tags in root group: %s', number_of_root_tags)
        self.parse_tags(
            number_of_root_tags,
            group_name="root",
            group_dict=self.tags_dict)

    def parse_header(self):
        self.dm_version = iou.read_long(self.f, "big")
        if self.dm_version not in (3, 4):
            raise NotImplementedError(
                "Currently we only support reading DM versions 3 and 4 but "
                "this file "
                "seems to be version %s " % self.dm_version)
        filesizeB = self.read_l_or_q(self.f, "big")
        is_little_endian = iou.read_long(self.f, "big")

        _logger.info('DM version: %i', self.dm_version)
        _logger.info('size %i B', filesizeB)
        _logger.info('Is file Little endian? %s', bool(is_little_endian))
        if bool(is_little_endian):
            self.endian = 'little'
        else:
            self.endian = 'big'

    def parse_tags(self, ntags, group_name='root', group_dict=None):
        """Parse the DM file into a dictionary.

        """
        if group_dict is None: #pragma: no cover
            group_dict = {}    # currently this default value is not used in this script
        unnammed_data_tags = 0
        unnammed_group_tags = 0
        for tag in range(ntags):
            _logger.debug('Reading tag name at address: %s', self.f.tell())
            tag_header = self.parse_tag_header()
            tag_name = tag_header['tag_name']

            skip = True if (group_name == "ImageData" and
                            tag_name == "Data") else False
            _logger.debug('Tag name: %s', tag_name[:20])
            _logger.debug('Tag ID: %s', tag_header['tag_id'])

            if tag_header['tag_id'] == 21:  # it's a TagType (DATA)
                if not tag_name:
                    tag_name = 'Data%i' % unnammed_data_tags
                    unnammed_data_tags += 1

                _logger.debug('Reading data tag at address: %s', self.f.tell())

                # Start reading the data
                # Raises IOError if it is wrong
                self.check_data_tag_delimiter()
                infoarray_size = self.read_l_or_q(self.f, 'big')
                _logger.debug("Infoarray size: %s", infoarray_size)
                if infoarray_size == 1:  # Simple type
                    _logger.debug("Reading simple data")
                    etype = self.read_l_or_q(self.f, "big")
                    data = self.read_simple_data(etype)
                elif infoarray_size == 2:  # String
                    _logger.debug("Reading string")
                    enctype = self.read_l_or_q(self.f, "big")
                    if enctype != 18:
                        raise IOError("Expected 18 (string), got %i" % enctype)
                    string_length = self.parse_string_definition()
                    data = self.read_string(string_length, skip=skip)
                elif infoarray_size == 3:  # Array of simple type
                    _logger.debug("Reading simple array")
                    # Read array header
                    enctype = self.read_l_or_q(self.f, "big")
                    if enctype != 20:  # Should be 20 if it is an array
                        raise IOError("Expected 20 (string), got %i" % enctype)
                    size, enc_eltype = self.parse_array_definition()
                    data = self.read_array(size, enc_eltype, skip=skip)
                elif infoarray_size > 3:
                    enctype = self.read_l_or_q(self.f, "big")
                    if enctype == 15:  # It is a struct
                        _logger.debug("Reading struct")
                        definition = self.parse_struct_definition()
                        _logger.debug("Struct definition %s", definition)
                        data = self.read_struct(definition, skip=skip)
                    elif enctype == 20:  # It is an array of complex type
                        # Read complex array info
                        # The structure is
                        # 20 <4>, ?  <4>, enc_dtype <4>, definition <?>,
                        # size <4>
                        enc_eltype = self.read_l_or_q(self.f, "big")
                        if enc_eltype == 15:  # Array of structs
                            _logger.debug("Reading array of structs")
                            definition = self.parse_struct_definition()
                            size = self.read_l_or_q(self.f, "big")
                            _logger.debug("Struct definition: %s", definition)
                            _logger.debug("Array size: %s", size)
                            data = self.read_array(
                                size=size,
                                enc_eltype=enc_eltype,
                                extra={"definition": definition},
                                skip=skip)
                        elif enc_eltype == 18:  # Array of strings
                            _logger.debug("Reading array of strings")
                            string_length = \
                                self.parse_string_definition()
                            size = self.read_l_or_q(self.f, "big")
                            data = self.read_array(
                                size=size,
                                enc_eltype=enc_eltype,
                                extra={"length": string_length},
                                skip=skip)
                        elif enc_eltype == 20:  # Array of arrays
                            _logger.debug("Reading array of arrays")
                            el_length, enc_eltype = \
                                self.parse_array_definition()
                            size = self.read_l_or_q(self.f, "big")
                            data = self.read_array(
                                size=size,
                                enc_eltype=enc_eltype,
                                extra={"size": el_length},
                                skip=skip)

                else:  # Infoarray_size < 1
                    raise IOError("Invalided infoarray size ", infoarray_size)

                group_dict[tag_name] = data

            elif tag_header['tag_id'] == 20:  # it's a TagGroup (GROUP)
                if not tag_name:
                    tag_name = 'TagGroup%i' % unnammed_group_tags
                    unnammed_group_tags += 1
                _logger.debug(
                    'Reading Tag group at address: %s',
                    self.f.tell())
                ntags = self.parse_tag_group(size=True)[2]
                group_dict[tag_name] = {}
                self.parse_tags(
                    ntags=ntags,
                    group_name=tag_name,
                    group_dict=group_dict[tag_name])
            else:
                _logger.debug('File address:', self.f.tell())
                raise DM3TagIDError(tag_header['tag_id'])

    def get_data_reader(self, enc_dtype):
        # _data_type dictionary.
        # The first element of the InfoArray in the TagType
        # will always be one of _data_type keys.
        # the tuple reads: ('read bytes function', 'number of bytes', 'type')

        dtype_dict = {
            2: (iou.read_short, 2, 'h'),
            3: (iou.read_long, 4, 'l'),
            4: (iou.read_ushort, 2, 'H'),  # dm3 uses ushorts for unicode chars
            5: (iou.read_ulong, 4, 'L'),
            6: (iou.read_float, 4, 'f'),
            7: (iou.read_double, 8, 'd'),
            8: (iou.read_boolean, 1, 'B'),
            # dm3 uses chars for 1-Byte signed integers
            9: (iou.read_char, 1, 'b'),
            10: (iou.read_byte, 1, 'b'),   # 0x0a
            11: (iou.read_long_long, 8, 'q'),  # long long, new in DM4
            # unsigned long long, new in DM4
            12: (iou.read_ulong_long, 8, 'Q'),
            15: (self.read_struct, None, 'struct',),  # 0x0f
            18: (self.read_string, None, 'c'),  # 0x12
            20: (self.read_array, None, 'array'),  # 0x14
        }
        return dtype_dict[enc_dtype]

    def skipif4(self, n=1):
        if self.dm_version == 4:
            self.f.seek(4 * n, 1)

    @property
    def read_l_or_q(self):
        if self.dm_version == 4:
            return iou.read_long_long
        else:
            return iou.read_long

    def parse_array_definition(self):
        """Reads and returns the element type and length of the array.

        The position in the file must be just after the
        array encoded dtype.

        """
        enc_eltype = self.read_l_or_q(self.f, "big")
        length = self.read_l_or_q(self.f, "big")
        return length, enc_eltype

    def parse_string_definition(self):
        """Reads and returns the length of the string.

        The position in the file must be just after the
        string encoded dtype.
        """
        return self.read_l_or_q(self.f, "big")

    def parse_struct_definition(self):
        """Reads and returns the struct definition tuple.

        The position in the file must be just after the
        struct encoded dtype.

        """
        length = self.read_l_or_q(self.f, "big")
        nfields = self.read_l_or_q(self.f, "big")
        definition = ()
        for ifield in range(nfields):
            length2 = self.read_l_or_q(self.f, "big")
            definition += (self.read_l_or_q(self.f, "big"),)

        return definition

    def read_simple_data(self, etype):
        """Parse the data of the given DM3 file f
        with the given endianness (byte order).
        The infoArray iarray specifies how to read the data.
        Returns the tuple (file address, data).
        The tag data is stored in the platform's byte order:
        'little' endian for Intel, PC; 'big' endian for Mac, Motorola.
        If skip != 0 the data is actually skipped.
        """
        data = self.get_data_reader(etype)[0](self.f, self.endian)
        if isinstance(data, str):
            data = hyperspy.misc.utils.ensure_unicode(data)
        return data

    def read_string(self, length, skip=False):
        """Read a string defined by the infoArray iarray from file f with a
        given endianness (byte order). endian can be either 'big' or 'little'.

        If it's a tag name, each char is 1-Byte;
        if it's a tag data, each char is 2-Bytes Unicode,
        """
        if skip is True:
            offset = self.f.tell()
            self.f.seek(length, 1)
            return {'size': length,
                    'size_bytes': size_bytes,
                    'offset': offset,
                    'endian': self.endian, }
        data = b''
        if self.endian == 'little':
            s = iou.L_char
        elif self.endian == 'big':
            s = iou.B_char
        for char in range(length):
            data += s.unpack(self.f.read(1))[0]
        try:
            data = data.decode('utf8')
        except BaseException:
            # Sometimes the dm3 file strings are encoded in latin-1
            # instead of utf8
            data = data.decode('latin-1', errors='ignore')
        return data

    def read_struct(self, definition, skip=False):
        """Read a struct, defined by iarray, from file f
        with a given endianness (byte order).
        Returns a list of 2-tuples in the form
        (fieldAddress, fieldValue).
        endian can be either 'big' or 'little'.

        """
        field_value = []
        size_bytes = 0
        offset = self.f.tell()
        for dtype in definition:
            if dtype in self.simple_type:
                if skip is False:
                    data = self.get_data_reader(dtype)[0](self.f, self.endian)
                    field_value.append(data)
                else:
                    sbytes = self.get_data_reader(dtype)[1]
                    self.f.seek(sbytes, 1)
                    size_bytes += sbytes
            else:
                raise DM3DataTypeError(dtype)
        if skip is False:
            return tuple(field_value)
        else:
            return {'size': len(definition),
                    'size_bytes': size_bytes,
                    'offset': offset,
                    'endian': self.endian, }

    def read_array(self, size, enc_eltype, extra=None, skip=False):
        """Read an array, defined by iarray, from file f
        with a given endianness (byte order).
        endian can be either 'big' or 'little'.

        """
        eltype = self.get_data_reader(enc_eltype)[0]  # same for all elements
        if skip is True:
            if enc_eltype not in self._complex_type:
                size_bytes = self.get_data_reader(enc_eltype)[1] * size
                data = {"size": size,
                        "endian": self.endian,
                        "size_bytes": size_bytes,
                        "offset": self.f.tell()}
                self.f.seek(size_bytes, 1)  # Skipping data
            else:
                data = eltype(skip=skip, **extra)
                self.f.seek(data['size_bytes'] * (size - 1), 1)
                data['size'] = size
                data['size_bytes'] *= size
        else:
            if enc_eltype in self.simple_type:  # simple type
                data = [eltype(self.f, self.endian)
                        for element in range(size)]
                if enc_eltype == 4 and data:  # it's actually a string
                    data = "".join([chr(i) for i in data])
            elif enc_eltype in self._complex_type:
                data = [eltype(**extra)
                        for element in range(size)]
        return data

    def parse_tag_group(self, size=False):
        """Parse the root TagGroup of the given DM3 file f.
        Returns the tuple (is_sorted, is_open, n_tags).
        endian can be either 'big' or 'little'.
        """
        is_sorted = iou.read_byte(self.f, "big")
        is_open = iou.read_byte(self.f, "big")
        if self.dm_version == 4 and size:
            # Just guessing that this is the size
            size = self.read_l_or_q(self.f, "big")
        n_tags = self.read_l_or_q(self.f, "big")
        return bool(is_sorted), bool(is_open), n_tags

    def parse_tag_header(self):
        tag_id = iou.read_byte(self.f, "big")
        tag_name_length = iou.read_short(self.f, "big")
        tag_name = self.read_string(tag_name_length)
        return {'tag_id': tag_id,
                'tag_name_length': tag_name_length,
                'tag_name': tag_name, }

    def check_data_tag_delimiter(self):
        self.skipif4(2)
        delimiter = self.read_string(4)
        if delimiter != '%%%%':
            raise DM3TagTypeError(delimiter)

    def get_image_dictionaries(self):
        """Returns the image dictionaries of all images in the file except
        the thumbnails.

        Returns
        -------
        dict, None

        """
        if 'ImageList' not in self.tags_dict:
            return None
        if "Thumbnails" in self.tags_dict:
            thumbnail_idx = [tag['ImageIndex'] for key, tag in
                             self.tags_dict['Thumbnails'].items()]
        else:
            thumbnail_idx = []
        images = [image for key, image in
                  self.tags_dict['ImageList'].items()
                  if not int(key.replace("TagGroup", "")) in
                  thumbnail_idx]
        return images


class ImageObject(object):

    def __init__(self, imdict, file, order="C", record_by=None):
        self.imdict = DictionaryTreeBrowser(imdict)
        self.file = file
        self._order = order if order else "C"
        self._record_by = record_by

    @property
    def shape(self):
        dimensions = self.imdict.ImageData.Dimensions
        shape = tuple([dimension[1] for dimension in dimensions])
        return shape[::-1]  # DM uses image indexing X, Y, Z...

    # For some image stacks created using plugins in Digital Micrograph
    # the metadata under Calibrations.Dimension would not reflect the
    # actual dimensions in the dataset, leading to these images not
    # loading properly. To allow HyperSpy to load these files, any missing
    # dimensions in the metadata is appended with "dummy" values.
    # This is done for the offsets, scales and units properties, using
    # the len_diff variable
    @property
    def offsets(self):
        dimensions = self.imdict.ImageData.Calibrations.Dimension
        len_diff = len(self.shape) - len(dimensions)
        origins = np.array([dimension[1].Origin for dimension in dimensions])
        origins = np.append(origins, (0.0,) * len_diff)
        return -1 * origins[::-1] * self.scales

    @property
    def scales(self):
        dimensions = self.imdict.ImageData.Calibrations.Dimension
        len_diff = len(self.shape) - len(dimensions)
        scales = np.array([dimension[1].Scale for dimension in dimensions])
        scales = np.append(scales, (1.0,) * len_diff)
        return scales[::-1]

    @property
    def units(self):
        dimensions = self.imdict.ImageData.Calibrations.Dimension
        len_diff = len(self.shape) - len(dimensions)
        return (tuple([dimension[1].Units
                       if dimension[1].Units else ""
                       for dimension in dimensions]) + ('',) * len_diff)[::-1]

    @property
    def names(self):
        names = [t.Undefined] * len(self.shape)
        indices = list(range(len(self.shape)))

        if self.signal_type == "EELS":
            if "eV" in self.units:
                names[indices.pop(self.units.index("eV"))] = "Energy loss"
        elif self.signal_type in ("EDS", "EDX"):
            if "keV" in self.units:
                names[indices.pop(self.units.index("keV"))] = "Energy"
        elif self.signal_type == "CL":
            if "nm" in self.units:
                names[indices.pop(self.units.index("nm"))] = "Wavelength"
        for index, name in zip(indices[::-1], ("x", "y", "z")):
            names[index] = name
        return names

    @property
    def title(self):
        title = self.imdict.get_item("Name", "")
        # ``if title else ""`` below is there to account for when Name
        # contains an empty list.
        # See https://github.com/hyperspy/hyperspy/issues/1937
        return title if title else ""

    @property
    def record_by(self):
        if self._record_by is not None:
            return self._record_by
        if len(self.scales) == 1:
            return "spectrum"
        elif (('ImageTags.Meta_Data.Format' in self.imdict and
               self.imdict.ImageTags.Meta_Data.Format in ("Spectrum image",
                                                          "Spectrum")) or (
                "ImageTags.spim" in self.imdict)) and len(self.scales) == 2:
            return "spectrum"
        else:
            return "image"

    @property
    def to_spectrum(self):
        if (('ImageTags.Meta_Data.Format' in self.imdict and
                self.imdict.ImageTags.Meta_Data.Format == "Spectrum image") or
                ("ImageTags.spim" in self.imdict)) and len(self.scales) > 2:
            return True
        else:
            return False

    @property
    def order(self):
        return self._order

    @property
    def intensity_calibration(self):
        ic = self.imdict.ImageData.Calibrations.Brightness.as_dictionary()
        if not ic['Units']:
            ic['Units'] = ""
        return ic

    @property
    def dtype(self):
        # Signal2D data types (Signal2D Object chapter on DM help)#
        # key = DM data type code
        # value = numpy data type
        if self.imdict.ImageData.DataType == 4:
            raise NotImplementedError(
                "Reading data of this type is not implemented.")

        imdtype_dict = {
            0: 'not_implemented',  # null
            1: 'int16',
            2: 'float32',
            3: 'complex64',
            5: 'float32',  # not numpy: 8-Byte packed complex (FFT data)
            6: 'uint8',
            7: 'int32',
            8: np.dtype({'names': ['B', 'G', 'R', 'A'],
                         'formats': ['u1', 'u1', 'u1', 'u1']}),
            9: 'int8',
            10: 'uint16',
            11: 'uint32',
            12: 'float64',
            13: 'complex128',
            14: 'bool',
            23: np.dtype({'names': ['B', 'G', 'R', 'A'],
                          'formats': ['u1', 'u1', 'u1', 'u1']}),
            27: 'complex64',  # not numpy: 8-Byte packed complex (FFT data)
            28: 'complex128',  # not numpy: 16-Byte packed complex (FFT data)
        }
        return imdtype_dict[self.imdict.ImageData.DataType]

    @property
    def signal_type(self):
        md_signal = self.imdict.get_item('ImageTags.Meta_Data.Signal', "")
        if md_signal == 'X-ray':
            return "EDS_TEM"
        elif md_signal == 'CL'  or 'ImageTags.Acquisition.Monarc_Spectrometer' in self.imdict:
            return "CL"
        # 'ImageTags.spim.eels' is Orsay's tag group
        elif md_signal == 'EELS' or 'ImageTags.spim.eels' in self.imdict:
            return "EELS"
        else:
            return ""

    def _get_data_array(self):
        need_to_close = False
        if self.file.closed:
            self.file = open(self.filename, "rb")
            need_to_close = True
        self.file.seek(self.imdict.ImageData.Data.offset)
        count = self.imdict.ImageData.Data.size
        if self.imdict.ImageData.DataType in (27, 28):  # Packed complex
            count = int(count / 2)
        data = np.fromfile(self.file,
                           dtype=self.dtype,
                           count=count)
        if need_to_close:
            self.file.close()
        return data

    @property
    def size(self):
        if self.imdict.ImageData.DataType in (27, 28):  # Packed complex
            if self.imdict.ImageData.Data.size % 2:
                raise IOError(
                    "ImageData.Data.size should be an even integer for "
                    "this datatype.")
            else:
                return int(self.imdict.ImageData.Data.size / 2)
        else:
            return self.imdict.ImageData.Data.size

    def get_data(self):
        if isinstance(self.imdict.ImageData.Data, np.ndarray):
            return self.imdict.ImageData.Data
        data = self._get_data_array()
        if self.imdict.ImageData.DataType in (27, 28):  # New packed complex
            return self.unpack_new_packed_complex(data)
        elif self.imdict.ImageData.DataType == 5:  # Old packed compled
            return self.unpack_packed_complex(data)
        elif self.imdict.ImageData.DataType in (8, 23):  # ABGR
            # Reorder the fields
            data = data[['R', 'G', 'B', 'A']].astype(
                [('R', 'u1'), ('G', 'u1'), ('B', 'u1'), ('A', 'u1')])
        return data.reshape(self.shape, order=self.order)

    def unpack_new_packed_complex(self, data):
        packed_shape = (self.shape[0], int(self.shape[1] / 2 + 1))
        data = data.reshape(packed_shape, order=self.order)
        return np.hstack((data[:, ::-1], np.conjugate(data[:, 1:-1])))

    def unpack_packed_complex(self, tmpdata):
        shape = self.shape
        if shape[0] != shape[1] or len(shape) > 2:
            raise IOError(
                'Unable to read this DM file in packed complex format. '
                'Please report the issue to the HyperSpy developers providing '
                'the file if possible')
        N = int(self.shape[0] / 2)      # think about a 2Nx2N matrix
        # create an empty 2Nx2N ndarray of complex
        data = np.zeros(shape, dtype="complex64")

        # fill in the real values:
        data[N, 0] = tmpdata[0]
        data[0, 0] = tmpdata[1]
        data[N, N] = tmpdata[2 * N ** 2]  # Nyquist frequency
        data[0, N] = tmpdata[2 * N ** 2 + 1]  # Nyquist frequency

        # fill in the non-redundant complex values:
        # top right quarter, except 1st column
        for i in range(N):  # this could be optimized
            start = 2 * i * N + 2
            stop = start + 2 * (N - 1) - 1
            step = 2
            realpart = tmpdata[start:stop:step]
            imagpart = tmpdata[start + 1:stop + 1:step]
            data[i, N + 1:2 * N] = realpart + imagpart * 1j
        # 1st column, bottom left quarter
        start = 2 * N
        stop = start + 2 * N * (N - 1) - 1
        step = 2 * N
        realpart = tmpdata[start:stop:step]
        imagpart = tmpdata[start + 1:stop + 1:step]
        data[N + 1:2 * N, 0] = realpart + imagpart * 1j
        # 1st row, bottom right quarter
        start = 2 * N ** 2 + 2
        stop = start + 2 * (N - 1) - 1
        step = 2
        realpart = tmpdata[start:stop:step]
        imagpart = tmpdata[start + 1:stop + 1:step]
        data[N, N + 1:2 * N] = realpart + imagpart * 1j
        # bottom right quarter, except 1st row
        start = stop + 1
        stop = start + 2 * N * (N - 1) - 1
        step = 2
        realpart = tmpdata[start:stop:step]
        imagpart = tmpdata[start + 1:stop + 1:step]
        complexdata = realpart + imagpart * 1j
        data[
            N +
            1:2 *
            N,
            N:2 *
            N] = complexdata.reshape(
            N -
            1,
            N,
            order=self.order)

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
        data[N + 1:2 * N, 1:N] = np.conjugate(data[-N - 1:-2 * N:-1, -1:-N:-1])

        return data

    def get_axes_dict(self):
        return [{'name': name,
                 'size': size,
                 'index_in_array': i,
                 'scale': scale,
                 'offset': offset,
                 'units': str(units), }
                for i, (name, size, scale, offset, units) in enumerate(
                    zip(self.names, self.shape, self.scales, self.offsets,
                        self.units))]

    def get_metadata(self, metadata=None):
        if metadata is None:
            metadata = {}
        if "General" not in metadata:
            metadata['General'] = {}
        if "Signal" not in metadata:
            metadata['Signal'] = {}
        metadata['General']['title'] = self.title
        metadata["Signal"]['record_by'] = self.record_by
        metadata["Signal"]['signal_type'] = self.signal_type
        return metadata

    def _get_quantity(self, units):
        quantity = "Intensity"
        if len(units) == 0:
            units = ""
        elif units == 'e-':
            units = "Counts"
            quantity = "Electrons"
        if self.signal_type == 'EDS_TEM':
            quantity = "X-rays"
        if len(units) != 0:
            units = " (%s)" % units
        return "%s%s" % (quantity, units)

    def _get_mode(self, mode):
        if 'STEM' in mode:
            return 'STEM'
        elif 'SEM' in mode:
            return 'SEM'
        else:
            return 'TEM'

    def _get_time(self, time):
        try:
            dt = dateutil.parser.parse(time)
            return dt.time().isoformat()
        except BaseException:
            _logger.warning(f"Time string '{time}' could not be parsed.")

    def _get_date(self, date):
        try:
            dt = dateutil.parser.parse(date)
            return dt.date().isoformat()
        except BaseException:
            _logger.warning(f"Date string '{date}' could not be parsed.")

    def _get_microscope_name(self, ImageTags):
        locations = (
            "Session_Info.Microscope",
            "Microscope_Info.Name",
            "Microscope_Info.Microscope",
        )
        for loc in locations:
            mic = ImageTags.get_item(loc)
            if mic and mic != "[]":
                return mic
        _logger.info("Microscope name not present")
        return None

    def _parse_string(self, tag, convert_to_float=False, tag_name=None):
        if len(tag) == 0:
            return None
        elif convert_to_float:
            try:
                return float(tag)
            # In case the string can't be converted to float
            except BaseException:
                if tag_name is None:
                    warning = "Metadata could not be parsed."
                else:
                    warning = f"Metadata '{tag_name}' could not be parsed."
                _logger.warning(warning)
                return None
        else:
            return tag

    def _get_exposure_time(self, tags):
        # for GMS 2 and quantum/enfinium, the  "Integration time (s)" tag is
        # only present for single spectrum acquisition;  for maps we need to
        # compute exposure * number of frames
        # same holds for some types of CL measurements
        if 'Integration_time_s' in tags.keys():
            return float(tags["Integration_time_s"])
        elif 'Exposure_s' in tags.keys():
            frame_number = 1
            if "Number_of_frames" in tags.keys():
                frame_number = float(tags["Number_of_frames"])
            return float(tags["Exposure_s"]) * frame_number
        else:
            _logger.info("EELS/CL exposure time can't be read.")

    def _get_CL_detector_type(self, tags):
        if "Acquisition_Mode" in tags and tags["Acquisition_Mode"] == "Parallel dispersive":
            return "CCD"
        elif "Acquisition_Mode" in tags and tags["Acquisition_Mode"] == "Serial dispersive":
            return "PMT"
        else:
            _logger.info("CL detector type can't be read.")

    def get_mapping(self):
        if 'source' in self.imdict.ImageTags.keys():
            # For stack created with the stack builder plugin
            tags_path = 'ImageList.TagGroup0.ImageTags.source.Tags at creation'
            image_tags_dict = self.imdict.ImageTags.source['Tags at creation']
        else:
            # Standard tags
            tags_path = 'ImageList.TagGroup0.ImageTags'
            image_tags_dict = self.imdict.ImageTags
        is_scanning = "DigiScan" in image_tags_dict.keys()
        # check if instrument is SEM or TEM 
        if "Microscope Info" in self.imdict.ImageTags and "Illumination Mode" in self.imdict.ImageTags["Microscope Info"]:
            microscope = "SEM" if self._get_mode(self.imdict.ImageTags["Microscope Info"]["Illumination Mode"]) == "SEM" else "TEM"
        else:
            microscope = "TEM"
        mapping = {
            "{}.DataBar.Acquisition Date".format(tags_path): (
                "General.date", self._get_date),
            "{}.DataBar.Acquisition Time".format(tags_path): (
                "General.time", self._get_time),
            "{}.Microscope Info.Voltage".format(tags_path): (
                "Acquisition_instrument.%s.beam_energy" % microscope, lambda x: x / 1e3),
            "{}.Microscope Info.Stage Position.Stage Alpha".format(tags_path): (
                "Acquisition_instrument.%s.Stage.tilt_alpha" % microscope, None),
            "{}.Microscope Info.Stage Position.Stage Beta".format(tags_path): (
                "Acquisition_instrument.%s.Stage.tilt_beta" % microscope, None),
            "{}.Microscope Info.Stage Position.Stage X".format(tags_path): (
                "Acquisition_instrument.%s.Stage.x" % microscope, lambda x: x * 1e-3),
            "{}.Microscope Info.Stage Position.Stage Y".format(tags_path): (
                "Acquisition_instrument.%s.Stage.y" % microscope, lambda x: x * 1e-3),
            "{}.Microscope Info.Stage Position.Stage Z".format(tags_path): (
                "Acquisition_instrument.%s.Stage.z" % microscope, lambda x: x * 1e-3),
            "{}.Microscope Info.Illumination Mode".format(tags_path): (
                "Acquisition_instrument.%s.acquisition_mode" % microscope, self._get_mode),
            "{}.Microscope Info.Probe Current (nA)".format(tags_path): (
                "Acquisition_instrument.%s.beam_current" % microscope, None),
            "{}.Session Info.Operator".format(tags_path): (
                "General.authors", self._parse_string),
            "{}.Session Info.Specimen".format(tags_path): (
                "Sample.description", self._parse_string),
        }

        if "Microscope_Info" in image_tags_dict.keys():
            is_TEM = is_diffraction = None
            if "Illumination_Mode" in image_tags_dict['Microscope_Info'].keys(
            ):
                is_TEM = (
                    'TEM' == image_tags_dict.Microscope_Info.Illumination_Mode)
            if "Imaging_Mode" in image_tags_dict['Microscope_Info'].keys():
                is_diffraction = (
                    'DIFFRACTION' == image_tags_dict.Microscope_Info.Imaging_Mode)

            if is_TEM:
                if is_diffraction:
                    mapping.update({
                        "{}.Microscope Info.Indicated Magnification".format(tags_path): (
                            "Acquisition_instrument.TEM.camera_length",
                            None),
                    })
                else:
                    mapping.update({
                        "{}.Microscope Info.Indicated Magnification".format(tags_path): (
                            "Acquisition_instrument.TEM.magnification",
                            None),
                    })
            else:
                mapping.update({
                    "{}.Microscope Info.STEM Camera Length".format(tags_path): (
                        "Acquisition_instrument.%s.camera_length" % microscope,
                        None),
                    "{}.Microscope Info.Indicated Magnification".format(tags_path): (
                        "Acquisition_instrument.%s.magnification" % microscope,
                        None),
                })

            mapping.update({
                tags_path: (
                    "Acquisition_instrument.%s.microscope" % microscope,
                    self._get_microscope_name),
            })
        if "SI" in self.imdict.ImageTags.keys():
            mapping.update({
                "{}.SI.Acquisition.Date".format(tags_path): (
                    "General.date",
                    self._get_date),
                "{}.SI.Acquisition.Start time".format(tags_path): (
                    "General.time",
                    self._get_time),
            })
        if self.signal_type == "EELS":
            if is_scanning:
                mapped_attribute = 'dwell_time'
            else:
                mapped_attribute = 'exposure'
            mapping.update({
                "{}.EELS.Acquisition.Date".format(tags_path): (
                    "General.date",
                    self._get_date),
                "{}.EELS.Acquisition.Start time".format(tags_path): (
                    "General.time",
                    self._get_time),
                "{}.EELS.Experimental Conditions.".format(tags_path) +
                "Collection semi-angle (mrad)": (
                    "Acquisition_instrument.TEM.Detector.EELS.collection_angle",
                    None),
                "{}.EELS.Experimental Conditions.".format(tags_path) +
                "Convergence semi-angle (mrad)": (
                    "Acquisition_instrument.TEM.convergence_angle",
                    None),
                "{}.EELS.Acquisition".format(tags_path): (
                    "Acquisition_instrument.TEM.Detector.EELS.%s" % mapped_attribute,
                    self._get_exposure_time),
                "{}.EELS.Acquisition.Number_of_frames".format(tags_path): (
                    "Acquisition_instrument.TEM.Detector.EELS.frame_number",
                    None),
                "{}.EELS_Spectrometer.Aperture_label".format(tags_path): (
                    "Acquisition_instrument.TEM.Detector.EELS.aperture_size",
                    lambda string: self._parse_string(string.replace('mm', ''),
                                                      convert_to_float=True,
                                                      tag_name='Aperture_label')),
                "{}.EELS Spectrometer.Instrument name".format(tags_path): (
                    "Acquisition_instrument.TEM.Detector.EELS.spectrometer",
                    None),
            })
        elif self.signal_type == "EDS_TEM":
            mapping.update({
                "{}.EDS.Acquisition.Date".format(tags_path): (
                    "General.date",
                    self._get_date),
                "{}.EDS.Acquisition.Start time".format(tags_path): (
                    "General.time",
                    self._get_time),
                "{}.EDS.Detector_Info.Azimuthal_angle".format(tags_path): (
                    "Acquisition_instrument.TEM.Detector.EDS.azimuth_angle",
                    None),
                "{}.EDS.Detector_Info.Elevation_angle".format(tags_path): (
                    "Acquisition_instrument.TEM.Detector.EDS.elevation_angle",
                    None),
                "{}.EDS.Solid_angle".format(tags_path): (
                    "Acquisition_instrument.TEM.Detector.EDS.solid_angle",
                    None),
                "{}.EDS.Live_time".format(tags_path): (
                    "Acquisition_instrument.TEM.Detector.EDS.live_time",
                    None),
                "{}.EDS.Real_time".format(tags_path): (
                    "Acquisition_instrument.TEM.Detector.EDS.real_time",
                    None),
            })
        elif self.signal_type == "CL":
            mapping.update({
                "{}.CL.Acquisition.Date".format(tags_path): (
                    "General.date", self._get_date),
                "{}.CL.Acquisition.Start_time".format(tags_path): (
                    "General.time", self._get_time),
                "{}.Meta_Data.Acquisition_Mode".format(tags_path): (
                    "Acquisition_instrument.Spectrometer.acquisition_mode", None),
                "{}.Meta_Data.Format".format(tags_path): (
                    "Signal.format", None),
                "{}.Meta_Data".format(tags_path): (
                    "Acquisition_instrument.Detector.detector_type",
                    self._get_CL_detector_type),
                "{}.Acquisition.Monarc_Spectrometer.Grating".format(tags_path): (
                    "Acquisition_instrument.Spectrometer.Grating.groove_density",
                    lambda string: self._parse_string(string,
                                                      convert_to_float=True,
                                                      tag_name='Grating')),
                "{}.CL.Acquisition.Dispersion_grating_(lines/mm)".format(tags_path): (
                    "Acquisition_instrument.Spectrometer.Grating.groove_density", None),
                "{}.Acquisition.Monarc_Spectrometer.Slit_Width".format(tags_path): (
                    "Acquisition_instrument.Spectrometer.entrance_slit_width", None),
                "{}.Acquisition.Monarc_Spectrometer.Bandpass".format(tags_path): (
                    "Acquisition_instrument.Spectrometer.bandpass", None),
                # Parallel spectrum
                "{}.CL.Acquisition.Central_wavelength_(nm)".format(tags_path): (
                    "Acquisition_instrument.Spectrometer.central_wavelength", None),
                "{}.CL.Acquisition.Exposure_(s)".format(tags_path): (
                    "Acquisition_instrument.Detector.exposure_per_frame", None),
                "{}.CL.Acquisition.Number_of_frames".format(tags_path): (
                    "Acquisition_instrument.Detector.frames", None),
                "{}.CL.Acquisition".format(tags_path): (
                    "Acquisition_instrument.Detector.integration_time",
                    self._get_exposure_time),
                "{}.CL.Acquisition.Saturation_fraction".format(tags_path): (
                    "Acquisition_instrument.Detector.saturation_fraction", None),
                "{}.Acquisition.Parameters.High_Level.Binning".format(tags_path): (
                    "Acquisition_instrument.Detector.binning", None),
                "{}.Acquisition.Parameters.High_Level.CCD_Read_Area".format(tags_path): (
                    "Acquisition_instrument.Detector.sensor_roi", None),
                "{}.Acquisition.Parameters.High_Level.Processing".format(tags_path): (
                    "Acquisition_instrument.Detector.processing", None),
                "{}.Acquisition.Device.CCD.Pixel_Size_um".format(tags_path): (
                    "Acquisition_instrument.Detector.pixel_size", 
                    lambda x: x[0] if (isinstance(x, tuple) and x[0] == x[1]) else x),
                # Serial Spectrum
                "{}.CL.Acquisition.Acquisition_begin".format(tags_path): (
                    "General.date", self._get_date),
                "{}.CL.Acquisition.Dwell_time_(s)".format(tags_path): (
                    "Acquisition_instrument.Detector.integration_time", None),
                "{}.CL.Acquisition.Start_wavelength_(nm)".format(tags_path): (
                    "Acquisition_instrument.Spectrometer.start_wavelength", None),
                "{}.CL.Acquisition.Step-size_(nm)".format(tags_path): (
                    "Acquisition_instrument.Spectrometer.step_size", None),
                # PMT image
                "{}.Acquisition.Monarc_Spectrometer.PMT_HV".format(tags_path): (
                    "Acquisition_instrument.Detector.pmt_voltage", None),
                "{}.DigiScan.Sample Time".format(tags_path): (
                    "Acquisition_instrument.%s.dwell_time" % microscope,
                    lambda x: x / 1e6),
                # SI
                "{}.DataBar.Acquisition_Date".format(tags_path): (
                    "General.date", self._get_date),
                "{}.DataBar.Acquisition_Time".format(tags_path): (
                    "General.time", self._get_time),
                "{}.SI.Acquisition.SI_Application_Mode.Name".format(tags_path): (
                    "Acquisition_instrument.Spectrum_image.mode", None),
                "{}.SI.Acquisition.Artefact_Correction.Spatial_Drift.Periodicity".format(tags_path): (
                    "Acquisition_instrument.Spectrum_image.drift_correction_periodicity", None),
                "{}.SI.Acquisition.Artefact_Correction.Spatial_Drift.Units".format(tags_path): (
                    "Acquisition_instrument.Spectrum_image.drift_correction_units", None),
            })
        elif "DigiScan" in image_tags_dict.keys():
            mapping.update({
                "{}.DigiScan.Sample Time".format(tags_path): (
                    "Acquisition_instrument.%s.dwell_time" % microscope,
                    lambda x: x / 1e6),
            })
        else:
            mapping.update({
                "{}.Acquisition.Parameters.Detector.".format(tags_path) +
                "exposure_s": (
                    "Acquisition_instrument.TEM.Camera.exposure",
                    None),
            })
        mapping.update({
            "ImageList.TagGroup0.ImageData.Calibrations.Brightness.Units": (
                "Signal.quantity",
                self._get_quantity),
            "ImageList.TagGroup0.ImageData.Calibrations.Brightness.Scale": (
                "Signal.Noise_properties.Variance_linear_model.gain_factor",
                None),
            "ImageList.TagGroup0.ImageData.Calibrations.Brightness.Origin": (
                "Signal.Noise_properties.Variance_linear_model.gain_offset",
                None),
        })
        return mapping


def file_reader(filename, record_by=None, order=None, lazy=False,
                optimize=True):
    """Reads a DM3/4 file and loads the data into the appropriate class.
    data_id can be specified to load a given image within a DM3/4 file that
    contains more than one dataset.

    Parameters
    ----------
    record_by: Str
        One of: SI, Signal2D
    order : Str
        One of 'C' or 'F'
    lazy : bool, default False
        Load the signal lazily.
    %s
    """

    with open(filename, "rb") as f:
        dm = DigitalMicrographReader(f)
        dm.parse_file()
        images = [ImageObject(imdict, f, order=order, record_by=record_by)
                  for imdict in dm.get_image_dictionaries()]
        imd = []
        del dm.tags_dict['ImageList']
        dm.tags_dict['ImageList'] = {}

        for image in images:
            dm.tags_dict['ImageList'][
                'TagGroup0'] = image.imdict.as_dictionary()
            axes = image.get_axes_dict()
            mp = image.get_metadata()
            mp['General']['original_filename'] = os.path.split(filename)[1]
            post_process = []
            if image.to_spectrum is True:
                post_process.append(lambda s: s.to_signal1D(optimize=optimize))
            post_process.append(lambda s: s.squeeze())
            if lazy:
                image.filename = filename
                from dask.array import from_delayed
                import dask.delayed as dd
                val = dd(image.get_data, pure=True)()
                data = from_delayed(val, shape=image.shape,
                                    dtype=image.dtype)
            else:
                data = image.get_data()
            # in the event there are multiple signals contained within this
            # DM file, it is important to make a "deepcopy" of the metadata
            # and original_metadata, since they are changed in each iteration
            # of the "for image in images" loop, and using shallow copies
            # will result in the final signal's metadata being used for all
            # of the contained signals
            imd.append(
                {'data': data,
                 'axes': axes,
                 'metadata': deepcopy(mp),
                 'original_metadata': deepcopy(dm.tags_dict),
                 'post_process': post_process,
                 'mapping': image.get_mapping(),
                 })

    return imd
    file_reader.__doc__ %= (OPTIMIZE_ARG.replace('False', 'True'))
