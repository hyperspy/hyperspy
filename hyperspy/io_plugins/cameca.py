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


import os
import logging

import numpy as np

try:
    from collections import OrderedDict

    ordict = True
except ImportError:
    ordict = False
from hyperspy.misc.array_tools import sarray2dict

_logger = logging.getLogger(__name__)

im_extensions = ('im', 'IM')
im_info_extensions = ('im_chk', "IM_CHK")
# Plugin characteristics
# ----------------------
format_name = 'CAMECA'
description = 'Format used for the Cameca NanoSIMS'
full_support = False
# Recognised file extension
file_extensions = im_extensions
default_extension = 0

# Writing capabilities
writes = False


# ----------------------


def get_endian(file):
    """
    Check endian by seeing how large the value in bytes 8:12 are

    Parameters
    ----------
    file: file object

    Returns
    -------
    endian: string, either ">" (big-endian) or "<" (small-endian) depending on OS that saved the file

    """
    file.seek(8)
    header_size = np.fromfile(file,
                              dtype=np.dtype(">u4"),
                              count=1)
    if header_size < 2e6:
        endian = ">"  # (big-endian)
    else:
        endian = "<"  # (small-endian)
    return endian


def get_header_dtype_list(file, endian):
    """Parse header info from file

    Parameters
    ----------
    file: file object
    endian: string, either ">" (big-endian) or "<" (small-endian) depending on OS that saved the file
    Returns
    -------
    header: np.ndarray, dictionary-like object of image properties

    """


    # Read the first part of the header
    header_list1 = [
        ("release", endian + "u4"),
        ("analysis_type", endian + "u4"),
        ("header_size", endian + "u4"),
        ("sample_type", endian + "u4"),
        ("data_present", endian + "u4"),
        ("stage_position_x", endian + "i4"),
        ("stage_position_y", endian + "i4"),
        ("analysis_name", endian + "S32"),
        ("username", endian + "S16"),
        ("samplename", endian + "S16"),
        ("date", endian + "S16"),
        ("time", endian + "S16"),
        ("filename", endian + "S16"),
        ("analysis_duration", endian + "u4"),
        ("cycle_number", endian + "u4"),
        ("scantype", endian + "u4"),
        ("magnification", endian + "u2"),
        ("sizetype", endian + "u2"),
        ("size_detector", endian + "u2"),
        ("no_used", endian + "u2"),
        ("beam_blanking", endian + "u4"),
        ("pulverisation", endian + "u4"),
        ("pulve_duration", endian + "u4"),
        ("auto_cal_in_anal", endian + "u4"),
        ("autocal", endian + "S72"),
        ("sig_reference", endian + "u4"),
        ("sigref", endian + "S156"),
        ("number_of_masses", endian + "u4"),
    ]
    file.seek(0)
    header1 = np.fromfile(file,
                          dtype=np.dtype(header_list1),
                          count=1)

    # Once we know the type of the OffsetArrayOffset, we can continue reading
    # the 2nd part of the header
    file.seek(header1["header_size"] - 78)

    header_list2 = [
        ("width_pixels", endian + "u2"),
        ("height_pixels", endian + "u2"),
        ("pixel_size", endian + "u2"),
        ("number_of_images", endian + "u2"),
        ("number_of_planes", endian + "u2"),
        ("raster", endian + "u4"),
        ("nickname", endian + "S64"),
    ]

    header2 = np.fromfile(file,
                          dtype=np.dtype(header_list2),
                          count=1)

    header_list3 = [
        ("mass_name", endian + "S64"),
        ("placeholder2", endian + "S128"),
    ]


    if endian == ">": # big-endian
        offset = 452 + 8;
    elif endian == "<": # small-endian
        offset = 412 + 192 + 56

    print(endian)
    mass_names = []
    file.seek(offset + 56)
    for i in range(header1["number_of_masses"][0]):
        header3 = np.fromfile(file,
                              dtype=np.dtype(header_list3),
                              count=1)
        mass_name = header3["mass_name"][0].decode()
        mass_name = "".join(mass_name.split("\x00"))

        if mass_name == "":
            mass_name = "SE"

        mass_names.append(mass_name)

    print(mass_names)

    header1 = sarray2dict(header1)
    header2 = sarray2dict(header2)

    header = {}

    for key in header1:
        header[key] = header1[key]

    for key in header2:
        header[key] = header2[key]

    header["mass_names"] = mass_names

    file.seek(0)
    return header


def file_reader(filename, *args, **kwds):
    ext = os.path.splitext(filename)[1][1:]
    if ext in im_extensions:
        return im_reader(filename, *args, **kwds)
        # elif ext in emi_extensions:
        # return emi_reader(filename, *args, **kwds)


def im_reader(filename, *args, **kwds):
    """Reads the information from the file and returns it in the HyperSpy
    required format.

    """
    header, data = load_im_file(filename)

    # Image mode

    axes = []
    array_shape = []
    chk_exists = False
    if chk_exists == True:
        # set units based on that info
        units = "unitsfromchkfile"
    else:
        units = "um"

    # Z axis
    axes.append({
        'name': 'z',
        'offset': 0,
        'scale': 1,
        'units': "",
        'size': header["number_of_planes"],
    })
    array_shape.append(header["number_of_planes"])

    # Y axis
    axes.append({
        'name': 'y',
        'offset': 0,
        'scale': header["raster"] / header['height_pixels'],
        'units': units,
        'size': header["height_pixels"],
    })
    array_shape.append(header["height_pixels"])

    # X axis
    axes.append({
        'name': 'x',
        'offset': 0,
        'scale': header["raster"] / header['width_pixels'],
        'units': units,
        'size': header["width_pixels"],
    })

    array_shape.append(header['width_pixels'])

    # If the acquisition stops before finishing the job, the stored file will
    # contain only zeroes in all remaining slices. Better remove them.
    from pprint import pprint

    dictionary_list = []
    for i in range(header["number_of_masses"]):
        dc = data[i]
        print(dc.shape)
        # Set? original_metadata = {}

        dictionary = {
            'data': dc,
            'metadata': {
                'General': {
                    'title': header["mass_names"][i],
                    'original_filename': header["filename"],
                },
                "Signal": {
                    'signal_type': "",
                },
            },
            'axes': axes,
        }
        dictionary_list.append(dictionary)
    # Return a list of dictionaries
    pprint(dictionary_list)
    return dictionary_list


def load_im_file(filename):
    _logger.info("Opening the file: %s", filename)
    with open(filename, 'rb') as f:
        # Check endian of bytes, as it depends on the OS that saved the file
        endian = get_endian(f)
        header = get_header_dtype_list(f, endian=endian)

        for key in header:

            if type(header[key]) == np.bytes_:
                header[key] = header[key].decode()

        # Read the first element of data offsets
        f.seek(header["header_size"])
        # Data can either be of data type uint16 or uint32 - maybe even uint64

        datadtype = endian + "u" + str(header["pixel_size"])

        data = np.fromfile(f,
                           dtype=datadtype,
                           count=header["number_of_masses"] * header["number_of_planes"] * header["width_pixels"] *
                                 header["height_pixels"])

        #data = data[0]
        data = data.astype(float)

        # Reshape into shape (images*planes, width, height)
        data = data.reshape(header["number_of_masses"] * header["number_of_planes"], header["width_pixels"],
                            header["height_pixels"])


        data = np.array([data[i::header["number_of_masses"]] for i in range(header["number_of_masses"])])

    return header, data
