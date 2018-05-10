# -*- coding: utf-8 -*-
# Copyright 2007-2018 The HyperSpy developers
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

import logging
import numpy as np
import dask.array as da


_logger = logging.getLogger(__name__)
# Plugin characteristics
# ----------------------
format_name = 'Merlin binary'
description = 'Read data from Merlin binary files'
full_support = False
# Recognised file extension
file_extensions = ['mib', 'MIB']
default_extension = 0

# Writing capabilities:
writes = False


def _get_dtype_from_header_string(header_string):
    header_split_list = header_string.split(",")
    dtype_string = header_split_list[6]
    if dtype_string == 'U16':
        dtype = ">u2"
    else:
        print("dtype {0} not recognized, trying unsigned 16 bit".format(dtype_string))
        dtype = ">u2"
    return dtype

def _get_detector_pixel_size(header_string):
    header_split_list = header_string.split(",")
    det_x_string = header_split_list[4]
    det_y_string = header_split_list[5]
    try:
        det_x = int(det_x_string)
        det_y = int(det_y_string)
    except:
        print("detector size strings {0} and {1} not recognized, trying 256 x 256".format(
            det_x_string, det_y_string))
        det_x, det_y = 256, 256
    if det_x == 256:
        det_x_value = det_x
    elif det_x == 512:
        det_x_value = det_x
    else:
        print("detector x size {0} not recognized, trying 256".format(det_x))
        det_x_value = 256
    if det_y == 256:
        det_y_value = det_y
    elif det_y == 512:
        det_y_value = det_y
    else:
        print("detector y size {0} not recognized, trying 256".format(det_y))
        det_y_value = 256
    return(det_x_value, det_y_value)


def file_reader(filename, probe_x, probe_y, lazy=True):
    _logger.debug("Reading Merlin binary file: %s" % filename)

    f = open(filename, 'r')
    header_string = f.read(50)
    f.close()
    dtype = _get_dtype_from_header_string(header_string)
    det_x, det_y = _get_detector_pixel_size(header_string)

    value_between_frames = 192
    total_frame_size = det_x*det_y + value_between_frames
    flyback_pixels = 1

    data_array = np.memmap(filename, dtype=dtype, mode='r',)
    number_of_values = (total_frame_size*probe_x*(probe_y + flyback_pixels))
    data_array = data_array[:number_of_values]
    data_array = data_array.reshape(probe_y, probe_x + flyback_pixels, total_frame_size)
    data_array = data_array[:, :, value_between_frames:]
    data_array = data_array.reshape(probe_y, probe_x + flyback_pixels, det_y, det_x)
    data_array = data_array[:, :probe_x, :, :]

    data = da.from_array(data_array, chunks=(16, 16, 16, 16))

    # Create the axis objects for each axis
    names = ['Probe x', 'Probe y', 'Detector x', 'Detector y']
    navigate = [True, True, False, False]
    data_shape = [probe_x, probe_y, det_x, det_y]
    axes = [{'size': shape,
             'name': name,
             'scale': 1,
             'offset': 0.0,
             'units': '',
             'navigate': nav}
            for shape, name, nav in zip(data_shape, names, navigate)]

    dictionary = {'data': data,
                  'axes': axes}

    return [dictionary, ]
