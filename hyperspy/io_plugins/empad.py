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
import ast
import xml.etree.ElementTree as ET
import numpy as np
import logging
import pint

from hyperspy.misc.io.tools import convert_xml_to_dict


_logger = logging.getLogger(__name__)
_ureg = pint.UnitRegistry()


# Plugin characteristics
# ----------------------
format_name = 'empad'
description = ''
full_support = False
# Recognised file extension
file_extensions = ['xml', 'XML']
default_extension = 0

 # Writing capabilities:
writes = False


def _read_raw(info, fp, mmap_mode='c'):

    width = info['width']
    height = info['height']

    data = np.memmap(fp,
                     dtype='<f4',
                     mode=mmap_mode)

    if 'series_count' in info.keys():   # stack of images
        size = (info['series_count'], height, width)
        data = data.reshape(size)
    else:  # 2D x 2D
        size = (info['scan_x'], info['scan_y'], height, width)
        data = data.reshape(size)
    return data


def _parse_xml(filename):
    tree = ET.parse(filename)
    om = convert_xml_to_dict(tree.getroot())

    info = {'raw_filename': om.root.raw_file.filename,
            'width':128,
            'height':130,
            'record-by':'image'}   
    if om.has_item('root.count'):
        # Stack of images
        info.update({'series_count':int(om.root.scan_parameters.series_count)})
    elif om.has_item('root.pix_x') and om.has_item('root.pix_y'):
        # 2D x 2D
        info.update({'scan_x':int(om.root.pix_x),
                     'scan_y':int(om.root.pix_y)})
    else:
        raise IOError("Unsupported Empad file: the scan parameters can not "
                      "imported.")

    return om, info


def _convert_scale_units(value, units, factor=1):
    v = np.float(value) * _ureg(units)
    converted_v = (factor * v).to_compact()
    converted_value = converted_v.magnitude / factor
    converted_units = '{:~}'.format(converted_v.units)

    return converted_value, converted_units

def file_reader(filename, lazy=False, mmap_mode='c', **kwds):

    om, info = _parse_xml(filename)
    dname, fname = os.path.split(filename)

    md = {
        'General': {'original_filename': fname,
                    'title': os.path.splitext(fname)[0],
                    },
        "Signal": {'signal_type': '',
                   'record_by': 'image'},
    }

    if om.has_item('root.timestamp.isoformat'):
        date, time = om.root.timestamp.isoformat.split('T')
        md['General'].update({"date":date, "time":time})

    units = ['1/nm', '1/nm']
    scales = [1, 1]
    origins = [-64, -64]
    axes = []
    index_in_array = 0
    names = ['height', 'width']

    if 'series_count' in info.keys():
        names = ['series_count'] + names
        units.insert(0, 'ms')
        scales.insert(0, 1)
        origins.insert(0, 0)
    else:
        names = ['scan_x', 'scan_y'] + names
        units.insert(0, '')
        units.insert(0, '')
        scales.insert(0, 1)
        scales.insert(0, 1)
        origins.insert(0, 0)
        origins.insert(0, 0)

    sizes = [info[name] for name in names]

    if not 'series_count' in info.keys():
        try:
            fov = ast.literal_eval(
                om.root.iom_measurements.opticsget_full_scan_field_of_view)
            for i in range(2):
                value = fov[i] / sizes[i]
                scales[i], units[i] = _convert_scale_units(value, 'm', sizes[i])
        except BaseException:
            _logger.warning("The scale of the navigation axes can't be read.")

    try:
        value = float(om.root.iom_measurements.calibrated_pixelsize)
        ps, unit = _convert_scale_units(value, '1/m', sizes[i])
        for i in [-1, -2]:
            scales[i] = float(om.root.iom_measurements.calibrated_pixelsize)
    except BaseException:
        _logger.warning("The scale of the signal axes can't be read.")

    for i in range(len(names)):
        if sizes[i] > 1:
            axes.append({
                'size': sizes[i],
                'index_in_array': index_in_array,
                'name': names[i],
                'scale': scales[i],
                'offset': origins[i],
                'units': units[i],
            })
            index_in_array += 1

    raw_filename = os.path.join(dname, info['raw_filename'])
    data = _read_raw(info, raw_filename, mmap_mode=mmap_mode)

    dictionary = {
        'data': data.squeeze(),
        'axes': axes,
        'metadata': md,
        'original_metadata': om.as_dictionary()
    }

    return [dictionary, ]
