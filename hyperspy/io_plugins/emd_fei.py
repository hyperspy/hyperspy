# -*- coding: utf-8 -*-
# Copyright 2007-2015 The HyperSpy developers
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

import warnings

import h5py
import numpy as np
from traits.api import Undefined

from hyperspy.misc.utils import ensure_unicode
from hyperspy.axes import AxesManager

import json


# Plugin characteristics
# ----------------------
format_name = 'EMD FEI'
description = \
    'File reader for the EMD FEI format'

full_support = False
# Recognised file extension
file_extensions = ['emd']
default_extension = 0

# Writing capabilities
writes = False

def file_reader(filename, mode='r', driver='core',
                backing_store=False, **kwds):
    with h5py.File(filename, mode=mode, driver=driver) as f:
        experiments = []
        exp_dict_list = []
        for experiment_key, experiment_group in f.iteritems():
            if isinstance(experiment_group, h5py.Group):
                exp_dict = emdimagegroup2signaldict(experiment_key, experiment_group)
                exp_dict_list.extend(exp_dict)
        return exp_dict_list

def _get_emd_metadata_from_dict(metadata_string):
    original_metadata_dict = json.loads(metadata_string)
    axes_offset = original_metadata_dict['_values']['Scale/Spatial/Start']
    axes_scale = original_metadata_dict['_values']['Scale/Spatial/Step']
    axes_units = original_metadata_dict['_units']['Scale/Spatial/Step']
    camera_length = original_metadata_dict['_values']['Stem/Acquisition/CameraLength']
    beam_energy = original_metadata_dict['_values']['Microscope/Gun/HT']
    
    temp_metadata_dict = {
            'axes': {
                'offset':axes_offset,
                'scale':axes_scale,
                'units':axes_units},
            'camera_length': camera_length,
            'beam_energy': beam_energy}
    return(temp_metadata_dict)

def _make_axes(data_shape, axes_dict):
    offset_list = axes_dict['offset']
    scale_list = axes_dict['scale']
    units_list = axes_dict['units']

    axes_list = []
    for offset, scale, units, size in zip(
            offset_list.iteritems(),
            scale_list.iteritems(),
            units_list.iteritems(),
            data_shape):
        temp_axes = []
        temp_axes = {
                'size': size,
                'offset': offset[1],
                'scale': scale[1],
                'units': units[1],
                'name': units[0]
                }
        axes_list.append(temp_axes)
    return(axes_list)
    
def emdimagegroup2signaldict(experiment_key, group):
    image_data_list = []
    if 'images' in group.keys():
        for image_group in group['images'].itervalues():
            if 'image' in image_group.keys():
                temp_image_data = {'data':image_group['image'].value}

                metadata_tag_string = image_group.attrs['links']
                metadata_tag = json.loads(metadata_tag_string)['meta_d']
                original_metadata = group['metadata'][metadata_tag].attrs['metadata']

                data_shape = temp_image_data['data'].shape
                metadata_dict = _get_emd_metadata_from_dict(original_metadata)

                axes_list = _make_axes(data_shape, metadata_dict['axes'])

                temp_image_data['axes'] = axes_list

                temp_image_data['metadata'] = {}

                image_data_list.append(temp_image_data)

    if 'edx' in group.keys():
        temp_data = {}
        temp_spectrum_data = []
        for spectrum in group['edx']['spectrum'].itervalues():
            if 'data' in spectrum.keys():
                temp_spectrum_data.append(spectrum['data'].value)
            else:
                if 'compressed_cube' in spectrum.keys():
                    temp_spectrum_data.append(
                        spectrum['compressed_cube'].value)

        temp_data['metadata'] = {}
        temp_data['data'] = np.array(temp_spectrum_data)

        image_data_list.append(temp_data)

    return(image_data_list)
