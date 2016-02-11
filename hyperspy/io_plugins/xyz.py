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

import warnings
import datetime

import numpy as np

from scipy.ndimage.filters import gaussian_filter
from hyperspy.misc.elements import elements


# Plugin characteristics
# ----------------------
format_name = 'xyz'
description = ''

full_support = False
# Recognised file extension
file_extensions = ['xyz']
default_extension = 0

# Writing capabilities
writes = False

# -----------------------
# File format description
# -----------------------
#

def file_reader(filename):
    atom_array = np.loadtxt(
            filename,
            skiprows=2,
            comments='EOF',
            dtype={
                'names': ('atom_type', 'x', 'y', 'z'),
                'formats': ('S4', 'f8', 'f8', 'f8')})

    image_array, axes_dict = _generate_image_from_3d_points(atom_array)
    return image_array, axes_dict

def _find_nearest_index(array,value):
    idx = (np.abs(array-value)).argmin()
    return(idx)

def _generate_image_from_3d_points(
        atom_array, 
        image_size=(1024,1024), 
        projection_axis='z',
        gaussian_blur=0.5):
    image_array = np.zeros((1024, 1024))

    if projection_axis == 'z':
        axis0 = 'x'
        axis1 = 'y'
    elif projection_axis == 'y':
        axis0 = 'x'
        axis1 = 'z'
    elif projection_axis == 'x':
        axis0 = 'y'
        axis1 = 'z'

    scale_axis0 = (atom_array[axis0].max() - atom_array[axis0].min())/image_size[0]
    scale_axis1 = (atom_array[axis1].max() - atom_array[axis1].min())/image_size[1] 

    offset_axis0 = atom_array[axis0].min()
    offset_axis1 = atom_array[axis1].min()

    axis0_position_array = np.arange(image_size[0])*scale_axis0 + offset_axis0
    axis1_position_array = np.arange(image_size[1])*scale_axis1 + offset_axis1

    for atom in atom_array:
        atom_Z = elements[atom['atom_type']]['General_properties']['Z']
        
        index_axis0 = _find_nearest_index(axis0_position_array, atom[axis0])
        index_axis1 = _find_nearest_index(axis1_position_array, atom[axis1])

        image_array[index_axis0, index_axis1] += atom_Z

    gaussian_filter(image_array, gaussian_blur/scale_axis0, output=image_array)

    axis0_dict = {
            'size':image_size[0],
            'name':axis0,
            'scale':scale_axis0,
            'offset':offset_axis0}
    axis1_dict = {
            'size':image_size[1],
            'name':axis1,
            'scale':scale_axis1,
            'offset':offset_axis1}

    axes_dict = [axis0_dict, axis1_dict]
    return(image_array, axes_dict)
