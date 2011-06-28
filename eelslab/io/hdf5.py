# -*- coding: utf-8 -*-
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

import h5py

import numpy as np

from eelslab import messages

# Plugin characteristics
# ----------------------
format_name = 'HDF5'
description = 'The default file format for EELSLab based on the HDF5 standard' 

full_suport = False
# Recognised file extension
file_extensions = ['hdf5', 'hdf', 'h5', 'he5']
default_extension = 0
# Reading capabilities
reads_images = True
reads_spectrum = True
reads_spectrum_image = True
# Writing capabilities
writes_images = True
writes_spectrum = True
writes_spectrum_image = True

# -----------------------
# File format description
# -----------------------
# The root must contain a group called Experiments
# The experiments group can contain any number of subgroups
# Each subgroup it is an experiment or signal
# Each subgroup must contain at least one dataset called data
# The data is an array of arbitrary dimension
# In addition a number equal to the number of dimensions of the data dataset
# + 1 of empty groups called coordinates followed by a number must exists 
# with the following attributes:
#    'name' 
#    'offset' 
#    'scale' 
#    'units' 
#    'size'
#    'index_in_array' : 1
# The experiment group contains a number of attributes that will be directly 
# assigned as class attributes of the Signal instance. In addition the 
# experiment groups may contain an 'extra_parameters' subgroup that will be 
# assigned to the 'extra_parameters' attribute of the Signal instance as a 
# dictionary
# The Experiments group can contain attributes that may be common to all the 
# experiments and that will be accessible as attribures of the Experiments
# instance

not_valid_format = 'The file is not a valid EELSLab hdf5 file'

def file_reader(filename, data_type, mode = 'r', driver = 'core', 
                backing_store = False, **kwds):
            
    f = h5py.File(filename, mode = mode, driver = driver)
    # If the file has been created with EELSLab it should cointain a folder 
    # Experiments.
    experiments = []
    exp_dict_list = []
    datasets = []
    if 'Experiments' in f:
        for ds in f['Experiments']:
            if isinstance(f['Experiments'][ds], h5py.Group):
                if 'data' in f['Experiments'][ds]:
                    experiments.append(ds)
        if not experiments:
            f.close()
            raise IOError(not_valid_format)
        # Parse the file
        for experiment in experiments:
            exg = f['Experiments'][experiment]
            exp = {}
            exp['data'] = exg['data'][:]
            coordinates = []
            for i in range(len(exp['data'].shape)):
                try:
                    print('coordinate-%i' % i)
                    coordinates.append(dict(exg['coordinate-%i' % i].attrs))
                except KeyError:
                    f.close()
                    raise IOError(not_valid_format)
            exp['parameters'] = dict(exg.attrs)
            exp['extra_parameters'] = dict(exg['extra_parameters'].attrs)
            exp['coordinates'] = coordinates
            exp['data_type'] = 'Signal'
            exp_dict_list.append(exp)
            
    else:
        # Eventually there will be the possibility of loading the datasets of 
        # any hdf5 file
        pass
    f.close()
    return exp_dict_list
                                    
def file_writer(filename, signal, *args, **kwds):
    f = h5py.File(filename, mode = 'w-')
    exps = f.create_group('Experiments')
    expg = exps.create_group(signal.name)
    expg.create_dataset('data', data = signal.data)
    i = 0
    for coordinate in signal.coordinates.coordinates:
        coord_group = expg.create_group('coordinate-%s' % i)
        coord_group.attrs['name'] =  str(coordinate.name)
        coord_group.attrs['offset'] =  coordinate.offset
        coord_group.attrs['scale'] =  coordinate.scale
        coord_group.attrs['units'] =  coordinate.units
        coord_group.attrs['size'] = coordinate.size
        coord_group.attrs['index_in_array'] = coordinate.index_in_array
        i += 1
    extra_par = expg.create_group('extra_parameters')
    for key, value in signal.extra_parameters.iteritems():
        extra_par.attrs[key] = value
        
    f.close()
    