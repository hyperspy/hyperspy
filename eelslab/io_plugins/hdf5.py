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

def file_reader(filename, record_by, mode = 'r', driver = 'core', 
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
            axes = []
            for i in xrange(len(exp['data'].shape)):
                try:
                    print('axis-%i' % i)
                    axes.append(dict(exg['axis-%i' % i].attrs))
                except KeyError:
                    f.close()
                    raise IOError(not_valid_format)
            exp['mapped_parameters'] = dict(exg['mapped_parameters'].attrs)
            exp['original_parameters'] = dict(exg['original_parameters'].attrs)
            exp['axes'] = axes
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
    expg = exps.create_group(signal.mapped_parameters.name)
    expg.create_dataset('data', data = signal.data)
    i = 0
    for axis in signal.axes_manager.axes:
        coord_group = expg.create_group('axis-%s' % i)
        coord_group.attrs['name'] =  str(axis.name)
        coord_group.attrs['offset'] =  axis.offset
        coord_group.attrs['scale'] =  axis.scale
        coord_group.attrs['units'] =  axis.units
        coord_group.attrs['size'] = axis.size
        coord_group.attrs['index_in_array'] = axis.index_in_array
        i += 1
    mapped_par = expg.create_group('mapped_parameters')
    for key, value in signal.mapped_parameters.__dict__.iteritems():
        try:
            mapped_par.attrs[key] = value
        except:
            if value is not None:
                print "HDF5 File saver: WARNING: could not save data at: "
                print "  mapped_parameters.%s"%key
                print "    value:"
                print "      ",value
                print "    type:"
                print "      ",type(value)
                print "The saved HDF5 file has lost information from your original data.  \
Please make sure it is not something important.\n"            
    original_par = expg.create_group('original_parameters')
    for key, value in signal.original_parameters.iteritems():
        try:
            original_par.attrs[key] = value
        except:
            if value is not None:
                print "HDF5 File saver: WARNING: could not save data at: "
                print "  original_parameters.%s"%key
                print "    value:"
                print "      ",value
                print "    type:"
                print "      ",type(value)
                print "The saved HDF5 file has lost information from your original data.  \
Please make sure it is not something important.\n"
    omit_keys=['data', 'axes_manager', 'mapped_parameters', 'original_parameters']
    attributes = expg.create_group('attributes')
    for key, value in signal.__dict__.iteritems():
        # store only attributes that we don't handle another way.
        # These attributes are obsolete - everything mapped should be stored
        # under mapped_parameters.

        if key not in omit_keys:
            try:
                attributes.attrs[key] = value
                print "Warning - deprecated use of attribute on Signal object."
                print "  Developer message: consider moving %s to the \
mapped_parameters attribute of the Signal class or derived subclass."%key
            except:
                if value is not None:
                    print "HDF5 File saver: WARNING: could not save attribue: "
                    print "  %s"%key
                    print "    value:"
                    print "      ",value
                    print "    type:"
                    print "      ",type(value)
                    print "The saved HDF5 file has lost information from your original data.  \
Please make sure it is not something important.\n"                
    f.close()
    

