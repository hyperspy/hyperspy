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

import h5py

import numpy as np

from hyperspy import messages
from hyperspy.misc.utils import ensure_unicode

# Plugin characteristics
# ----------------------
format_name = 'HDF5'
description = \
    'The default file format for Hyperspy based on the HDF5 standard' 

full_suport = False
# Recognised file extension
file_extensions = ['hdf', 'h4', 'hdf4', 'h5', 'hdf5', 'he4', 'he5']
default_extension = 4

# Writing capabilities
writes = True

# -----------------------
# File format description
# -----------------------
# The root must contain a group called Experiments
# The experiments group can contain any number of subgroups
# Each subgroup is an experiment or signal
# Each subgroup must contain at least one dataset called data
# The data is an array of arbitrary dimension
# In addition a number equal to the number of dimensions of the data
# dataset + 1 of empty groups called coordinates followed by a number
# must exists with the following attributes:
#    'name' 
#    'offset' 
#    'scale' 
#    'units' 
#    'size'
#    'index_in_array'
# The experiment group contains a number of attributes that will be
# directly assigned as class attributes of the Signal instance. In
# addition the experiment groups may contain 'original_parameters' and 
# 'mapped_parameters'subgroup that will be 
# assigned to the same name attributes of the Signal instance as a 
# Dictionary Browsers
# The Experiments group can contain attributes that may be common to all
# the experiments and that will be accessible as attribures of the
# Experimentsinstance

not_valid_format = 'The file is not a valid Hyperspy hdf5 file'

def file_reader(filename, record_by, mode = 'r', driver = 'core', 
                backing_store = False, **kwds):
    with h5py.File(filename, mode=mode, driver=driver) as f:
        # If the file has been created with Hyperspy it should cointain a
        # folder Experiments.
        experiments = []
        exp_dict_list = []
        if 'Experiments' in f:
            for ds in f['Experiments']:
                if isinstance(f['Experiments'][ds], h5py.Group):
                    if 'data' in f['Experiments'][ds]:
                        experiments.append(ds)
            if not experiments:
                raise IOError(not_valid_format)
            # Parse the file
            for experiment in experiments:
                exg = f['Experiments'][experiment]
                exp=hdfgroup2signaldict(exg)
                exp_dict_list.append(exp)
        else:
            # Eventually there will be the possibility of loading the
            # datasets of any hdf5 file
            raise IOError('This is not a Hyperspy HDF5')
        return exp_dict_list

def hdfgroup2signaldict(group):
    exp = {}
    exp['data'] = group['data'][:]
    axes = []
    for i in xrange(len(exp['data'].shape)):
        try:
            axes.append(dict(group['axis-%i' % i].attrs))
        except KeyError:
            raise IOError(not_valid_format)
    for axis in axes:
        for key, item in axis.iteritems():
            axis[key] = ensure_unicode(item)
    exp['mapped_parameters'] = hdfgroup2dict(
        group['mapped_parameters'], {})
    exp['original_parameters'] = hdfgroup2dict(
        group['original_parameters'], {})
    exp['axes'] = axes
    exp['attributes']={}
    if 'learning_results' in group.keys():
        exp['attributes']['learning_results'] = \
            hdfgroup2dict(group['learning_results'],{})
    if 'peak_learning_results' in group.keys():
        exp['attributes']['peak_learning_results'] = \
            hdfgroup2dict(group['peak_learning_results'],{})
        
    # Load the decomposition results written with the old name,
    # mva_results
    if 'mva_results' in group.keys():
        exp['attributes']['learning_results'] = hdfgroup2dict(
            group['mva_results'],{})
    if 'peak_mva_results' in group.keys():
        exp['attributes']['peak_learning_results']=hdfgroup2dict(
            group['peak_mva_results'],{})
    # Replace the old signal and name keys with their current names
    if 'signal' in exp['mapped_parameters']:
        exp['mapped_parameters']['signal_type'] = \
            exp['mapped_parameters']['signal']
        del exp['mapped_parameters']['signal']
        
    if 'name' in exp['mapped_parameters']:
        exp['mapped_parameters']['title'] = \
            exp['mapped_parameters']['name']
        del exp['mapped_parameters']['name']
    
    # If the title was not defined on writing the Experiment is 
    # then called __unnamed__. The next "if" simply sets the title
    # back to the empty string
    if '__unnamed__' == exp['mapped_parameters']['title']:
        exp['mapped_parameters']['title'] = ''
        
    return exp

def dict2hdfgroup(dictionary, group, compression = None):
    from hyperspy.misc.utils import DictionaryBrowser
    from hyperspy.signal import Signal
    for key, value in dictionary.iteritems():
        if isinstance(value, dict):
            dict2hdfgroup(value, group.create_group(key), 
                          compression = compression)
        elif isinstance(value, DictionaryBrowser):
            dict2hdfgroup(value.as_dictionary(),
                          group.create_group(key),
                          compression = compression)
        elif isinstance(value, Signal):
            if key.startswith('_sig_'):
                try:
                    write_signal(value,group[key])
                except:
                    write_signal(value,group.create_group(key))
            else:
                write_signal(value,group.create_group('_sig_'+key))
        elif isinstance(value, np.ndarray):
            group.create_dataset(key,
                                 data=value,
                                 compression = compression)
        elif value is None:
            group.attrs[key] = '_None_'
        elif isinstance(value, basestring):
            group.attrs[key] = value.encode('utf8',
                                            errors='ignore')
        else:
            try:
                group.attrs[key] = value
            except:
                print("The hdf5 writer could not write the following "
                "information in the file")
                print('%s : %s' % (key, value))
            
def hdfgroup2dict(group, dictionary = {}):
    for key, value in group.attrs.iteritems():
        if type(value) is np.string_:
            if value == '_None_':
                value = None
            else:
                try:
                    value = value.decode('utf8')
                except UnicodeError:
                    # For old files
                    value = value.decode('latin-1')
                    
        elif type(value) is np.ndarray and \
                value.dtype == np.dtype('|S1'):
            value = value.tolist()
        # skip signals - these are handled below.
        if key.startswith('_sig_'):
            pass
        else:
            dictionary[key] = value
    if not isinstance(group,h5py.Dataset):
        for key in group.keys():
            if key.startswith('_sig_'):
                dictionary[key[5:]] = hdfgroup2signaldict(group[key])
            elif isinstance(group[key],h5py.Dataset):
                dictionary[key]=np.array(group[key])
            else:
                dictionary[key] = {}
                hdfgroup2dict(group[key], dictionary[key])
    return dictionary

def write_signal(signal,group, compression='gzip'):
    group.create_dataset('data',
                         data=signal.data,
                         compression=compression)
    for axis in signal.axes_manager.axes:
        axis_dict = axis.get_axis_dictionary()
        # For the moment we don't store the navigate attribute
        del(axis_dict['navigate'])
        coord_group = group.create_group(
            'axis-%s' % axis.index_in_array)
        dict2hdfgroup(axis_dict, coord_group, compression = compression)
    mapped_par = group.create_group('mapped_parameters')
    dict2hdfgroup(signal.mapped_parameters.as_dictionary(), 
                  mapped_par, compression = compression)
    original_par = group.create_group('original_parameters')
    dict2hdfgroup(signal.original_parameters.as_dictionary(), 
                  original_par, compression = compression)
    learning_results = group.create_group('learning_results')
    dict2hdfgroup(signal.learning_results.__dict__, 
                  learning_results, compression = compression)
    if hasattr(signal,'peak_learning_results'):
        peak_learning_results = group.create_group(
            'peak_learning_results')
        dict2hdfgroup(signal.peak_learning_results.__dict__, 
                  peak_learning_results, compression = compression)
                                    
def file_writer(filename, signal, compression = 'gzip', *args, **kwds):
    with h5py.File(filename, mode = 'w') as f:
        exps = f.create_group('Experiments')
        group_name = signal.mapped_parameters.title if \
                     signal.mapped_parameters.title else '__unnamed__'
        expg = exps.create_group(group_name)
        write_signal(signal,expg, compression = compression)
