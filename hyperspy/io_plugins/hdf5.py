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

from distutils.version import StrictVersion
import warnings
import datetime

import h5py
import numpy as np
from traits.api import Undefined

from hyperspy.misc.utils import ensure_unicode
from hyperspy.axes import AxesManager

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
version = "1.1"

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
# addition the experiment groups may contain 'original_metadata' and
# 'metadata'subgroup that will be
# assigned to the same name attributes of the Signal instance as a
# Dictionary Browsers
# The Experiments group can contain attributes that may be common to all
# the experiments and that will be accessible as attribures of the
# Experimentsinstance

not_valid_format = 'The file is not a valid Hyperspy hdf5 file'

current_file_version = None  # Format version of the file being read
default_version = StrictVersion(version)


def get_hspy_format_version(f):
    if "file_format_version" in f.attrs:
        version = f.attrs["file_format_version"]
        if isinstance(version, float):
            version = str(round(version, 2))
    elif "Experiments" in f:
        # Chances are that this is a HSpy hdf5 file version 1.0
        version = "1.0"
    else:
        raise IOError(not_valid_format)
    return StrictVersion(version)


def file_reader(filename, record_by, mode='r', driver='core',
                backing_store=False, **kwds):
    with h5py.File(filename, mode=mode, driver=driver) as f:
        # Getting the format version here also checks if it is a valid HSpy
        # hdf5 file, so the following two lines must not be deleted or moved
        # elsewhere.
        global current_file_version
        current_file_version = get_hspy_format_version(f)
        global default_version
        if current_file_version > default_version:
            warnings.warn("This file was written using a newer version of "
                          "HyperSpy. I will attempt to load it, but, "
                          "if I fail, "
                          "it is likely that I will be more successful at this "
                          "and other tasks if you upgrade me.")

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
                exp = hdfgroup2signaldict(exg)
                exp_dict_list.append(exp)
        else:
            raise IOError('This is not a valid HyperSpy HDF5 file. '
                          'You can still load the data using a hdf5 reader, '
                          'e.g. h5py, and manually create a Signal. '
                          'Please, refer to the User Guide for details')
        return exp_dict_list


def hdfgroup2signaldict(group):
    global current_file_version
    global default_version
    if current_file_version < StrictVersion("1.2"):
        metadata = "mapped_parameters"
        original_metadata = "original_parameters"
    else:
        metadata = "metadata"
        original_metadata = "original_metadata"

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
    exp['metadata'] = hdfgroup2dict(
        group[metadata], {})
    exp['original_metadata'] = hdfgroup2dict(
        group[original_metadata], {})
    exp['axes'] = axes
    exp['attributes'] = {}
    if 'learning_results' in group.keys():
        exp['attributes']['learning_results'] = \
            hdfgroup2dict(group['learning_results'], {})
    if 'peak_learning_results' in group.keys():
        exp['attributes']['peak_learning_results'] = \
            hdfgroup2dict(group['peak_learning_results'], {})

    # If the title was not defined on writing the Experiment is
    # then called __unnamed__. The next "if" simply sets the title
    # back to the empty string
    if '__unnamed__' == exp['metadata']['title']:
        exp['metadata']['title'] = ''

    if current_file_version < StrictVersion("1.1"):
        # Load the decomposition results written with the old name,
        # mva_results
        if 'mva_results' in group.keys():
            exp['attributes']['learning_results'] = hdfgroup2dict(
                group['mva_results'], {})
        if 'peak_mva_results' in group.keys():
            exp['attributes']['peak_learning_results'] = hdfgroup2dict(
                group['peak_mva_results'], {})
        # Replace the old signal and name keys with their current names
        if 'signal' in exp['metadata']:
            exp['metadata']['signal_type'] = \
                exp['metadata']['signal']
            del exp['metadata']['signal']

        if 'name' in exp['metadata']:
            exp['metadata']['title'] = \
                exp['metadata']['name']
            del exp['metadata']['name']

    if current_file_version < StrictVersion("1.2"):
        if '_internal_parameters' in exp['metadata']:
            exp['metadata']['_Internal_parameters'] = \
                exp['metadata']['_internal_parameters']
            del exp['metadata']['_internal_parameters']

    return exp


def dict2hdfgroup(dictionary, group, compression=None):
    from hyperspy.misc.utils import DictionaryTreeBrowser
    from hyperspy.signal import Signal
    for key, value in dictionary.iteritems():
        if isinstance(value, dict):
            dict2hdfgroup(value, group.create_group(key),
                          compression=compression)
        elif isinstance(value, DictionaryTreeBrowser):
            dict2hdfgroup(value.as_dictionary(),
                          group.create_group(key),
                          compression=compression)
        elif isinstance(value, Signal):
            if key.startswith('_sig_'):
                try:
                    write_signal(value, group[key])
                except:
                    write_signal(value, group.create_group(key))
            else:
                write_signal(value, group.create_group('_sig_' + key))
        elif isinstance(value, np.ndarray):
            group.create_dataset(key,
                                 data=value,
                                 compression=compression)
        elif value is None:
            group.attrs[key] = '_None_'
        elif isinstance(value, str):
            try:
                # Store strings as unicode using the default encoding
                group.attrs[key] = unicode(value)
            except UnicodeEncodeError:
                pass
        elif isinstance(value, AxesManager):
            dict2hdfgroup(value.as_dictionary(),
                          group.create_group('_hspy_AxesManager_'
                                             + key),
                          compression=compression)
        elif isinstance(value, (datetime.date, datetime.time)):
            group.attrs["_datetime_" + key] = repr(value)
        elif value is Undefined:
            continue
        else:
            try:
                group.attrs[key] = value
            except:
                print("The hdf5 writer could not write the following "
                      "information in the file")
                print('%s : %s' % (key, value))


def hdfgroup2dict(group, dictionary={}):
    for key, value in group.attrs.iteritems():
        if isinstance(value, (np.string_, str)):
            if value == '_None_':
                value = None
        elif isinstance(value, np.bool_):
            value = bool(value)

        elif isinstance(value, np.ndarray) and \
                value.dtype == np.dtype('|S1'):
            value = value.tolist()
        # skip signals - these are handled below.
        if key.startswith('_sig_'):
            pass
        elif key.startswith('_datetime_'):
            dictionary[key.replace("_datetime_", "")] = eval(value)
        else:
            dictionary[key] = value
    if not isinstance(group, h5py.Dataset):
        for key in group.keys():
            if key.startswith('_sig_'):
                from hyperspy.io import dict2signal
                dictionary[key[len('_sig_'):]] = (
                    dict2signal(hdfgroup2signaldict(group[key])))
            elif isinstance(group[key], h5py.Dataset):
                dictionary[key] = np.array(group[key])
            elif key.startswith('_hspy_AxesManager_'):
                dictionary[key[len('_hspy_AxesManager_'):]] = \
                    AxesManager([i
                                 for k, i in sorted(iter(
                                     hdfgroup2dict(group[key]).iteritems()))])
            else:
                dictionary[key] = {}
                hdfgroup2dict(group[key], dictionary[key])
    return dictionary


def write_signal(signal, group, compression='gzip'):
    if default_version < StrictVersion("1.2"):
        metadata = "mapped_parameters"
        original_metadata = "original_parameters"
    else:
        metadata = "metadata"
        original_metadata = "original_metadata"

    group.create_dataset('data',
                         data=signal.data,
                         compression=compression)
    for axis in signal.axes_manager._axes:
        axis_dict = axis.get_axis_dictionary()
        # For the moment we don't store the navigate attribute
        del(axis_dict['navigate'])
        coord_group = group.create_group(
            'axis-%s' % axis.index_in_array)
        dict2hdfgroup(axis_dict, coord_group, compression=compression)
    mapped_par = group.create_group(metadata)
    metadata_dict = signal.metadata.as_dictionary()
    if default_version < StrictVersion("1.2"):
        metadata_dict["_internal_parameters"] = \
            metadata_dict.pop("_Internal_parameters")
    dict2hdfgroup(metadata_dict,
                  mapped_par, compression=compression)
    original_par = group.create_group(original_metadata)
    dict2hdfgroup(signal.original_metadata.as_dictionary(),
                  original_par, compression=compression)
    learning_results = group.create_group('learning_results')
    dict2hdfgroup(signal.learning_results.__dict__,
                  learning_results, compression=compression)
    if hasattr(signal, 'peak_learning_results'):
        peak_learning_results = group.create_group(
            'peak_learning_results')
        dict2hdfgroup(signal.peak_learning_results.__dict__,
                      peak_learning_results, compression=compression)


def file_writer(filename,
                signal,
                compression='gzip',
                *args, **kwds):
    with h5py.File(filename, mode='w') as f:
        f.attrs['file_format'] = "HyperSpy"
        f.attrs['file_format_version'] = version
        exps = f.create_group('Experiments')
        group_name = signal.metadata.title if \
            signal.metadata.title else '__unnamed__'
        expg = exps.create_group(group_name)
        write_signal(signal, expg, compression=compression)
