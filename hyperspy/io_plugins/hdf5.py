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
    'The default file format for HyperSpy based on the HDF5 standard'

full_support = False
# Recognised file extension
file_extensions = ['hdf', 'h4', 'hdf4', 'h5', 'hdf5', 'he4', 'he5']
default_extension = 4

# Writing capabilities
writes = True
version = "1.3"

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
# the experiments and that will be accessible as attributes of the
# Experiments instance
#
# New in v1.3
# -----------
# - Added support for lists, tuples and binary strings

not_valid_format = 'The file is not a valid HyperSpy hdf5 file'

current_file_version = None  # Format version of the file being read
default_version = StrictVersion(version)


def get_hspy_format_version(f):
    if "file_format_version" in f.attrs:
        version = f.attrs["file_format_version"]
        if isinstance(version, bytes):
            version = version.decode()
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
            warnings.warn(
                "This file was written using a newer version of the "
                "HyperSpy hdf5 file format. I will attempt to load it, but, "
                "if I fail, it is likely that I will be more successful at "
                "this and other tasks if you upgrade me.")

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

    exp = {'data': group['data'][:]}
    axes = []
    for i in range(len(exp['data'].shape)):
        try:
            axes.append(dict(group['axis-%i' % i].attrs))
            axis = axes[-1]
            for key, item in axis.items():
                axis[key] = ensure_unicode(item)
        except KeyError:
            break
    if len(axes) != len(exp['data'].shape):  # broke from the previous loop
        try:
            axes = [i for k, i in sorted(iter(hdfgroup2dict(
                group['_list_' + str(len(exp['data'].shape)) + '_axes'],
                {}).items()))]
        except KeyError:
            raise IOError(not_valid_format)
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
    if "General" in exp["metadata"] and "title" in exp["metadata"]["General"]:
        if '__unnamed__' == exp['metadata']['General']['title']:
            exp['metadata']["General"]['title'] = ''

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
            if "Signal" not in exp["metadata"]:
                exp["metadata"]["Signal"] = {}
            exp['metadata']["Signal"]['signal_type'] = \
                exp['metadata']['signal']
            del exp['metadata']['signal']

        if 'name' in exp['metadata']:
            if "General" not in exp["metadata"]:
                exp["metadata"]["General"] = {}
            exp['metadata']['General']['title'] = \
                exp['metadata']['name']
            del exp['metadata']['name']

    if current_file_version < StrictVersion("1.2"):
        if '_internal_parameters' in exp['metadata']:
            exp['metadata']['_HyperSpy'] = \
                exp['metadata']['_internal_parameters']
            del exp['metadata']['_internal_parameters']
            if 'stacking_history' in exp['metadata']['_HyperSpy']:
                exp['metadata']['_HyperSpy']["Stacking_history"] = \
                    exp['metadata']['_HyperSpy']['stacking_history']
                del exp['metadata']['_HyperSpy']["stacking_history"]
            if 'folding' in exp['metadata']['_HyperSpy']:
                exp['metadata']['_HyperSpy']["Folding"] = \
                    exp['metadata']['_HyperSpy']['folding']
                del exp['metadata']['_HyperSpy']["folding"]
        if 'Variance_estimation' in exp['metadata']:
            if "Noise_properties" not in exp["metadata"]:
                exp["metadata"]["Noise_properties"] = {}
            exp['metadata']['Noise_properties']["Variance_linear_model"] = \
                exp['metadata']['Variance_estimation']
            del exp['metadata']['Variance_estimation']
        if "TEM" in exp["metadata"]:
            if "Acquisition_instrument" not in exp["metadata"]:
                exp["metadata"]["Acquisition_instrument"] = {}
            exp["metadata"]["Acquisition_instrument"]["TEM"] = \
                exp["metadata"]["TEM"]
            del exp["metadata"]["TEM"]
            tem = exp["metadata"]["Acquisition_instrument"]["TEM"]
            if "EELS" in tem:
                if "dwell_time" in tem:
                    tem["EELS"]["dwell_time"] = tem["dwell_time"]
                    del tem["dwell_time"]
                if "dwell_time_units" in tem:
                    tem["EELS"]["dwell_time_units"] = tem["dwell_time_units"]
                    del tem["dwell_time_units"]
                if "exposure" in tem:
                    tem["EELS"]["exposure"] = tem["exposure"]
                    del tem["exposure"]
                if "exposure_units" in tem:
                    tem["EELS"]["exposure_units"] = tem["exposure_units"]
                    del tem["exposure_units"]
                if "Detector" not in tem:
                    tem["Detector"] = {}
                tem["Detector"] = tem["EELS"]
                del tem["EELS"]
            if "EDS" in tem:
                if "Detector" not in tem:
                    tem["Detector"] = {}
                if "EDS" not in tem["Detector"]:
                    tem["Detector"]["EDS"] = {}
                tem["Detector"]["EDS"] = tem["EDS"]
                del tem["EDS"]
            del tem
        if "SEM" in exp["metadata"]:
            if "Acquisition_instrument" not in exp["metadata"]:
                exp["metadata"]["Acquisition_instrument"] = {}
            exp["metadata"]["Acquisition_instrument"]["SEM"] = \
                exp["metadata"]["SEM"]
            del exp["metadata"]["SEM"]
            sem = exp["metadata"]["Acquisition_instrument"]["SEM"]
            if "EDS" in sem:
                if "Detector" not in sem:
                    sem["Detector"] = {}
                if "EDS" not in sem["Detector"]:
                    sem["Detector"]["EDS"] = {}
                sem["Detector"]["EDS"] = sem["EDS"]
                del sem["EDS"]
            del sem

        if "Sample" in exp["metadata"] and "Xray_lines" in exp[
                "metadata"]["Sample"]:
            exp["metadata"]["Sample"]["xray_lines"] = exp[
                "metadata"]["Sample"]["Xray_lines"]
            del exp["metadata"]["Sample"]["Xray_lines"]

        for key in ["title", "date", "time", "original_filename"]:
            if key in exp["metadata"]:
                if "General" not in exp["metadata"]:
                    exp["metadata"]["General"] = {}
                exp["metadata"]["General"][key] = exp["metadata"][key]
                del exp["metadata"][key]
        for key in ["record_by", "signal_origin", "signal_type"]:
            if key in exp["metadata"]:
                if "Signal" not in exp["metadata"]:
                    exp["metadata"]["Signal"] = {}
                exp["metadata"]["Signal"][key] = exp["metadata"][key]
                del exp["metadata"][key]

    return exp


def dict2hdfgroup(dictionary, group, **kwds):
    from hyperspy.misc.utils import DictionaryTreeBrowser
    from hyperspy.signal import BaseSignal

    def parse_structure(key, group, value, _type, **kwds):
        try:
            # Here we check if there are any signals in the container, as casting a long list of signals to a
            # numpy array takes a very long time. So we check if there are any,
            # and save numpy the trouble
            if np.any([isinstance(t, BaseSignal) for t in value]):
                tmp = np.array([[0]])
            else:
                tmp = np.array(value)
        except ValueError:
            tmp = np.array([[0]])
        if tmp.dtype is np.dtype('O') or tmp.ndim is not 1:
            dict2hdfgroup(dict(zip(
                [str(i) for i in range(len(value))], value)),
                group.create_group(_type + str(len(value)) + '_' + key),
                **kwds)
        elif tmp.dtype.type is np.unicode_:
            group.create_dataset(_type + key,
                                 tmp.shape,
                                 dtype=h5py.special_dtype(vlen=str),
                                 **kwds)
            group[_type + key][:] = tmp[:]
        else:
            group.create_dataset(
                _type + key,
                data=tmp,
                **kwds)

    for key, value in dictionary.items():
        if isinstance(value, dict):
            dict2hdfgroup(value, group.create_group(key),
                          **kwds)
        elif isinstance(value, DictionaryTreeBrowser):
            dict2hdfgroup(value.as_dictionary(),
                          group.create_group(key),
                          **kwds)
        elif isinstance(value, BaseSignal):
            if key.startswith('_sig_'):
                try:
                    write_signal(value, group[key])
                except:
                    write_signal(value, group.create_group(key))
            else:
                write_signal(value, group.create_group('_sig_' + key))
        elif isinstance(value, np.ndarray):
            group.create_dataset(key, data=value, **kwds)
        elif value is None:
            group.attrs[key] = '_None_'
        elif isinstance(value, bytes):
            try:
                # binary string if has any null characters (otherwise not
                # supported by hdf5)
                _ = value.index(b'\x00')
                group.attrs['_bs_' + key] = np.void(value)
            except ValueError:
                group.attrs[key] = value.decode()
        elif isinstance(value, str):
            group.attrs[key] = value
        elif isinstance(value, AxesManager):
            dict2hdfgroup(value.as_dictionary(),
                          group.create_group('_hspy_AxesManager_' + key),
                          **kwds)
        elif isinstance(value, (datetime.date, datetime.time)):
            group.attrs["_datetime_" + key] = repr(value)
        elif isinstance(value, list):
            if len(value):
                parse_structure(key, group, value, '_list_', **kwds)
            else:
                group.attrs['_list_empty_' + key] = '_None_'
        elif isinstance(value, tuple):
            if len(value):
                parse_structure(key, group, value, '_tuple_', **kwds)
            else:
                group.attrs['_tuple_empty_' + key] = '_None_'

        elif value is Undefined:
            continue
        else:
            try:
                group.attrs[key] = value
            except:
                print("The hdf5 writer could not write the following "
                      "information in the file")
                print('%s : %s' % (key, value))


def hdfgroup2dict(group, dictionary=None):
    if dictionary is None:
        dictionary = {}
    for key, value in group.attrs.items():
        if isinstance(value, bytes):
            value = value.decode()
        if isinstance(value, (np.string_, str)):
            if value == '_None_':
                value = None
        elif isinstance(value, np.bool_):
            value = bool(value)
        elif isinstance(value, np.ndarray) and value.dtype.char == "S":
            # Convert strings to unicode
            value = value.astype("U")
            if value.dtype.str.endswith("U1"):
                value = value.tolist()
        # skip signals - these are handled below.
        if key.startswith('_sig_'):
            pass
        elif key.startswith('_list_empty_'):
            dictionary[key[len('_list_empty_'):]] = []
        elif key.startswith('_tuple_empty_'):
            dictionary[key[len('_tuple_empty_'):]] = ()
        elif key.startswith('_bs_'):
            dictionary[key[len('_bs_'):]] = value.tostring()
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
                ans = np.array(group[key])
                if ans.dtype.char == "S":
                    ans = ans.astype("U")
                kn = key
                if key.startswith("_list_"):
                    ans = ans.tolist()
                    kn = key[6:]
                elif key.startswith("_tuple_"):
                    ans = tuple(ans.tolist())
                    kn = key[7:]
                dictionary[kn] = ans
            elif key.startswith('_hspy_AxesManager_'):
                dictionary[key[len('_hspy_AxesManager_'):]] = \
                    AxesManager([i
                                 for k, i in sorted(iter(
                                     hdfgroup2dict(group[key]).items()))])
            elif key.startswith('_list_'):
                dictionary[key[7 + key[6:].find('_'):]] = \
                    [i for k, i in sorted(iter(
                        hdfgroup2dict(group[key], {}).items()))]
            elif key.startswith('_tuple_'):
                dictionary[key[8 + key[7:].find('_'):]] = tuple(
                    [i for k, i in sorted(iter(
                        hdfgroup2dict(group[key], {}).items()))])
            else:
                dictionary[key] = {}
                hdfgroup2dict(group[key], dictionary[key])
    return dictionary


def write_signal(signal, group, **kwds):
    if default_version < StrictVersion("1.2"):
        metadata = "mapped_parameters"
        original_metadata = "original_parameters"
    else:
        metadata = "metadata"
        original_metadata = "original_metadata"
    if 'compression' not in kwds:
        kwds['compression'] = 'gzip'

    group.create_dataset('data',
                         data=signal.data,
                         **kwds)
    for axis in signal.axes_manager._axes:
        axis_dict = axis.get_axis_dictionary()
        # For the moment we don't store the navigate attribute
        del(axis_dict['navigate'])
        coord_group = group.create_group(
            'axis-%s' % axis.index_in_array)
        dict2hdfgroup(axis_dict, coord_group, **kwds)
    mapped_par = group.create_group(metadata)
    metadata_dict = signal.metadata.as_dictionary()
    if default_version < StrictVersion("1.2"):
        metadata_dict["_internal_parameters"] = \
            metadata_dict.pop("_HyperSpy")
    dict2hdfgroup(metadata_dict, mapped_par, **kwds)
    original_par = group.create_group(original_metadata)
    dict2hdfgroup(signal.original_metadata.as_dictionary(), original_par,
                  **kwds)
    learning_results = group.create_group('learning_results')
    dict2hdfgroup(signal.learning_results.__dict__,
                  learning_results, **kwds)
    if hasattr(signal, 'peak_learning_results'):
        peak_learning_results = group.create_group(
            'peak_learning_results')
        dict2hdfgroup(signal.peak_learning_results.__dict__,
                      peak_learning_results, **kwds)


def file_writer(filename,
                signal,
                *args, **kwds):
    with h5py.File(filename, mode='w') as f:
        f.attrs['file_format'] = "HyperSpy"
        f.attrs['file_format_version'] = version
        exps = f.create_group('Experiments')
        group_name = signal.metadata.General.title if \
            signal.metadata.General.title else '__unnamed__'
        # / is a invalid character, see #942
        if "/" in group_name:
            group_name = group_name.replace("/", "-")
        expg = exps.create_group(group_name)
        if 'compression' not in kwds:
            kwds['compression'] = 'gzip'
        write_signal(signal, expg, **kwds)
