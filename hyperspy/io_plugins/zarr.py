# -*- coding: utf-8 -*-
# Copyright 2007-2021 The HyperSpy developers
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

from distutils.version import LooseVersion
import warnings
import logging
import datetime
import ast

import zarr
import numpy as np
import dask.array as da
from traits.api import Undefined
from hyperspy.misc.utils import ensure_unicode, multiply, get_object_package_info
from hyperspy.axes import AxesManager
import numcodecs


_logger = logging.getLogger(__name__)


# Plugin characteristics
# ----------------------
format_name = 'ZArr'
description = \
    'The default file format for HyperSpy based on the zarr standard'
full_support = False
# Recognised file extension
file_extensions = ['zarr']
default_extension = 0
# Writing capabilities
writes = True
version = "3.1"

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
# CHANGES
#
# v3.1
# - move metadata.Signal.binned attribute to axes.is_binned parameter
#
# v3.0
# - add Camera and Stage node
# - move tilt_stage to Stage.tilt_alpha
#
# v2.2
# - store more metadata as string: date, time, notes, authors and doi
# - store quantity for intensity axis
#
# v2.1
# - Store the navigate attribute.
# - record_by is stored only for backward compatibility but the axes navigate
#   attribute takes precendence over record_by for files with version >= 2.1
# v1.3
# ----
# - Added support for lists, tuples and binary strings

not_valid_format = 'The file is not a valid HyperSpy zarr file'

current_file_version = None  # Format version of the file being read
default_version = LooseVersion(version)



def file_reader(filename,
                lazy=False,
                **kwds):
    """Read data from zarr files saved with the hyperspy zarr format specification

    Parameters
    ----------
    filename: str
    lazy: bool
        Load image lazily using dask
    **kwds, optional
    """
    mode = kwds.pop('mode', 'r')
    f = zarr.open(filename, mode=mode, **kwds)
    models_with_signals = []
    standalone_models = []
    if 'Analysis/models' in f:
        try:
            m_gr = f.require_group('Analysis/models')
            for model_name in m_gr:
                if '_signal' in m_gr[model_name].attrs:
                    key = m_gr[model_name].attrs['_signal']
                    # del m_gr[model_name].attrs['_signal']
                    res = zarrgroup2dict(
                        m_gr[model_name],
                        lazy=lazy)
                    del res['_signal']
                    models_with_signals.append((key, {model_name: res}))
                else:
                    standalone_models.append(
                        {model_name: zarrgroup2dict(
                            m_gr[model_name], lazy=lazy)})
        except TypeError:
            raise IOError(not_valid_format)

    experiments = []
    exp_dict_list = []
    if 'Experiments' in f:
        for ds in f['Experiments']:
            if isinstance(f['Experiments'][ds], zarr.Group):
                if 'data' in f['Experiments'][ds]:
                    experiments.append(ds)
        # Parse the file
        for experiment in experiments:
            exg = f['Experiments'][experiment]
            exp = zarrgroup2signaldict(exg, lazy)
            # assign correct models, if found:
            _tmp = {}
            for (key, _dict) in reversed(models_with_signals):
                if key == exg.name:
                    _tmp.update(_dict)
                    models_with_signals.remove((key, _dict))
            exp['models'] = _tmp

            exp_dict_list.append(exp)

    for _, m in models_with_signals:
        standalone_models.append(m)

    exp_dict_list.extend(standalone_models)
    if not len(exp_dict_list):
        raise IOError('This is not a valid HyperSpy Zarr file. '
                      'You can still load the data using zarr reader, '
                      'e.g. zarr, and manually create a Signal. '
                      'Please, refer to the User Guide for details')
    return exp_dict_list


def zarrgroup2signaldict(group, lazy=False):
    global current_file_version
    global default_version
    metadata = "metadata"
    original_metadata = "original_metadata"

    exp = {'metadata': zarrgroup2dict(
        group[metadata], lazy=lazy),
        'original_metadata': zarrgroup2dict(
            group[original_metadata], lazy=lazy),
        'attributes': {} }
    if "package" in group.attrs:
        exp["package"] = group.attrs["package"]
        exp["package_version"] = group.attrs["package_version"]
    else:
        exp["package"] = ""
        exp["package_version"] = ""

    data = group['data']
    if lazy:
        data = da.from_array(data, chunks=data.chunks)
        exp['attributes']['_lazy'] = True
    else:
        data = np.asanyarray(data)
    exp['data'] = data
    axes = []
    for i in range(len(exp['data'].shape)):
        try:
            axes.append(dict(group['axis-%i' % i].attrs))
            axis = axes[-1]
            for key, item in axis.items():
                if isinstance(item, np.bool_):
                    axis[key] = bool(item)
                else:
                    axis[key] = ensure_unicode(item)
        except KeyError:
            break
    if len(axes) != len(exp['data'].shape):  # broke from the previous loop
        try:
            axes = [i for k, i in sorted(iter(zarrgroup2dict(
                group['_list_' + str(len(exp['data'].shape)) + '_axes'],
                lazy=lazy).items()))]
        except KeyError:
            raise IOError(not_valid_format)
    exp['axes'] = axes
    if 'learning_results' in group.keys():
        exp['attributes']['learning_results'] = \
            zarrgroup2dict(
                group['learning_results'],
                lazy=lazy)
    if 'peak_learning_results' in group.keys():
        exp['attributes']['peak_learning_results'] = \
            zarrgroup2dict(
                group['peak_learning_results'],
                lazy=lazy)

    # If the title was not defined on writing the Experiment is
    # then called __unnamed__. The next "if" simply sets the title
    # back to the empty string
    if "General" in exp["metadata"] and "title" in exp["metadata"]["General"]:
        if '__unnamed__' == exp['metadata']['General']['title']:
            exp['metadata']["General"]['title'] = ''
    return exp


def dict2zarrgroup(dictionary, group, **kwds):
    "Recursive writer of dicts and signals"

    from hyperspy.misc.utils import DictionaryTreeBrowser
    from hyperspy.signal import BaseSignal

    def parse_structure(key, group, value, _type, **kwds):
        try:
            # Here we check if there are any signals in the container, as
            # casting a long list of signals to a numpy array takes a very long
            # time. So we check if there are any, and save numpy the trouble
            if np.any([isinstance(t, BaseSignal) for t in value]):
                tmp = np.array([[0]])
            else:
                tmp = np.array(value)
        except ValueError:
            tmp = np.array([[0]])
        if tmp.dtype == np.dtype('O') or tmp.ndim != 1:
            dict2zarrgroup(dict(zip(
                [str(i) for i in range(len(value))], value)),
                group.create_group(_type + str(len(value)) + '_' + key),
                **kwds)
        elif tmp.dtype.type is np.unicode_:
            if _type + key in group:
                del group[_type + key]
            group.create_dataset(_type + key,
                                 data=tmp,
                                 dtype=object,
                                 object_codec=numcodecs.JSON(),
                                 **kwds)
        else:
            if _type + key in group:
                del group[_type + key]
            group.create_dataset(
                _type + key,
                data=tmp,
                **kwds)

    for key, value in dictionary.items():
        if isinstance(value, dict):
            dict2zarrgroup(value, group.create_group(key),
                           **kwds)
        elif isinstance(value, DictionaryTreeBrowser):
            dict2zarrgroup(value.as_dictionary(),
                           group.create_group(key),
                           **kwds)
        elif isinstance(value, BaseSignal):
            kn = key if key.startswith('_sig_') else '_sig_' + key
            write_signal(value, group.require_group(kn))
        elif isinstance(value, (np.ndarray, zarr.Array, da.Array)):
            overwrite_dataset(group, value, key, **kwds)
        elif value is None:
            group.attrs[key] = '_None_'
        elif isinstance(value, bytes):
            try:
                # binary string if has any null characters (otherwise not
                # supported by zarr)
                value.index(b'\x00')
                group.attrs['_bs_' + key] = None
            except ValueError:
                group.attrs[key] = value.decode()
        elif isinstance(value, str):
            group.attrs[key] = value
        elif isinstance(value, AxesManager):
            dict2zarrgroup(value.as_dictionary(),
                           group.create_group('_hspy_AxesManager_' + key),
                           **kwds)
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
                if isinstance(value, np.int64):
                    value = int(value)
                group.attrs[key] = value
            except BaseException:
                _logger.exception(
                    "The zarr writer could not write the following "
                    "information in the file: %s : %s", key, value)


def get_signal_chunks(shape, dtype, signal_axes=None):
    """Function that calculates chunks for the signal,
     preferably at least one chunk per signal space.

    Parameters
    ----------
    shape : tuple
        the shape of the dataset to be sored / chunked
    dtype : {dtype, string}
        the numpy dtype of the data
    signal_axes: {None, iterable of ints}
        the axes defining "signal space" of the dataset. If None, the default
        zarr chunking is performed.
    """
    typesize = np.dtype(dtype).itemsize
    if signal_axes is None:
        return None
    # chunk size larger than 1 Mb https://zarr.readthedocs.io/en/stable/tutorial.html#chunk-optimizations
    # shooting for 100 Mb chunks
    total_size = np.prod(shape)*typesize
    if total_size < 1e8:  # 1 mb
        return None
    CHUNK_MAX = 1024 * 1024
    want_to_keep = multiply([shape[i] for i in signal_axes]) * typesize
    if want_to_keep >= CHUNK_MAX:
        chunks = [1 for _ in shape]
        for i in signal_axes:
            chunks[i] = shape[i]
        return tuple(chunks)

    chunks = [i for i in shape]
    idx = 0
    navigation_axes = tuple(i for i in range(len(shape)) if i not in
                            signal_axes)
    nchange = len(navigation_axes)
    while True:
        chunk_bytes = multiply(chunks) * typesize

        if chunk_bytes < CHUNK_MAX:
            break

        if multiply([chunks[i] for i in navigation_axes]) == 1:
            break
        change = navigation_axes[idx % nchange]
        chunks[change] = np.ceil(chunks[change] / 2.0)
        idx += 1
    return tuple(int(x) for x in chunks)


def overwrite_dataset(group, data, key, signal_axes=None, chunks=None, **kwds):
    if chunks is None:
        if isinstance(data, da.Array):
            # For lazy dataset, by default, we use the current dask chunking
            chunks = tuple([c[0] for c in data.chunks])
        else:
            # If signal_axes=None, use automatic h5py chunking, otherwise
            # optimise the chunking to contain at least one signal per chunk
            chunks = get_signal_chunks(data.shape, data.dtype, signal_axes)
    got_data = False
    if data.dtype == np.dtype('O'):
        # For saving ragged array
        # https://zarr.readthedocs.io/en/stable/tutorial.html?highlight=ragged%20array#ragged-arrays
        if chunks is None:
            chunks == 1
        these_kwds = kwds.copy()
        these_kwds.update(dict(dtype=object,
                               exact=True,
                               chunks=chunks))
        dset = group.require_dataset(key,
                                     data.shape,
                                     object_codec=numcodecs.VLenArray(int),
                                     **these_kwds)
        got_data=True

    while not got_data:
        try:
            these_kwds = kwds.copy()
            these_kwds.update(dict(shape=data.shape,
                                   dtype=data.dtype,
                                   exact=True,
                                   chunks=chunks,
                                   ))

            # If chunks is True, the `chunks` attribute of `dset` below
            # contains the chunk shape guessed by h5py
            dset = group.require_dataset(key, **these_kwds)
            got_data = True
        except TypeError:
            # if the shape or dtype/etc do not match,
            # we delete the old one and create new in the next loop run
            del group[key]
    if dset == data:
        # just a reference to already created thing
        pass
    else:
        _logger.info(f"Chunks used for saving: {dset.chunks}")
        if isinstance(data, da.Array):
            if data.chunks != dset.chunks:
                data = data.rechunk(dset.chunks)
            path = group._store.dir_path()+"/"+dset.path
            data.to_zarr(url=path,
                         overwrite=True,
                         **kwds)  # add in compression etc
        elif data.dtype == np.dtype('O'):
            group[key][:] = data[:]  # check lazy
        else:
            path = group._store.dir_path()+"/"+dset.path
            dset = zarr.open_array(path,
                                   mode="w",
                                   shape=data.shape,
                                   dtype=data.dtype,
                                   chunks=chunks,
                                   **kwds)
            dset[:] = data

def zarrgroup2dict(group, dictionary=None, lazy=False):
    if dictionary is None:
        dictionary = {}
    for key, value in group.attrs.items():
        if isinstance(value, bytes):
            value = value.decode()
        if isinstance(value, (np.string_, str)):
            if value == '_None_':
                value = None
        elif isinstance(value, np.bool_) or isinstance(value, bool):
            value = bool(value)
        elif isinstance(value, np.ndarray) and value.dtype.char == "S":
            # Convert strings to unicode
            value = value.astype("U")
            if value.dtype.str.endswith("U1"):
                value = value.tolist()
        elif isinstance(value, np.int):
            value = int(value)
        # skip signals - these are handled below.
        if key.startswith('_sig_'):
            pass
        elif key.startswith('_list_empty_'):
            dictionary[key[len('_list_empty_'):]] = []
        elif key.startswith('_tuple_empty_'):
            dictionary[key[len('_tuple_empty_'):]] = ()
        elif key.startswith('_bs_'):
            dictionary[key[len('_bs_'):]] = value.tobytes()
        # The following two elif stataments enable reading date and time from
        # v < 2 of HyperSpy's metadata specifications
        elif key.startswith('_datetime_date'):
            date_iso = datetime.date(
                *ast.literal_eval(value[value.index("("):])).isoformat()
            dictionary[key.replace("_datetime_", "")] = date_iso
        elif key.startswith('_datetime_time'):
            date_iso = datetime.time(
                *ast.literal_eval(value[value.index("("):])).isoformat()
            dictionary[key.replace("_datetime_", "")] = date_iso
        else:
            dictionary[key] = value
    if not isinstance(group, zarr.Array):
        for key in group.keys():
            if key.startswith('_sig_'):
                from hyperspy.io import dict2signal
                dictionary[key[len('_sig_'):]] = (
                    dict2signal(zarrgroup2signaldict(
                        group[key], lazy=lazy)))
            elif isinstance(group[key], zarr.Array):
                dat = group[key]
                kn = key
                if key.startswith("_list_"):
                    #dat = dat.asstr()[:]
                    ans = np.array(dat)
                    ans = ans.tolist()
                    kn = key[6:]
                elif key.startswith("_tuple_"):
                    ans = np.array(dat)
                    ans = tuple(ans.tolist())
                    kn = key[7:]
                elif dat.dtype.char == "S":
                    ans = np.array(dat)
                    try:
                        ans = ans.astype("U")
                    except UnicodeDecodeError:
                        # There are some strings that must stay in binary,
                        # for example dill pickles. This will obviously also
                        # let "wrong" binary string fail somewhere else...
                        pass
                elif lazy:
                    ans = da.from_array(dat, chunks=dat.chunks)
                else:
                    ans = np.array(dat)
                dictionary[kn] = ans
            elif key.startswith('_hspy_AxesManager_'):
                dictionary[key[len('_hspy_AxesManager_'):]] = AxesManager(
                    [i for k, i in sorted(iter(
                        zarrgroup2dict(
                            group[key], lazy=lazy).items()
                    ))])
            elif key.startswith('_list_'):
                dictionary[key[7 + key[6:].find('_'):]] = \
                    [i for k, i in sorted(iter(
                        zarrgroup2dict(
                            group[key], lazy=lazy).items()
                    ))]
            elif key.startswith('_tuple_'):
                dictionary[key[8 + key[7:].find('_'):]] = tuple(
                    [i for k, i in sorted(iter(
                        zarrgroup2dict(
                            group[key], lazy=lazy).items()
                    ))])
            else:
                dictionary[key] = {}
                zarrgroup2dict(
                    group[key],
                    dictionary[key],
                    lazy=lazy)
    return dictionary


def write_signal(signal, group, **kwds):
    """Writes a hyperspy signal to a zarr group"""

    group.attrs.update(get_object_package_info(signal))
    if default_version < LooseVersion("1.2"):
        metadata = "mapped_parameters"
        original_metadata = "original_parameters"
    else:
        metadata = "metadata"
        original_metadata = "original_metadata"

    if 'compressor' not in kwds:
        kwds['compressor'] = None

    for axis in signal.axes_manager._axes:
        axis_dict = axis.get_axis_dictionary()
        coord_group = group.create_group(
            'axis-%s' % axis.index_in_array)
        dict2zarrgroup(axis_dict, coord_group, **kwds)
    mapped_par = group.create_group(metadata)
    metadata_dict = signal.metadata.as_dictionary()
    overwrite_dataset(group, signal.data, 'data',
                      signal_axes=signal.axes_manager.signal_indices_in_array,
                      **kwds)
    if default_version < LooseVersion("1.2"):
        metadata_dict["_internal_parameters"] = \
            metadata_dict.pop("_HyperSpy")
    # Remove chunks from the kwds since it wouldn't have the same rank as the
    # dataset and can't be used
    kwds.pop('chunks', None)
    dict2zarrgroup(metadata_dict, mapped_par, **kwds)
    original_par = group.create_group(original_metadata)
    dict2zarrgroup(signal.original_metadata.as_dictionary(), original_par,
                   **kwds)
    learning_results = group.create_group('learning_results')
    dict2zarrgroup(signal.learning_results.__dict__,
                  learning_results, **kwds)
    if hasattr(signal, 'peak_learning_results'):
        peak_learning_results = group.create_group(
            'peak_learning_results')
        dict2zarrgroup(signal.peak_learning_results.__dict__,
                      peak_learning_results, **kwds)

    if len(signal.models):
        model_group = group.file.require_group('Analysis/models')
        dict2zarrgroup(signal.models._models.as_dictionary(),
                      model_group, **kwds)
        for model in model_group.values():
            model.attrs['_signal'] = group.name


def file_writer(filename, signal, *args, **kwds):
    """Writes data to hyperspy's zarr format

    Parameters
    ----------
    filename: str
    signal: a BaseSignal instance
    *args, optional
    **kwds, optional
    """
    store = zarr.storage.NestedDirectoryStore(filename,)
    f = zarr.group(store=store, overwrite=True)
    f.attrs['file_format'] = "Zarr"
    f.attrs['file_format_version'] = version
    exps = f.create_group('Experiments')
    group_name = signal.metadata.General.title if \
        signal.metadata.General.title else '__unnamed__'
    # / is a invalid character, see #942
    if "/" in group_name:
        group_name = group_name.replace("/", "-")
    expg = exps.create_group(group_name)

    # Add record_by metadata for backward compatibility
    smd = signal.metadata.Signal
    if signal.axes_manager.signal_dimension == 1:
        smd.record_by = "spectrum"
    elif signal.axes_manager.signal_dimension == 2:
        smd.record_by = "image"
    else:
        smd.record_by = ""
    try:
        write_signal(signal, expg, **kwds)
    except BaseException:
        raise
    finally:
        del smd.record_by


