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
from zarr import Array as Dataset
from zarr import open as File
import numpy as np
import dask.array as da
from traits.api import Undefined
from hyperspy.misc.utils import ensure_unicode, multiply, get_object_package_info
from hyperspy.axes import AxesManager
from hyperspy.io_plugins.hspy import hdfgroup2signaldict, file_reader, write_signal, overwrite_dataset, get_signal_chunks
import numcodecs

from hyperspy.io_plugins.hspy import version

_logger = logging.getLogger(__name__)


# Plugin characteristics
# ----------------------
format_name = 'ZSpy'
description = \
    'A default file format for HyperSpy based on the zarr standard'
full_support = False
# Recognised file extension
file_extensions = ['zspy']
default_extension = 0
# Writing capabilities
non_uniform_axis = True
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
# addition the experiment groups may contain 'original_metadata' and
# 'metadata'subgroup that will be
# assigned to the same name attributes of the Signal instance as a
# Dictionary Browsers
# The Experiments group can contain attributes that may be common to all
# the experiments and that will be accessible as attributes of the
# Experiments instance

def get_object_dset(group, data, key, chunks, **kwds):
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
        return data, dset


def store_data(data, dset, group, key, chunks, **kwds):
    if isinstance(data, da.Array):
        if data.chunks != dset.chunks:
            data = data.rechunk(dset.chunks)
        path = group._store.dir_path() + "/" + dset.path
        data.to_zarr(url=path,
                     overwrite=True,
                     **kwds)  # add in compression etc
    elif data.dtype == np.dtype('O'):
        group[key][:] = data[:]  # check lazy
    else:
        path = group._store.dir_path() + "/" + dset.path
        dset = zarr.open_array(path,
                               mode="w",
                               shape=data.shape,
                               dtype=data.dtype,
                               chunks=chunks,
                               **kwds)
        dset[:] = data

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
def file_writer(filename, signal, *args, **kwds):
    """Writes data to hyperspy's zarr format
    Parameters
    ----------
    filename: str
    signal: a BaseSignal instance
    *args, optional
    **kwds, optional
    """
    if "compressor" not in kwds:
        from numcodecs import Blosc
        kwds["compressor"] = Blosc(cname='zstd', clevel=1)
    store = zarr.storage.NestedDirectoryStore(filename,)
    f = zarr.group(store=store, overwrite=True)
    f.attrs['file_format'] = "ZSpy"
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
        write_signal(signal, expg, f, **kwds)
    except BaseException:
        raise
    finally:
        del smd.record_by