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

from packaging.version import Version
import warnings
import logging
from functools import partial
import h5py
import numpy as np
import dask.array as da
from h5py import Dataset, File, Group

from hyperspy.io_plugins.hierarchical import HierarchicalWriter, HierarchicalReader, _overwrite_dataset, version
from hyperspy.misc.utils import multiply

_logger = logging.getLogger(__name__)


# Plugin characteristics
# ----------------------
format_name = 'HSPY'
description = \
    'The default file format for HyperSpy based on the HDF5 standard'
full_support = False
# Recognised file extension
file_extensions = ['hspy', 'hdf5']
default_extension = 0
# Writing capabilities
writes = True
non_uniform_axis = True
version = version
# ----------------------

# -----------------------
# File format description
# -----------------------
# The root must contain a group called Experiments.
# The experiments group can contain any number of subgroups.
# Each subgroup is an experiment or signal.
# Each subgroup must contain at least one dataset called data.
# The data is an array of arbitrary dimension.
# In addition, a number equal to the number of dimensions of the data
# dataset + 1 of empty groups called coordinates followed by a number
# must exist with the following attributes:
#    'name'
#    'offset'
#    'scale'
#    'units'
#    'size'
#    'index_in_array'
# Alternatively to 'offset' and 'scale', the coordinate groups may
# contain an 'axis' vector attribute defining the axis points.
# The experiment group contains a number of attributes that will be
# directly assigned as class attributes of the Signal instance. In
# addition the experiment groups may contain 'original_metadata' and
# 'metadata'-subgroup that will be assigned to the same name attributes 
# of the Signal instance as a Dictionary Browser.
# The Experiments group can contain attributes that may be common to all
# the experiments and that will be accessible as attributes of the
# Experiments instance.
#
# CHANGES
#
# v3.1
# - add read support for non-uniform DataAxis defined by 'axis' vector
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
# - Store the navigate attribute
# - record_by is stored only for backward compatibility but the axes navigate
#   attribute takes precendence over record_by for files with version >= 2.1
# v1.3
# ----
# - Added support for lists, tuples and binary strings

not_valid_format = 'The file is not a valid HyperSpy hdf5 file'

current_file_version = None  # Format version of the file being read
default_version = Version(version)


def get_signal_chunks(shape,
                      dtype,
                      signal_axes=None):
    """Function that calculates chunks for the signal, preferably at least one
    chunk per signal space.

    Parameters
    ----------
    shape : tuple
        the shape of the dataset to be sored / chunked
    dtype : {dtype, string}
        the numpy dtype of the data
    signal_axes: {None, iterable of ints}
        the axes defining "signal space" of the dataset. If None, the default
        h5py chunking is performed.
    """
    typesize = np.dtype(dtype).itemsize
    if signal_axes is None:
        return h5py._hl.filters.guess_chunk(shape, None, typesize)

    # largely based on the guess_chunk in h5py
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


def _get_object_dset(group, data, key, chunks, **kwds):
    """Creates a Dataset or Zarr Array object for saving ragged data
    Parameters
    ----------
    self
    group
    data
    key
    chunks
    kwds

    Returns
    -------

    """
    # For saving ragged array
    if chunks is None:
        chunks = 1
    dset = group.require_dataset(key,
                                 chunks,
                                 dtype=h5py.special_dtype(vlen=data[0].dtype),
                                 **kwds)
    return dset


def _store_data(data, dset, group, key, chunks, **kwds):
    if isinstance(data, da.Array):
        if data.chunks != dset.chunks:
            data = data.rechunk(dset.chunks)
        da.store(data, dset)
    elif data.flags.c_contiguous:
        dset.write_direct(data)
    else:
        dset[:] = data


overwrite_dataset = partial(_overwrite_dataset,
                            get_signal_chunks=get_signal_chunks,
                            get_object_dset=_get_object_dset,
                            store_data=_store_data)


class HyperspyReader(HierarchicalReader):
    def __init__(self, file):
        super().__init__(file)
        self.Dataset = Dataset
        self.Group = Group
        self.unicode_kwds = {"dtype": h5py.special_dtype(vlen=str)}


class HyperspyWriter(HierarchicalWriter):
    """An object used to simplify and orgainize the process for
    writing a hyperspy signal.  (.hspy format)
    """
    def __init__(self,
                 file,
                 signal,
                 expg,
                 **kwds):
        super().__init__(file,
                       signal,
                       expg,
                       **kwds)
        self.Dataset = Dataset
        self.Group = Group
        self.unicode_kwds = {"dtype": h5py.special_dtype(vlen=str)}
        self.ragged_kwds = {"dtype": h5py.special_dtype(vlen=signal.data[0].dtype)}
        self.overwrite_dataset = overwrite_dataset

def file_reader(
                filename,
                lazy=False,
                **kwds):
    """Read data from hdf5 files saved with the hyperspy hdf5 format specification

     Parameters
    ----------
    filename: str
    lazy: bool
        Load image lazily using dask
    **kwds, optional
    """
    try:
        # in case blosc compression is used
        import hdf5plugin
    except ImportError:
        pass
    mode = kwds.pop('mode', 'r')
    f = File(filename, mode=mode, **kwds)
    # Getting the format version here also checks if it is a valid HSpy
    # hdf5 file, so the following two lines must not be deleted or moved
    # elsewhere.
    reader = HyperspyReader(f)
    if reader.version > Version(version):
        warnings.warn(
            "This file was written using a newer version of the "
            "HyperSpy hdf5 file format. I will attempt to load it, but, "
            "if I fail, it is likely that I will be more successful at "
            "this and other tasks if you upgrade me.")
    exp_dict_list = reader.read(lazy=lazy)
    if not lazy:
        f.close()
    return exp_dict_list


def file_writer(filename, signal, *args, **kwds):
    """Writes data to hyperspy's hdf5 format

    Parameters
    ----------
    filename: str
    signal: a BaseSignal instance
    *args, optional
    **kwds, optional
    """
    if 'compression' not in kwds:
        kwds['compression'] = 'gzip'
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

        # Add record_by metadata for backward compatibility
        smd = signal.metadata.Signal
        if signal.axes_manager.signal_dimension == 1:
            smd.record_by = "spectrum"
        elif signal.axes_manager.signal_dimension == 2:
            smd.record_by = "image"
        else:
            smd.record_by = ""
        try:
            if "shuffle" not in kwds:
                kwds["shuffle"] = True
            writer = HyperspyWriter(f, signal, expg, **kwds)
            writer.write()
            #write_signal(signal, expg, **kwds)
        except BaseException:
            raise
        finally:
            del smd.record_by
