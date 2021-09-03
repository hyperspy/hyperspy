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
from zarr import Array, Group
from zarr import open as File
import numpy as np
import dask.array as da
from traits.api import Undefined
from hyperspy.misc.utils import ensure_unicode, multiply, get_object_package_info
from hyperspy.axes import AxesManager
from hyperspy.io_plugins.hspy import HyperspyReader, HyperspyWriter
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


class ZspyWriter(HyperspyWriter):
    def __init__(self,
                 file,
                 signal,
                 expg, **kwargs):
        super().__init__(file, signal, expg, **kwargs)
        self.Dataset = Array

    def write_signal(self,
                     signal,
                     group,
                     **kwds):
        """Overrides the hyperspy store data function for using zarr as the backend
        """

        group.attrs.update(get_object_package_info(signal))
        metadata = "metadata"
        original_metadata = "original_metadata"

        if 'compressor' not in kwds:
            kwds['compressor'] = None

        for axis in signal.axes_manager._axes:
            axis_dict = axis.get_axis_dictionary()
            coord_group = group.create_group(
                'axis-%s' % axis.index_in_array)
            self.dict2hdfgroup(axis_dict, coord_group, **kwds)
        mapped_par = group.create_group(metadata)
        metadata_dict = signal.metadata.as_dictionary()
        self.overwrite_dataset(group, signal.data, 'data',
                          signal_axes=signal.axes_manager.signal_indices_in_array,
                          **kwds)
        # Remove chunks from the kwds since it wouldn't have the same rank as the
        # dataset and can't be used
        kwds.pop('chunks', None)
        self.dict2hdfgroup(metadata_dict, mapped_par, **kwds)
        original_par = group.create_group(original_metadata)
        self.dict2hdfgroup(signal.original_metadata.as_dictionary(), original_par,
                      **kwds)
        learning_results = group.create_group('learning_results')
        self.dict2hdfgroup(signal.learning_results.__dict__,
                      learning_results, **kwds)

        if len(signal.models):
            model_group = self.file.require_group('Analysis/models')
            self.dict2hdfgroup(signal.models._models.as_dictionary(),
                               model_group, **kwds)
            for model in model_group.values():
                model.attrs['_signal'] = group.name

    def store_data(self,
                   data,
                   dset,
                   group,
                   key,
                   chunks,
                   **kwds):
        """Overrides the hyperspy store data function for using zarr as the backend
        """
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

    def get_object_dset(self, group, data, key, chunks, **kwds):
        """Overrides the hyperspy get object dset function for using zarr as the backend
        """
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

    def get_signal_chunks(self, shape, dtype, signal_axes=None):
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
        total_size = np.prod(shape) * typesize
        if total_size < 1e8:  # 1 mb
            return None

    def parse_structure(self, key, group, value, _type, **kwds):
        from hyperspy.signal import BaseSignal
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
            self.dict2hdfgroup(dict(zip(
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
        writer = ZspyWriter(f, signal, expg, **kwds)
        writer.write()
    except BaseException:
        raise
    finally:
        del smd.record_by


def file_reader(filename,
                lazy=False,
                **kwds):
    """Read data from zspy files saved with the hyperspy zspy format specification
    Parameters
    ----------
    filename: str
    lazy: bool
        Load image lazily using dask
    **kwds, optional
    """
    mode = kwds.pop('mode', 'r')
    f = zarr.open(filename, mode=mode, **kwds)
    reader = HyperspyReader(f, Group, Dataset=Array)
    if reader.version > version:
        warnings.warn(
            "This file was written using a newer version of the "
            "HyperSpy zspy file format. I will attempt to load it, but, "
            "if I fail, it is likely that I will be more successful at "
            "this and other tasks if you upgrade me.")
    return reader.read(lazy=lazy)
