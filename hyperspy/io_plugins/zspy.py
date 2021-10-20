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

import warnings
import logging
from packaging.version import Version
from collections import MutableMapping

import dask.array as da
import numcodecs
import numpy as np
import zarr

from hyperspy.io_plugins._hierarchical import (
    HierarchicalWriter, HierarchicalReader, version
    )


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


class ZspyReader(HierarchicalReader):
    def __init__(self, file):
        super(ZspyReader, self).__init__(file)
        self.Dataset = zarr.Array
        self.Group = zarr.Group


class ZspyWriter(HierarchicalWriter):
    target_size = 1e8
    def __init__(self,
                 file,
                 signal,
                 expg, **kwargs):
        super().__init__(file, signal, expg, **kwargs)
        self.Dataset = zarr.Array
        self.unicode_kwds = {"dtype": object, "object_codec": numcodecs.JSON()}
        self.ragged_kwds = {"dtype": object,
                            "object_codec": numcodecs.VLenArray(int),
                            "exact":  True}
        self.target_size = 1e8

    @staticmethod
    def _get_object_dset(group, data, key, chunks, **kwds):
        """Creates a Zarr Array object for saving ragged data"""
        these_kwds = kwds.copy()
        these_kwds.update(dict(dtype=object,
                               exact=True,
                               chunks=chunks))
        dset = group.require_dataset(key,
                                     data.shape,
                                     object_codec=numcodecs.VLenArray(int),
                                     **these_kwds)
        return dset

    @staticmethod
    def _store_data(data, dset, group, key, chunks, **kwds):
        """Overrides the hyperspy store data function for using zarr format."""
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


def file_writer(filename,
                signal,
                *args,
                **kwds):
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
    if ("write_to_storage" in kwds and kwds["write_to_storage"]) or isinstance(filename, MutableMapping):
        if isinstance(filename, MutableMapping):
            store = filename.path
        else:
            store = filename
    else:
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
    reader = ZspyReader(f)
    if reader.version > Version(version):
        warnings.warn(
            "This file was written using a newer version of the "
            "HyperSpy zspy file format. I will attempt to load it, but, "
            "if I fail, it is likely that I will be more successful at "
            "this and other tasks if you upgrade me.")
    return reader.read(lazy=lazy)
