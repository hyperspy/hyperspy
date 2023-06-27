# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import logging
from collections.abc import MutableMapping

import dask.array as da
import numcodecs
import zarr

from hyperspy.io_plugins._hierarchical import (
    HierarchicalWriter, HierarchicalReader, version
    )


_logger = logging.getLogger(__name__)


# Plugin characteristics
# ----------------------
format_name = 'ZSpy'
description = 'A default file format for HyperSpy based on the zarr standard'
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

    _file_type = format_name.lower()

    def __init__(self, file):
        super().__init__(file)
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
    def _store_data(data, dset, group, key, chunks):
        """Write data to zarr format."""
        if isinstance(data, da.Array):
            if data.chunks != dset.chunks:
                data = data.rechunk(dset.chunks)
            # lock=False is necessary with the distributed scheduler
            data.store(dset, lock=False)
        else:
            dset[:] = data


def file_writer(filename, signal, close_file=True, **kwds):
    """Writes data to hyperspy's zarr format.

    Parameters
    ----------
    filename : str
        The name of the file used to save the signal.
    signal : a BaseSignal instance
        The signal to save.
    chunks : tuple of integer or None, default: None
        Define the chunking used for saving the dataset. If None, calculates
        chunks for the signal, with preferably at least one chunk per signal
        space.
    compressor : numcodecs compression
        The default is to use a Blosc compressor.
    close_file : bool, default: True
        Close the file after writing.
    write_dataset : bool, default: True
        If True, write the data, otherwise, don't write it. Useful to
        save attributes without having to write the whole dataset.
    **kwds
        The keyword argument are passed to the
        :py:meth:`zarr.hierarchy.Group.require_dataset` function.
    """
    if "compressor" not in kwds:
        from numcodecs import Blosc
        kwds["compressor"] = Blosc(
            cname='zstd', clevel=1, shuffle=Blosc.SHUFFLE
            )

    if isinstance(filename, MutableMapping):
        store = filename
    else:
        store = zarr.storage.NestedDirectoryStore(filename,)
    write_dataset = kwds.get('write_dataset', True)
    if not isinstance(write_dataset, bool):
        raise ValueError("`write_dataset` argument must a boolean.")
    mode = 'w' if kwds.get('write_dataset', True) else 'a'

    _logger.debug(f'File mode: {mode}')
    _logger.debug(f'Zarr store: {store}')

    f = zarr.open_group(store=store, mode=mode)
    f.attrs['file_format'] = "ZSpy"
    f.attrs['file_format_version'] = version
    exps = f.require_group('Experiments')
    group_name = signal.metadata.General.title if \
        signal.metadata.General.title else '__unnamed__'
    # / is a invalid character, see #942
    if "/" in group_name:
        group_name = group_name.replace("/", "-")
    expg = exps.require_group(group_name)

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

    if isinstance(store, (zarr.ZipStore, zarr.DBMStore, zarr.LMDBStore)):
        if close_file:
            store.close()
        else:
            store.flush()


def file_reader(filename, lazy=False, **kwds):
    """Read data from zspy files saved with the hyperspy zspy format
    specification.

    Parameters
    ----------
    filename: str
    lazy: bool
        Load image lazily using dask
    **kwds, optional
    """
    mode = kwds.pop('mode', 'r')
    try:
        f = zarr.open(filename, mode=mode, **kwds)
    except BaseException:
        _logger.error(
            "The file can't be read. It may be possible that the zspy file is "
            "saved with a different store than a zarr directory store. Try "
            "passing a different zarr store instead of the file name."
            )
        raise

    reader = ZspyReader(f)

    return reader.read(lazy=lazy)
