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
from packaging.version import Version
from pathlib import Path

import dask.array as da
import h5py

from hyperspy.io_plugins._hierarchical import (
    # hyperspy.io_plugins.hspy.get_signal_chunks is in the hyperspy public API
    HierarchicalWriter, HierarchicalReader, version, get_signal_chunks
    )


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


class HyperspyReader(HierarchicalReader):

    _file_type = format_name.lower()

    def __init__(self, file):
        super().__init__(file)
        self.Dataset = h5py.Dataset
        self.Group = h5py.Group
        self.unicode_kwds = {"dtype": h5py.special_dtype(vlen=str)}


class HyperspyWriter(HierarchicalWriter):
    """
    An object used to simplify and organize the process for
    writing a hyperspy signal.  (.hspy format)
    """
    target_size = 1e6

    def __init__(self, file, signal, expg, **kwds):
        super().__init__(file, signal, expg, **kwds)
        self.Dataset = h5py.Dataset
        self.Group = h5py.Group
        self.unicode_kwds = {"dtype": h5py.special_dtype(vlen=str)}
        self.ragged_kwds = {"dtype": h5py.special_dtype(vlen=signal.data[0].dtype)}


    @staticmethod
    def _store_data(data, dset, group, key, chunks):
        if isinstance(data, da.Array):
            if data.chunks != dset.chunks:
                data = data.rechunk(dset.chunks)
            da.store(data, dset)
        elif data.flags.c_contiguous:
            dset.write_direct(data)
        else:
            dset[:] = data

    @staticmethod
    def _get_object_dset(group, data, key, chunks, **kwds):
        """Creates a h5py dataset object for saving ragged data"""
        # For saving ragged array
        if chunks is None:
            chunks = 1
        dset = group.require_dataset(key,
                                     chunks,
                                     dtype=h5py.special_dtype(vlen=data[0].dtype),
                                     **kwds)
        return dset


def file_reader(filename, lazy=False, **kwds):
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
    f = h5py.File(filename, mode=mode, **kwds)

    reader = HyperspyReader(f)
    exp_dict_list = reader.read(lazy=lazy)
    if not lazy:
        f.close()

    return exp_dict_list


def file_writer(filename, signal, close_file=True, **kwds):
    """Writes data to hyperspy's hdf5 format

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
    close_file : bool, default: True
        Close the file after writing.
    write_dataset : bool, default: True
        If True, write the data, otherwise, don't write it. Useful to
        save attributes without having to write the whole dataset.
    **kwds
        The keyword argument are passed to the
        :py:meth:`h5py.Group.require_dataset` function.
    """
    if 'compression' not in kwds:
        kwds['compression'] = 'gzip'

    if "shuffle" not in kwds:
        # Use shuffle by default to improve compression
        kwds["shuffle"] = True

    folder = signal.tmp_parameters.get_item('original_folder', '')
    fname = signal.tmp_parameters.get_item('original_filename', '')
    ext = signal.tmp_parameters.get_item('original_extension', '')
    original_path = Path(folder, f"{fname}.{ext}")

    f = None
    if (signal._lazy and Path(filename).absolute() == original_path):
        f = signal._get_file_handle(warn=False)
        if f is not None and f.mode == 'r':
            # when the file is read only, force to reopen it in writing mode
            raise OSError("File opened in read only mode. To overwrite file "
                          "with lazy signal, use `mode='a'` when loading the "
                          "signal.")

    if f is None:
        write_dataset = kwds.get('write_dataset', True)
        if not isinstance(write_dataset, bool):
            raise ValueError("`write_dataset` argument must a boolean.")
        # with "write_dataset=False", we need mode='a', otherwise the dataset
        # will be flushed with using 'w' mode
        mode = kwds.get('mode', 'w' if write_dataset else 'a')
        if mode != 'a' and not write_dataset:
            raise ValueError("`mode='a'` is required to use "
                             "`write_dataset=False`.")
        f = h5py.File(filename, mode=mode)

    f.attrs['file_format'] = "HyperSpy"
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
        writer = HyperspyWriter(f, signal, expg, **kwds)
        writer.write()
    except BaseException:
        raise
    finally:
        del smd.record_by

    if close_file:
        f.close()

overwrite_dataset = HyperspyWriter.overwrite_dataset
