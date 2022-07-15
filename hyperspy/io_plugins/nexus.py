"""Nexus file reading, writing and inspection."""
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
#
import logging
import warnings
import os

import dask.array as da
import h5py
import numpy as np
import pprint
import traits.api as t

from hyperspy.io_plugins.hspy import (get_signal_chunks, overwrite_dataset)
from hyperspy.exceptions import VisibleDeprecationWarning


_logger = logging.getLogger(__name__)

# Plugin characteristics
# ----------------------
format_name = 'Nexus'
description = \
    'Read NXdata sets from Nexus files and metadata. Data and metadata can '\
    'also be examined from general hdf5 files'
full_support = False
# Recognised file extension
file_extensions = ['nxs', 'NXS']
default_extension = 0
# Writing capabilities:
writes = True
non_uniform_axis = False
# ----------------------



def _byte_to_string(value):
    """Decode a byte string.

    Parameters
    ----------
    value :  byte str

    Returns
    -------
    str
        decoded version of input value

    """
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError:
        text = value.decode('latin-1')
    return text.replace('\x00', '').rstrip()


def _parse_from_file(value, lazy=False):
    """To convert values from the hdf file to compatible formats.

    When reading string arrays we convert or keep string arrays as
    byte_strings (some io_plugins only supports byte-strings arrays so this
    ensures inter-compatibility across io_plugins)
    Arrays of length 1 - return the single value stored.
    Large datasets are returned as dask arrays if lazy=True.

    Parameters
    ----------
    value : input read from hdf file (array,list,tuple,string,int,float)
    lazy  : bool  {default: False}
        The lazy flag is only applied to values of size >=2

    Returns
    -------
    str,int, float, ndarray dask Array
        parsed value.

    """
    toreturn = value
    if isinstance(value, h5py.Dataset):
        if value.size < 2:
            toreturn = value[...].item()
        else:
            if lazy:
                if value.chunks:
                    toreturn = da.from_array(value, value.chunks)
                else:
                    chunks = get_signal_chunks(value.shape, value.dtype)
                    toreturn = da.from_array(value, chunks)
            else:
                toreturn = np.array(value)

    if isinstance(toreturn, np.ndarray) and value.shape == (1,):
        toreturn = toreturn[0]
    if isinstance(toreturn, bytes):
        toreturn = _byte_to_string(toreturn)
    if isinstance(toreturn, (int, float)):
        toreturn = toreturn
    if isinstance(toreturn, (np.ndarray)) and toreturn.dtype.char == "U":
        toreturn = toreturn.astype("S")
    return toreturn


def _parse_to_file(value):
    """Convert to a suitable format for writing to HDF5.

    For example unicode values are not compatible with hdf5 so conversion to
    byte strings is required.

    Parameters
    ----------
    value - input object to write to the hdf file

    Returns
    -------
    parsed value

    """
    totest = value
    toreturn = totest
    if isinstance(totest, (bytes, int, float)):
        toreturn = value
    if isinstance(totest, (list, tuple)):
        totest = np.array(value)
    if isinstance(totest, np.ndarray) and totest.dtype.char == "U":
        toreturn = np.array(totest).astype("S")
    elif isinstance(totest, (np.ndarray, da.Array)):
        toreturn = totest
    if isinstance(totest, str):
        toreturn = totest.encode("utf-8")
        toreturn = np.string_(toreturn)
    return toreturn


def _text_split(s, sep):
    """Split a string based of list of seperators.

    Parameters
    ----------
    s   : str
    sep : str  - seperator or list of seperators e.g. '.' or ['_','/']

    Returns
    -------
    list
       String sections split based on the seperators

    """
    stack = [s]
    for char in sep:
        pieces = []
        for substr in stack:
            pieces.extend(substr.split(char))
        stack = pieces
    if '' in stack:
        stack.remove('')
    return stack


def _getlink(h5group, rootkey, key):
    """Return the link target path.

    If a hdf group is a soft link or has a target attribute
    this method will return the target path. If no link is found
    return None.

    Returns
    -------
    str
        Soft link path if it exists, otherwise None

    """
    _target = None
    if rootkey != '/':
        if isinstance(h5group, h5py.Group):
            _link = h5group.get(key, getlink=True)
            if isinstance(_link, h5py.SoftLink):
                _target = _link.path
        if 'target' in h5group.attrs.keys():
            _target = _parse_from_file(h5group.attrs['target'])
            if not _target.startswith('/'):
                _target = '/' + _target
            if _target == rootkey:
                _target = None

    return _target


def _get_nav_list(data, dataentry):
    """Get the list with information of each axes of the dataset

    Parameters
    ----------
    data : hdf dataset
        the dataset to be loaded.
    dataentry : hdf group
        the group with corresponding attributes.

    Returns
    -------
    nav_list : list
        contains information about each axes.
    """

    detector_index = 0
    nav_list = []
    # list indices...
    axis_index_list = []
    if "axes" in dataentry.attrs.keys():
        axes_key = dataentry.attrs["axes"]
        axes_list = ["."]*data.ndim
        if isinstance(axes_key, np.ndarray):
            axes_keys = axes_key[:data.ndim]
            for i, num in enumerate(axes_keys):
                axes_list[i] = _parse_from_file(num)
        elif isinstance(axes_key, (str, bytes)):
            axes_list = _parse_from_file(axes_key).split(',')[:data.ndim]
        else:
            axes_list[0] = _parse_from_file(axes_key)

        named_axes = list(range(len(axes_list)))
        for i, ax in enumerate(axes_list):
            if ax != ".":
                index_name = ax + "_indices"
                if index_name in dataentry.attrs:
                    ind_in_array = int(dataentry.attrs[index_name])
                else:
                    ind_in_array = i
                axis_index_list.append(ind_in_array)
                if "units" in dataentry[ax].attrs:
                    units = _parse_from_file(dataentry[ax].attrs["units"])
                else:
                    units = ""

                navigation = True
                named_axes.remove(ind_in_array)

                if _is_numeric_data(dataentry[ax]):
                    if dataentry[ax].size > 1:
                        if _is_linear_axis(dataentry[ax]):
                            nav_list.append({
                                'size': data.shape[ind_in_array],
                                'index_in_array': ind_in_array,
                                'name': ax,
                                'scale': abs(dataentry[ax][1] -
                                             dataentry[ax][0]),
                                'offset': min(dataentry[ax][0],
                                              dataentry[ax][-1]),
                                'units': units,
                                'navigate': navigation
                                })
                        else:
                            nav_list.append({
                                'size': data.shape[ind_in_array],
                                'index_in_array': ind_in_array,
                                'name': ax,
                                'scale': 1,
                                'offset': 0,
                                'navigate': navigation
                                })
                    else:
                        nav_list.append({
                            'size': 1,
                            'index_in_array': ind_in_array,
                            'name': ax,
                            'scale': 1,
                            'offset': dataentry[ax][0],
                            'units': units,
                            'navigate': True
                            })
            else:
                if len(data.shape) == len(axes_list):
                    nav_list.append({
                            'size': data.shape[named_axes[detector_index]],
                            'index_in_array': named_axes[detector_index],
                            'scale': 1,
                            'offset': 0.0,
                            'units': '',
                            'navigate': False
                           })
                    detector_index = detector_index+1

    return nav_list


def _extract_hdf_dataset(group, dataset, lazy=False):
    """Import data from hdf path.

    Parameters
    ----------
    group : hdf group
        group from which to load the dataset
    dataset : str
        path to the dataset within the group
    lazy    : bool {default:True}
        If true use lazy opening, if false read into memory

    Returns
    -------
    dict
        A signal dictionary which can be used to instantiate a signal.

    """

    data = group[dataset]

    # exclude the dataset tagged by the signal attribute to avoid extracting
    # duplicated dataset, which is already loaded when loading NeXus data
    if 'signal' in data.parent.attrs.keys():
        data_key = data.parent.attrs['signal']
        if isinstance(data_key, bytes):
            data_key = data_key.decode()
        if dataset.split('/')[-1] == data_key:
            return

    nav_list = _get_nav_list(data, data.parent)

    if lazy:
        if "chunks" in data.attrs.keys():
            chunks = data.attrs["chunks"]
        else:
            signal_axes = [d['index_in_array'] for d in nav_list
                           if not d['navigate']]
            chunks = get_signal_chunks(data.shape, data.dtype, signal_axes)
        data_lazy = da.from_array(data, chunks=chunks)
    else:
        data_lazy = np.array(data)

    dictionary = {'data': data_lazy, 'metadata': {}, 'original_metadata': {},
                  'axes': nav_list}

    return dictionary


def _nexus_dataset_to_signal(group, nexus_dataset_path, lazy=False):
    """Load an NXdata set as a hyperspy signal.

    Parameters
    ----------
    group : hdf group containing the NXdata
    nexus_data_path : str
        Path to the NXdata set in the group
    lazy : bool, default : True
        lazy loading of data

    Returns
    -------
    dict
        A signal dictionary which can be used to instantiate a signal.

    """

    interpretation = None
    dataentry = group[nexus_dataset_path]
    if "signal" in dataentry.attrs.keys():
        if _is_int(dataentry.attrs["signal"]):
            data_key = "data"
        else:
            data_key = dataentry.attrs["signal"]
    else:
        _logger.info("No signal attr associated with NXdata will\
                     try assume signal name is data")
        if "data" not in dataentry.keys():
            raise ValueError("Signal attribute not found in NXdata and "
                             "attempt to find a default \"data\" key failed")
        else:
            data_key = "data"

    if "interpretation" in dataentry.attrs.keys():
        interpretation = _parse_from_file(dataentry.attrs["interpretation"])

    data = dataentry[data_key]
    nav_list = _get_nav_list(data, dataentry)

    if lazy:
        if "chunks" in data.attrs.keys():
            chunks = data.attrs["chunks"]
        else:
            signal_axes = [d['index_in_array'] for d in nav_list
                           if not d['navigate']]
            chunks = get_signal_chunks(data.shape, data.dtype, signal_axes)
        data_lazy = da.from_array(data, chunks=chunks)
    else:
        data_lazy = np.array(data)

    if not nav_list:
        for i in range(data.ndim):
            nav_list.append({
                    'size': data_lazy.shape[i],
                    'index_in_array': i,
                    'scale': 1,
                    'offset': 0.0,
                    'units': '',
                    'navigate': True
                   })
    title = _text_split(nexus_dataset_path, '/')[-1]
    metadata = {'General': {'title': title}}

    #
    # if interpretation - reset the nav axes
    # assume the last dimensions are the signal
    #
    if interpretation:
        for x in nav_list:
            x["navigate"] = True
        if interpretation == "spectrum":
            nav_list[-1]["navigate"] = False
        elif interpretation == "image":
            nav_list[-1]["navigate"] = False
            nav_list[-2]["navigate"] = False

    dictionary = {'data': data_lazy,
                  'axes': nav_list,
                  'metadata': metadata}
    return dictionary


def file_reader(filename, lazy=False, dataset_key=None, dataset_path=None,
                metadata_key=None,
                skip_array_metadata=False,
                nxdata_only=False,
                hardlinks_only=False,
                use_default=False,
                **kwds):
    """Read NXdata class or hdf datasets from a file and return signal(s).

    Note
    ----
    Loading all datasets can result in a large number of signals
    Please review your datasets and use the dataset_key to target
    the datasets of interest.
    "keys" is a special keywords and prepended with "fix" in the metadata
    structure to avoid any issues.

    Datasets are all arrays with size>2 (arrays, lists)

    Parameters
    ----------
    filename : str
        Input filename
    dataset_key  : None, str, list of strings, default : None
        If None all datasets are returned.
        If a string or list of strings is provided only items
        whose path contain the string(s) are returned. For example
        dataset_key = ["instrument", "Fe"] will return
        data entries with instrument or Fe in their hdf path.
    dataset_path : None, str, list of strings, default : None
        If None, no absolute path is searched.
        If a string or list of strings is provided items with the absolute
        paths specified will be returned. For example, dataset_path =
        ['/data/spectrum/Mn'], it returns the exact dataset with this path.
        It is not filtered by dataset_key, i.e. with dataset_key = ['Fe'],
        it still returns the specific dataset at '/data/spectrum/Mn'. It is
        empty if no dataset matching the absolute path provided is present.
    metadata_key: : None, str, list of strings, default : None
        Only return items from the original metadata whose path contain the
        strings .e.g metadata_key = ["instrument", "Fe"] will return
        all metadata entries with "instrument" or "Fe" in their hdf path.
    skip_array_metadata : bool, default : False
        Whether to skip loading metadata with an array entry. This is useful
        as metadata may contain large array that is redundant with the data.
    nxdata_only : bool, default : False
        If True only NXdata will be converted into a signal
        if False NXdata and any hdf datasets will be loaded as signals
    hardlinks_only : bool, default : False
        If True any links (soft or External) will be ignored when loading.
    use_default : bool, default : False
        If True and a default NXdata is defined in the file load this as a
        signal. This will ignore the other keyword options. If True and no
        default is defined the file will be loaded according to
        the keyword options.

    Returns
    -------
    dict : signal dictionary or list of signal dictionaries


    See Also
    --------
    * :py:meth:`~.io_plugins.nexus.list_datasets_in_file`
    * :py:meth:`~.io_plugins.nexus.read_metadata_from_file`


    """
    # search for NXdata sets...

    mapping = kwds.get('mapping', {})
    original_metadata = {}
    learning = {}
    fin = h5py.File(filename, "r")
    signal_dict_list = []

    if 'dataset_keys' in kwds:
        warnings.warn("The `dataset_keys` keyword is deprecated. "
                      "Use `dataset_key` instead.", VisibleDeprecationWarning)
        dataset_key = kwds['dataset_keys']

    if 'dataset_paths' in kwds:
        warnings.warn("The `dataset_paths` keyword is deprecated. "
                      "Use `dataset_path` instead.", VisibleDeprecationWarning)
        dataset_path = kwds['dataset_paths']

    if 'metadata_keys' in kwds:
        warnings.warn("The `metadata_keys` keyword is deprecated. "
                      "Use `metadata_key` instead.", VisibleDeprecationWarning)
        metadata_key = kwds['metadata_keys']

    dataset_key = _check_search_keys(dataset_key)
    dataset_path = _check_search_keys(dataset_path)
    metadata_key = _check_search_keys(metadata_key)
    original_metadata = _load_metadata(fin, lazy=lazy,
                                       skip_array_metadata=skip_array_metadata)
    # some default values...
    nexus_data_paths = []
    hdf_data_paths = []
    # check if a default dataset is defined
    if use_default:
        nexus_data_paths, hdf_data_paths = _find_data(fin,
                                                      search_keys=None,
                                                      hardlinks_only=False)
        nxentry = None
        nxdata = None
        if "attrs" in original_metadata:
            if "default" in original_metadata["attrs"]:
                nxentry = original_metadata["attrs"]["default"]
            else:
                rootlist = list(original_metadata.keys())
                rootlist.remove("attrs")
                if rootlist and len(rootlist) == 1:
                    nxentry == rootlist[0]
            if nxentry:
                if "default" in original_metadata[nxentry]["attrs"]:
                    nxdata = original_metadata[nxentry]["attrs"]["default"]
            if nxentry and nxdata:
                nxdata = "/"+nxentry+"/"+nxdata
            if nxdata:
                hdf_data_paths = []
                nexus_data_paths = [nxpath for nxpath in nexus_data_paths
                                    if nxdata in nxpath]

    # if no default found then search for the data as normal
    if not nexus_data_paths and not hdf_data_paths:
        nexus_data_paths, hdf_data_paths = \
            _find_data(fin, search_keys=dataset_key,
                       hardlinks_only=hardlinks_only,
                       absolute_path=dataset_path)

    for data_path in nexus_data_paths:
        dictionary = _nexus_dataset_to_signal(fin, data_path, lazy=lazy)
        entryname = _text_split(data_path, "/")[0]
        dictionary["mapping"] = mapping
        title = dictionary["metadata"]["General"]["title"]
        if entryname in original_metadata:
            if metadata_key is None:
                dictionary["original_metadata"] = \
                    original_metadata[entryname]
            else:
                dictionary["original_metadata"] = \
                    _find_search_keys_in_dict(original_metadata,
                                              search_keys=metadata_key)
            # test if it's a hyperspy_nexus format and update metadata
            # as appropriate.
            if "attrs" in original_metadata and \
                    "file_writer" in original_metadata["attrs"]:
                if original_metadata["attrs"]["file_writer"] == \
                        "hyperspy_nexus_v3":
                    orig_metadata = original_metadata[entryname]
                    if "auxiliary" in orig_metadata:
                        oma = orig_metadata["auxiliary"]
                        if "learning_results" in oma:
                            learning = oma["learning_results"]
                            dictionary["attributes"] = {}
                            dictionary["attributes"]["learning_results"] = \
                                learning
                        if "original_metadata" in oma:
                            if metadata_key is None:
                                dictionary["original_metadata"] = \
                                    (oma["original_metadata"])
                            else:
                                dictionary["original_metadata"] = \
                                    _find_search_keys_in_dict(
                                        (oma["original_metadata"]),
                                        search_keys=metadata_key)
                            # reconstruct the axes_list for axes_manager
                            for k, v in oma['original_metadata'].items():
                                if k.startswith('_sig_'):
                                    hyper_ax = [ax_v for ax_k, ax_v in v.items()
                                                if ax_k.startswith('_axes')]
                                    oma['original_metadata'][k]['axes'] = hyper_ax
                        if "hyperspy_metadata" in oma:
                            hyper_metadata = oma["hyperspy_metadata"]
                            hyper_metadata.update(dictionary["metadata"])
                            dictionary["metadata"] = hyper_metadata
        else:
            dictionary["original_metadata"] = {}

        signal_dict_list.append(dictionary)

    if not nxdata_only:
        for data_path in hdf_data_paths:
            datadict = _extract_hdf_dataset(fin, data_path, lazy=lazy)
            if datadict:
                title = data_path[1:].replace('/', '_')
                basic_metadata = {'General':
                                  {'original_filename':
                                   os.path.split(filename)[1],
                                   'title': title}}
                datadict["metadata"].update(basic_metadata)
                signal_dict_list.append(datadict)

    return signal_dict_list


def _is_linear_axis(data):
    """Check if the data is linearly incrementing.

    Parameters
    ----------
    data : dask or numpy array

    Returns
    -------
    bool
       True or False

    """
    steps = np.diff(data)
    est_steps = np.array([steps[0]]*len(steps))
    return np.allclose(est_steps, steps, rtol=1.0e-5)


def _is_numeric_data(data):
    """Check that data contains numeric data.

    Parameters
    ----------
    data : dask or numpy array

    Returns
    -------
    bool
        True or False

    """
    try:
        data.astype(float)
        return True
    except ValueError:
        return False


def _is_int(s):
    """Check that s in an integer.

    Parameters
    ----------
    s : python object to test

    Returns
    -------
    bool
        True or False

    """
    try:
        int(s)
        return True
    except ValueError:
        return False


def _check_search_keys(search_keys):
    if type(search_keys) is str:
        return [search_keys]
    elif type(search_keys) is list:
        if all(isinstance(key, str) for key in search_keys):
            return search_keys
        else:
            raise ValueError("key list provided is not a list of strings")
    elif search_keys is None:
        return search_keys
    else:
        raise ValueError("search keys must be None, a string, "
                         "or a list of strings")


def _find_data(group, search_keys=None, hardlinks_only=False,
               absolute_path=None):
    """Read from a nexus or hdf file and return a list of the dataset entries.

    The method iterates through group attributes and returns NXdata or
    hdf datasets of size >=2 if they're not already NXdata blocks
    and returns a list of the entries
    This is a convenience method to inspect a file to see which datasets
    are present rather than loading all the sets in the file as signals
    h5py.visit or visititems does not visit soft
    links or external links so an implementation of a recursive
    search is required. See https://github.com/h5py/h5py/issues/671


    Parameters
    ----------
    group : hdf group or File
    search_keys  : string, list of strings or None, default: None
        Only return items which contain the strings
        .e.g search_list = ["instrument","Fe"] will return
        hdf entries with instrument or Fe in their hdf path.
    hardlinks_only : bool , default : False
        Option to ignore links (soft or External) within the file.
    absolute_path : string, list of strings or None, default: None
        Return items with the exact specified absolute path

    Returns
    -------
    nx_dataset_list, hdf_dataset_list
        nx_dataset_list is a list of all NXdata paths
        hdf_dataset_list is a list of all hdf_datasets not linked to an
        NXdata set.

    """
    _check_search_keys(search_keys)
    _check_search_keys(absolute_path)
    all_hdf_datasets = []
    unique_hdf_datasets = []
    all_nx_datasets = []
    unique_nx_datasets = []
    rootname = ""

    def find_data_in_tree(group, rootname):
        for key, value in group.items():
            if rootname != "":
                rootkey = rootname + "/" + key
            else:
                rootkey = "/" + key
            if isinstance(value, h5py.Group):
                target = _getlink(group, rootkey, key)
                if "NX_class" in value.attrs:
                    if value.attrs["NX_class"] == b"NXdata" \
                            and "signal" in value.attrs.keys():
                        all_nx_datasets.append(rootkey)
                        if target is None:
                            unique_nx_datasets.append(rootkey)
                if hardlinks_only:
                    if target is None:
                        find_data_in_tree(value, rootkey)
                else:
                    find_data_in_tree(value, rootkey)
            else:
                if isinstance(value, h5py.Dataset):
                    if value.size >= 2:
                        target = _getlink(group, rootkey, key)
                        if not(value.dtype.type is str or
                                value.dtype.type is object):
                            all_hdf_datasets.append(rootkey)
                            if target is None:
                                unique_hdf_datasets.append(rootkey)
    # need to use custom recursive function as visititems in h5py
    # does not visit links
    find_data_in_tree(group, rootname)

    if search_keys is None and absolute_path is None:
        # return all datasets
        if hardlinks_only:
            # return only the stored data, no linked data
            return unique_nx_datasets, unique_hdf_datasets
        else:
            return all_nx_datasets, all_hdf_datasets

    elif type(search_keys) is list or type(absolute_path) is list:
        if hardlinks_only:
            # return only the stored data, no linked data
            nx_datasets = unique_nx_datasets
            hdf_datasets = unique_hdf_datasets
        else:
            nx_datasets = all_nx_datasets
            hdf_datasets = all_hdf_datasets

    matched_hdf = set()
    matched_nexus = set()
    # return data having the specified absolute paths
    if absolute_path is not None:
        matched_hdf.update([j for j in hdf_datasets
                            if any(s==j for s in absolute_path)])
        matched_nexus.update([j for j in nx_datasets
                              if any(s==j for s in absolute_path)])
    # return data which contains a search string
    if search_keys is not None:
        matched_hdf.update([j for j in hdf_datasets
                            if any(s in j for s in search_keys)])
        matched_nexus.update([j for j in nx_datasets
                              if any(s in j for s in search_keys)])

    matched_nexus = list(matched_nexus)
    matched_hdf = list(matched_hdf)

    return matched_nexus, matched_hdf


def _load_metadata(group, lazy=False, skip_array_metadata=False):
    """Search through a hdf group and return the group structure.

    h5py.visit or visititems does not visit soft
    links or external links so an implementation of a recursive
    search is required. See https://github.com/h5py/h5py/issues/671

    Parameters
    ----------
    group : hdf group
        location to load the metadata from
    lazy : bool , default : False
        Option for lazy loading
    skip_array_metadata : bool, default : False
        whether to skip loading array metadata

    Returns
    -------
    dict
        dictionary of group contents


    """
    rootname = ""

    def find_meta_in_tree(group, rootname, lazy=False,
                          skip_array_metadata=False):
        tree = {}
        for key, item in group.attrs.items():
            new_key = _fix_exclusion_keys(key)
            if "attrs" not in tree.keys():
                tree["attrs"] = {}
            tree["attrs"][new_key] = _parse_from_file(item, lazy=lazy)

        for key, item in group.items():
            if rootname != "":
                rootkey = rootname + "/" + key
            else:
                rootkey = "/" + key
            new_key = _fix_exclusion_keys(key)
            if type(item) is h5py.Dataset:
                if item.attrs:
                    if new_key not in tree.keys():
                        tree[new_key] = {}
                    # avoid loading array as metadata
                    if skip_array_metadata:
                        if item.size < 2 and item.ndim == 0:
                            tree[new_key]["value"] = _parse_from_file(item,
                                                                      lazy=lazy)
                    else:
                        tree[new_key]["value"] = _parse_from_file(item,
                                                                  lazy=lazy)

                    for k, v in item.attrs.items():
                        if "attrs" not in tree[new_key].keys():
                            tree[new_key]["attrs"] = {}
                        tree[new_key]["attrs"][k] = _parse_from_file(v,
                                                                     lazy=lazy)
                else:
                    # this is to support hyperspy where datasets are not saved
                    # with attributes
                    if skip_array_metadata:
                        if item.size < 2 and item.ndim == 0:
                            tree[new_key] = _parse_from_file(item, lazy=lazy)
                    else:
                        tree[new_key] = _parse_from_file(item, lazy=lazy)

            elif type(item) is h5py.Group:
                if "NX_class" in item.attrs:
                    if item.attrs["NX_class"] != b"NXdata":
                        tree[new_key] = find_meta_in_tree(item, rootkey,
                                                          lazy=lazy,
                                      skip_array_metadata=skip_array_metadata)
                else:
                    tree[new_key] = find_meta_in_tree(item, rootkey,
                                                      lazy=lazy,
                                      skip_array_metadata=skip_array_metadata)

        return tree
    extracted_tree = find_meta_in_tree(group, rootname, lazy=lazy,
                                       skip_array_metadata=skip_array_metadata)
    return extracted_tree


def _fix_exclusion_keys(key):
    """Exclude hyperspy specific keys.

    Signal and DictionaryBrowser break if a
    a key is a dict method - e.g. {"keys":2.0}.

    This method prepends the key with ``fix_`` so the information is
    still present to work around this issue

    Parameters
    ----------
    key : str

    Returns
    -------
    str

    """
    if key.startswith("keys"):
        return "fix_"+key
    else:
        return key


def _find_search_keys_in_dict(tree, search_keys=None):
    """Search through a dict for search keys.

    This is a convenience method to inspect a file for a value
    rather than loading the file as a signal

    Parameters
    ----------
    tree         : h5py File object
    search_keys  : string or list of strings
        Only return items which contain the strings
        .e.g search_keys = ["instrument","Fe"] will return
        hdf entries with instrument or Fe in their hdf path.

    Returns
    -------
    dict
        When search_list is specified only full paths
        containing one or more search_keys will be returned

    """
    _check_search_keys(search_keys)
    metadata_dict = {}
    rootname = ""

    # recursive function
    def find_searchkeys_in_tree(myDict, rootname):
        for key, value in myDict.items():
            if rootname != "":
                rootkey = rootname + "/" + key
            else:
                rootkey = key
            if type(search_keys) is list \
                    and any([s1 in rootkey for s1 in search_keys]):
                mod_keys = _text_split(rootkey, (".", "/"))
                # create the key, values in the dict
                p = metadata_dict
                for d in mod_keys[:-1]:
                    p = p.setdefault(d, {})
                p[mod_keys[-1]] = value
            if isinstance(value, dict):
                find_searchkeys_in_tree(value, rootkey)

    if search_keys is None:
        return tree
    else:
        find_searchkeys_in_tree(tree, rootname)
    return metadata_dict


def _write_nexus_groups(dictionary, group, skip_keys=None, **kwds):
    """Recursively iterate throuh dictionary and write groups to nexus.

    Parameters
    ----------
    dictionary : dict
        dictionary contents to store to hdf group
    group : hdf group
        location to store dictionary
    skip_keys : str or list of str
        the key(s) to skip when writing into the group
    **kwds : additional keywords
       additional keywords to pass to h5py.create_dataset method

    """
    if skip_keys is None:
        skip_keys = []
    elif isinstance(skip_keys, str):
        skip_keys = [skip_keys]

    for key, value in dictionary.items():
        if key == 'attrs' or key in skip_keys:
            # we will handle attrs later... and skip unwanted keys
            continue
        if isinstance(value, dict):
            if "attrs" in value:
                if "NX_class" in value["attrs"] and \
                        value["attrs"]["NX_class"] == "NXdata":
                    continue
            if 'value' in value.keys() \
                and not isinstance(value["value"], dict) \
                    and len(set(list(value.keys()) + ["attrs", "value"])) == 2:
                value = value["value"]
            else:
                _write_nexus_groups(value, group.require_group(key),
                                    skip_keys=skip_keys, **kwds)
        if isinstance(value, (list, tuple, np.ndarray, da.Array)):
            if all(isinstance(v, dict) for v in value):
                # a list of dictionary is from the axes of HyperSpy signal
                for i, ax_dict in enumerate(value):
                    ax_key = '_axes_' + str(i)
                    _write_nexus_groups(ax_dict, group.require_group(ax_key),
                                        skip_keys=skip_keys, **kwds)
            else:
                data = _parse_to_file(value)
                overwrite_dataset(group, data, key, chunks=None, **kwds)
        elif isinstance(value, (int, float, str, bytes)):
            group.create_dataset(key, data=_parse_to_file(value))
        else:
            if value is not None and value != t.Undefined and key not in group:
                _write_nexus_groups(value, group.require_group(key),
                                    skip_keys=skip_keys, **kwds)


def _write_nexus_attr(dictionary, group, skip_keys=None):
    """Recursively iterate through dictionary and write "attrs" dictionaries.

    This step is called after the groups and datasets have been created

    Parameters
    ----------
    dictionary : dict
        Input dictionary to be written to the hdf group
    group : hdf group
        location to store the attrs sections of the dictionary

    """
    if skip_keys is None:
        skip_keys = []
    elif isinstance(skip_keys, str):
        skip_keys = [skip_keys]

    for key, value in dictionary.items():
        if key == 'attrs':

            for k, v in value.items():
                group.attrs[k] = _parse_to_file(v)
        else:
            if isinstance(value, dict):
                if "attrs" in value:
                    if "NX_class" in value["attrs"] and \
                            value["attrs"]["NX_class"] == "NXdata":
                        continue
                if key not in skip_keys:
                    _write_nexus_attr(dictionary[key], group[key],
                                      skip_keys=skip_keys)


def read_metadata_from_file(filename, metadata_key=None,
                            lazy=False, verbose=False,
                            skip_array_metadata=False):
    """Read the metadata from a nexus or hdf file.

    This method iterates through the file and returns a dictionary of
    the entries.
    This is a convenience method to inspect a file for a value
    rather than loading the file as a signal.

    Parameters
    ----------
    filename : str
        path of the file to read
    metadata_key  : None,str or list_of_strings , default : None
        None will return all datasets found including linked data.
        Providing a string or list of strings will only return items
        which contain the string(s).
        For example, search_keys = ["instrument","Fe"] will return
        hdf entries with "instrument" or "Fe" in their hdf path.
    verbose: bool, default : False
        Pretty Print the results to screen
    skip_array_metadata : bool, default : False
        Whether to skip loading array metadata. This is useful as a lot of
        large array may be present in the metadata and it is redundant with
        dataset itself.

    Returns
    -------
    dict
        Metadata dictionary.

    See Also
    --------
    * :py:meth:`~.io_plugins.nexus.file_reader`
    * :py:meth:`~.io_plugins.nexus.file_writer`
    * :py:meth:`~.io_plugins.nexus.list_datasets_in_file`


    """
    search_keys = _check_search_keys(metadata_key)
    fin = h5py.File(filename, "r")
    # search for NXdata sets...
    # strip out the metadata (basically everything other than NXdata)
    stripped_metadata = _load_metadata(fin, lazy=lazy,
                                       skip_array_metadata=skip_array_metadata)
    stripped_metadata = _find_search_keys_in_dict(stripped_metadata,
                                                  search_keys=search_keys)
    if verbose:
        pprint.pprint(stripped_metadata)

    fin.close()
    return stripped_metadata


def list_datasets_in_file(filename, dataset_key=None,
                          hardlinks_only=False,
                          verbose=True):
    """Read from a nexus or hdf file and return a list of the dataset paths.

    This method is used to inspect the contents of a Nexus file.
    The method iterates through group attributes and returns NXdata or
    hdf datasets of size >=2 if they're not already NXdata blocks
    and returns a list of the entries.
    This is a convenience method to inspect a file to list datasets
    present rather than loading all the datasets in the file as signals.

    Parameters
    ----------
    filename : str
        path of the file to read
    dataset_key  : str, list of strings or None , default: None
        If a str or list of strings is provided only return items whose
        path contain the strings.
        For example, dataset_key = ["instrument", "Fe"] will only return
        hdf entries with "instrument" or "Fe" somewhere in their hdf path.
    hardlinks_only : bool, default : False
        If true any links (soft or External) will be ignored when loading.
    verbose : boolean, default : True
        Prints the results to screen


    Returns
    -------
    list
        list of paths to datasets


    See Also
    --------
    * :py:meth:`~.io_plugins.nexus.file_reader`
    * :py:meth:`~.io_plugins.nexus.file_writer`
    * :py:meth:`~.io_plugins.nexus.read_metadata_from_file`


    """
    search_keys = _check_search_keys(dataset_key)
    fin = h5py.File(filename, "r")
    # search for NXdata sets...
    # strip out the metadata (basically everything other than NXdata)
    nexus_data_paths, hdf_dataset_paths = \
        _find_data(fin, search_keys=search_keys, hardlinks_only=hardlinks_only)
    if verbose:
        if nexus_data_paths:
            print("NXdata found")
            for nxd in nexus_data_paths:
                print(nxd)
        else:
            print("NXdata not found")
        if hdf_dataset_paths:
            print("Unique HDF datasets found")
            for hdfd in hdf_dataset_paths:
                print(hdfd, fin[hdfd].shape)
        else:
            print("No HDF datasets not found or data is captured by NXdata")
    fin.close()
    return nexus_data_paths, hdf_dataset_paths


def _write_signal(signal, nxgroup, signal_name, **kwds):
    """Store the signal data as an NXdata dataset.

    Parameters
    ----------
    signal : Hyperspy signal
    nxgroup : HDF group
        Entry at which to save signal data
    signal_name : str
        Name  under which to store the signal entry in the file

    """
    smd = signal.metadata.Signal
    if signal.axes_manager.signal_dimension == 1:
        smd.record_by = "spectrum"
    elif signal.axes_manager.signal_dimension == 2:
        smd.record_by = "image"
    else:
        smd.record_by = ""

    nxdata = nxgroup.require_group(signal_name)
    nxdata.attrs["NX_class"] = _parse_to_file("NXdata")
    nxdata.attrs["signal"] = _parse_to_file("data")
    if smd.record_by:
        nxdata.attrs["interpretation"] = _parse_to_file(smd.record_by)
    overwrite_dataset(nxdata, signal.data, "data", chunks=None,
                      signal_axes=signal.axes_manager.signal_indices_in_array,
                      **kwds)
    axis_names = [_parse_to_file(".")] * len(signal.axes_manager.shape)
    for i, axis in enumerate(signal.axes_manager._axes):
        if axis.name != t.Undefined:
            axname = axis.name
            axindex = [axis.index_in_array]
            indices = _parse_to_file(axis.name + "_indices")
            nxdata.attrs[indices] = _parse_to_file(axindex)
            ax = nxdata.require_dataset(axname, data=axis.axis,
                                        shape=axis.axis.shape,
                                        dtype=axis.axis.dtype)
            if axis.units != t.Undefined:
                ax.attrs['units'] = axis.units
            axis_names[axis.index_in_array] = axname

    nxdata.attrs["axes"] = _parse_to_file(axis_names)
    return nxdata


def file_writer(filename,
                signals,
                save_original_metadata=True,
                skip_metadata_key=None,
                use_default=False,
                *args, **kwds):
    """Write the signal and metadata as a nexus file.

    This will save the signal in NXdata format in the file.
    As the form of the metadata can vary and is not validated it will
    be stored as an NXcollection (an unvalidated collection)

    Parameters
    ----------
    filename : str
        Path of the file to write
    signals : signal or list of signals
        Signal(s) to be written
    save_original_metadata : bool , default : False
          Option to save hyperspy.original_metadata with the signal.
          A loaded Nexus file may have a large amount of data
          when loaded which you may wish to omit on saving
    skip_metadata_key : str or list of str, default : None
        the key(s) to skip when it is saving original metadata. This is useful
        when some metadata's keys are to be ignored.
    use_default : bool , default : False
          Option to define the default dataset in the file.
          If set to True the signal or first signal in the list of signals
          will be defined as the default (following Nexus v3 data rules).

    See Also
    --------
    * :py:meth:`~.io_plugins.nexus.file_reader`
    * :py:meth:`~.io_plugins.nexus.list_datasets_in_file`
    * :py:meth:`~.io_plugins.nexus.read_metadata_from_file`

    """
    if not isinstance(signals, list):
        signals = [signals]

    with h5py.File(filename, mode='w') as f:
        f.attrs['file_format'] = "nexus"
        f.attrs['file_writer'] = "hyperspy_nexus_v3"
        if 'compression' not in kwds:
            kwds['compression'] = 'gzip'
        if use_default:
            f.attrs["default"] = "entry1"
        #
        # write the signals
        #

        for i, sig in enumerate(signals):
            nxentry = f.create_group("entry%d" % (i + 1))
            nxentry.attrs["NX_class"] = _parse_to_file("NXentry")

            signal_name = sig.metadata.General.title \
                if sig.metadata.General.title else 'unnamed__%d' % i
            if "/" in signal_name:
                signal_name = signal_name.replace("/", "_")
            if signal_name.startswith("__"):
                signal_name = signal_name[2:]

            if i == 0 and use_default:
                nxentry.attrs["default"] = signal_name

            nxaux = nxentry.create_group("auxiliary")
            nxaux.attrs["NX_class"] = _parse_to_file("NXentry")
            _write_signal(sig, nxentry, signal_name, **kwds)

            if sig.learning_results:
                nxlearn = nxaux.create_group('learning_results')
                nxlearn.attrs["NX_class"] = _parse_to_file("NXcollection")
                learn = sig.learning_results.__dict__
                _write_nexus_groups(learn, nxlearn, **kwds)
                _write_nexus_attr(learn, nxlearn)
            #
            # write metadata
            #
            if save_original_metadata:
                if sig.original_metadata:
                    ometa = sig.original_metadata.as_dictionary()

                    nxometa = nxaux.create_group('original_metadata')
                    nxometa.attrs["NX_class"] = _parse_to_file("NXcollection")
                    # write the groups and structure
                    _write_nexus_groups(ometa, nxometa,
                                        skip_keys=skip_metadata_key, **kwds)
                    _write_nexus_attr(ometa, nxometa,
                                      skip_keys=skip_metadata_key)

            if sig.metadata:
                meta = sig.metadata.as_dictionary()

                nxometa = nxaux.create_group('hyperspy_metadata')
                nxometa.attrs["NX_class"] = _parse_to_file("NXcollection")
                # write the groups and structure
                _write_nexus_groups(meta, nxometa, **kwds)
                _write_nexus_attr(meta, nxometa)
