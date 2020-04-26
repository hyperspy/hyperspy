# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
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
#
#
#    - The characteristics of the IO plugin as the following python variables:
#
#        # Plugin characteristics
#        # ----------------------
#	format_name = <String>
#        description = <String>
#        full_support = <Bool>	# Whether all the Hyperspy features are supported
#        # Recognised file extension
#        file_extensions = <Tuple of string>
#        default_extension = <Int>	# Index of the extension that will be used by default
#        # Reading capabilities
#        reads_images = <Bool>
#        reads_spectrum = <Bool>
#        reads_spectrum_image = <Bool>
#        # Writing capabilities
#        writes_images = <Bool>
#        writes_spectrum = <Bool>
#        writes_spectrum_image = <Bool>
#
#    - A function called file_reader with at least one attribute: filename
#
#    - A function called file_writer with at least two attributes: 
#        filename and object2save in that order.

import logging
import numpy as np
import dask.array as da
import os
import h5py 
import sys
import pprint
#from hyperspy.io_plugins.hspy import overwrite_dataset,get_signal_chunks
from hyperspy.io_plugins.hspy import get_signal_chunks

NX_ENCODING = sys.getfilesystemencoding()

_logger = logging.getLogger(__name__)
# Plugin characteristics

format_name = 'Nexus'
description = \
    'Read NXdata, hdf datasets and metadata from hdf5 or Nexus files'
full_support = False
# Recognised file extension
file_extensions = ['nxs','NXS','h5','hdf','hdf5']
default_extension = 0
# Writing capabilities:
writes = True

def _byte_to_string(value):
    """
    
    decode a byte string

    Parameters
    ----------
    value :  byte str
    
    Returns
    -------
    str, decoded version of input value

    """
    try:
        text = value.decode(NX_ENCODING)
    except UnicodeDecodeError:
        if NX_ENCODING == 'utf-8':
            text = value.decode('latin-1')
        else:
            text = value.decode('utf-8')
    return text.replace('\x00','').rstrip()

def _parse_from_file(value,lazy=True):
    """ 
    
    To convert bytes to strings and
    to convert arrays of byte strings to lists of strings 
    
    Parameters
    ----------
    value - input object read from hdf file
    
    
    Returns
    -------
    parsed value
    
    """
    toreturn = value
    if isinstance(value, np.ndarray) and value.shape == (1,):
        toreturn = value[0]        
    elif isinstance(value, bytes):
        toreturn= _byte_to_string(value)
    elif isinstance(value, (np.int,np.float)):
        toreturn= value
    elif isinstance(value, (np.ndarray)) and value.dtype.char == "S":
        try:
            toreturn = np.array(value).astype("U")
        except UnicodeDecodeError:
            # There are some strings that must stay in binary,
            # for example dill pickles. This will obviously also
            # let "wrong" binary string fail somewhere else...
            pass
    elif isinstance(value,h5py.Dataset):    
        # test if its an array of byte string  s
        if value.dtype.char == "S":
            try:
                toreturn = np.array(value).astype("U")
            except UnicodeDecodeError:
                # There are some strings that must stay in binary,
                # for example dill pickles. This will obviously also
                # let "wrong" binary string fail somewhere else...
                pass
        else:
           if value.size<2:
               toreturn = value[...].item()
           else:
               if lazy:
                   if value.chunks:
                       toreturn = da.from_array(value,value.chunks)
                   else:
                       chunks = _guess_chunks(value)
                       toreturn = da.from_array(value,chunks)
                        
               else:
                   toreturn = np.array(value)
    return toreturn


def _text_split(s, sep):
    """
    
    Split a string based of list of seperators
    
    Parameters
    ----------
    s   : str 
    sep : str  - seperator or list of seperators e.g. '.' or ['_','/']
    
    Returns
    -------
    list containing string sections split based on the seperators
    
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

def _getlink(h5group,rootkey):
    
    """Return the link target path and filename.

    If a hdf group is a link to an external file or a soft link  
    this method will return the target path within 
    the external file, the external file name and if the external
    file name is an absolute path or not
    e.g. if /entry/data points to /raw_entry/tem/data in raw.nxs
    returns /raw_entry/tem/data, raw.nxs, False 
    
    
    Returns
    -------
    str, str, bool
        Link path, filename, and boolean that is True if an absolute file
        path is given.
    
    """
    _target, _filename, _abspath = None, None, False
    if rootkey != '/':
        if isinstance(h5group,h5py.Group):
            _link = h5group.get(rootkey, getlink=True)
            if isinstance(_link, h5py.ExternalLink):
                _target, _filename = _link.path, _link.filename
                _abspath = os.path.isabs(_filename)
            elif isinstance(_link, h5py.SoftLink):
                _target = _link.path
        if 'target' in h5group.attrs:
            _target = _parse_from_file(h5group.attrs['target'])
            if not _target.startswith('/'):
                _target = '/' + _target
            if _target == rootkey:
                _target = None
    return _target, _filename, _abspath


def _extract_hdf_dataset(tree,dataset,lazy=True):
    """
    
    Import data from hdf path 
    
    Parameters
    ----------
    tree : hdf group from which to load the dataset
    dataset : str   path to the dataset within tree
    lazy    : bool  if true use lazy opening, if false read into memory
    
    Returns
    -------
    data - dask or numpy array
    
    """
    data = tree[dataset]
    if data.dtype.type is np.string_ or data.dtype.type is np.object_:
        return None
    if(lazy):
        if "chunks" in data.attrs.keys():
             chunks = data.attrs["chunks"]
        else:
            chunks = _guess_chunks(data)
        data_lazy = da.from_array(data, chunks=chunks)
    else:
       data_lazy = np.array(data)    
    return data_lazy

    
def _nexus_dataset_to_signal(tree,nexus_dataset,lazy=True):
    """
    """
    detector_index = 0
    dataentry = tree[nexus_dataset]
    if "signal" in dataentry.attrs.keys():
        if _is_int(dataentry.attrs["signal"]):
            data_key  = "data"
        else:
            data_key  = dataentry.attrs["signal"]
    else:
        _logger.info("No signal attr associated with NXdata will\
                     try assume signal name is data")
        if "data" not in  dataentry.keys():
            raise ValueError("Signal attribute not found in NXdata and\
                             attempt to find a default \"data\" key failed")
        else:            
            data_key  = "data"
    data = dataentry[data_key] 
    nav_list = []
    # list indices...
    axis_index_list=[]
    if "axes" in dataentry.attrs.keys():           
        axes_key  = dataentry.attrs["axes"]
        axes_list = [_parse_from_file(num) for num in axes_key]
        named_axes=list(range(len(axes_list)))
        for i,ax in enumerate(axes_list):
            if ax != ".":
                index_name = ax + "_indices"
                ind_in_array = int(dataentry.attrs[index_name])
                axis_index_list.append(ind_in_array)
                if "units" in dataentry[ax].attrs:
                    units= _parse_from_file(dataentry[ax].attrs["units"])
                else:
                    units=""
                named_axes.remove(ind_in_array)
                if _is_numeric_data(dataentry[ax]):
                    if _is_linear_axis(dataentry[ax]):
                        nav_list.append({   
                            'size': data.shape[ind_in_array],
                            'index_in_array': ind_in_array,
                            'name': ax,
                            'scale': abs(dataentry[ax][1]-\
                                         dataentry[ax][0]),
                            'offset': min(dataentry[ax][0],\
                                          dataentry[ax][-1] ),
                            'units': units,
                            'navigate': True,
                            })
                    else:
                        nav_list.append({   
                            'size': data.shape[ind_in_array],
                            'index_in_array': ind_in_array,
                            'name': ax,
                            'scale': 1,
                            'offset': 0,
                            'navigate': True,
                            })
                         # when non-linear axes supported...
#                        nav_list.append({   
#                            'index_in_array': ind_in_array,
#                            'name': axname,
#                            'axis' : dataentry[ax],
#                            'units': units,
#                            'navigate': True,
#                            })
                    
        for i,ax in enumerate(axes_list):
            if ax == ".":
                if(len(data.shape)==len(axes_list)):
                    nav_list.append({   
                            'size': data.shape[named_axes[detector_index]],
                            'index_in_array': named_axes[detector_index],
                            'name': "detector_%d"%detector_index,
                            'scale': 1,
                            'offset': 0.0,
                            'units': '',
                            'navigate': False,
                           })
                    detector_index=detector_index+1
                            
        if(lazy):
            if "chunks" in data.attrs.keys():
                 chunks = data.attrs["chunks"]
            else:
                chunks = _guess_chunks(data)
            data_lazy = da.from_array(data, chunks=chunks)
        else:
           data_lazy = np.array(data)    
        # adjust the navigate boolean based on 
        # ifer the dataset isjudged to be 1D or 2D
        signal_type = _guess_signal_type(data_lazy)
        if signal_type == "Signal2D":
            nav_list[-1]["navigate"] = False
            nav_list[-2]["navigate"] = False
        if signal_type == "Signal1D":
            nav_list[-1]["navigate"] = False
        title = _text_split(nexus_dataset, '/')[-1]
        print("nexus_dataset",nexus_dataset,title)
        metadata = {'General': {'title': title},\
                    "Signal": {'signal_type': signal_type}}
        if nav_list:      
            dictionary = {'data': data_lazy,
                          'axes': nav_list,
                          'metadata':metadata}
        else:
            dictionary = {'data': data_lazy,
                          'metadata': metadata}                    
    return dictionary


def file_reader(filename,lazy=True, dset_search_keys=None,
                meta_search_keys=None,
                load_nxdata_only=True,
                small_metadata_only=True,
                **kwds):
    
    """Reads NXdata class or hdf datasets from a file and returns signal
    
    Note
    ----
    Loading all datasets can result in a large number of signals
    Please review your datasets and use the dset_search_keys to target
    the datasets of interest. 

    Datasets are all arrays with size>2 (arrays, lists)
    Metadata is all data with size < 2 (int, float, strings,lists)    
    
    Parameters
    ----------
    myDict : dict or h5py.File object
    dset_search_keys  : str list or None  
        Only return items whose path contain the strings
        .e.g search_list = ["instrument","Fe"] will return
        data entries with instrument or Fe in their hdf path.
    meta_search_keys: : str list or None
        Only return items whose path contain the strings
        .e.g search_list = ["instrument","Fe"] will return
        all metadata entries with instrument or Fe in their hdf path.
    load_nxdata_only : bool  If true only NXdata will be converted into a signal
                 if false NXdata and any hdf datasets not bound within the
                 NXdata sets will be loaded as signals
    small_metadata_only : bool, default:True  If true arrays of size<2
                 will be loaded into the metadata structure. If false larger 
                 arrays will be also loaded into the metadata
    

    Returns
    -------
    signal dictionary or list of signal dictionaries of the form
    {'data': data,
     'metadata': metadata,
     'original_metadata':original_metadata,
     'mapping':mapping}
    

    See also 
    --------
    get_datasets_from_file 
                 
   
    """
    # search for NXdata sets...
    if dset_search_keys == None:
        dset_search_keys=[""]
    if meta_search_keys == None:
        meta_search_keys=[""]
    mapping  = kwds.get('mapping',{})   
    metadata={}
    original_metadata={}
    fin = h5py.File(filename,"r")
    signal_dict_list = []

    
    nexus_data_paths, hdf_data_paths = _find_data(fin,
                                                  search_keys=dset_search_keys)
    # strip out the metadata (basically everything other than NXdata)
    #
    # Be careful not to load a large number of datasets
    # Check against an upper limit and that the user has
    # sub-selected the datasets to load used dset_search_keys
    #
    all_metadata = load_dict_from_h5py(fin)
    stripped_metadata = _find_search_keys_in_dict(all_metadata,
                                                  search_keys=meta_search_keys)

    if "metadata" in all_metadata.keys():
        if "original_metadata" in stripped_metadata["metadata"]:
            original_metadata = stripped_metadata["metadata"]["original_metadata"]
        if "hyperspy_metadata" in stripped_metadata["metadata"]:
            metadata = stripped_metadata["metadata"]["hyperspy_metadata"]
    else:
        original_metadata = all_metadata
    original_metadata = _find_search_keys_in_dict(original_metadata,
                                            search_keys=meta_search_keys)
        
    
    for data_path in nexus_data_paths:        
        dictionary = _nexus_dataset_to_signal(fin,data_path,lazy)
        dictionary["original_metadata"] = original_metadata
        dictionary["metadata"].update(metadata)        
        dictionary["mapping"] = mapping
        
        signal_dict_list.append(dictionary)
    if not load_nxdata_only:
        for data_path in hdf_data_paths:        
            data = _extract_hdf_dataset(fin,data_path,lazy)
            if data is not None:
                signal_type =  _guess_signal_type(data)
                title = _text_split(data_path, '/')[-1]
                basic_metadata = {'General': {'original_filename': \
                          os.path.split(filename)[1],\
                'title': title},\
                "Signal": {'signal_type': signal_type}}
                dictionary = {'data': data,
                              'metadata': basic_metadata,
                              'original_metadata':original_metadata,
                              'mapping':mapping}                
                signal_dict_list.append(dictionary)

        
    return signal_dict_list
        
def _is_linear_axis(data):
    
    """
    
    Check if the data is linearly incrementing

    Parameter
    ---------
    data : dask or numpy array

    Returns
    -------
    True or False    
    
    """
    steps = np.diff(data)
    est_steps = np.array([steps[0]]*len(steps))
    return np.allclose(est_steps,steps,rtol=1.0e-5)

def _is_number(a):
    """
    
    Check if the value is a number
    # will be True also for 'NaN'
    
    Parameter
    ---------
    a : python object to be tested
    
    Returns
    -------
    True or False    

    """
    try:
        float(a)
        return True
    except ValueError:
        return False

def _is_numeric_data(data):
    """
 
    Check that data contains numeric data

    Parameter
    ---------
    data : dask or numpy array
    
    Returns
    -------
    True or False    

    """
    try:
        data.astype(float)
        return True
    except ValueError:
        return False
 
    
def _is_int(s):
    """
    Check that s in an integer

    Parameter
    ---------
    s : python object to test
    
    Returns
    -------
    True or False    
    
    """
    try: 
        int(s)
        return True
    except ValueError:
        return False

        
def _find_data(myDict,search_keys=None):
    """
    
    Read from a nexus or hdf file and return a list 
    of the dataset entries
    The method iterates through group attributes and returns NXdata or 
    hdf datasets of size >=2 if they're not already NXdata blocks
    and returns a list of the entries
    This is a convenience method to inspect a file to see which datasets
    are present rather than loading all the sets in the file as signals
    
    Parameters
    ----------
    myDict : dict or h5py.File object
    search_keys  : str list or None  
        Only return items which contain the strings
        .e.g search_list = ["instrument","Fe"] will return
        hdf entries with instrument or Fe in their hdf path.
    
    Returns
    -------
    nx_dataset_list, hdf_dataset_list
    
    nx_dataset_list is a list of all NXdata paths
    hdf_dataset_list is a list of all hdf_datasets not linked to an NXdata set
        
    """
    hdf_datasets = []
    nx_datasets = []
    rootname=""
    def find_data_in_tree(myDict,rootname):
        for key, value in myDict.items():
            if rootname != "":
                rootkey = rootname +"/"+ key
            else:
                rootkey = "/" + key
            if isinstance(value,h5py.Dataset):  
                if value.size >= 2:
                    target,b,c, = _getlink(value,rootkey)
                    if target is not None:
                        if search_keys:
                            if any(s in rootkey for s in search_keys):
                                hdf_datasets.append(target)
                        else:
                            hdf_datasets.append(target)
            elif isinstance(value,h5py.Group):
                if "NX_class" in value.attrs:
                    if value.attrs["NX_class"] == b"NXdata" \
                        and "signal" in value.attrs.keys(): 
                        if search_keys:
                            if any(s in rootkey for s in search_keys):
                                nx_datasets.append(rootkey)
                        else:
                            nx_datasets.append(rootkey) 
                find_data_in_tree(value,rootkey)

    # need to use custom recursive function as visititems in h5py
    # does not visit links
    find_data_in_tree(myDict,rootname)
    nx_datasets = list(dict.fromkeys(nx_datasets))
    hdf_datasets = list(dict.fromkeys(hdf_datasets))
    return nx_datasets,hdf_datasets


def _fix_exclusion_keys(key):
    """
    
    Exclude hyperspy specific keys
    Signal and DictionaryBrowser break if a
    a key is a dict method - e.g. {"keys":2.0} 
    
    This method prepends the key with "fix" so the information is
    still present to work around this issue
    
    Parameter
    ---------
    key : str 
    
    Returns
    -------
    key or key prepended with 'fix_' if key is 
    in a set of excluded of keywords
    
    """
    exclusion_list=['_sig_','_list_empty_','_tuple_empty_',
                    '_bs_','_datetime_date',"_datetime_","keys"]
    
    if any([key.startswith(x) for x in exclusion_list]):
        return"fix_"+key
    else:
        return key




def _find_search_keys_in_dict(tree,search_keys=None):
    
    """    
    Search through a dict and return a dictionary
    whose full path contains one of the search keys

    This is a convenience method to inspect a file for a value
    rather than loading the file as a signal
    
    Parameters
    ----------
    tree         : h5py File object 
    search_keys  : str list or None  
        Only return items which contain the strings
        .e.g search_keys = ["instrument","Fe"] will return
        hdf entries with instrument or Fe in their hdf path.
    
    Returns
    -------
    Dictionary. When search_list is specified only full paths
    containing one or more search_keys will be returned 
        
    """
    if search_keys == None:
        search_keys=[""]
    metadata_dict = {}
    rootname=""
    # recursive function
    
    def find_searchkeys_in_tree(myDict,rootname):
        for key, value in myDict.items():
            if rootname != "":
                rootkey = rootname +"/"+ key
            else:
                rootkey = key            
            if any(s in rootkey for s in search_keys):
                mod_keys = _text_split(rootkey, (".","/") )                        
                # create the key, values in the dict
                p = metadata_dict
                for d in mod_keys[:-1]:
                    p = p.setdefault(d,{})
                p[mod_keys[-1]] = value
            if isinstance(value,dict):
                find_searchkeys_in_tree(value,rootkey)
    find_searchkeys_in_tree(tree,rootname)
    return metadata_dict



def load_dict_from_h5py(group,lazy=True,small_metadata_only=True):
    """
    Search through a group identifying metadata
    Metadata is classed as data of size < 2
    which should covers all descriptive details - 
    strings, floats, ints 

    Paramaters
    ----------
    group : hdf group or dict
    small_metadata_only : bool, default : true
        If true only return items of size <2 (no arrays) as metadata

    Returns
    -------
    dictionary of group contents
    
    """
    
    return _find_metadata_in_tree(group,small_metadata_only=small_metadata_only)

# recursive function
def _find_metadata_in_tree(group,lazy=True,small_metadata_only=True):
    """

    Search through a group identifying metadata
    Metadata is classed as data of size < 2
    which should covers all descriptive details - 
    strings, floats, ints 

    Paramaters
    ----------
    group : hdf group or dict
    small_metadata_only : bool, default : true
        If true only return items of size <2 (no arrays) as metadata

    Returns
    -------
    dictionary of group contents
    
        
    """
    
    tree={}
    for key,item in group.attrs.items():
        new_key=_fix_exclusion_keys(key)
        if "attrs" not in tree.keys():
            tree["attrs"]={}
        tree["attrs"][new_key] =  _parse_from_file(item,lazy=lazy)
    for key,item in group.items():
        new_key=key
        new_key =_fix_exclusion_keys(key)
        if type(item) is h5py._hl.dataset.Dataset:
            if (small_metadata_only and item.size < 2) or \
                (not small_metadata_only):
                if new_key not in tree.keys():
                    tree[new_key]={}
                tree[new_key]["value"] = _parse_from_file(item,lazy=lazy)
                for key,item in group.attrs.items():
                    if "attrs" not in tree.keys():
                        tree["attrs"]={}
                    tree["attrs"][new_key] =  _parse_from_file(item,lazy=lazy)
                
        elif type(item) is h5py._hl.group.Group:
            tree[new_key]=load_dict_from_h5py(item,small_metadata_only=small_metadata_only)
    return tree


    
def _guess_chunks(data):
    """
    
    Guess an appropriate chunk layout for a dataset, given its shape and
    the size of each element in bytes.  Will allocate chunks only as large
    as MAX_SIZE.  Chunks are generally close to some power-of-2 fraction of
    each axis, slightly favoring bigger values for the last index.    
    
    Parameters
    ----------
    data  -  numpy, dask array or h5py.Dataset
    
    Returns
    -------
    tuple - Estimated chunk sizes for each axis
    
    """
    chunks= h5py._hl.filters.guess_chunk(data.shape, None, np.dtype(data.dtype).itemsize)
    return chunks

############################################################
# Write values to hdf5 file
############################################################
def _write_array(group, a, name):
    overwrite_dataset(group, a, name)
    if "attrs" in a.keys():
        _write_dict_attrs(group,a["attrs"])

def _write_string(group, s, name):
    group.create_dataset(name, data=np.asarray(s), dtype=h5py.new_vlen(str))

def _write_list(group,value,key):
    if len(value):
        cdict = dict(zip([str(i) for i in range(len(value))], value))
        _write_dict(group, cdict, '_list_' + str(len(value)) + '_' + key)
    else:
        group.attrs['_list_empty_' + key] = '_None_'

def _write_bytes(group, value, key):
    try:
        # binary string if has any null characters (otherwise not
        # supported by hdf5)
        value.index(b'\x00')
        group.attrs['_bs_' + key] = np.void(value)
    except ValueError:
        group.attrs[key] = value.decode()

def _write_tuple(group, value, key):
    if len(value):
        cdict = dict(zip([str(i) for i in range(len(value))], value))
        _write_dict(group, cdict, '_tuple_' + str(len(value)) + '_' + key)
    else:
        group.attrs['_tuple_empty_' + key] = '_None_'

def _write_None(group, a, name):
    group.require_dataset(name, data = np.zeros((1,)))

def _write_scalar(group, a, name):
    overwrite_dataset(group, a, name)
    
def _write_dict_attrs(group,value):
    for k,v in value.items():
        group.attrs[k]=v

def _write_dict(group, value, name):
    if 'value' in value.keys() and not isinstance(value["value"],dict):
        _write_to_nexus(group, value["value"], name)
        if "attrs" in value.keys():
            _write_dict_attrs(group,value["attrs"])
    else:
        dset = group.require_group(name)
        if "attrs" in value.keys():
            _write_dict_attrs(dset,value["attrs"])
        for k,v in value.items():
            _write_to_nexus(dset.require_group(k), v, k)
            


#overwrite_dataset(group, data, key, signal_axes=None, chunks=None, **kwds):
def overwrite_dataset(group, data, key, signal_axes=None, chunks=None, **kwds):
    if chunks is None:
        if signal_axes is None:
            # Use automatic h5py chunking
            chunks = True
        else:
            # Optimise the chunking to contain at least one signal per chunk
            chunks = get_signal_chunks(data.shape, data.dtype, signal_axes)

    if data.dtype == np.dtype('O'):
        # For saving ragged array
        # http://docs.h5py.org/en/stable/special.html#arbitrary-vlen-data
        group.require_dataset(key,
                              chunks,
                              dtype=h5py.special_dtype(vlen=data[0].dtype),
                              **kwds)
        group[key][:] = data[:]

    maxshape = tuple(None for _ in data.shape)

    got_data = False
    while not got_data:
        try:
            these_kwds = kwds.copy()
            these_kwds.update(dict(shape=data.shape,
                                   dtype=data.dtype,
                                   exact=True,
                                   maxshape=maxshape,
                                   chunks=chunks,
                                   shuffle=True,))

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
        _logger.info("Chunks used for saving: %s" % str(dset.chunks))
        if isinstance(data, da.Array):
            da.store(data.rechunk(dset.chunks), dset)
        elif data.flags.c_contiguous:
            dset.write_direct(data)
        else:
            dset[:] = data
    return dset

def _write_to_nexus(group,a,name):
    
    if isinstance(a,str):
        _write_string(group,a,name)
    elif isinstance(a,dict):
        _write_dict(group,a,name)
    elif isinstance(a, list):
        _write_list(group,a,name)
    elif isinstance(a,tuple):
        _write_tuple(group,a,name)
    elif isinstance(a,(np.ndarray,da.Array)):
        _write_array(group,a,name)
    elif np.isscalar(a):
        _write_scalar(group, a ,name)
    elif a is None:
        _write_None(group, a, name)


def _read_dict(dset):
    d = {}
    for k,v in  dset.items():
        d[k] = _read_nexus(v)
    _read_attrs(dset,d)
    return d

def _read_attrs(dset,dictionary):
    if "attrs" in dset.keys():
        dictionary["attrs"]={}
        for k,v in  dset.attrs.items():
            dictionary["attrs"][k] = _read_nexus(v)
    return dictionary

def _read_list(dset):
    l = []
    keys = dset.keys()
    keys.sort()
    for k in keys:
        l.append(_read_nexus(dset[k]))
    return l

def _read_array(dset,sl=None):
    if sl is not None:
        return dset[sl]
    else:
        return dset[...]

def _read_scalar(dset):
    try:
        return dset[...].item()
    except:
        return dset[...]

def _read_str(dset):
    return dset.value

def _read_nexus(dset):
    # Treat groups as dicts
    if isinstance(dset, h5py.Group):
        _read_dict(dset)
        _read_attrs(dset)
    elif isinstance(dset,h5py.Group):
        for key in group.keys():
            if isinstance(group[key], h5py.Dataset):
                dat = group[key]
                kn = key
                if key.startswith("_list_"):
                    _read_list()
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
                        hdfgroup2dict(
                            group[key], lazy=lazy).items()
                    ))])
            elif key.startswith('_list_'):
                dictionary[key[7 + key[6:].find('_'):]] = \
                    [i for k, i in sorted(iter(
                        hdfgroup2dict(
                            group[key], lazy=lazy).items()
                    ))]
            elif key.startswith('_tuple_'):
                dictionary[key[8 + key[7:].find('_'):]] = tuple(
                    [i for k, i in sorted(iter(
                        hdfgroup2dict(
                            group[key], lazy=lazy).items()
                    ))])
            else:
                dictionary[key] = {}
                hdfgroup2dict(
                    group[key],
                    dictionary[key],
                    lazy=lazy)
                
    if (dset_type is None) and (type(dset) is h5py.Group):
        dset_type = 'dict'        
    if dset_type == 'dict':
        val = _read_dict(dset)
    elif dset_type == 'list':
        val = _read_list(dset)
    elif dset_type == 'array':
        val = _read_array(dset)
    elif dset_type == 'arraylist':
        val = [x for x in _read_array(dset)]
    elif dset_type == 'tuple':
        val = tuple(_read_list(dset))
    elif dset_type == 'arraytuple':
        val = tuple(_read_array(dset).tolist())
    elif dset_type == 'string':
        val = _read_str(dset)
    elif dset_type == 'unicode':
        val = _read_str(dset)
    elif dset_type == 'scalar':
        val = _read_scalar(dset)
    elif dset_type == 'None':
        val = None
    elif dset_type is None:
        val = _read_array(dset)
    else:
        raise RuntimeError('Unsupported data type : %s' % dset_type)
    return val


def write_nexus_groups(dictionary, group,small_metadata_only=True, **kwds):
    """
    Recursively iterate through dictionary and 
    create the groups and datasets
    A hdf dataset can have a value and a list of attrs and this doesn't
    quite fit with a python dict
    A dict for a dataset can therefore has a "value" key and "attrs" key
    if the "value" key is present the dataset is created set to this value 
    and then attrs for the dataset are added later.
    
    Parameters
    ----------
    dictionary : dict - structure to write to the group
    group      : hdf group - group in which to store the dictionary 
    **kwds     : optional keywords for h5py create_dataset methods

    """

    for key, value in dictionary.items():
        if key == 'value' and not isinstance(value,dict):
            continue
        if key == 'attrs':
            continue
        if isinstance(value, dict):
            if 'value' in value.keys() and not isinstance(value["value"],dict):
                if isinstance(value["value"],(int,float,str,bytes)):
                    group.create_dataset(key,data=value["value"])     
                elif isinstance(value["value"],(np.ndarray, h5py.Dataset, da.Array)):
                    group.require_dataset(key,data=value["value"],
                                      shape=value["value"].shape,
                                   dtype=value["value"].dtype)                
            else:
                write_nexus_groups(value, group.require_group(key),
                          **kwds)
        elif isinstance(value, (np.ndarray, h5py.Dataset, da.Array)):
            if small_metadata_only and value.size<2:
                overwrite_dataset(group, value, key, **kwds)        
        elif isinstance(value, (int,float,str,bytes)):
            group.attrs[key] = value


def write_nexus_attr(dictionary, group, **kwds):
    """
    
    Recursively iterate through dictionary and 
    write "attrs" dictionaries to the nexus file
    This step is called after the groups and datasets
    have been created
    
    """
    for key, value in dictionary.items():
        if key == 'attrs':
            for k, v in value.items():
                group.attrs[k] = v        
        else:
            if isinstance(value, dict):
                write_nexus_attr(dictionary[key], group[key],
                          **kwds)

def _guess_signal_type(data):
    """
    
    Checks last 2 dimensions
    An area detector will be square, > 255 pixels each side and
    if rectangular ratio of sides is at most 4.
    An EDS detector will generally by ndetectors*2048 or ndetectors*4096
    the ratio test will be X10-1000  
    
    Paramaters
    ----------
    data : dask or numpy array
    
    Returns
    -------
    Estimate of signal type - "Signal1D", "Signal2D" or "BaseSignal"
    
    """
    if len(data.shape) == 1:
        signal_type = "Signal1D"
    elif len(data.shape) == 2:
        signal_type = "Signal2D"
    else:
        n1 = data.shape[-1]
        n2 = data.shape[-2]
        if n2>127 and n1 >127 and n1/n2 < 1.5:
            signal_type = "Signal2D"
        elif n1 > 127 and n2 < n1:
            signal_type = "Signal1D"
        else:
            signal_type = "BaseSignal"
    return signal_type
        


def get_metadata_in_file(filename,search_keys=None,
                        small_metadata_only=False,
                        verbose=True):
    """
    
    Read the metadata from a nexus or hdf file       
    The method iterates through group attributes and 
    Datasets of size < 2 (i.e. single values)
    and returns a dictionary of the entries
    This is a convenience method to inspect a file for a value
    rather than loading the file as a signal
    
    Parameters
    ------------
    filename : str  path of the file to read
    search_list  : str list or None  
        Only return items which contain the strings
        .e.g search_list = ["instrument","Fe"] will return
        hdf entries with instrument or Fe in their hdf path.
    verbose: bool  Print the results to screen 
    
    Returns
    -------
    Metadata dictionary. When search_list is specified only items
    matching the search keys will be returned 
        
    """
    if search_keys == None:
        search_keys=[""]
    fin = h5py.File(filename,"r")
    # search for NXdata sets...
    # strip out the metadata (basically everything other than NXdata)
    #metadata = _find_hdf_metadata(fin,search_keys=search_keys)
    metadata = load_dict_from_h5py(fin,small_metadata_only)
    stripped_metadata = _find_search_keys_in_dict(metadata,search_keys=search_keys)

    if verbose:
        pprint.pprint(stripped_metadata)
    return stripped_metadata    


def get_datasets_in_file(filename,search_keys=None,
                          verbose=True):
    """
    
    Read from a nexus or hdf file and return a list of the paths
    to the dataset entries. This method is used to inspect the contents
    of a Nexus file.
    The method iterates through group attributes and returns NXdata or 
    hdf datasets of size >=2 if they're not already NXdata blocks
    and returns a list of the entries
    This is a convenience method to inspect a file to see what datasets
    are present rather than loading all the sets in the file as signals
    
    Parameters
    ----------
    filename : str   path of the file to read
    search_keys  : str list or None  
        Only return items which contain the strings
        .e.g search_list = ["instrument","Fe"] will return
        hdf entries with instrument or Fe in their hdf path.
    verbose : boolean, default : True  - prints the datasets 
    
    Returns
    -------
    list of paths to datasets. When search_list is specified only those paths in the 
    hdf file match the search will be returned 
        
    """
    if search_keys == None:
        search_keys=[""]
    fin = h5py.File(filename,"r")
    # search for NXdata sets...
    # strip out the metadata (basically everything other than NXdata)
    nexus_data_paths,hdf_dataset_paths = \
        _find_data(fin,search_keys=search_keys)
    if verbose:
        if nexus_data_paths:
            print("NXdata found")
            for nxd in nexus_data_paths:
                print(nxd)
        else:
            print("NXdata not found")
        if hdf_dataset_paths:    
            print("HDF datasets found")
            for hdfd in hdf_dataset_paths:
                print(hdfd,fin[hdfd].shape)
        else:
            print("No HDF datasets not found or data is captured by NXdata")
    fin.close() 
    return nexus_data_paths,hdf_dataset_paths


def write_signal(signal, nxgroup, signal_name,**kwds):
    """

    Store the signal data as an NXdata dataset 
    
    Parameters
    ----------
    signal   : Hyperspy signal
    nxgroup  : HDF group entry at which to save signal data   
            
    
    """
    smd = signal.metadata.Signal
    if signal.axes_manager.signal_dimension == 1:
        smd.record_by = "spectrum"
    elif signal.axes_manager.signal_dimension == 2:
        smd.record_by = "image"
    else:
        smd.record_by = ""

    nxdata = nxgroup.require_group(signal_name)
    nxdata.attrs["NX_class"] = b"NXdata"    
    nxdata.attrs[u"signal"] = b"data"
    overwrite_dataset(nxdata, signal.data, u"data", chunks=None, **kwds)
    axis_names=[b"."]*len(signal.axes_manager.shape)
    for i,axis in enumerate(signal.axes_manager._axes):
        try:
            axname  = axis.name + "_indices"
        except:
            axname  = "axis"+str(i)            
        axindex = axis.index_in_array
        indices = axname+"_indices"
        nxdata.attrs[bytes(indices, 'utf-8')] = axindex
        nxdata.require_dataset(axname,data=axis.axis,shape=axis.axis.shape,
                               dtype=axis.axis.dtype)
        axis_names[axis.index_in_array]=bytes(axname, 'utf-8')
    nxdata.attrs[u"axes"] = axis_names
    

def file_writer(filename,
                signal,
                *args, **kwds):
    """
    
    Write the signal as a nexus file
    Current this will save the signal as NXdata in the file.
    If the metadata was loaded from a Nexus file the structure will
    be preserved when saved.
    All other metadata will be written as hdf groups, datasets or attributes
    reflecting the structure of the metadata 

    Parameters
    ----------
    filename : str   path of the file to write
    signal   : hyperspy signal or list of hyperspy signals
    
    Returns
    -------
    Dataset list. When search_list is specified only items whose path in the 
    hdf file match the search will be returned 
        
    
    """
    with h5py.File(filename, mode='w') as f:
        f.attrs['file_format'] = "Nexus"

        # 
        # if want to store hyperspy metadata 
        # basically move any hyperspy metadata into the original_metadata
        # dict       
        signal_name = signal.metadata.General.title if \
        signal.metadata.General.title else '__unnamed__'
        if "/" in signal_name:
            signal_name = signal_name.replace("/", "_")

        try:
            #
            # write the signal
            #
            nxentry = f.create_group('signal')
            nxentry.attrs[u"NX_class"] = u"NXentry"
            write_signal(signal, nxentry, signal_name,**kwds)
            #
            # write metadata 
            #
            nxentry = f.create_group('metadata')
            nxentry.attrs["NX_class"] = b"NXentry"                
            if signal.original_metadata:
                nxometa = nxentry.create_group('original_metadata')
                nxometa.attrs["NX_class"] = b"NXcollection"    
                orig_meta = signal.original_metadata.as_dictionary()
                # write the groups and structure
                write_nexus_groups(orig_meta,nxometa)
                write_nexus_attr(orig_meta,nxometa)
            if signal.metadata:
                nxmeta = nxentry.create_group('hyperspy_metadata')
                nxmeta.attrs["NX_class"] = b"NXcollection"    
                write_nexus_groups(signal.metadata.as_dictionary(),nxmeta)

        except BaseException:
            raise
        finally:
            f.close()
