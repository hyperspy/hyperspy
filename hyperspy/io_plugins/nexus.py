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
from hyperspy.io_plugins.hspy import overwrite_dataset
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

    """
    try:
        text = value.decode(NX_ENCODING)
    except UnicodeDecodeError:
        if NX_ENCODING == 'utf-8':
            text = value.decode('latin-1')
        else:
            text = value.decode('utf-8')
    return text.replace('\x00','').rstrip()

def _parse_value(value):
    """ 
    
    To convert bytes to strings and
    to convert arrays of byte strings to lists of strings 
    
    """
    toreturn = value
    if isinstance(value, np.ndarray) and value.shape == (1,):
        toreturn = value[0]        
    if isinstance(value, bytes):
        toreturn= _byte_to_string(value)
    if isinstance(value, (np.int,np.float)):
        toreturn= value
    if isinstance(value,(np.ndarray)):    
        # test if its an array of byte strings
        if isinstance(value[0],bytes):
           toreturn=[]
           for v in value:
               toreturn.append(_byte_to_string(v))               
        else:
           toreturn = value
    return toreturn


def _text_split(s, sep):
    """
    
    Split a string based of list of seperators
    
    
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
            _target = _parse_value(h5group.attrs['target'])
            if not _target.startswith('/'):
                _target = '/' + _target
            if _target == rootkey:
                _target = None
    return _target, _filename, _abspath


def _extract_hdf_dataset(tree,dataset,lazy=True):
    """
    
    Import data from hdf path 
    
    """
    data = tree[dataset]
    if data.dtype.type is np.string_ or data.dtype.type is np.object_:
        return None
    if(lazy):
        if "chunks" in data.attrs.keys():
             chunks = data.attrs["chunks"]
        else:
            chunks = guess_chunks(data)
        data_lazy = da.from_array(data, chunks=chunks)
    else:
       data_lazy = data    
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
        axes_list = [_parse_value(num) for num in axes_key]
        named_axes=list(range(len(axes_list)))
        for i,ax in enumerate(axes_list):
            if ax != ".":
                index_name = ax + "_indices"
                ind_in_array = int(dataentry.attrs[index_name])
                axis_index_list.append(ind_in_array)
                if "units" in dataentry[ax].attrs:
                    units= _parse_value(dataentry[ax].attrs["units"])
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
                chunks = guess_chunks(data)
            data_lazy = da.from_array(data, chunks=chunks)
        else:
           data_lazy = data    
        # adjust the navigate boolean based on 
        # ifer the dataset isjudged to be 1D or 2D
        signal_type = guess_signal_type(data_lazy)
        if signal_type == "Signal2D":
            nav_list[-1]["navigate"] = False
            nav_list[-2]["navigate"] = False
        if signal_type == "Signal1D":
            nav_list[-1]["navigate"] = False
        title = _text_split(nexus_dataset, '/')[-1]
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
                nxdata_only=True,
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
    nxdata_ony : bool  If true only NXdata will be converted into a signal
                 if false NXdata and any hdf datasets not bound within the
                 NXdata sets will be loaded. 

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
    original_metadata = _find_hdf_metadata(fin,meta_search_keys)
        
    for data_path in nexus_data_paths:        
        metadata={}
        dictionary = _nexus_dataset_to_signal(fin,data_path,lazy)
        dictionary["original_metadata"] = original_metadata
        dictionary["mapping"] = mapping
        signal_dict_list.append(dictionary)
    if not nxdata_only:
        for data_path in hdf_data_paths:        
            data = _extract_hdf_dataset(fin,data_path,lazy)
            if data is not None:
                signal_type =  guess_signal_type(data)
                title = _text_split(data_path, '/')[-1]
                metadata = {'General': {'original_filename': \
                          os.path.split(filename)[1],\
                'title': title},\
                "Signal": {'signal_type': signal_type}}
                dictionary = {'data': data,
                              'metadata': metadata,
                              'original_metadata':original_metadata,
                              'mapping':mapping}                
                signal_dict_list.append(dictionary)

        
    return signal_dict_list
        
def _is_linear_axis(data):
    
    """
    
    Check if the data is linearly incrementing
    
    """
    steps = np.diff(data)
    est_steps = np.array([steps[0]]*len(steps))
    return np.allclose(est_steps,steps,rtol=1.0e-5)

def _is_number(a):
    """
    
    Check if the value is a number
    # will be True also for 'NaN'
    
    """
    try:
        float(a)
        return True
    except ValueError:
        return False

def _is_numeric_data(data):
    """
 
    Check that data contains numeric data
    
    """
    try:
        data.astype(float)
        return True
    except ValueError:
        return False
 
    
def _is_int(s):
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
    nx_link_targets = []
    rootname=""
    def find_data_in_tree(myDict,rootname):
        for key, value in myDict.items():
            if rootname != "":
                rootkey = rootname +"/"+ key
            else:
                rootkey = "/" + key
            if value:
                if "NX_class" in value.attrs:
                    if value.attrs["NX_class"] == b"NXdata" \
                        and "signal" in value.attrs.keys(): 
                        if search_keys:
                            if any(s in rootkey for s in search_keys):
                                nx_datasets.append(rootkey)
                        else:
                            nx_datasets.append(rootkey) 
                        find_data_in_tree(value,rootkey)
                        continue
            if isinstance(value,h5py.Dataset):
                
                if value.size >= 2:
                    target,b,c, = _getlink(value,rootkey)
                    if target is not None:
                        nx_link_targets.append(target)
                    if search_keys:
                        if any(s in rootkey for s in search_keys):
                            hdf_datasets.append(rootkey)                            
                    else:
                        hdf_datasets.append(rootkey)

            if isinstance(value,h5py.Group):
                find_data_in_tree(value,rootkey)
    # need to use custom recursive function as visititems in h5py
    # does not visit links
    find_data_in_tree(myDict,rootname)
    clean_hdf_datasets=[]
    # remove duplicate keys
    nx_link_targets = list(dict.fromkeys(nx_link_targets))
    nx_datasets = list(dict.fromkeys(nx_datasets))
    hdf_datasets = list(dict.fromkeys(hdf_datasets))
    # remove sets already in nexus 
    clean_hdf_datasets = [x for x in hdf_datasets if x not in nx_link_targets]
    # remove sets already in nexus 
    cleaner_hdf_datasets = []
    for x in clean_hdf_datasets:
        if not any(y in x for y in nx_datasets):
            cleaner_hdf_datasets.append(x)
    
    return nx_datasets,cleaner_hdf_datasets


def fix_exclusion_keys(key):
    """
    
    Exclude hyperspy specific keys
    Signal and DictionaryBrowser look for specific keys
    - if they appear in a hdf file or nexus file they
    cause an exception
    This method prepends the key with "fix" so the information is
    still present
    
    """
    exclusion_list=['_sig_','_list_empty_','_tuple_empty_',
                    '_bs_','_datetime_date',"_datetime_","keys"]
    
    if any([key.startswith(x) for x in exclusion_list]):
        return"fix_"+key
    else:
        return key

    
def _find_hdf_metadata(fin,search_keys=None):
    
    """    
    Read the metadata from a nexus or hdf file       
    The method iterates through group attributes and 
    Datasets of size < 2 (i.e. single values)
    and returns a dictionary of the entries
    This is a convenience method to inspect a file for a value
    rather than loading the file as a signal
    
    Parameters
    ------------
    fin : h5py File object 
    search_list  : str list or None  
        Only return items which contain the strings
        .e.g search_list = ["instrument","Fe"] will return
        hdf entries with instrument or Fe in their hdf path.
        This excudes attrs
    search_datasets = True
    search 
    
    Returns
    -------
    Metadata dictionary. When search_list is specified only items
    matching the search will be returned 
        
    """
    if search_keys == None:
        search_keys=[""]
    metadata_dict = {}
    rootname=""
    # recursive function
    def find_data_in_tree(myDict,rootname):
        for key, value in myDict.items():
            if rootname != "":
                rootkey = rootname +"/"+ key
            else:
                rootkey = key            

            if isinstance(value,h5py.Dataset): 
                if value.size < 2:
                   if any(s in rootkey for s in search_keys):
                        data = value[...]
                        output = _parse_value(data.item())
                        mod_keys = _text_split(rootkey, (".","/") )                        
                        # create the key, values in the dict
                        p = metadata_dict
                        for d in mod_keys:
                            d=fix_exclusion_keys(d)
                            p = p.setdefault(d,{})
                        p["value"] = output
                        # skip signals - these are handled below.
                        if value.attrs.keys():
                            p = p.setdefault("attrs",{})
                            for d,v in value.attrs.items():
                                d =fix_exclusion_keys(d)
                                p[d] = _parse_value(v)
            elif isinstance(value,h5py.Group):
                if any(s in rootkey or s in value.attrs.keys()
                    for s in search_keys):
                    mod_keys = _text_split(rootkey, (".","/") )
                    # create the key, values in the dict
                    p = metadata_dict
                    for d in mod_keys:
                        d=fix_exclusion_keys(d)
                        p = p.setdefault(d,{})
                    if value.attrs.keys():
                        p=p.setdefault("attrs",{})
                        for d,v in value.attrs.items():
                            d=fix_exclusion_keys(d)
                            p[d] = _parse_value(v)                                      
                find_data_in_tree(value,rootkey)  

    find_data_in_tree(fin,rootname)
    return metadata_dict

def guess_chunks(data):
    chunks= h5py._hl.filters.guess_chunk(data.shape, None, np.dtype(data.dtype).itemsize)
    return chunks


def write_nexus_groups(dictionary, group, **kwds):
    """
    Recursively iterate through dictionary and 
    create the groups and datasets
    A hdf dataset can have a value and a list of attrs and this doesn't
    quite fit with a python dict
    A dict for a dataset can therefore has a "value" key and "attrs" key
    if the "value" key is present the dataset is created set to this value 
    and then attrs for the dataset are added later.

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
            overwrite_dataset(group, value, key, **kwds)        
        elif isinstance(value, (int,float,str,bytes)):
            if isinstance(group,h5py.Dataset):
                group.attrs[key] = value
            else:
                group.require_dataset(key,data=value,shape=value.shape,
                                   dtype=value.dtype)        



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

def guess_signal_type(data):
    """
    
    Checks last 2 dimensions
    An area detector will be square, > 255 pixels each side and
    if rectangular ratio of sides is at most 4.
    An EDS detector will generally by ndetectors*2048 or ndetectors*4096
    the ratio test will be X10-1000  
    
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
    metadata = _find_hdf_metadata(fin,search_keys=search_keys)
    fin.close()
    if verbose:
        pprint.pprint(metadata)
    return metadata    


def get_datasets_in_file(filename,search_keys=None,
                          verbose=True):
    """
    
    Read from a nexus or hdf file and return a list 
    of the dataset entries
    The method iterates through group attributes and returns NXdata or 
    hdf datasets of size >=2 if they're not already NXdata blocks
    and returns a list of the entries
    This is a convenience method to inspect a file to see what datasets
    are present rather than loading all the sets in the file as signals
    
    Parameters
    ------------
    filename : str   path of the file to read
    search_keys  : str list or None  
        Only return items which contain the strings
        .e.g search_list = ["instrument","Fe"] will return
        hdf entries with instrument or Fe in their hdf path.
    verbose : boolean, default : True  - prints the datasets 
    
    Returns
    -------
    Dataset list. When search_list is specified only items whose path in the 
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
                print(hdfd)
        else:
            print("No HDF datasets not found or data is captured by NXdata")
    
    return nexus_data_paths,hdf_dataset_paths


def write_signal(signal, nxgroup, **kwds):
    """

    Store the signal data as nexus file by
    storing the data as an NXdata dataset 
    along with the metadata
    
    
    """
    group_name = signal.metadata.General.title if \
        signal.metadata.General.title else '__unnamed__'
    if "/" in group_name:
        group_name = group_name.replace("/", "-")
    smd = signal.metadata.Signal
    if signal.axes_manager.signal_dimension == 1:
        smd.record_by = "spectrum"
    elif signal.axes_manager.signal_dimension == 2:
        smd.record_by = "image"
    else:
        smd.record_by = ""

    nxdata = nxgroup.require_group(group_name)
    nxdata.attrs["NX_class"] = b"NXdata"    
    nxdata.attrs[u"signal"] = b"data"
    overwrite_dataset(nxdata, signal.data, u"data", chunks=None, **kwds)
    axis_names=[b"."]*len(signal.axes_manager.shape)
    for i,axis in enumerate(signal.axes_manager._axes):
        try:
            axname  = axis.name + "_indices"
        except:
            axname  = "axis"+str(i)#+ "_indices"            
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
    
    Write the signal to a nexus file
    This currently writes the
    signal.original_metadata structure, preserving the structure
    and the signal data and axes as an NXdata set in "/entry/data"
    
    """
    with h5py.File(filename, mode='w') as f:
        f.attrs['file_format'] = "Nexus"
        nxentry = f.create_group('entry')
        nxentry.attrs[u"NX_class"] = u"NXentry"

        # 
        # if want to store hyperspy metadata 
        # basically move any hyperspy metadata into the original_metadata
        # dict
        #if signal.metadata:
        #    if not "hyperspy_metadata" in signal.original_metadata:
        #        signal.original_metadata["hyperspy_metadata"]={}
        #    signal.original_metadata.hyperspy_metadata.add_dictionary(signal.metadata.as_dictionary())
        # now remove the hyperspy metadata... 
        #del signal.original_metadata.hyperspy_metadata
        
        try:
            write_signal(signal, nxentry, **kwds)
        except BaseException:
            raise
        try:
            if signal.original_metadata:
                for key in signal.original_metadata.keys():
                    if key not in f:
                        entry = f.create_group(key)
                    else:
                        entry = nxentry
                    # write the groups and structure
                    write_nexus_groups(signal.original_metadata[key].as_dictionary(),entry)
                    write_nexus_attr(signal.original_metadata[key].as_dictionary(),entry)

                    # add the attributes
                    
        except BaseException:
            raise
        finally:
            f.close()
