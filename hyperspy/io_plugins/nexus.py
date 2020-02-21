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
NX_ENCODING = sys.getfilesystemencoding()

_logger = logging.getLogger(__name__)
# Plugin characteristics

format_name = 'Nexus'
description = \
    'Read NXdata, hdf datasets and metadata from hdf5 or Nexus files'
full_support = False
# Recognised file extension
file_extensions = ['nxs','NXS','h5','hdf5']
default_extension = 0
# Writing capabilities:
writes = True


def _to_text(value):
    """Return a unicode string in both Python 2 and 3"""
    if isinstance(value, np.ndarray) and value.shape == (1,):
        value = value[0]
    if isinstance(value, bytes):
        try:
            text = value.decode(NX_ENCODING)
        except UnicodeDecodeError:
            if NX_ENCODING == 'utf-8':
                text = value.decode('latin-1')
            else:
                text = value.decode('utf-8')
    return text.replace('\x00','').rstrip()

def _text_split(s, sep):
    stack = [s]
    for char in sep:
        pieces = []
        for substr in stack:
            pieces.extend(substr.split(char))
        stack = pieces
    if '' in stack:
        stack.remove('')
    return stack


def nested_set(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value

def _getlink(h5group,rootkey):
    """Return the link target path and filename.

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
            _target = _to_text(h5group.attrs['target'])
            if not _target.startswith('/'):
                _target = '/' + _target
            if _target == rootkey:
                _target = None
    return _target, _filename, _abspath


def _nexus_dataset_to_signal_axes(tree,nexus_dataset):
    detector_index = 0
    dataentry = tree[nexus_dataset]
    if "signal" in dataentry.attrs.keys():
        if RepresentsInt(dataentry.attrs["signal"]):
            data_key  = "data"
        else:
            data_key  = dataentry.attrs["signal"]#.decode("ascii")
    else:
        _logger.info("No signal attr associated with NXdata so\
                     assuming signal name is data")
        data_key  = "data"
    data = dataentry[data_key] 
    nav_list = []
    if "axes" in dataentry.attrs.keys():           
        axes_key  = dataentry.attrs["axes"]
        axes_list = [num for num in axes_key]
        named_axes=list(range(len(axes_list)))
        for i,ax in enumerate(axes_list):
            if ax != ".":
                axname = ax
                index_name = ax + "_indices"
                ind_in_array = int(dataentry.attrs[index_name])
                if "units" in dataentry[ax].attrs:
                    units= dataentry[ax].attrs["units"]
                else:
                    units=""
                named_axes.remove(ind_in_array)
                if is_numeric_data(dataentry[ax]):
                    if is_linear_axis(dataentry[ax]):
                        nav_list.append({   
                            'size': data.shape[ind_in_array],
                            'index_in_array': ind_in_array,
                            'name': axname,
                            'scale': abs(dataentry[ax][1]-\
                                         dataentry[ax][0]),
                            'offset': min(dataentry[ax][0],\
                                          dataentry[ax][-1] ),
                            'units': units,
                            'navigate': True,
                            })
                    else:
                        nav_list.append({   
                            'index_in_array': ind_in_array,
                            'name': axname,
                            'axis' : dataentry[ax],
                            'units': units,
                            'navigate': True,
                            })
                    
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
    return nav_list



def file_reader(filename,lazy=True, dset_search_keys=None,
                ignore_non_linear_dims=True,
                map_datasets_to_signal=None,
                max_number_of_datasets=5,
                **kwargs):
    
    """
    
    Additional keywords:
    linked_files : list of filenames linked to the main nexus file
          Nexus allows for links to files but the paths in the file can be
          absolute. When moving data from linux to windows or backup these
          links can break
    
    
    """
    # search for NXdata sets...
    datakeys = kwargs.get('datakey',"")    
    
    fin = h5py.File(filename)
    original_metadata = find_hdf_metadata(fin)
    signal_dict_list = []
    if map_datasets_to_signal:
        dictionary = _map_datasets_to_signal(map_datasets)
        signal_dict_list.append(dictionary)
    else:
        nexus_data_paths, hdf_data_paths = _find_data(fin,datakeys)
        # strip out the metadata (basically everything other than NXdata)
        #
        # Be careful not to load a large number of datasets
        # Check against an upper limit and that the user has
        # sub-selected the datasets to load used dset_search_keys
        #
        if nexus_data_paths:
            if(len(nexus_data_paths)>max_number_of_datasets):
                raise ValueError("Number of nexus datasets you had asked to load\
                                 exceeds the max_number_of_datasets - please\
                                 use dset_search_keys to filter the number of \
                                 datasets of increases the max_number_of_datasets")
        else:
            if(len(hdf_data_paths)>max_number_of_datasets):
                raise ValueError("Number of hdf datasets you had asked to load\
                                 exceeds the max_number_of_datasets - please\
                                 use dset_search_keys to filter the number of \
                                 datasets of increases the max_number_of_datasets")            
        if nexus_data_paths:
            
            for data_path in nexus_data_paths:        
                metadata={}
                dataentry = fin[data_path]
                dictionary = _nexus_dataset_to_signal(dataentry)
                signal_dict_list.append(dictionary)
        else:
            for data_path in hdf_data_paths:        
                dataentry = fin[data_path]
                dictionary = _hdf_dataset_to_signal(dataentry)
                signal_dict_list.append(dictionary)
                

        
    return signal_dict_list
        
def is_linear_axis(data):
    
    """
    Check if the axis is linear
    
    
    """
    steps = np.diff(data)
    est_steps = np.array([steps[0]]*len(steps))
    return np.allclose(est_steps,steps,rtol=1.0e-5)

def is_number(a):
    # will be True also for 'NaN'
    try:
        float(a)
        return True
    except ValueError:
        return False

def is_numeric_data(data):
    
    for x in data:
        if not is_number(x):
            return False
    return True

        
    
# The parameters
def RepresentsInt(s):
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
    This is a convenience method to inspect a file to see what datasets
    are present rather than loading all the sets in the file as signals
    
    Parameters
    ------------
    myDict : dict or h5p.File object
    search_list  : str list or None  
        Only return items which contain the strings
        .e.g search_list = ["instrument","Fe"] will return
        hdf entries with instrument or Fe in their hdf path.
    
    Returns
    -------
    Dataset list. When search_list is specified only items whose path in the 
    hdf file match the search will be returned 
        
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

            if "NX_class" in value.attrs:
                if value.attrs["NX_class"] == b"NXdata":
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
    # need to use custom recursive function as visititems does not visit links
    find_data_in_tree(myDict,rootname)
    clean_hdf_datasets=[]
    # remove duplicate keys
    nx_link_targets = list(dict.fromkeys(nx_link_targets))
    nx_datasets = list(dict.fromkeys(nx_datasets))
    hdf_datasets = list(dict.fromkeys(hdf_datasets))
    # remove sets already in nexus 
    clean_hdf_datasets = [x for x in hdf_datasets if x not in nx_link_targets]
            
    return nx_datasets,clean_hdf_datasets

def find_hdf_metadata(fin,search_list=[""],
                      search_datasets=True,
                      search_groups=True):
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
    search_datasets = True
    search 
    
    Returns
    -------
    Metadata dictionary. When search_list is specified only items
    matching the search will be returned 
        
    """

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
                   if any(s in rootkey for s in search_list):
                        data = value[...]
                        output = data.item()
                        keys = _text_split(rootkey, (".","/") )
                        # create the key, values in the dict
                        p = metadata_dict
                        for key in keys:
                            p = p.setdefault(key,{})
                        p["value"] = output
                        for dkey,dvalue in value.attrs.items():
                            if isinstance(dvalue,bytes):
                                dvalue = dvalue.decode(NX_ENCODING)                       
                            p[dkey] = dvalue                          
            elif isinstance(value,h5py.Group):# and search_groups:
                keys = _text_split(rootkey, (".","/") )
                # create the key, values in the dict
                p = metadata_dict
                for dkey in keys:
                    p = p.setdefault(dkey,{})
                for dkey,dvalue in value.attrs.items():
                    if isinstance(dvalue,bytes):
                        dvalue = dvalue.decode(NX_ENCODING)
                    p[key] = dvalue                                      
                find_data_in_tree(value,rootkey)  

    find_data_in_tree(fin,rootname)
    return metadata_dict

def guess_chunks(data):
    chunks= h5py._hl.filters.guess_chunk(data.shape, None, np.dtype(data.dtype).itemsize)
    return chunks


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
        else:
            signal_type = "Signal1D"
    return signal_type
        


def get_metadata_in_file(filename,search_list=None,
                        search_datasets=True,
                        search_group_attributes=True):
    """
    
    Read the metadata from a nexus or hdf file       
    The method iterates through group attributes and 
    Datasets of size < 2 (i.e. single values)
    and returns a dictionary of the entries
    This is a convenience method to inspect a file for a value
    rather than loading the file as a signal
    
    Parameters
    ------------
    filename : str  -  name of the file to read
    search_list  : str list or None  
        Only return items which contain the strings
        .e.g search_list = ["instrument","Fe"] will return
        hdf entries with instrument or Fe in their hdf path.
    search_datasets = True
    search 
    
    Returns
    -------
    Metadata dictionary. When search_list is specified only items
    matching the search will be returned 
        
    """
    if search_list == None:
        search_list=[""]
    fin = h5py.File(filename,"r")
    # search for NXdata sets...
    # strip out the metadata (basically everything other than NXdata)
    metadata = find_hdf_metadata(fin,search_list=search_list)
    fin.close()
    pprint.pprint(metadata)
    return metadata    


def get_datasets_in_file(filename,search_list=None,
                          search_datasets=True,
                          search_group_attributes=True,
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
    filename : str  -  name of the file to read
    search_list  : str list or None  
        Only return items which contain the strings
        .e.g search_list = ["instrument","Fe"] will return
        hdf entries with instrument or Fe in their hdf path.
    search_datasets = True
    verbose : boolean, default : True  - prints the datasets 
    
    Returns
    -------
    Dataset list. When search_list is specified only items whose path in the 
    hdf file match the search will be returned 
        
    """
    if search_list == None:
        search_list=[""]
    fin = h5py.File(filename,"r")
    # search for NXdata sets...
    # strip out the metadata (basically everything other than NXdata)
    nexus_data_paths,hdf_dataset_paths = \
        _find_data(fin,search_list=search_list)
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

    nxdata = nxgroup.create_group(group_name)
    nxdata.attrs["NX_class"] = u"NXdata"    
    nxdata.attrs[u"signal"] = u"data"
    nxdata.create_dataset(u"data", data=signal.data)
    axis_names=[b"."]*len(signal.axes_manager.shape)
    for i,axis in enumerate(signal.axes_manager.navigation_axes):
        axname  = axis.name + "_indices"
        axindex = axis.index_in_array
        nxdata.attrs[axname] = axindex
        nxdata.create_dataset(axis.name,data=axis.axis)
        axis_names[axis.index_in_array]=bytes(axis.name, 'utf-8')
    nxdata.attrs[u"axes"] = axis_names
    


def file_writer(filename,
                signal,
                *args, **kwds):
    with h5py.File(filename, mode='w') as f:
        f.attrs['file_format'] = "Nexus"
        nxentry = f.create_group('entry')
        nxentry.attrs[u"NX_class"] = u"NXentry"
        try:
            write_signal(signal, nxentry, **kwds)
        except BaseException:
            raise
        finally:
            f.close()
