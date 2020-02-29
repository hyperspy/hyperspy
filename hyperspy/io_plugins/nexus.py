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
file_extensions = ['nxs','NXS','h5','hdf','hdf5']
default_extension = 0
# Writing capabilities:
writes = True


def _to_text(value):
    """ To convert bytes to strings where needed """
    toreturn = value
    if isinstance(value, np.ndarray) and value.shape == (1,):
        toreturn = value[0]        
    if isinstance(value, bytes):
        try:
            text = value.decode(NX_ENCODING)
        except UnicodeDecodeError:
            if NX_ENCODING == 'utf-8':
                text = value.decode('latin-1')
            else:
                text = value.decode('utf-8')
        toreturn= text.replace('\x00','').rstrip()
    if isinstance(value, (np.int,np.float,np.ndarray)):
        toreturn= value
        
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
    detector_index = 0
    dataentry = tree[nexus_dataset]
    if "signal" in dataentry.attrs.keys():
        if RepresentsInt(dataentry.attrs["signal"]):
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
        axes_list = [_to_text(num) for num in axes_key]
        named_axes=list(range(len(axes_list)))
        for i,ax in enumerate(axes_list):
            if ax != ".":
                index_name = ax + "_indices"
                ind_in_array = int(dataentry.attrs[index_name])
                axis_index_list.append(ind_in_array)
                if "units" in dataentry[ax].attrs:
                    units= _to_text(dataentry[ax].attrs["units"])
                else:
                    units=""
                named_axes.remove(ind_in_array)
                if is_numeric_data(dataentry[ax]):
                    if is_linear_axis(dataentry[ax]):
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
            
        metadata = {'General': {'title': nexus_dataset},\
                    "Signal": {'signal_type': signal_type}}
        if nav_list:      
            dictionary = {'data': data_lazy,
                          'axes': nav_list,
                          'metadata':metadata}
        else:
            dictionary = {'data': data_lazy,
                          'metadata': metadata}                    
    return dictionary

def _map_datasets_to_signal(tree,dataset_map,lazy):
    """
    
    dataset_map is a dictionary similar to an NXdata set
    {
    
    "signal" : "/entry/result/data"
    "axes" :  ["/entry/result/sample_x","/entry/result/sample_y","."]
    "axes_indices" : [0,1,2]
    "signal_type"  : "Signal1D"      
    
    """
    
    if "signal" not in dataset_map:
        raise ValueError("signal key not found in dataset_map")
        
    dataentry = dataset_map["signal"] 
    data = tree[dataentry]
    detector_index = 0
    nav_list = []
    if "axes" in dataset_map.keys():           
        axes_list  = tree[dataset_map["axes"]]
        named_axes=list(range(len(axes_list)))
        for i,ax in enumerate(axes_list):
            if ax != ".":
                axname = ax
                ind_in_array = dataset_map["axes_indices"][i]
                if "units" in tree[ax].attrs:
                    units= dataentry[ax].attrs["units"]
                else:
                    units=""
                named_axes.remove(ind_in_array)
                if is_numeric_data(tree[ax]):
                    if is_linear_axis(dataentry[ax]):
                        nav_list.append({   
                            'size': data.shape[ind_in_array],
                            'index_in_array': ind_in_array,
                            'name': axname,
                            'scale': abs(tree[ax][1]-\
                                         tree[ax][0]),
                            'offset': min(tree[ax][0],\
                                          tree[ax][-1] ),
                            'units': units,
                            'navigate': True,
                            })
                    else:
                        nav_list.append({   
                            'size': data.shape[ind_in_array],
                            'index_in_array': ind_in_array,
                            'name': axname,
                            'scale': 1,
                            'offset': 0,
                            'navigate': True,
                            })
                         # when non-linear axes supported...
#                        nav_list.append({   
#                            'index_in_array': ind_in_array,
#                            'name': axname,
#                            'axis' : tree[ax],
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
        signal_type = dataset_map["signal_type"]
            
        metadata = {'General': {'title': dataset_map["signal"]},\
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
                mapping_datasets_to_signal=None,
                **kwds):
    
    """
    
    Reads NXdata class or hdf datasets from a file and returns signal
    Reads all data of size<2 as metadata

    This does not return or use NX Classes as the objective is conversion
    into singals rather than interpretting Nexus Objects or
    Nexus application definitions
    
    Follows Nexus documentation definition of a nxdata class
    https://manual.nexusformat.org/classes/base_classes/NXdata.html#nxdata
    
   
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
    if mapping_datasets_to_signal:
        dictionary = _map_datasets_to_signal(fin,mapping_datasets_to_signal
                                             ,lazy)
        dictionary["original_metadata"] = original_metadata
        dictionary["mapping"] = mapping
        signal_dict_list.append(dictionary)
    else:
        for data_path in hdf_data_paths:        
            data = _extract_hdf_dataset(fin,data_path,lazy)
            if data:
                signal_type =  guess_signal_type(data)
                metadata = {'General': {'original_filename': \
                          os.path.split(filename)[1],\
                'title': data_path},\
                "Signal": {'signal_type': signal_type}}
                dictionary = {'data': data,
                              'metadata': metadata,
                              'original_metadata':original_metadata,
                              'mapping':mapping}                
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
    This is a convenience method to inspect a file to see which datasets
    are present rather than loading all the sets in the file as signals
    
    Parameters
    ------------
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
    # need to use custom recursive function as visititems does not visit links
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
    
def _find_hdf_metadata(fin,search_keys=[""],
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
                   if any(s in rootkey for s in search_keys):
                        data = value[...]
                        output = _to_text(data.item())
                        dkeys = _text_split(rootkey, (".","/") )
                        #keys = _text_split(rootkey, ("/") )
                        
                        # create the key, values in the dict
                        p = metadata_dict
                        for dkey in dkeys:
                            dkey=fix_exclusion_keys(dkey)
                            p = p.setdefault(dkey,{})
                        p["value"] = output
                        # skip signals - these are handled below.
                        if value.attrs.keys():
                            #p = p.setdefault("attrs",{})
                            for dkey,dvalue in value.attrs.items():
                                dkey =fix_exclusion_keys(dkey)
                                dvalue = _to_text(dvalue) 
                                p[dkey] = dvalue
            elif isinstance(value,h5py.Group):
                if any(s in rootkey for s in search_keys):
                    dkeys = _text_split(rootkey, (".","/") )
                    # create the key, values in the dict
                    p = metadata_dict
                    for dkey in dkeys:
                        dkey=fix_exclusion_keys(dkey)
                        p = p.setdefault(dkey,{})
                    if value.attrs.keys():
                        #p=p.setdefault("attrs",{})
                        for dkey,dvalue in value.attrs.items():
                            dkey=fix_exclusion_keys(dkey)
                            dvalue = _to_text(dvalue)                        
                            p[dkey] = dvalue                                      
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
        elif n1 > 127 and n2 < n1:
            signal_type = "Signal1D"
        else:
            signal_type = "BaseSignal"
    return signal_type
        


def get_metadata_in_file(filename,search_keys=None,
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
    if search_keys == None:
        search_keys=[""]
    fin = h5py.File(filename,"r")
    # search for NXdata sets...
    # strip out the metadata (basically everything other than NXdata)
    metadata = _find_hdf_metadata(fin,search_keys=search_keys)
    fin.close()
    pprint.pprint(metadata)
    return metadata    


def get_datasets_in_file(filename,search_keys=None,
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

    nxdata = nxgroup.create_group(group_name)
    nxdata.attrs["NX_class"] = b"NXdata"    
    nxdata.attrs[u"signal"] = b"data"
    overwrite_dataset(nxdata, signal.data, u"data", chunks=None, **kwds)
    #nxdata.create_dataset(u"data", data=signal.data)
    axis_names=[b"."]*len(signal.axes_manager.shape)
    for i,axis in enumerate(signal.axes_manager.navigation_axes):
        try:
            axname  = axis.name + "_indices"
        except:
            axname  = "axis"+str(i)#+ "_indices"            
        axindex = axis.index_in_array
        indices = axname+"_indices"
        nxdata.attrs[bytes(indices, 'utf-8')] = axindex
        nxdata.create_dataset(axname,data=axis.axis)
        axis_names[axis.index_in_array]=bytes(axname, 'utf-8')
    nxdata.attrs[u"axes"] = axis_names
    

def overwrite_dataset(group, data, key, chunks=None, **kwds):
    if chunks is None:
        # Optimise the chunking to contain at least one signal per chunk
        chunks = guess_chunks(data)

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


def dict2hdfgroup(dictionary, group, **kwds):
    from hyperspy.misc.utils import DictionaryTreeBrowser
    from hyperspy.signal import BaseSignal

    def parse_structure(key, group, value, _type, **kwds):
        try:
            tmp = np.array(value)
        except ValueError:
            tmp = np.array([[0]])
        if tmp.dtype == np.dtype('O') or tmp.ndim != 1:
            dict2hdfgroup(dict(zip(
                [str(i) for i in range(len(value))], value)),
                group.create_group(_type + str(len(value)) + '_' + key),
                **kwds)
        elif tmp.dtype.type is np.unicode_:
            if _type + key in group:
                del group[_type + key]
            group.create_dataset(_type + key,
                                 tmp.shape,
                                 dtype=h5py.special_dtype(vlen=str),
                                 **kwds)
            group[_type + key][:] = tmp[:]
        else:
            if _type + key in group:
                del group[_type + key]
            group.create_dataset(
                _type + key,
                data=tmp,
                **kwds)

    for key, value in dictionary.items():
        if isinstance(value, dict):
            dict2hdfgroup(value, group.create_group(key),
                          **kwds)
        elif isinstance(value, DictionaryTreeBrowser):
            dict2hdfgroup(value.as_dictionary(),
                          group.create_group(key),
                          **kwds)
        elif isinstance(value, BaseSignal):
            kn = key if key.startswith('_sig_') else '_sig_' + key
            write_signal(value, group.require_group(kn))
        elif isinstance(value, (np.ndarray, h5py.Dataset, da.Array)):
            overwrite_dataset(group, value, key, **kwds)
        elif value is None:
            group.attrs[key] = '_None_'
        elif isinstance(value, bytes):
            try:
                # binary string if has any null characters (otherwise not
                # supported by hdf5)
                value.index(b'\x00')
                group.attrs['_bs_' + key] = np.void(value)
            except ValueError:
                group.attrs[key] = value.decode()
        elif isinstance(value, str):
            group.attrs[key] = value
        elif isinstance(value, list):
            if len(value):
                parse_structure(key, group, value, '_list_', **kwds)
            else:
                group.attrs['_list_empty_' + key] = '_None_'
        elif isinstance(value, tuple):
            if len(value):
                parse_structure(key, group, value, '_tuple_', **kwds)
            else:
                group.attrs['_tuple_empty_' + key] = '_None_'
        else:
            try:
                group.attrs[key] = value
            except BaseException:
                _logger.exception(
                    "The hdf5 writer could not write the following "
                    "information in the file: %s : %s", key, value)

def file_writer(filename,
                signal,
                *args, **kwds):
          
    with h5py.File(filename, mode='w') as f:
        f.attrs['file_format'] = "Nexus"
        nxentry = f.create_group('entry')
        nxentry.attrs[u"NX_class"] = u"NXentry"
        #try:
        #    if signal.metadata:
        #        dict2hdfgroup(signal.metadata.as_dictionary(),nxentry) 
        #except BaseException:
        #    raise
        if signal.metadata:
            if not "hyperspy_metadata" in signal.original_metadata:
                signal.original_metadata["hyperspy_metadata"]={}
            signal.original_metadata.hyperspy_metadata.add_dictionary(signal.metadata.as_dictionary())
        try:
            if signal.original_metadata:
                for key in signal.original_metadata.keys():
                    if key not in f:
                        entry = f.create_group(key)
                    else:
                        entry = nxentry
                    dict2hdfgroup(signal.original_metadata[key].as_dictionary(),entry)
        except BaseException:
            raise

        try:
            write_signal(signal, nxentry, **kwds)
        except BaseException:
            raise
        finally:
            f.close()
