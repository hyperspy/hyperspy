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
import nexusformat.nexus.tree as nx
import numpy as np
import dask.array as da
import os
import h5py 
import sys
import six
NX_ENCODING = sys.getfilesystemencoding()

_logger = logging.getLogger(__name__)
# Plugin characteristics

format_name = 'Nexus'
description = \
    'Read NXdata groups from Nexus files,all other groups added as metadata'
full_support = False
# Recognised file extension
file_extensions = ['nxs','NXS','h5']
default_extension = 0
# Writing capabilities:
writes = True



def nx_data_search(myDict,rootname=""):
    """
    Recursively search the results of an nx tree (a nested dict) 
    to find nxdata sets and return a list of keys pointing to those NXdata sets    
    """
    datalist = []
    def find_data_in_tree():
        for key, value in myDict.items():
            if rootname != "":
                rootkey = rootname +"/"+ key
            else:
                rootkey = key
            if isinstance(value,nx.NXdata):
                datalist.append(rootkey)
            else:
                if isinstance(value,nx.NXgroup):
                    find_data_in_tree(value,rootkey)
    find_data_in_tree(myDict,rootname)
    return datalist


def text(value):
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
    elif six.PY3:
        text = str(value)
    return text.replace('\x00','').rstrip()

def nx_metadata_search(a_dict,rootname=None):
    """
    
    Recursively search the results of an nx tree (a nested dict) 
    to find all non-nxdata information and return a dict of this metadata
    
    """
    new_dict = {}
    if rootname == None:
        rootname = ""
    for k, v in a_dict.items():
        if isinstance(v,nx.NXdata):
            continue
        else:
            if isinstance(v,(dict,nx.NXgroup)):
                rootkey = rootname +"/"+ k
                v = nx_metadata_search(v,rootname=rootkey)
            elif isinstance(v,nx.NXfield):
                v=v.nxvalue
            new_dict[k] = v
    return new_dict

def tsplit(s, sep):
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
    
def nx_metadata_search_for_keys(a_dict,nxdict_keys):
    """
    Recursively search the results of an nx tree (a nested dict) 
    to find all non-nxdata information and return a dict of this metadata
    
    """
    dct = {}
    for item in nxdict_keys:
        correct_item = item.replace('.','/')
        if correct_item in a_dict:
            p = dct
            keys = tsplit(correct_item, (".","/") )
            v=a_dict[correct_item]
            for key in keys[:-1]:
                p = p.setdefault(key,{})
            # this is a workaround for a fault on
            # windows pc's....
            for i in range(2):
                try:
                        p[keys[-1]] = v.nxvalue
                except:
                        p[keys[-1]] = {}
    return dct

   
def nexus_metadata_reader(filename,**kwargs):
    """
    
    Read the metadata from a nexus file       
    
    """
    fin = nx.load(filename)
    # search for NXdata sets...
    mapping = kwargs.get('mapping',{})
    # strip out the metadata (basically everything other than NXdata)
    metadata = nx_metadata_search_for_keys(fin,mapping.keys())
    
    return metadata    

def read_h5_datasets(fin,datakeys):

    datalist=[]
    def visitor_func(name, node):
        if isinstance(node,h5py.Dataset):
            if node.size > 2:
                for s in datakeys:
                    if s in node.name:
                        datalist.append(node)
        else:
            pass
    fin.visititems(visitor_func)
    return datalist
    

def dataset_to_signal_axes(tree,dataset):
    detector_index = 0
    dataentry = tree[dataset]
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
                ind_in_array = int(NXfield_to_float(dataentry.attrs[index_name]))
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
                            'scale': abs(NXfield_to_float(dataentry[ax][1])-\
                                         NXfield_to_float(dataentry[ax][0])),
                            'offset': min(NXfield_to_float(dataentry[ax][0]),\
                                          NXfield_to_float(dataentry[ax][-1]) ),
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

def find_nexus_data(myDict,datakeys):
    """
    Recursively search the results of an nx tree (a nested dict) 
    to find nxdata sets and return a list of keys pointing to those NXdata sets    
    """
    datalist = []
    rootname=""
    def find_data_in_tree(myDict,rootname):
        for key, value in myDict.items():
            if rootname != "":
                rootkey = rootname +"/"+ key
            else:
                rootkey = key
            if isinstance(value,nx.NXdata):
                datalist.append(rootkey)
            else:
                if isinstance(value,nx.NXgroup):
                    find_data_in_tree(value,rootkey)
    find_data_in_tree(myDict,rootname)
    return_list=[]
    for data in datalist:
        for dkey in datakeys:
            if dkey in data:
                return_list.append(data)
    return datalist


def file_reader(filename,lazy=True,**kwargs):
    """
    
    Additional keywords:
    linked_files : list of filenames linked to the main nexus file
          Nexus allows for links to files but the paths in the file can be
          absolute. When moving data from linux to windows or backup these
          links can break
    
    
    """
    ext=os.path.extension = os.path.splitext(filename)[1]
    # search for NXdata sets...
    mapping = kwargs.get('mapping',{})
    datakeys = kwargs.get('datakey',"")    
    
    if ext == "h5":
        fin = h5py.File(filename)
        datalist=read_h5_datasets(fin,datakeys)
        nav_list=[]
    else:
        fin = nx.load(filename)
        datalist = find_nexus_data(fin,datakeys)
    # strip out the metadata (basically everything other than NXdata)
    original_metadata = nx_metadata_search_for_keys(fin,mapping.keys())

    signal_dict_list = []
    for data_name in datalist:        
        metadata={}
        dataentry = fin[data_name]
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
        nav_list = dataset_to_signal_axes(fin,data_name)    
        if lazy:
            chunks=guess_chunks(data)
            data_lazy = da.from_array(data, chunks=chunks)
            data_lazy = da.squeeze(data_lazy)
        else:
            data_lazy = np.squeeze(data)

        # now extract some metadata from the nexus file....
        signal_type = guess_signal_type(data_lazy)
        if nav_list:
            if signal_type == "Signal1D":
                nav_list[-1]["navigate"]=False
            else:
                nav_list[-1]["navigate"]=False
                nav_list[-2]["navigate"]=False
        name= os.path.split(data_name)[-1]
        new_metadata = {'General': {'original_filename': \
                                os.path.split(filename)[1],\
                    'title': name},\
                    "Signal": {'signal_type': signal_type}}

        metadata.update(new_metadata)
        if nav_list:      
            dictionary = {'data': data_lazy,
                          'axes': nav_list,
                          'metadata': metadata,
                          'original_metadata':original_metadata,
                          'mapping':mapping}
        else:
            dictionary = {'data': data_lazy,
                          'metadata': metadata,
                          'original_metadata':original_metadata,
                          'mapping':mapping}
            
        signal_dict_list.append(dictionary)
        
    return signal_dict_list
        
def guess_chunks(data):
    if(len(data.shape)==5):
        near = int(0.5+
                       max(data.shape[-5],
                           data.shape[-4],
                           data.shape[-3])/2.0)
        near = roundup10(near)
        chunks = (
                    1,near,near,data.shape[-2],data.shape[-1])
    elif(len(data.shape)==4):
        near = int(0.5+
                    max(data.shape[-4],
                       data.shape[-3])/2.0)
        near = roundup10(near)
        chunks = (
                    near,near,data.shape[-2],data.shape[-1])
    elif(len(data.shape)==3):
        near = int(0.5+
                   max(data.shape[-3],
                       data.shape[-2])/2.0)
        near = roundup10(near)
        chunks = (
                    near,near,data.shape[-1])
    elif(len(data.shape)==2):
        chunks = (
                    data.shape[-2],data.shape[-1])    
    return chunks
        
def roundup10(x):
    return int(np.ceil(x / 10.0)) * 10

def NXfield_to_float(value): 
    if isinstance(value,nx.NXfield):
        value = value.nxdata
        if isinstance(value,(np.ndarray,list)):
            if len(value)==1:
                value = value[0]
    return value

def is_linear_axis(data):
    
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

def guess_signal_type(data):
    """
    
    Checks last 2 dimensions
    An area detector will be square, > 255 pixels each side and
    if rectangular ratio of sides is at most 4.
    An EDS detector will generally by ndetectors*2048 or ndetectors*4096
    the ratio test will be X10-1000  
    
    """
    n1 = data.shape[-1]
    n2 = data.shape[-2]
#    if len(data.shape)==2:
#        signal_type = "Signal2D"
#    else:        
    if n2>127 and n1 >127 and n1/n2 < 1.5:
        signal_type = "Signal2D"
    else:
        signal_type = "Signal1D"
    return signal_type
        
    
# The parameters
def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

        


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
