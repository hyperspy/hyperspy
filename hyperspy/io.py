# -*- coding: utf-8 -*-
# Copyright 2007-2011 The Hyperspy developers
#
# This file is part of  Hyperspy.
#
#  Hyperspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  Hyperspy.  If not, see <http://www.gnu.org/licenses/>.

import os
import glob
import tempfile
import os.path as path

import numpy as np

from hyperspy import messages
import hyperspy.defaults_parser
from hyperspy.io_plugins import (msa, digital_micrograph, fei, mrc,
    ripple, tiff)
from hyperspy.gui.tools import Load
from hyperspy.misc.utils import (ensure_directory, DictionaryBrowser, 
    strlist2enumeration)
from hyperspy.misc.natsort import natsorted
import hyperspy.misc.utils_varia

io_plugins = [msa, digital_micrograph, fei, mrc, ripple, tiff]

#try:
#    from hyperspy.io_plugins import fits
#    io_plugins.append(fits)
#except ImportError:
#    messages.information('The FITS IO features are not available')
try:
    from hyperspy.io_plugins import netcdf
    io_plugins.append(netcdf)
except ImportError:
    pass
    # NetCDF is obsolate and is only provided for users who have
    # old EELSLab files. Therefore, we print no message if it is not
    # available
    #~ messages.information('The NetCDF IO features are not available')
    
try:
    from hyperspy.io_plugins import hdf5
    io_plugins.append(hdf5)
except ImportError:
    messages.warning('The HDF5 IO features are not available. '
    'It is highly reccomended to install h5py')
    
try:
    from hyperspy.io_plugins import image
    io_plugins.append(image)
except ImportError:
    messages.information('The Image (PIL) IO features are not available')

default_write_ext = set()
for plugin in io_plugins:
    if plugin.writes:
        
        default_write_ext.add(
            plugin.file_extensions[plugin.default_extension])

def load(filenames=None, record_by=None, signal_type=None, 
         stack=False, mmap=False, mmap_dir=None, **kwds):
    """
    Load potentially multiple supported file into an hyperspy structure
    Supported formats: HDF5, msa, Gatan dm3, Ripple (rpl+raw)
    FEI ser and emi and hdf5, tif and a number of image formats.
    
    Any extra keyword is passed to the corresponsing reader. For 
    available options see their individual documentation.
    
    Parameters
    ----------
    filenames :  None, str or list of strings
        The filename to be loaded. If None, a window will open to select
        a file to load. If a valid filename is passed in that single
        file is loaded. If multiple file names are passed in
        a list, a list of objects or a single object containing the data
        of the individual files stacked are returned. This behaviour is
        controlled by the `stack` parameter (see bellow). Multiple
        files can be loaded by using simple shell-style wildcards, 
        e.g. 'my_file*.msa' loads all the files that starts
        by 'my_file' and has the '.msa' extension.
    record_by : None | 'spectrum' | 'image' 
        Manually set the way in which the data will be read. Possible
        values are 'spectrum' or 'image'.
    signal_type : str
        Manually set the signal type of the data. Although only setting
        signal type to 'EELS' will currently change the way the data is 
        loaded, it is good practice to set this parameter so it can be 
        stored when saving the file. Please note that, if the 
        signal_type is already defined in the file the information 
        will be overriden without warning.
    stack : bool
        If True and multiple filenames are passed in, stacking all
        the data into a single object is attempted. All files must match
        in shape. It is possible to store the data in a memory mapped
        temporary file instead of in memory setting mmap_mode. The title is set
        to the name of the folder containing the files.
        
    mmap: bool
        If True and stack is True, then the data is stored
        in a memory-mapped temporary file.The memory-mapped data is 
        stored on disk, and not directly loaded into memory.  
        Memory mapping is especially useful for accessing small 
        fragments of large files without reading the entire file into 
        memory.
    mmap_dir : string
        If mmap_dir is not None, and stack and mmap are True, the memory
        mapped file will be created in the given directory,
        otherwise the default directory is used.
        
    Returns
    -------
    Signal instance or list of signal instances

    Examples
    --------
    Loading a single file providing the signal type:
    
    >>> d = load('file.dm3', signal_type='XPS')
    
    Loading a single file and overriding its default record_by:
    
    >>> d = load('file.dm3', record_by='Image')
    
    Loading multiple files:
    
    >>> d = load('file1.dm3','file2.dm3')
    
    Loading multiple files matching the pattern:
    
    >>>d = load('file*.dm3')

    """
    
    kwds['record_by'] = record_by
    if filenames is None:
        if hyperspy.defaults_parser.preferences.General.interactive is True:
            load_ui = Load()
            load_ui.edit_traits()
            if load_ui.filename:
                filenames = load_ui.filename
        else:
            raise ValueError("No file provided to reader and "
            "interactive mode is disabled")
        if filenames is None:
            raise ValueError("No file provided to reader")
        
    if isinstance(filenames, basestring):
        filenames=natsorted([f for f in glob.glob(filenames)
                             if os.path.isfile(f)])
        if not filenames:
            raise ValueError('No file name matches this pattern')
    elif not isinstance(filenames, (list, tuple)):
        raise ValueError(
        'The filenames parameter must be a list, tuple, string or None')
    if not filenames:
        raise ValueError('No file provided to reader.')
        return None
    else:
        if len(filenames) > 1:
            messages.information('Loading individual files')
        if stack is True:
            original_shape = None
            for i, filename in enumerate(filenames):
                obj = load_single_file(filename, output_level=0,
                    signal_type=signal_type, **kwds)
                if original_shape is None:
                    original_shape = obj.data.shape
                    record_by = obj.mapped_parameters.record_by
                    stack_shape = tuple([len(filenames),]) + original_shape
                    tempf = None
                    if mmap is False:
                        data = np.empty(stack_shape,
                                           dtype=obj.data.dtype)
                    else:
                        #filename = os.path.join(tempfile.mkdtemp(),
                                             #'newfile.dat')
                        tempf = tempfile.NamedTemporaryFile(
                                                        dir=mmap_dir)
                        data = np.memmap(tempf,
                                         dtype=obj.data.dtype,
                                         mode = 'w+',
                                         shape=stack_shape,)
                    signal = type(obj)(
                        {'data' : data})
                    # Store the temporary file in the signal class to
                    # avoid its deletion when garbage collecting
                    if tempf is not None:
                        signal._data_temporary_file = tempf
                    signal.axes_manager.axes[1:] = obj.axes_manager.axes
                    signal.axes_manager._set_axes_index_in_array_from_position()
                    eaxis = signal.axes_manager.axes[0]
                    eaxis.name = 'stack_element'
                    eaxis.navigate = True
                    signal.mapped_parameters = obj.mapped_parameters
                    # Get the title from the folder name
                    signal.mapped_parameters.title = \
                    os.path.split(
                        os.path.split(
                            os.path.abspath(filenames[0])
                                     )[0]
                                  )[1]
                    signal.original_parameters = DictionaryBrowser({})
                    signal.original_parameters.add_node('stack_elements')
                if obj.data.shape != original_shape:
                    raise IOError(
                "Only files with data of the same shape can be stacked")
                
                signal.data[i,...] = obj.data
                signal.original_parameters.stack_elements.add_node(
                    'element%i' % i)
                node = signal.original_parameters.stack_elements[
                    'element%i' % i]
                node.original_parameters = \
                    obj.original_parameters.as_dictionary()
                node.mapped_parameters = \
                    obj.mapped_parameters.as_dictionary()
                del obj
            messages.information('Individual files loaded correctly')
            signal.print_summary()
            objects = [signal,]
        else:
            objects=[load_single_file(filename, output_level=0,
                     signal_type=signal_type, **kwds) 
                for filename in filenames]
            
        if hyperspy.defaults_parser.preferences.General.plot_on_load:
            for obj in objects:
                obj.plot()
        if len(objects) == 1:
            objects = objects[0]
    return objects


def load_single_file(filename, record_by=None, output_level=2, 
    signal_type=None, **kwds):
    """
    Load any supported file into an Hyperspy structure
    Supported formats: netCDF, msa, Gatan dm3, Ripple (rpl+raw)
    FEI ser and emi and hdf5.

    Parameters
    ----------

    filename : string
        File name (including the extension)
    record_by : {None, 'spectrum', 'image'}
        If None (default) it will try to guess the data type from the file,
        if 'spectrum' the file will be loaded as an Spectrum object
        If 'image' the file will be loaded as an Image object
    output_level : int
        If 0, do not output file loading text.
        If 1, output simple file summary (data type and shape)
        If 2, output more diagnostic output (e.g. number of tags for DM3 files)
    """
    extension = os.path.splitext(filename)[1][1:]

    i = 0
    while extension.lower() not in io_plugins[i].file_extensions and \
        i < len(io_plugins) - 1:
        i += 1
    if i == len(io_plugins):
        # Try to load it with the python imaging library
        reader = image
        try:
            return load_with_reader(filename, reader, record_by, 
                signal_type=signal_type, **kwds)
        except:
            messages.warning_exit('File type not supported')
    else:
        reader = io_plugins[i]
        return load_with_reader(filename, reader, record_by,
                    signal_type=signal_type,
                    output_level=output_level, **kwds)


def load_with_reader(filename, reader, record_by=None,
        signal_type=None, output_level=1, **kwds):
    from hyperspy.signals.image import Image
    from hyperspy.signals.spectrum import Spectrum
    from hyperspy.signals.eels import EELSSpectrum
    if output_level>1:
        messages.information('Loading %s ...' % filename)
    
    file_data_list = reader.file_reader(filename,
                                        record_by=record_by,
                                        output_level=output_level,
                                        **kwds)
    objects = []
    for file_data_dict in file_data_list:
        if record_by is not None:
            file_data_dict['mapped_parameters']['record_by'] = record_by
        # The record_by can still be None if it was not defined by the reader
        if file_data_dict['mapped_parameters']['record_by'] is None:
            print "No data type provided.  Defaulting to image."
            file_data_dict['mapped_parameters']['record_by']= 'image'

        if signal_type is not None:
            file_data_dict['mapped_parameters']['signal_type'] = signal_type

        if file_data_dict['mapped_parameters']['record_by'] == 'image':
            s = Image(file_data_dict)
        else:
            if ('signal_type' in file_data_dict['mapped_parameters'] 
                and file_data_dict['mapped_parameters']['signal_type'] 
                == 'EELS'):
                s = EELSSpectrum(file_data_dict)
            else:
                s = Spectrum(file_data_dict)
        folder, filename = os.path.split(os.path.abspath(filename))
        filename, extension = os.path.splitext(filename)
        s.tmp_parameters.folder = folder
        s.tmp_parameters.filename = filename
        s.tmp_parameters.extension = extension.replace('.','')
        objects.append(s)
        s.print_summary()

    if len(objects) == 1:
        objects = objects[0]
    if output_level>1:
        messages.information('%s correctly loaded' % filename)
    return objects

def save(filename, signal, overwrite=None, **kwds):
    extension = os.path.splitext(filename)[1][1:]
    if extension == '':
        extension = \
            hyperspy.defaults_parser.preferences.General.default_file_format
        filename = filename + '.' + \
            hyperspy.defaults_parser.preferences.General.default_file_format
    writer = None
    for plugin in io_plugins:
        if extension.lower() in plugin.file_extensions:
            writer = plugin
            break

    if writer is None:
        raise ValueError('.%s does not correspond ' % extension + 
        'of any supported format. Supported file extensions are: %s ' % 
                    strlist2enumeration(default_write_ext))
    else:
        # Check if the writer can write
        sd = signal.axes_manager.signal_dimension
        nd = signal.axes_manager.navigation_dimension
        if writer.writes is False:
                raise ValueError('Writing to this format is not '
                'supported, supported file extensions are: %s ' % 
                    strlist2enumeration(default_write_ext))
        if writer.writes is not True and (sd, nd) not in writer.writes:
            yes_we_can = [plugin.format_name for plugin in io_plugins 
                if plugin.writes is True or
                plugin.writes is not False and 
                (sd, nd) in plugin.writes]
            raise ValueError('This file format cannot write this data. '
            'The following formats can: %s' % 
            strlist2enumeration(yes_we_can))
        ensure_directory(filename)
        if overwrite is None:
            overwrite = hyperspy.misc.utils_varia.overwrite(filename)
        if overwrite is True:
            writer.file_writer(filename, signal, **kwds)
            print('The %s file was created' % filename)
            folder, filename = os.path.split(os.path.abspath(filename))
            signal.tmp_parameters.set_item('folder', folder)
            signal.tmp_parameters.set_item('filename', 
                                           os.path.splitext(filename)[0])
            signal.tmp_parameters.set_item('extension', extension)
