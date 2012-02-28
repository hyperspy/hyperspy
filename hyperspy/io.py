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

from hyperspy import messages
import hyperspy.defaults_parser
from hyperspy.io_plugins import msa, digital_micrograph, fei, mrc, ripple
from hyperspy.gui.tools import Load
from hyperspy.misc.utils import ensure_directory


io_plugins = [msa, digital_micrograph, fei, mrc, ripple]

#try:
#    from hyperspy.io_plugins import fits
#    io_plugins.append(fits)
#except ImportError:
#    messages.information('The FITS IO features are not available')
try:
    from hyperspy.io_plugins import netcdf
    io_plugins.append(netcdf)
except ImportError:
    messages.information('The NetCDF IO features are not available')
    
try:
    from hyperspy.io_plugins import hdf5
    io_plugins.append(hdf5)
except ImportError:
    messages.information('The HDF5 IO features are not available')
    
try:
    from hyperspy.io_plugins import image
    io_plugins.append(image)
except ImportError:
    messages.information('The Image (PIL) IO features are not available')

write_1d_exts, write_2d_exts, write_3d_exts, write_xd_exts = [], [], [], []
default_write_ext = set()
for plugin in io_plugins:

    if plugin.writes_1d is True:
        write_1d_exts.extend(plugin.file_extensions)
        default_write_ext.add(plugin.file_extensions[plugin.default_extension])
    if plugin.writes_2d is True:
        write_2d_exts.extend(plugin.file_extensions)
        default_write_ext.add(plugin.file_extensions[plugin.default_extension])
    if plugin.writes_3d is True:
        write_3d_exts.extend(plugin.file_extensions)
        default_write_ext.add(plugin.file_extensions[plugin.default_extension])
    if plugin.writes_xd is True:
        write_xd_exts.extend(plugin.file_extensions)
        default_write_ext.add(plugin.file_extensions[plugin.default_extension])

def load(*filenames, **kwds):
    """
    Load potentially multiple supported file into an hyperspy structure
    Supported formats: netCDF, msa, Gatan dm3, Ripple (rpl+raw)
    FEI ser and emi and hdf5.
    
    If no parameter is passed and the interactive mode is enabled the a window 
    ui is raised.
    
    Parameters
    ----------
    *filenames : if multiple file names are passed in, they get aggregated to
    a Signal class that has members for each file, plus a data set that
    consists of stacked input files. That stack has one dimension more than
    the input files. All files must match in size, number of dimensions, and
    type/extension.

    record_by : Str 
        Manually set the way in which the data will be read. Possible values are
        'spectrum' or 'image'. Please note that most of the times it is better 
        to leave Hyperspy to decide this.
        
    signal_type : Str
        Manually set the signal type of the data. Although only setting signal 
        type to 'EELS' will currently change the way the data is loaded, it is 
        good practice to set this parameter so it can be stored when saving the 
        file. Please note that, if the signal_type is already defined in the 
        file the information will be overriden without warning.

    Example usage:
        Loading a single file providing the signal type:
            d=load('file.dm3', signal_type = 'XPS')
        Loading a single file and overriding its default record_by:
            d=load('file.dm3', record_by='Image')
        Loading multiple files:
            d=load('file1.dm3','file2.dm3')

    """

    if len(filenames)<1 and hyperspy.defaults_parser.preferences.General.interactive is True:
            load_ui = Load()
            load_ui.edit_traits()
            if load_ui.filename:
                filenames = (load_ui.filename,)
    if len(filenames)<1:
        messages.warning('No file provided to reader.')
        return None
    elif len(filenames)==1:
        if '*' in filenames[0]:
            from glob import glob
            filenames=sorted(glob(filenames[0]))
        else:
            f=load_single_file(filenames[0], **kwds)
            return f
    import hyperspy.signals.aggregate as agg
    objects=[load_single_file(filename, output_level=0, is_agg = True, **kwds) 
        for filename in filenames]

    obj_type=objects[0].mapped_parameters.record_by
    if obj_type=='image':
        if len(objects[0].data.shape)==3:
            # feeding 3d objects creates cell stacks
            agg_sig=agg.AggregateCells(*objects)
        else:
            agg_sig=agg.AggregateImage(*objects)
    elif 'spectrum' in obj_type:
        agg_sig=agg.AggregateSpectrum(*objects)
    else:
        agg_sig=agg.Aggregate(*objects)
    if hyperspy.defaults_parser.preferences.General.plot_on_load is True:
        agg_sig.plot()
    return agg_sig


def load_single_file(filename, record_by=None, output_level=1, is_agg = False, 
    **kwds):
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
    while extension not in io_plugins[i].file_extensions and \
        i < len(io_plugins) - 1:
        i += 1
    if i == len(io_plugins):
        # Try to load it with the python imaging library
        reader = image
        try:
            return load_with_reader(filename, reader, record_by, **kwds)
        except:
            messages.warning_exit('File type not supported')
    else:
        reader = io_plugins[i]
        return load_with_reader(filename, reader, record_by, 
                                output_level=output_level, is_agg = is_agg,
                                **kwds)


def load_with_reader(filename, reader, record_by = None, signal_type = None,
                     output_level=1, is_agg = False, **kwds):
    from hyperspy.signals.image import Image
    from hyperspy.signals.spectrum import Spectrum
    from hyperspy.signals.eels import EELSSpectrum
    if output_level>1:
        messages.information(reader.description)
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
            if 'signal_type' in file_data_dict['mapped_parameters'] and \
                file_data_dict['mapped_parameters']['signal_type'] == 'EELS':
                s = EELSSpectrum(file_data_dict)
            else:
                s = Spectrum(file_data_dict)
        objects.append(s)
        print s
        if hyperspy.defaults_parser.preferences.General.plot_on_load is True \
            and is_agg is False:
            s.plot()
    if len(objects) == 1:
        objects = objects[0]
    return objects


def save(filename, signal, **kwds):
    extension = os.path.splitext(filename)[1][1:]
    i = 0
    if extension == '':
        extension = \
            hyperspy.defaults_parser.preferences.General.default_file_format
        filename = filename + '.' + \
            hyperspy.defaults_parser.preferences.General.default_file_format
    while extension not in io_plugins[i].file_extensions and \
        i < len(io_plugins) - 1:
        i += 1
    if i == len(io_plugins):
        messages.warning_exit('File type not supported')
    else:
        writer = io_plugins[i]
        # Check if the writer can write
        ensure_directory(filename)
        writer.file_writer(filename, signal, **kwds)
        print('The %s file was created' % filename)
