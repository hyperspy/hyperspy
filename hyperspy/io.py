# -*- coding: utf-8 -*-
# Copyright © 2007 Francisco Javier de la Peña
#
# This file is part of Hyperspy.
#
# Hyperspy is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Hyperspy; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  
# USA

import os

from hyperspy import messages
from hyperspy.defaults_parser import defaults
from hyperspy.io_plugins import (netcdf, msa, dm3_data_plugin, fei, mrc, image, 
ripple, hdf5)

io_plugins = (netcdf, msa, dm3_data_plugin, fei, mrc, image, ripple, hdf5)
def load(*filenames, **kwds):
    """
    Load potentially multiple supported file into an EELSLab structure
    Supported formats: netCDF, msa, Gatan dm3, Ripple (rpl+raw)
    FEI ser and emi and hdf5.
        
    *filenames : if multiple file names are passed in, they get aggregated to a Signal class
        that has members for each file, plus a data set that consists of stacked input files.
        That stack has one dimension more than the input files.
        All files must match in size, number of dimensions, and type/extension.

    *kwds : any specified parameters.  Currently, the only interesting one here is
        data_type, to manually force the outcome Signal to a particular type.

    Example usage:
        Loading a single file:
            d=load('file.dm3')
        Loading a single file and overriding its default data_type:
            d=load('file.dm3',data_type='Image')
        Loading multiple files:
            d=load('file1.dm3','file2.dm3')

    """
    if len(filenames)<1:
        messages.warning_exit('No file provided to reader.')
        return None
    elif len(filenames)==1:
        return load_single_file(filenames[0],**kwds)
    else:
        import eelslab.signals.aggregate as agg
        objects=[load_single_file(filename,**kwds) for filename in filenames]

        obj_type=objects[0].__class__.__name__
        if obj_type=='Image':
            if len(objects[0].data.shape)==3:
                # feeding 3d objects creates cell stacks
                agg_sig=agg.AggregateCells(*objects)
            else:
                agg_sig=agg.AggregateImage(*objects)
        elif obj_type=='Spectrum':
            agg_sig=agg.AggregateSpectrum(*objects)
        else:
            agg_sig=agg.Aggregate(*objects)
        return agg_sig            
        
def load_single_file(filename, record_by=None, **kwds):
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
    """
    extension = os.path.splitext(filename)[1][1:]

    i = 0
    while extension not in io_plugins[i].file_extensions and \
        i < len(io_plugins) - 1: i += 1
    if i == len(io_plugins):
        # Try to load it with the python imaging library
        reader = image
        try:
            return load_with_reader(filename, reader, record_by, **kwds)
        except:
            messages.warning_exit('File type not supported')
    else:
        reader = io_plugins[i]
        return load_with_reader(filename, reader, record_by, **kwds)
        
def load_with_reader(filename, reader, record_by = None, signal = None, **kwds):
    from hyperspy.signals.image import Image
    from hyperspy.signals.spectrum import Spectrum
    from hyperspy.signals.eels import EELSSpectrum
    messages.information(reader.description)    
    file_data_list = reader.file_reader(filename,
                                         record_by=record_by,
                                        **kwds)
    objects = []
    for file_data_dict in file_data_list:
        if record_by is not None:
            file_data_dict['mapped_parameters']['record_by'] = record_by
        # The record_by can still be None if it was not defined by the reader
        if file_data_dict['mapped_parameters']['record_by'] is None:
            print "No data type provided.  Defaulting to image."
            file_data_dict['mapped_parameters']['record_by']  = 'image'
            
        if signal is not None:
            file_data_dict['mapped_parameters']['signal'] = signal

        if file_data_dict['mapped_parameters']['record_by'] == 'image':
            s = Image(file_data_dict)  
        else:
            if file_data_dict['mapped_parameters']['signal'] == 'EELS':
                s = EELSSpectrum(file_data_dict)
            else:
                s = Spectrum(file_data_dict)
        if defaults.plot_on_load is True:
            s.plot()
        objects.append(s)
        
    if len(objects) == 1:
        objects = objects[0]
    return objects
    
def save(filename, signal, format = 'hdf5', **kwds):
    extension = os.path.splitext(filename)[1][1:]
    i = 0
    if extension == '':
        extension = format
        filename = filename + '.' + format
    while extension not in io_plugins[i].file_extensions and \
        i < len(io_plugins) - 1: i += 1
    if i == len(io_plugins):
        messages.warning_exit('File type not supported')
    else:
        writer = io_plugins[i]
        # Check if the writer can write 
        writer.file_writer(filename, signal, **kwds)
