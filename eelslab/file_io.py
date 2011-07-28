# -*- coding: utf-8 -*-
# Copyright © 2007 Francisco Javier de la Peña
#
# This file is part of EELSLab.
#
# EELSLab is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# EELSLab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EELSLab; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  
# USA

import os

from eelslab import messages
from eelslab.defaults_parser import defaults
from eelslab.io import netcdf, msa, dm3_data_plugin, fei, bin, mrc, image, ripple, hdf5

io_plugins = (netcdf, msa, dm3_data_plugin, fei, bin, mrc, image, ripple,
              hdf5)

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
        from eelslab.signals.aggregate import AggregateImage, AggregateCells
        objects=[load_single_file(filename,**kwds) for filename in filenames]

        obj_type=objects[0].__class__.__name__
        if obj_type=='Image':
            if len(objects[0].data.shape)==3:
                # feeding 3d objects creates cell stacks
                agg_sig=AggregateCells(*objects)
            else:
                agg_sig=AggregateImage(*objects)
        elif obj_type=='Spectrum':
            agg_sig=Spectrum()
        else:
            agg_sig=Signal()
        return agg_sig            
        
def load_single_file(filename, **kwds):
    """
    Load any supported file into an EELSLab structure
    Supported formats: netCDF, msa, Gatan dm3, Ripple (rpl+raw)
    FEI ser and emi and hdf5.

    Parameters
    ----------

    filename : string
        File name (including the extension)
    data_type : {None, 'SI', 'Image'}
        If None (default) it will try to guess the data type from the file,
        if 'SI' the file will be loaded as an Spectrum object
        If 'Image' the file will be loaded as an Image object
    """
    if not 'data_type' in kwds.keys():
        kwds['data_type']=None
    extension = os.path.splitext(filename)[1][1:]

    i = 0
    while extension not in io_plugins[i].file_extensions and \
        i < len(io_plugins) - 1: i += 1
    if i == len(io_plugins):
        # Try to load it with the python imaging library
        reader = image
        try:
            return load_with_reader(filename, reader, **kwds)
        except:
            messages.warning_exit('File type not supported')
    else:
        reader = io_plugins[i]
        return load_with_reader(filename, reader, **kwds)
        
def load_with_reader(filename, reader, data_type = None, **kwds):
    from eelslab.signals.image import Image
    from eelslab.signals.spectrum import Spectrum
    from eelslab.signal import Signal
    messages.information(reader.description)    
    file_data_list = reader.file_reader(filename,
                                         data_type=data_type,
                                        **kwds)
    objects = []
    for file_data_dict in file_data_list:
        if data_type is None:
            data_type = file_data_dict['mapped_parameters']['data_type']

        # The data_type can still be None if it was not defined by the reader
        if data_type is None:
                print "No data type provided.  Defaulting to Signal."
                data_type = 'SI'
                
        # We write the data type to the mapped_parameters to guarantee that it 
        # is coherent with the asigned class. Note that the original_parameters 
        # dict is not modified
        file_data_dict['mapped_parameters']['data_type'] = data_type
        
        if data_type == 'Image':
            s = Image(file_data_dict)  
        elif data_type is 'Spectrum':
            s=Spectrum(file_data_dict)
        else:
            print "defaulting to Signal"
            s = Signal(file_data_dict)
        if defaults.plot_on_load is True:
            s.plot()
        objects.append(s)
        
    if len(objects) == 1:
        objects = objects[0]
    return objects
    
def save(filename, signal, format = 'hdf5', only_view = False, **kwds):
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
