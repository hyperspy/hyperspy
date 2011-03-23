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

import messages
from defaults_parser import defaults
from io import netcdf, msa, dm3_data_plugin, fei, bin, mrc, pil, ripple, hdf5

io_plugins = (netcdf, msa, dm3_data_plugin, fei, bin, mrc, pil, ripple,
              hdf5)

def load(filename, data_type = None, **kwds):
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
    extension = os.path.splitext(filename)[1][1:]
    
    i = 0
    while extension not in io_plugins[i].file_extensions and \
        i < len(io_plugins) - 1: i += 1
    if i == len(io_plugins):
        # Try to load it with the python imaging library
        reader = pil
        try:
            return load_with_reader(filename, reader, data_type, **kwds)
        except:
            messages.warning_exit('File type not supported')
    else:
        reader = io_plugins[i]
        return load_with_reader(filename, reader, data_type, **kwds)
        
def load_with_reader(filename, reader, data_type = None, **kwds):
    from spectrum import Spectrum
    from signals.image import Image
    from signal import Signal
    messages.information(reader.description)    
    file_data_dict = reader.file_reader(filename,
                                         data_type=data_type,
                                         **kwds)
    objects = []
    data_type = file_data_dict['data_type']
    if data_type == 'SI':
        s = Spectrum(file_data_dict)
    elif data_type == 'Image':
        s = Image(file_data_dict)  
    elif data_type == 'Signal':
        s = Signal(file_data_dict)
    if defaults.plot_on_load is True:
        s.plot()
    objects.append(s)
        
    if len(objects) == 1:
        objects = objects[0]
    return objects
    
def save(filename, object2save, format = 'hdf5', **kwds):
    from spectrum import Spectrum
    from image import Image
    from signal import Signal
    
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
        if isinstance(object2save, Spectrum):
            if object2save.is_spectrum_image() is True and \
                writer.writes_spectrum_image is False:
                messages.warning_exit('SIs writing is not currently supported '
                'in the %s format' % writer.format_name) 
            elif object2save.is_spectrum_line() is True and \
                writer.writes_spectrum_image is False:
                messages.warning_exit('Spectrum line writing is not currently '
                'supported in the %s format' % writer.format_name) 
            elif object2save.is_single_spectrum() is True and \
                writer.writes_spectrum is False:
                messages.warning_exit('Spectrum writing is not currently '
                'supported in the %s format' % writer.format_name) 
            else:
                writer.file_writer(filename, object2save, **kwds)
        elif isinstance(object2save, Image):
            if writer.writes_images is False:
                messages.warning_exit('Image writing is not currently supported'
                ' in the %s format' % writer.format_name)
            else:
                writer.file_writer(filename, object2save, **kwds)
        elif isinstance(object2save, Signal):
            writer.file_writer(filename, object2save, **kwds)
    

## if file_extension in msa_extensions:
##     spectrum_dict, acquisition_dict = io.msa_reader(filename)
##     for key in spectrum_dict:
##         exec('self.%s = spectrum_dict[\'%s\']' % (key, key))
##     for key in acquisition_dict:
##         exec('self.acquisition_parameters.%s = acquisition_dict[\'%s\']' \
##         % (key, key))

## elif file_extension in dm3_extensions : 
##     spectrum_dict, acquisition_dict = io.dm3_reader(filename)
##     for key in spectrum_dict:
##         exec('self.%s = spectrum_dict[\'%s\']' % (key, key))
##     for key in acquisition_dict:
##         exec('self.acquisition_parameters.%s = acquisition_dict[\'%s\']' \
##         % (key, key))
##     # Swap the x and y axes if it is a vertical line scan.
##     self.get_dimensions_from_cube()
##     if self.xdimension == 1 and self.ydimension > 1:
##         self.swap_x_y()
##         print "Shape: ", self.data_cube.shape
##     
## elif file_extension in netcdf_extensions:
##     spectrum_dict, acquisition_dict, treatments_dict = \
##     io.netcdf_reader(filename)
##     for key in spectrum_dict:
##         exec('self.%s = spectrum_dict[\'%s\']' % (key, key))
##     for key in acquisition_dict:
##         exec('self.acquisition_parameters.%s = acquisition_dict[\'%s\']' \
##         % (key, key))
##         for key in treatments_dict:
##         exec('self.treatments.%s = treatments_dict[\'%s\']' \
##         % (key, key))
##     self.get_dimensions_from_cube()
##     print "Shape: ", self.data_cube.shape
##     print "History:"
##     for treatment in self.history:
##         print treatment

## elif file_extension in eelslab_extensions:
##     attributes = {}
##     float_keys = ['energyorigin', 'energyscale', 'xorigin', 'xscale', 
##     'yorigin', 'yscale']
##     int_keys = ['energydimension', 'xdimension', 'ydimension']
##     str_keys = ['yunits', 'xunits', 'energyunits']
##     ifile = open(filename,'r')
##     for line in ifile:
##         if line.split() != []:
##             print line.split()
##             key, value = line.split()
## 	    if key in float_keys:
## 		exec('self.%s = %f' % (key, float(value)))
## 	    elif key in int_keys:
## 		exec('self.%s = %i' % (key, int(value)))
## 	    elif key in str_keys:
## 		exec('self.%s = \'%s\'' % (key, str(value)))
##     self.data_cube = np.zeros((self.energydimension, self.xdimension, 
##     self.ydimension))
##     self.updateenergy_axis()
## elif file_extension in numpy_extensions:
##     self.data_cube = np.load(filename)
##     if len(self.data_cube.shape) == 1:
## 	self.data_cube = \
## 	self.data_cube.reshape(self.data_cube.shape[0], 1, 1)
##     elif len(self.data_cube.shape) == 2:
## 	self.data_cube = \
## 	self.data_cube.reshape(self.data_cube.shape[0], 
## 	self.data_cube.shape[1], 1)
##     elif len(self.data_cube.shape) > 3:
## 	messages.warning_exit(
## 	"Currently this format doesn't support dim > 3")
##     self.energydimension, self.xdimension, self.ydimension = \
##     self.data_cube.shape
##     self.xorigin = 0
##     self.xscale = 1
##     self.xunits = ""
##     self.yorigin = 0
##     self.yscale = 1
##     self.yunits = ""
##     self.energyorigin = 0
##     self.energyscale = 1
##     self.energyunits = "eV"
##     self.energy_axis = generate_axis(self.energyorigin,self.energyscale, 
##     self.energydimension)
##     
## else :
##     print "Unknown file format."
##     print "We read only msa, dm3 and NetCDF"
##     return 0
## if not hasattr(self, 'type'):
##     self.type = 'experiment'

## def save(filename, object2save):
## 	pass
