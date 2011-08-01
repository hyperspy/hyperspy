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


import numpy as np

no_netcdf = False
from eelslab import Release
from eelslab import messages
from eelslab.microscope import microscope
no_netcdf_message = 'Warning! In order to enjoy the netCDF Read/Write feature, '
'at least one of this packages must be installed: '
'python-pupynere, python-netcdf or python-netcdf4'
try:
    from netCDF4 import Dataset
    which_netcdf = 'netCDF4'
except:
    try:
        from netCDF3 import Dataset
        which_netcdf = 'netCDF3'
    except:
        try:
            from Scientific.IO.NetCDF import NetCDFFile as Dataset
            which_netcdf = 'Scientific Python'
        except:
            messages.warning(no_netcdf_message)
    
# Plugin characteristics
# ----------------------
format_name = 'netCDF'
description = ''
full_suport = True
file_extensions = ('nc', 'NC')
default_extension = 0

# Reading features
reads_images = True
reads_spectrum = True
reads_spectrum_image = True
# Writing features
writes_images = True
writes_spectrum = True
writes_spectrum_image = True
# ----------------------


attrib2netcdf = \
    {
    'energyorigin' : 'energy_origin',
    'energyscale' : 'energy_scale',
    'energyunits' : 'energy_units',
    'xorigin' : 'x_origin',
    'xscale' : 'x_scale',
    'xunits' : 'x_units',
    'yorigin' : 'y_origin',
    'yscale' : 'y_scale',
    'yunits' : 'y_units',
    'zorigin' : 'z_origin',
    'zscale' : 'z_scale',
    'zunits' : 'z_units',
    'exposure' : 'exposure',
    'title' : 'title',
    'binning' : 'binning',
    'readout_frequency' : 'readout_frequency',
    'ccd_height' : 'ccd_height',
    'blanking' : 'blanking'
    }
    
acquisition2netcdf = \
    {
    'exposure' : 'exposure',
    'binning' : 'binning',
    'readout_frequency' : 'readout_frequency',
    'ccd_height' : 'ccd_height',
    'blanking' : 'blanking',
    'gain' : 'gain', 
    'pppc' : 'pppc',
    }
    
treatments2netcdf = \
    {
    'dark_current' : 'dark_current', 
    'readout' : 'readout', 
    }
    
def file_reader(filename, *args, **kwds):
    if no_netcdf is True:
        messages.warning_exit(no_netcdf_message)
    
    ncfile = Dataset(filename,'r')
    
    if hasattr(ncfile, 'file_format_version'):
        if ncfile.file_format_version == 'EELSLab 0.1':
            dictionary = nc_eelslab_reader_0dot1(ncfile, *args, **kwds)
    else:
        ncfile.close()
        messages.warning_exit('Unsupported netCDF file')
        
    return (dictionary,)
        
def nc_eelslab_reader_0dot1(ncfile, *args, **kwds):
    calibration_dict, acquisition_dict , treatments_dict= {}, {}, {}
    dc = ncfile.variables['data_cube']
    calibration_dict['data_cube'] = dc[:]
    if 'history' in calibration_dict:
        calibration_dict['history'] = eval(ncfile.history)
    for attrib in attrib2netcdf.items():
        if hasattr(dc, attrib[1]):
            value = eval('dc.' + attrib[1])
            if type(value) is np.ndarray:
                calibration_dict[attrib[0]] = value[0]
            else:
                calibration_dict[attrib[0]] = value
        else:
            print "Warning: the \'%s\' attribute is not defined in the file\
            " % attrib[0]
    for attrib in acquisition2netcdf.items():
            if hasattr(dc, attrib[1]):
                value = eval('dc.' + attrib[1])
                if type(value) is np.ndarray:
                    acquisition_dict[attrib[0]] = value[0]
                else:
                    acquisition_dict[attrib[0]] = value
            else:
                print \
                "Warning: the \'%s\' attribute is not defined in the file\
            " % attrib[0]
    for attrib in treatments2netcdf.items():
            if hasattr(dc, attrib[1]):
                treatments_dict[attrib[0]] = eval('dc.' + attrib[1])
            else:
                print \
                "Warning: the \'%s\' attribute is not defined in the file\
            " % attrib[0]
    print "EELSLab NetCDF file correctly loaded" 
    dictionary = {'data_type' : ncfile.type, 'calibration' : calibration_dict, 
    'acquisition' : acquisition_dict, 'treatments' : treatments_dict}
    ncfile.close()
    return dictionary
    
def file_writer(filename, object2save, *args, **kwds):
    from eelslab import spectrum
    from eelslab import image
    if isinstance(object2save, spectrum.Spectrum):
        netcdf_spectrum_writer(filename, object2save, *args, **kwds)
    elif isinstance(object2save, image.Image):
        netcdf_image_writer(filename, object2save, *args, **kwds)
    else:
        messages.warning_exit('The object cannot be saved in the NetCDF format')

def netcdf_image_writer(filename, image, *args, **kwds):
    if no_netcdf:
        messages.warning_exit(no_netcdf_message)
    elif len(image.data_cube.squeeze().shape) == 3:
        x_dimension, y_dimension, z_dimension = image.data_cube.shape
        d_dtype = image.data_cube.dtype.char
        print "Saving the image in EELSLab netCDF format"
        ncfile = Dataset(filename,'w')
        setattr(ncfile, 'file_format_version', 'EELSLab 0.1')
        setattr(ncfile, 'eelslab_version', Release.version)
        setattr(ncfile, 'Conventions', 'http://www.eelslab.org')
#        setattr(ncfile, 'title', image.title)
        setattr(ncfile, 'type', 'Image')
        # Create the dimensions
        ncfile.createDimension('x', x_dimension)
        ncfile.createDimension('y', y_dimension)
        ncfile.createDimension('z', z_dimension)
        data_cube = ncfile.createVariable('data_cube', d_dtype, ('x', 'y', 'z'))
        
        for attrib in attrib2netcdf.items():
            if hasattr(image, attrib[0]):
                print "%s = %s" % (attrib[1], eval('image.' + attrib[0]))
                setattr(data_cube, attrib[1], eval('image.' + attrib[0]))
            else:
                print "Warning: the \'%s\' attribute is not defined" \
                % attrib[0]
        # write data to variable.
        data_cube[:] = image.data_cube
        # close the file.
        ncfile.close()
        print 'File saved'
    elif len(image.data_cube.squeeze().shape) == 2:
        dc = image.data_cube.squeeze()
        x_dimension, y_dimension = dc.shape
        d_dtype = dc.dtype.char
        print "Saving the image in EELSLab netCDF format"
        ncfile = Dataset(filename, 'w')
        setattr(ncfile, 'file_format_version', 'EELSLab 0.1')
        setattr(ncfile, 'Conventions', 'http://www.eelslab.org')
        setattr(ncfile, 'title', image.title)
        setattr(ncfile, 'type', 'Image')
        # Create the dimensions
        ncfile.createDimension('x', x_dimension)
        ncfile.createDimension('y', y_dimension)
        data_cube = ncfile.createVariable('data_cube', d_dtype, ('x', 'y'))
        for attrib in attrib2netcdf.items():
            if hasattr(image, attrib[0]):
                print "%s = %s" % (attrib[1], eval('image.' + attrib[0]))
                setattr(data_cube, attrib[1], eval('image.' + attrib[0]))
            else:
                print "Warning: the \'%s\' attribute is not defined" \
                % attrib[0]
        # write data to variable.
        data_cube[:] = image.data_cube
        # close the file.
        ncfile.close()
        print 'File saved'
    else:
        print "The images of dimension 2 or 3 are supported"
        print "This image has dimension ", len(image.data_cube.squeeze().shape)


def netcdf_spectrum_writer(filename, spectrum, 
    write_microscope_parameters = True, *args, **kwds):
    if no_netcdf is True:
        messages.warning_exit(no_netcdf_message)
    else:
        energy_dimension, x_dimension, y_dimension = spectrum.data_cube.shape
        d_dtype = spectrum.data_cube.dtype.char
        print "Writing to file in EELSLab netCDF format"
        ncfile = Dataset(filename, 'w')
        setattr(ncfile, 'file_format_version', 'EELSLab 0.1')
        setattr(ncfile, 'Conventions', 'http://www.eelslab.org')
#        setattr(ncfile, 'history', str(spectrum.history))
        setattr(ncfile, 'title', spectrum.title)
        setattr(ncfile, 'type', 'SI')
        # Create the dimensions
        ncfile.createDimension('energy', energy_dimension)
        ncfile.createDimension('x', x_dimension)
        ncfile.createDimension('y', y_dimension)
        data_cube = ncfile.createVariable('data_cube', d_dtype, ('energy', 
        'x', 'y'))
        
        for attrib in attrib2netcdf.items():
            if hasattr(spectrum, attrib[0]):
                print "%s = %s" % (attrib[1], eval('spectrum.' + attrib[0]))
                setattr(data_cube, attrib[1], eval('spectrum.' + attrib[0]))
            else:
                print "Warning: the \'%s\' attribute is not defined" \
                % attrib[0]
                
        for attrib in acquisition2netcdf.items():
            if hasattr(spectrum.acquisition_parameters, attrib[0]):
                value = \
                spectrum.acquisition_parameters.__getattribute__(attrib[0])
                print "%s = %s" % (attrib[1], 
                value)
                if type(value) is bool:
                    value = int(value)
                if value is not None:
                    setattr(data_cube, attrib[1], value)
                else:
                    print(attrib[1] + ' undefined')
                    
        for attrib in treatments2netcdf.items():
            if hasattr(spectrum.treatments, attrib[0]):
                value = eval('spectrum.treatments.' + attrib[0])
                print "%s = %s" % (attrib[1], 
                value)
                if value is not None:
                    setattr(data_cube, attrib[1], value)
            else:
                print "Warning: the \'%s\' attribute is not defined" \
                % attrib[0]
        if write_microscope_parameters is True:
            print "\nWarning: the microscope attributes will be written to the \
            file"
            print microscope
            print
            setattr(data_cube, 'convergence_angle', microscope.alpha)
            setattr(data_cube, 'collection_angle', microscope.beta)
            setattr(data_cube, 'beam_energy', microscope.E0)
            setattr(data_cube, 'collection_angle', microscope.beta)
            setattr(data_cube, 'pppc', microscope.pppc)
            setattr(data_cube, 'correlation_factor', 
            microscope.correlation_factor)
        # write data to variable.
        data_cube[:] = spectrum.data_cube
        # close the file.
        ncfile.close()
        print 'File saved'
