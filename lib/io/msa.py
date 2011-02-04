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

import locale
import time
import datetime

import numpy as np

from ..config_dir import os_name
from ..utils import generate_axis
from ..microscope import microscope
from .. import messages

# Plugin characteristics
# ----------------------
format_name = 'MSA'
description = ''
full_suport = False
file_extensions = ('msa', 'ems', 'mas', 'emsa', 'EMS', 'MAS', 'EMSA', 'MSA')
default_extension = 0

# Reading features
reads_images = False
reads_spectrum = True
reads_spectrum_image = False
# Writing features
writes_images = False
writes_spectrum = True
writes_spectrum_image = False
# ----------------------


def file_reader(filename, **kwds):
    calibration_dict = {}
    acquisition_dict = {}
    spectrum_file = open(filename)
    parameters = {}
    y = []
    for line in spectrum_file:
        if line[0] == "#":
            parameters[line[1:line.find(' ')]] = \
            line[line.find(' : ')+3:].strip()
        else :
            if parameters['DATATYPE'] == 'XY' :
                pair = line.strip().split(', ')
                if len(pair) != 2:
                    pair = line.strip().split('\t')
                y.append(float(pair[1]))
            elif parameters['DATATYPE'] == 'Y' :
                y.append(float(line))
                
    loc = locale.getlocale(locale.LC_TIME)
    
    if os_name == 'posix':
        locale.setlocale(locale.LC_TIME, ('en_US', 'UTF8'))
    elif os_name == 'windows':
        locale.setlocale(locale.LC_TIME, 'english')
    try:
        H, M = time.strptime(parameters['TIME'], "%H:%M")[3:5]
    except:
        H, M = time.strptime('00:00', "%H:%M")[3:5]
    calibration_dict['time'] = datetime.time(H, M)
    try:
        Y, M, D = time.strptime(parameters['DATE'], "%d-%b-%Y")[0:3]
    except:
        messages.warning('The date information could not be properly read'
        'The Ernst Ruska birthday is used instead')
        # Default to Ernst Ruska birth day
        Y, M, D = time.strptime('25-Dec-1906', "%d-%b-%Y")[0:3]
    calibration_dict['date'] = datetime.date(Y, M, D)
    locale.setlocale(locale.LC_TIME, loc) # restore saved locale

    calibration_dict['title'] = parameters['TITLE']
    calibration_dict['owner'] = parameters['OWNER']
    
    calibration_dict['xorigin'] = 0
    calibration_dict['xscale'] = 1
    calibration_dict['xdimension'] = 1
    calibration_dict['xunits'] = ""
    
    calibration_dict['yorigin'] = 0
    calibration_dict['yscale'] = 1
    calibration_dict['ydimension'] = 1
    calibration_dict['yunits'] = ""
    print_microscope = False
    if parameters.has_key('INTEGTIME'):
        acquisition_dict['exposure'] = float(parameters['INTEGTIME'])
        
    if parameters.has_key('CONVANGLE'):
       microscope.alpha = float(parameters['CONVANGLE'])
       print_microscope = True
    if parameters.has_key('COLLANGLE'):
       microscope.beta = float(parameters['COLLANGLE'])
       print_microscope = True
    if parameters.has_key('BEAMKV'):
       microscope.E0 = float(parameters['BEAMKV'])
       print_microscope = True
    if parameters.has_key('PPPC'):
       microscope.pppc = float(parameters['PPPC'])
       print_microscope = True
    if parameters.has_key('CORRFAC'):
       microscope.correlation_factor = float(parameters['CORRFAC'])
       print_microscope = True
    if print_microscope:
        print "\nWarning: Reading the microscope parameters from the file"
        print microscope
    calibration_dict['energyorigin'] = float(parameters['OFFSET'])
    calibration_dict['energyscale'] = float(parameters['XPERCHAN'])
    calibration_dict['energydimension'] = int(parameters['NPOINTS'])
    calibration_dict['energyunits'] =  parameters['XUNITS']
    calibration_dict['energy_axis'] = generate_axis(calibration_dict['energyorigin'],calibration_dict['energyscale'], 
    calibration_dict['energydimension'])
    calibration_dict['data_cube'] = np.zeros((calibration_dict['energydimension'],1,1))
    calibration_dict['data_cube'][:,0,0] = np.array(y)
    spectrum_file.close()
    dictionary = {'data_type' : 'SI', 'calibration' : calibration_dict, 
    'acquisition' : acquisition_dict}
    return [dictionary,]

def file_writer(filename, spectrum, write_microscope_parameters = True, 
format = 'Y', separator = ', '):    
    FORMAT = "EMSA/MAS Spectral Data File"
    VERSION = '1.0'
    keywords = {}
    if hasattr(spectrum, "title"):
        if len(spectrum.title) > 64 :
            print "The maximum lenght of the title is 64 char"
            print "The current title:\"", spectrum.title, "\" is too long"
            return 0
        else :
            keywords['TITLE'] = spectrum.title # max 64 char
    else :
        keywords['TITLE'] = 'Undefined'
    if hasattr(spectrum, "date"):
        loc = locale.getlocale(locale.LC_TIME)
        if os_name == 'posix':
            locale.setlocale(locale.LC_TIME, ('en_US', 'UTF8'))
        elif os_name == 'windows':
            locale.setlocale(locale.LC_TIME, 'english')
        keywords['DATE'] = spectrum.date.strftime("%d-%b-%Y")
        locale.setlocale(locale.LC_TIME, loc) # restore saved locale
    else:
        loc = locale.getlocale(locale.LC_TIME)
        if os_name == 'posix':
            locale.setlocale(locale.LC_TIME, ('en_US', 'UTF8'))
        elif os_name == 'windows':
            locale.setlocale(locale.LC_TIME, 'english')
        keywords['DATE'] = time.strftime("%d-%b-%Y")
        locale.setlocale(locale.LC_TIME, loc) # restore saved locale
    if hasattr(spectrum, "time"):
        keywords['TIME'] = spectrum.time.strftime("%H:%M")
    else:
        keywords['TIME'] = time.strftime("%H:%M")
    if hasattr(spectrum, "owner"):
        keywords['OWNER'] = spectrum.owner
    else:
        keywords['OWNER'] = 'Undefined'
    if spectrum.energydimension > 4096 :
        print "The MSA format does not support spectrum with more \
         than 4096 channels"
        print "The file was not saved"
        return
    else :
       keywords['NPOINTS'] = spectrum.energydimension
    keywords['NCOLUMNS'] = 1

    if hasattr(spectrum, "xunits"):
        keywords['XUNITS'] = spectrum.xunits
    else:
        print "The units x units are not set. Assuming eV"
        keywords['XUNITS'] = 'eV'           
    if hasattr(spectrum, "yunits"):
        keywords['YUNITS'] = spectrum.yunits
    else:
        print "The units y units are not defined"
        keywords['YUNITS'] = 'Undefined'
    if spectrum.acquisition_parameters.exposure is not None:
        keywords['INTEGTIME'] = spectrum.acquisition_parameters.exposure
    if format == 'XY':
        keywords['DATATYPE'] = 'XY'
    elif format == 'Y':
        keywords['DATATYPE'] = 'Y'
    keywords['XPERCHAN'] = spectrum.energyscale
    keywords['OFFSET'] = spectrum.energyorigin
    if write_microscope_parameters is True:
        print "\nWarning: the microscope attributes will be written to the file"
        print microscope
        print
        keywords['CONVANGLE'] = microscope.alpha
        keywords['COLLANGLE'] = microscope.beta
        keywords['BEAMKV'] = microscope.E0
        keywords['PPPC'] = microscope.pppc
        keywords['CORRFAC'] = microscope.correlation_factor
    file = open(filename, 'w')
    for keyword, value in keywords.items():
        file.write(u'#%-13s: %s \u000D\u000A' % (keyword, value))
    file.write(u'#SPECTRUM    : Spectral Data Starts Here\u000D\u000A')
    fmt="%g"
    i = 0
    ix = spectrum.coordinates.ix
    iy = spectrum.coordinates.iy
    if format == 'XY':
        
        for row in spectrum.data_cube[:,ix, iy]:
            file.write("%g%s%g" % (spectrum.energy_axis[i], separator, row))
            file.write(u'\u000D\u000A')
            i += 1
    elif format == 'Y':
        for row in spectrum.data_cube[:, ix, iy]:
            file.write(fmt % row)
            file.write(u'\u000D\u000A')
        
    file.write(u'#ENDOFDATA   : End Of Data and File')
    file.close()
    print "File saved"
