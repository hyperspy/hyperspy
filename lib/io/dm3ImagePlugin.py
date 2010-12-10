#!/usr/bin/env python
# -*- coding: latin-1 -*-

# Plugin to read the Gatan Digital Micrograph(TM) file format
#
# Copyright (c) 2010 Stefano Mazzucco.
# All rights reserved.
#
# This program is still at an early stage to be released, so the use of this
# code must be explicitly authorized by its author and cannot be shared for any reason.
#
# Once the program will be mature, it will be released under a GNU GPL license

from __future__ import with_statement #for Python versions < 2.6

import os
# import platform, sys, re, mmap, struct
import numpy as np

from dm3File import *
from temExceptions import *
from temUtils import Dimensions

# Plugin characteristics
# ----------------------
format_name = 'Digital Micrograph dm3'
description = ''
full_suport = False
# Recognised file extension
file_extensions = ('dm3', 'DM3')
default_extension = 0
# Reading features
reads_images = True
reads_spectrum = True
reads_spectrum_image = True
# Writing features
writes_images = False
writes_spectrum = False
writes_spectrum_image = False
# ----------------------

class dm3ImageFile(object):
    """ Class to handle Gatan Digital Micrograph (TM) files.
    """

    format = 'dm3'
    format_description = 'Gatan Digital Micrograph (TM) Version 3'

    # Image data types (Image Object chapter on DM help)#
    # key = DM data type code
    # value = numpy data type
    imdtype = {
        0 : 'not_implemented', # null
        1 : 'int16',
        2 : 'float32',
        3 : 'complex64',
        4 : 'not_implemented', # obsolete
        5 : "complex64_packed", # not numpy: 8-Byte packed complex (for FFT data)
        6 : 'uint8',
        7 : 'int32',
        8 : 'argb', # not numpy: 4-Byte RGB (alpha, R, G, B)
        9 : 'int8',
        10 : 'uint16',
        11 : 'uint32',
        12 : 'float64',
        13 : 'complex128',
        14 : 'bool',
        23 : 'rgb', # not numpy: 4-Byte RGB (0, R, G, B)
        }
    
    # important tags THIS DICTIONARY IS REALLY UGLY AND YOU KNOW IT!#
    itags = {
        # ENDIANNESS (BYTE ORDER)
        'lendiantag' : 'DM3.isLittleEndian',
        # DM3 NAME... is it really useful?
        'nametag' : 'DM3.DocumentObjectList.DocumentTags.Image Behavior.ImageList.__group__.__group__.ImageData.ImageTags.Name',
        # WIDTH (COLUMNS)
        'wtag' : 'DM3.DocumentObjectList.DocumentTags.Image Behavior.ImageList.__group__.__group__.ImageData.Calibrations.Dimensions.__data__.0',
        # WIDTH ORIGIN
        'worigtag' : 'DM3.DocumentObjectList.DocumentTags.Image Behavior.ImageList.__group__.__group__.ImageData.Calibrations.Brightness.Dimension.__group__.Origin',
        # WIDTH SCALE
        'wscaletag' : 'DM3.DocumentObjectList.DocumentTags.Image Behavior.ImageList.__group__.__group__.ImageData.Calibrations.Brightness.Dimension.__group__.Scale',
        # WIDTH UNITS
        'wunittag' : 'DM3.DocumentObjectList.DocumentTags.Image Behavior.ImageList.__group__.__group__.ImageData.Calibrations.Brightness.Dimension.__group__.Units', 
        # HEIGHT (ROWS)
        'htag' : 'DM3.DocumentObjectList.DocumentTags.Image Behavior.ImageList.__group__.__group__.ImageData.Calibrations.Dimensions.__data__.1',
        # HEIGHT ORIGIN
        'horigtag' : 'DM3.DocumentObjectList.DocumentTags.Image Behavior.ImageList.__group__.__group__.ImageData.Calibrations.Brightness.Dimension.__group__.__group__.Origin',
        # HEIGHT SCALE
        'hscaletag' : 'DM3.DocumentObjectList.DocumentTags.Image Behavior.ImageList.__group__.__group__.ImageData.Calibrations.Brightness.Dimension.__group__.__group__.Scale',
        # HEIGHT UNITS
        'hunittag' : 'DM3.DocumentObjectList.DocumentTags.Image Behavior.ImageList.__group__.__group__.ImageData.Calibrations.Brightness.Dimension.__group__.__group__.Units', 
        # DEPTH (usually spectral channels)
        'dtag' : 'DM3.DocumentObjectList.DocumentTags.Image Behavior.ImageList.__group__.__group__.ImageData.Calibrations.Dimensions.__data__',
        # DEPTH ORIGIN
        'dorigtag' : 'DM3.DocumentObjectList.DocumentTags.Image Behavior.ImageList.__group__.__group__.ImageData.Calibrations.Brightness.Dimension.__group__.__group__.__group__.Origin',
        # DEPTH SCALE
        'dscaletag' : 'DM3.DocumentObjectList.DocumentTags.Image Behavior.ImageList.__group__.__group__.ImageData.Calibrations.Brightness.Dimension.__group__.__group__.__group__.Scale',
        # DEPTH UNITS
        'dunittag' : 'DM3.DocumentObjectList.DocumentTags.Image Behavior.ImageList.__group__.__group__.ImageData.Calibrations.Brightness.Dimension.__group__.__group__.__group__.Units',         
        # BYTES/PIXEL... is it really useful?
        # 'pixdepthtag' : 'DM3.DocumentObjectList.DocumentTags.Image Behavior.ImageList.__group__.__group__.ImageData.Calibrations.Dimensions.PixelDepth',
        # IMAGE DATA
        'imgdatatag' : 'DM3.DocumentObjectList.DocumentTags.Image Behavior.ImageList.__group__.__group__.ImageData.Calibrations.Data',
        # IMAGE BRIGHTNESS ORIGIN (usually 0)
        'imgorigtag' : 'DM3.DocumentObjectList.DocumentTags.Image Behavior.ImageList.__group__.__group__.ImageData.Calibrations.Brightness.Origin',
        # IMAGE BRIGHTESS SCALE (usually 1)
        'imgscaletag' : 'DM3.DocumentObjectList.DocumentTags.Image Behavior.ImageList.__group__.__group__.ImageData.Calibrations.Brightness.Scale',
        # IMAGE BRIGHTNESS UNITS (usually None)
        'imgunittag' : 'DM3.DocumentObjectList.DocumentTags.Image Behavior.ImageList.__group__.__group__.ImageData.Calibrations.Brightness.Units',
        # IMAGE DATA TYPE
        'imdtypetag' : 'DM3.DocumentObjectList.DocumentTags.Image Behavior.ImageList.__group__.__group__.ImageData.Calibrations.DataType',
        # IMAGE ACQUISITION DATE
        'imgdatetag' : 'DM3.DocumentObjectList.DocumentTags.Image Behavior.ImageList.__group__.__group__.ImageData.ImageTags.Acquisition.DataBar.Acquisition Date',
        # IMAGE ACQUISITION TIME
        'imgtimetag' : 'DM3.DocumentObjectList.DocumentTags.Image Behavior.ImageList.__group__.__group__.ImageData.ImageTags.Acquisition.DataBar.Acquisition Time',
        # TEM ACTUAL MAGNIFICATION
        'magtag' : 'DM3.DocumentObjectList.DocumentTags.Image Behavior.ImageList.__group__.__group__.ImageData.ImageTags.Acquisition.DataBar.Microscope Info.Actual Magnification',
        # TEM NOMINAL MAGNIFICATION
        'nommagtag' : 'DM3.DocumentObjectList.DocumentTags.Image Behavior.ImageList.__group__.__group__.ImageData.ImageTags.Acquisition.DataBar.Microscope Info.Indicated Magnification',
        # TEM ILLUMINATION MODE
        'temillumtag' : 'DM3.DocumentObjectList.DocumentTags.Image Behavior.ImageList.__group__.__group__.ImageData.ImageTags.Acquisition.DataBar.Microscope Info.Illumination Mode',
        # TEM OPERATOIN MODE (e.g. imaging)
        'temmodetag' : 'DM3.DocumentObjectList.DocumentTags.Image Behavior.ImageList.__group__.__group__.ImageData.ImageTags.Acquisition.DataBar.Microscope Info.Items.Operation Mode',
        # TEM MODEL
        'temnametag' : 'DM3.DocumentObjectList.DocumentTags.Image Behavior.ImageList.__group__.__group__.ImageData.ImageTags.Acquisition.DataBar.Microscope Info.Items.Microscope',
        # TEM ACCELERATING VOLTAGE
        'temVtag' : 'DM3.DocumentObjectList.DocumentTags.Image Behavior.ImageList.__group__.__group__.ImageData.ImageTags.Acquisition.DataBar.Microscope Info.Items.Voltage',
        # CLUT (Color Look Up Table)
        # 'cluttag' : 'DM3.DocumentObjectList.__group__.AnnotationGroupList.ImageDisplayInfo.CLUT',
        # CLUT NAME
        # 'clutnametag' : 'DM3.DocumentObjectList.__group__.AnnotationGroupList.ImageDisplayInfo.CLUTName',
        # VSM: this is a SPIM Orsay key used by EELSLab
        'vsmtag' : 'DM3.DocumentObjectList.DocumentTags.Image Behavior.ImageList.__group__.__group__.ImageData.ImageTags.Microscope Info.Private.Processing.spim.detectors.eels.vsm',
        'exposuretag' : 'DM3.DocumentObjectList.DocumentTags.Image Behavior.ImageList.__group__.__group__.ImageData.ImageTags.Microscope Info.Private.Processing.spim.detectors.eels.dwell time',
        }

    lendiantag = itags['lendiantag']
    nametag = itags['nametag']
    wtag = itags['wtag']
    worigtag = itags['worigtag']
    wscaletag = itags['wscaletag']
    wunittag = itags['wunittag']
    htag = itags['htag']
    horigtag = itags['horigtag']
    hscaletag = itags['hscaletag']
    hunittag = itags['hunittag']    
    dtag = itags['dtag']
    dorigtag = itags['dorigtag']
    dscaletag = itags['dscaletag']
    dunittag = itags['dunittag']    
    imdtypetag = itags['imdtypetag']
    imgdatatag = itags['imgdatatag']
#     pixdepthtag = itags['pixdepthtag']    
#     cluttag = itags['cluttag']
    vsmtag = itags['vsmtag']
    exposuretag = itags['exposuretag']

    def __init__(self, fname):
        self.filename = fname
        self.info = '' # should be a dictionary with the microscope info
        self.mode = ''
        self.open()

    def __repr__(self):
        message = 'Instance of ' + repr(self.__class__)
        message += '\n' + self.mode + ' ' + str(self.imsize)
        return message

    def open(self):
        
        self.dataDict = openDM3(self.filename)

        self.name = self.dataDict[dm3ImageFile.nametag][1]

        if self.dataDict.has_key(dm3ImageFile.vsmtag):
            self.vsm = self.dataDict[dm3ImageFile.vsmtag][1]
            
        if self.dataDict.has_key(dm3ImageFile.exposuretag):
            self.exposure = self.dataDict[dm3ImageFile.exposuretag][1]
        
        if self.dataDict.has_key(dm3ImageFile.lendiantag):
            try:
                if self.dataDict[dm3ImageFile.lendiantag][1] == 1:
                    self.byte_order = 'little'
                else:
                    self.byte_order = 'big'
                self.endian = self.byte_order
            except:
                raise ByteOrderError, self.dataDict[dm3ImageFile.lendiantag][1]

        self.imdtype = dm3ImageFile.imdtype[self.dataDict[dm3ImageFile.imdtypetag][1]]

#         self.byte_offset = int(self.dataDict[dm3ImageFile.imgdatatag][0])

#         self.pixbyte = int(self.dataDict[dm3ImageFile.pixdepthtag][1])
        
#         CLUT is an array of struct
#         if self.dataDict.has_key(cluttag):
#             self.clut = self.dataDict[cluttag][1]
        # WIDTH
        self.width = Dimensions()
        if self.dataDict.has_key(dm3ImageFile.wtag):            
            self.width.size = int(self.dataDict[dm3ImageFile.wtag][1]) # i.e. COLUMNS
            self.width.origin = self.dataDict[dm3ImageFile.worigtag][1]
            self.width.scale = self.dataDict[dm3ImageFile.wscaletag][1]
            self.width.units = self.dataDict[dm3ImageFile.wunittag][1]
        # HEIGHT
        self.height = Dimensions()
        if self.dataDict.has_key(dm3ImageFile.htag):
            self.height.size = int(self.dataDict[dm3ImageFile.htag][1]) # i.e. ROWS
            self.height.origin = self.dataDict[dm3ImageFile.horigtag][1]
            self.height.scale = self.dataDict[dm3ImageFile.hscaletag][1]
            self.height.units = self.dataDict[dm3ImageFile.hunittag][1]
        # DEPTH
        self.depth = Dimensions()
        if self.dataDict.has_key(dm3ImageFile.dtag):
            self.depth.size = int(self.dataDict[dm3ImageFile.dtag][1])
            if self.width.size and self.height.size:
                self.depth.origin = self.dataDict[dm3ImageFile.dorigtag][1]
                self.depth.scale = self.dataDict[dm3ImageFile.dscaletag][1]
                self.depth.units = self.dataDict[dm3ImageFile.dunittag][1]

        # IMAGE MODE AND DIMENSIONS ATTRIBUTES
        if self.width.size and self.height.size and not self.depth.size:
            if (self.width.size == 1 or self.height.size == 1):
                self.mode += 'line_image'
            else:
                self.mode += 'image'
            self.imsize = (self.height.size, self.width.size) # SIZE = (ROWS, COLUMNS)
            self.units = [self.height.units, self.width.units]
            self.origin = [self.height.origin, self.width.origin]
            self.scale = [self.height.scale, self.width.scale]         
        elif not self.width.size and not self.height.size and self.depth.size:
            self.mode += 'spectrum'
            self.width = self.depth
            self.width.origin = self.dataDict[dm3ImageFile.worigtag][1]
            self.width.scale = self.dataDict[dm3ImageFile.wscaletag][1]
            self.width.units = self.dataDict[dm3ImageFile.wunittag][1] 
            self.depth = Dimensions()
            self.imsize = (self.width.size, )
            self.units = [self.width.units,]
            self.origin = [self.width.origin,]
            self.scale = [self.width.scale,]
        elif self.width.size and self.height.size and self.depth.size:
            self.mode += 'spim'
            self.imsize = (self.height.size, self.width.size, self.depth.size)
            self.units = [self.height.units, self.width.units, self.depth.units]
            self.origin = [self.height.origin, self.width.origin, self.depth.origin]
            self.scale = [self.height.scale, self.width.scale, self.depth.scale]
        else:
            self.mode = 'unknown'
            # raise an error?
        
        self.data = self.readImageData()
#         try:
#             self.data = self.readImageData()
#         except AttributeError:
#             print 'Could not read data.'
#             self.data = 'UNAVAILABLE'

    def readImageData(self):
        self.byte_offset = self.dataDict[dm3ImageFile.imgdatatag][0]
        self.imbytes = self.dataDict[dm3ImageFile.imgdatatag][1]
        if self.imdtype == 'not_implemented':
            raise AttributeError, self.imdtype
        with open(self.filename, 'r+b') as f:
            fmap = mmap.mmap(f.fileno(), (self.byte_offset + self.imbytes) , access=mmap.ACCESS_READ)
            fmap.seek(self.byte_offset)
            if 'rgb' in self.imdtype:
                data = self.readRGB(fmap)
            elif 'packed' in self.imdtype:
                data = self.readPackedComplex(fmap)
            else:
                if self.mode == 'spim':
                    data =  np.ndarray(self.imsize, self.imdtype, fmap.read(self.imbytes), order='F').transpose((1,0,2))
                else:
                    data =  np.ndarray(self.imsize, self.imdtype, fmap.read(self.imbytes), order='C')
            # fmap.flush()
            fmap.close()
            return data
            
    def readRGB(self, fmap):
        self.imsize = list(self.imsize)
        self.imsize.append(4)
        self.imsize = tuple(self.imsize)
        data = np.ndarray(self.imsize, 'uint8', fmap.read(self.imbytes), order='C') # (B, G, R, A)
        if self.imdtype == 'rgb':
            data = data[:, :, -2::-1] # (R, G, B)
            self.mode += '_rgb'
            self.imsize = list(self.imsize)
            self.imsize[-1] = self.imsize[-1] - 1
            self.imsize = tuple(self.imsize)
        elif self.imdtype == 'argb':
            data = np.concatenate((data[:, :, -2::-1], data[:, :, -1:]), axis=2) # (R, G, B, A)
            self.mode += '_rgba'
        return data

    def readPackedComplex(self, fmap):
        if self.width.size != self.height.size:
            raise AttributeError, "Packed complex format works only for a 2Nx2N image -> width == height"
        self.mode += '_FFT'
        N = self.width.size / 2      # think about a 2Nx2N matrix
        # read all the bytes as 1D array of 4-Byte float
        tmpdata =  np.ndarray( (self.imbytes/4, ), 'float32', fmap.read(self.imbytes), order='C')
        
        # create an empty 2Nx2N ndarray of complex
        data = np.zeros(self.imsize, 'complex64', 'F')
        
        # fill in the real values:
        data[N, 0] = tmpdata[0]
        data[0, 0] = tmpdata[1]
        data[N, N] = tmpdata[2*N**2] # Nyquist frequency
        data[0, N] = tmpdata[2*N**2+1] # Nyquist frequency
                
        # fill in the non-redundant complex values:
        # top right quarter, except 1st column
        for i in range(N):
            start = 2 * i * N + 2
            stop = start + 2 * (N - 1) - 1
            step = 2
            realpart = tmpdata[start:stop:step]
            imagpart = tmpdata[start+1:stop+1:step]
            data[i, N+1:2*N] = realpart + imagpart * 1j
        # 1st column, bottom left quarter
        start = 2 * N
        stop = start + 2 * N * (N - 1) - 1
        step = 2 * N
        realpart = tmpdata[start:stop:step]
        imagpart = tmpdata[start+1:stop+1:step]
        data[N+1:2*N, 0] = realpart + imagpart * 1j
        # 1st row, bottom right quarter
        start = 2 * N**2 + 2
        stop = start + 2 * (N - 1) - 1
        step = 2
        realpart = tmpdata[start:stop:step]
        imagpart = tmpdata[start+1:stop+1:step]
        data[N, N+1:2*N] = realpart + imagpart * 1j
        # bottom right quarter, except 1st row
        start = stop + 1
        stop = start + 2 * N * (N - 1) - 1
        step = 2
        realpart = tmpdata[start:stop:step]
        imagpart = tmpdata[start+1:stop+1:step]
        complexdata = realpart + imagpart * 1j
        data[N+1:2*N, N:2*N] = complexdata.reshape(N-1, N)

        # fill in the empty pixels: A(i)(j) = A(2N-i)(2N-j)*
        # 1st row, top left quarter, except 1st element
        data[0, 1:N] = np.conjugate(data[0, -1:-N:-1])
        # 1st row, bottom left quarter, except 1st element
        data[N, 1:N] = np.conjugate(data[N, -1:-N:-1])
        # 1st column, top left quarter, except 1st element
        data[1:N, 0] = np.conjugate(data[-1:-N:-1, 0])
        # 1st column, top right quarter, except 1st element
        data[1:N, N] = np.conjugate(data[-1:-N:-1, N])
        # top left quarter, except 1st row and 1st column
        data[1:N, 1:N] = np.conjugate(data[-1:-N:-1, -1:-N:-1])
        # bottom left quarter, except 1st row and 1st column
        data[N+1:2*N, 1:N] = np.conjugate(data[-N-1:-2*N:-1, -1:-N:-1])

        return data


def file_reader(filename, data_type=None):

    dm3 = dm3ImageFile(filename)
    
    calibration_dict = {}
    acquisition_dict = {}
    
    if 'image' in dm3.mode:
        data_type = 'Image'
    elif 'spim' in dm3.mode:
        data_type = 'SI'
    elif 'spectrum' in dm3. mode:
        raise IOError, "single spectra can't be loaded... yet"

    if dm3.name:
        calibration_dict['title'] = dm3.name
    else:
        calibration_dict['title'] =  os.path.splitext(filename)[0]

    data_cube = dm3.data

    # Scale the origins
    dm3.origin = np.asarray(dm3.origin)
    dm3.scale = np.asarray(dm3.scale)
    dm3.origin *= dm3.scale

    # Determine the dimensions
    dimensions = len(dm3.imsize)
    units = ['' for i in range(dimensions)]
    origins = np.zeros((dimensions), dtype = np.float)
    scales =  np.ones((dimensions), dtype = np.float)

    if data_type == 'SI': 
        print "Treating the data as an SI"

        # only Orsay Spim is supported for now
        if dm3.exposure:
            acquisition_dict['exposure'] = dm3.exposure
            
        if dm3.vsm:
            calibration_dict['vsm'] = float(dm3.vsm)

        # In EELSLab1 the first index must be the energy (this changes in EELSLab2)
        # Rearrange the data_cube and parameters to have the energy first
        if 'eV' in dm3.units: #could use regular expressions or compare to a 'energy units' dictionary/list
            energy_index = dm3.units.index('eV')
        elif 'keV' in dm3.units:
            energy_index = dm3.units.index('keV')
        else:
            energy_index = -1

        # In DM the origin is negative. Change it to positive
        dm3.origin[energy_index] *= -1
    
        data_cube = np.rollaxis(data_cube, energy_index, 0)
        origins = np.roll(dm3.origin, 1)
        scales = np.roll(dm3.scale, 1)
        units = np.roll(dm3.units, 1)

        # Store the calibration in the calibration dict
        origins_keys = ['energyorigin', 'xorigin', 'yorigin']
        scales_keys = ['energyscale', 'xscale', 'yscale']
        units_keys = ['energyunits', 'xunits', 'yunits']

        for value in origins:
            calibration_dict.__setitem__(origins_keys.pop(0), value)

        for value in scales:
            calibration_dict.__setitem__(scales_keys.pop(0), value)

        for value in units:
            calibration_dict.__setitem__(units_keys.pop(0), value)

    elif data_type == 'Image':
        print "Treating the data as an image"
        
        origins_keys = ['xorigin', 'yorigin', 'zorigin']
        scales_keys = ['xscale', 'yscale', 'zscale']
        units_keys = ['xunits', 'yunits', 'zunits']

        for value in origins:
            calibration_dict.__setitem__(origins_keys.pop(0), value)

        for value in scales:
            calibration_dict.__setitem__(scales_keys.pop(0), value)

        for value in units:
            calibration_dict.__setitem__(units_keys.pop(0), value)
    else:
        raise TypeError, "could not identify the file's data_type"

    calibration_dict['data_cube'] = data_cube
    
    dictionary = {
        'data_type' : data_type, 
        'calibration' : calibration_dict, 
        'acquisition' : acquisition_dict,
        'imported_parameters' : calibration_dict}
    
    return [dictionary, ]
