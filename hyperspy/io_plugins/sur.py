# -*- coding: utf-8 -*-
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

# Plugin to read the mountainsmap surface format (sur)
#Current state can bring support to the surface format if the file is an
#attolight hyperspectral map, but cannot bring write nor support for other
#mountainsmap files (.pro etc.). I need to write some tests, check whether the
#comments can be systematically parsed into metadata and write a support for
#original_metadata or other

import logging
#Dateutil allows to parse date but I don't think it's useful here
#import dateutil.parser

import numpy as np
#Commented for now because I don't know what purpose it serves
#import traits.api as t

from copy import deepcopy
import struct
import sys
import zlib
import os
import warnings

#Maybe later we can implement reading the class with the io utils tools instead
#of re-defining read functions in the class
#import hyperspy.misc.io.utils_readfile as iou

#This module will prove useful when we write the export function
#import hyperspy.misc.io.tools

#DictionaryTreeBrowser class handles the fancy metadata dictionnaries
#from hyperspy.misc.utils import DictionaryTreeBrowser
from hyperspy.exceptions import MountainsMapFileError

_logger = logging.getLogger(__name__)

# Plugin characteristics
# ----------------------
format_name = 'Digital Surf Surface sur'
description = """Read data from the proprietary .sur file format from Digital
Surf. Allows hyperspy to interact with the mountains map software"""
full_support = False #Check with the boys once this is done
# Recognised file extension
file_extensions = ('sur', 'SUR','pro','PRO')
default_extension = 0
# Writing features
writes = False #First we will check with the load
# ----------------------

class DigitalSurfHandler(object):
    """ Class to read Digital Surf MountainsMap files.

    Attributes
    ----------
    filename, signal_dict, _work_dict, _list_sur_file_content, _Object_type,
    _N_data_object, _N_data_channels, _initialized

    Methods
    -------
    parse_file, parse_header, get_image_dictionaries

    Class Variables
    ---------------
    _object_type : dict key: int containing the mountainsmap object types

    """
    #Object types
    _mountains_object_types = {
                                -1: "_ERROR"              ,
                                 0: "_UNKNOWN"            ,
                                 1: "_PROFILE"            ,
                                 2: "_SURFACE"            ,
                                 3: "_BINARYIMAGE"        ,
                                 4: "_PROFILESERIE"       ,
                                 5: "_SURFACESERIE"       ,
                                 6: "_MERIDIANDISC"       ,
                                 7: "_MULTILAYERPROFILE"  ,
                                 8: "_MULTILAYERSURFACE"  ,
                                 9: "_PARALLELDISC"       ,
                                10: "_INTENSITYIMAGE"     ,
                                11: "_INTENSITYSURFACE"   ,
                                12: "_RGBIMAGE"           ,
                                13: "_RGBSURFACE"         ,
                                14: "_FORCECURVE"         ,
                                15: "_SERIEOFFORCECURVE"  ,
                                16: "_RGBINTENSITYSURFACE",
                                20: "_SPECTRUM"           ,
                                21: "_HYPCARD"            ,
                                }

    def __init__(self, filename=None):

        #We do not need to check for file existence here because
        #io module implements it in the load function
        self.filename = filename

        #The signal_dict dictionnary has to be returned by the
        #file_reader function. Apparently original_metadata needs
        #to be set
        self.signal_dict = {'data': np.empty((0,0,0)),
                            'axes': [],
                            'metadata': {},
                            'original_metadata': {}
                            }

        #Dictionary to store, read and write fields in the binary file
        #defined in the MountainsMap SDK. Structure is
        # _work_dict['Field']['value'] : access field value
        # _work_dict['Field']['b_unpack_fn'](f) : unpack value from file f
        # _work_dict['Field']['b_pack_fn'](f,v): pack value v in file f
        self._work_dict = \
            {
            "_01_Signature":
                {
                'value':"DSCOMPRESSED",
                'b_unpack_fn': lambda f: self._get_str(f,12,"DSCOMPRESSED"),
                'b_pack_fn': lambda f,v: self._set_str(f,v,12),
                },
            "_02_Format":
                {
                'value':0,
                'b_unpack_fn': self._get_int16,
                'b_pack_fn': self._set_int16,
                },
            "_03_Number_of_Objects":
                {
                'value':1,
                'b_unpack_fn': self._get_int16,
                'b_pack_fn': self._set_int16,
                },
            "_04_Version":
                {
                'value':1,
                'b_unpack_fn': self._get_int16,
                'b_pack_fn': self._set_int16,
                },
            "_05_Object_Type":
                {
                'value':2,
                'b_unpack_fn': self._get_int16,
                'b_pack_fn': self._set_int16,
                },
            "_06_Object_Name":
                {
                'value':"",
                'b_unpack_fn': lambda f: self._get_str(f,30,"DOSONLY"),
                'b_pack_fn': lambda f,v: self._set_str(f,v,30),
                },
            "_07_Operator_Name":
                {
                'value':"",
                'b_unpack_fn': lambda f: self._get_str(f,30,""),
                'b_pack_fn': lambda f,v: self._set_str(f,v,30),
                },
            "_08_P_Size":
                {
                'value':1,
                'b_unpack_fn': self._get_int16,
                'b_pack_fn': self._set_int16,
                },
            "_09_Acquisition_Type":
                {
                'value':0,
                'b_unpack_fn': self._get_int16,
                'b_pack_fn': self._set_int16,
                },
            "_10_Range_Type":
                {
                'value':0,
                'b_unpack_fn': self._get_int16,
                'b_pack_fn': self._set_int16,
                },
            "_11_Special_Points":
                {
                'value':0,
                'b_unpack_fn': self._get_int16,
                'b_pack_fn': self._set_int16,
                },
            "_12_Absolute":
                {
                'value':0,
                'b_unpack_fn': self._get_int16,
                'b_pack_fn': self._set_int16,
                },
            "_13_Gauge_Resolution":
                {
                'value':0.0,
                'b_unpack_fn': self._get_float,
                'b_pack_fn': self._set_float,
                },
            "_14_W_Size":
                {
                'value':0,
                'b_unpack_fn': self._get_int32,
                'b_pack_fn': self._set_int32,
                },
            "_15_Size_of_Points":
                {
                'value':16,
                'b_unpack_fn':lambda f: self._get_int16(f,32),
                'b_pack_fn': self._set_int16,
                },
            "_16_Zmin":
                {
                'value':0,
                'b_unpack_fn':self._get_int32,
                'b_pack_fn':self._set_int32,
                },
            "_17_Zmax":
                {
                'value':0,
                'b_unpack_fn':self._get_int32,
                'b_pack_fn': self._set_int32,
                },
            "_18_Number_of_Points":
                {
                'value':0,
                'b_unpack_fn': self._get_int32,
                'b_pack_fn': self._set_int32,
                },
            "_19_Number_of_Lines":
                {
                'value':0,
                'b_unpack_fn':self._get_int32,
                'b_pack_fn':self._set_int32,
                },
            "_20_Total_Nb_of_Pts":
                {
                'value':0,
                'b_unpack_fn': self._get_int32,
                'b_pack_fn': self._set_int32
                },
            "_21_X_Spacing":
                {
                'value':1.0,
                'b_unpack_fn': self._get_float,
                'b_pack_fn': self._set_float,
                },
            "_22_Y_Spacing":
                {
                'value':1.0,
                'b_unpack_fn': self._get_float,
                'b_pack_fn': self._set_float,
                 },
            "_23_Z_Spacing":
                {
                'value':1.0,
                'b_unpack_fn': self._get_float,
                'b_pack_fn': self._set_float,
                },
            "_24_Name_of_X_Axis":
                {
                'value':'X',
                'b_unpack_fn': lambda f: self._get_str(f,16,"X"),
                'b_pack_fn': lambda f,v: self._set_str(f,v,16),
                },
            "_25_Name_of_Y_Axis":
                {
                'value':'Y',
                'b_unpack_fn': lambda f: self._get_str(f,16,"Y"),
                'b_pack_fn': lambda f,v: self._set_str(f,v,16),
                },
            "_26_Name_of_Z_Axis":
                {
                'value':'Z',
                'b_unpack_fn': lambda f: self._get_str(f,16,"Z"),
                'b_pack_fn': lambda f,v: self._set_str(f,v,16),
                },
            "_27_X_Step_Unit":
                {
                'value':'um',
                'b_unpack_fn': lambda f: self._get_str(f,16,"um"),
                'b_pack_fn': lambda f,v: self._set_str(f,v,16),
                },
            "_28_Y_Step_Unit":
                {
                'value':'um',
                'b_unpack_fn': lambda f: self._get_str(f,16,"um"),
                'b_pack_fn': lambda f,v: self._set_str(f,v,16),
                },
            "_29_Z_Step_Unit":
                {
                'value':'um',
                'b_unpack_fn': lambda f: self._get_str(f,16,"um"),
                'b_pack_fn': lambda f,v: self._set_str(f,v,16),
                },
            "_30_X_Length_Unit":
                {
                'value':'um',
                'b_unpack_fn': lambda f: self._get_str(f,16,"um"),
                'b_pack_fn': lambda f,v: self._set_str(f,v,16),
                },
            "_31_Y_Length_Unit":
                {
                'value':'um',
                'b_unpack_fn': lambda f: self._get_str(f,16,"um"),
                'b_pack_fn': lambda f,v: self._set_str(f,v,16),
                },
            "_32_Z_Length_Unit":
                {
                'value':'um',
                'b_unpack_fn': lambda f: self._get_str(f,16,"um"),
                'b_pack_fn': lambda f,v: self._set_str(f,v,16),
                },
            "_33_X_Unit_Ratio":
                {
                'value':1.0,
                'b_unpack_fn': self._get_float,
                'b_pack_fn': self._set_float,
                },
            "_34_Y_Unit_Ratio":
                {
                'value':1.0,
                'b_unpack_fn': self._get_float,
                'b_pack_fn': self._set_float,
                },
            "_35_Z_Unit_Ratio":
                {
                'value':1.0,
                'b_unpack_fn': self._get_float,
                'b_pack_fn': self._set_float,
                },
            "_36_Imprint":
                {
                'value':0,
                'b_unpack_fn': self._get_int16,
                'b_pack_fn': self._set_int16,
                },
            "_37_Inverted":
                {
                'value':0,
                'b_unpack_fn': self._get_int16,
                'b_pack_fn': self._set_int16,
                },
            "_38_Levelled":
                {
                'value':0,
                'b_unpack_fn': self._get_int16,
                'b_pack_fn': self._set_int16,
                },
            "_39_Obsolete":
                {
                'value':0,
                'b_unpack_fn': lambda f: self._get_bytes(f,12),
                'b_pack_fn': lambda f,v: self._set_bytes(f,v,12),
                },
            "_40_Seconds":
                {
                'value':0,
                'b_unpack_fn': self._get_int16,
                'b_pack_fn': self._set_int16,
                },
            "_41_Minutes":
                {
                'value':0,
                'b_unpack_fn': self._get_int16,
                'b_pack_fn': self._set_int16,
                },
            "_42_Hours":
                {
                'value':0,
                'b_unpack_fn': self._get_int16,
                'b_pack_fn': self._set_int16,
                },
            "_43_Day":
                {
                'value':0,
                'b_unpack_fn': self._get_int16,
                'b_pack_fn': self._set_int16,
                },
            "_44_Month":
                {
                'value':0,
                'b_unpack_fn': self._get_int16,
                'b_pack_fn': self._set_int16,
                },
            "_45_Year":
                {
                'value':0,
                'b_unpack_fn': self._get_int16,
                'b_pack_fn': self._set_int16,
                },
            "_46_Day_of_week":
                {
                'value':0,
                'b_unpack_fn': self._get_int16,
                'b_pack_fn': self._set_int16,
                },
            "_47_Measurement_duration":
                {
                'value':0.0,
                'b_unpack_fn': self._get_float,
                'b_pack_fn': self._set_float,
                },
            "_48_Compressed_data_size":
                {
                'value':0,
                'b_unpack_fn':self._get_uint32,
                'b_pack_fn':self._set_uint32,
                },
            "_49_Obsolete":
                {
                'value':0,
                'b_unpack_fn': lambda f: self._get_bytes(f,6),
                'b_pack_fn': lambda f,v: self._set_bytes(f,v,6),
                },
            "_50_Comment_size":
                {
                'value':0,
                'b_unpack_fn': self._get_int16,
                'b_pack_fn': self._set_int16,
                },
            "_51_Private_size":
                {
                'value':0,
                'b_unpack_fn': self._get_int16,
                'b_pack_fn': self._set_int16,
                },
            "_52_Client_zone":
                {
                'value':0,
                'b_unpack_fn': lambda f: self._get_bytes(f,128),
                'b_pack_fn': lambda f,v: self._set_bytes(f,v,128),
                },
            "_53_X_Offset":
                {
                'value':0.0,
                'b_unpack_fn': self._get_float,
                'b_pack_fn': self._set_float,
                },
            "_54_Y_Offset":
                {
                'value':0.0,
                'b_unpack_fn': self._get_float,
                'b_pack_fn': self._set_float,
                },
            "_55_Z_Offset":
                {
                'value':0.0,
                'b_unpack_fn': self._get_float,
                'b_pack_fn': self._set_float,
                },
            "_56_T_Spacing":\
                {
                'value':0.0,
                'b_unpack_fn': self._get_float,
                'b_pack_fn': self._set_float,
                },
            "_57_T_Offset":
                {
                'value':0.0,
                'b_unpack_fn': self._get_float,
                'b_pack_fn': self._set_float,
                },
            "_58_T_Axis_Name":
                {
                'value':'T',
                'b_unpack_fn': lambda f: self._get_str(f,13,"Wavelength"),
                'b_pack_fn': lambda f,v: self._set_str(f,v,13),
                },
            "_59_T_Step_Unit":
                {
                'value':'um',
                'b_unpack_fn': lambda f: self._get_str(f,13,"nm"),
                'b_pack_fn': lambda f,v: self._set_str(f,v,13),
                },
            "_60_Comment":
                {
                'value':0,
                'b_unpack_fn': self._unpack_comment,
                'b_pack_fn': self._pack_comment,
                },
            "_61_Private_zone":
                {
                'value':0,
                'b_unpack_fn': self._unpack_private,
                'b_pack_fn': self._pack_private,
                },
            "_62_points":
                {
                'value':0,
                'b_unpack_fn': self._unpack_data,
                'b_pack_fn': lambda f,v: 0, #Not implemented
                },
            }

        #List of all measurement
        self._list_sur_file_content = []

        #The surface files convention is that when saving multiple data
        #objects at once, they are all packed in the same binary file.
        #Every single object contains a full header with all the sections,
        # but only the first one contains the relevant infos about
        #object type, the number of objects in the file and other.
        #Hence they will be made attributes.
        #Object type
        self._Object_type = "_UNKNOWN"

        #Number of data objects in the file.
        self._N_data_object = 1
        self._N_data_channels = 1

    ### Read methods
    def _read_sur_file(self):
        """Read the binary, possibly compressed, content of the surface
        file. Surface files can be encoded as single or a succession
        of objects. The file is thus read iteratively and from metadata of the
        first file """

        with open(self.filename,'rb') as f:
            #We read the first object
            self._read_single_sur_object(f)
            #We append the first object to the content list
            self._append_work_dict_to_content()
            #Lookup how many objects are stored in the file and save
            self._N_data_object = self._get_work_dict_key_value("_03_Number_of_Objects")
            self._N_data_channels = self._get_work_dict_key_value('_08_P_Size')

            #Determine how many objects we need to read
            if self._N_data_channels>0 and self._N_data_object>0:
                N_objects_to_read = self._N_data_channels*self._N_data_object
            elif self._N_data_channels>0:
                N_objects_to_read = self._N_data_channels
            elif self._N_data_object>0:
                N_objects_to_read = self._N_data_object
            else:
                N_objects_to_read = 1

            #Lookup what object type we are dealing with and save
            self._Object_type = \
                DigitalSurfHandler._mountains_object_types[ \
                    self._get_work_dict_key_value("_05_Object_Type")]

            #if more than 1
            if N_objects_to_read > 1:
                #continue reading until everything is done
                for i in range(1,N_objects_to_read):
                    #We read an object
                    self._read_single_sur_object(f)
                    #We append it to content list
                    self._append_work_dict_to_content()

    def _read_single_sur_object(self,file):
        for key,val in self._work_dict.items():
            self._work_dict[key]['value'] = val['b_unpack_fn'](file)

    def _append_work_dict_to_content(self):
        """Save the values stored in the work dict in the surface file list"""
        datadict = deepcopy( \
            {key:val['value'] for key,val in self._work_dict.items()})
        self._list_sur_file_content.append(datadict)

    def _get_work_dict_key_value(self,key):
        return self._work_dict[key]['value']

    ### Signal dictionary methods
    def _build_sur_dict(self):
        """Create a signal dict with an unpacked object"""

        #If the signal is of the type spectrum or hypercard
        if self._Object_type in ["_HYPCARD",]:
            self._build_hyperspectral_map()
        elif self._Object_type in ["_SPECTRUM"]:
            self._build_spectrum()
        elif self._Object_type in ["_PROFILE"]:
            self._build_general_1D_data()
        elif self._Object_type in ["_PROFILESERIE"]:
            self._build_1D_series()
        elif self._Object_type in ["_SURFACE"]:
            self._build_surface()
        elif self._Object_type in ["_SURFACESERIE"]:
            self._build_surface_series()
        elif self._Object_type in ["_MULTILAYERSURFACE"]:
            self._build_surface_series()
        elif self._Object_type in ["_RGBSURFACE"]:
            self._build_RGB_surface()
        elif self._Object_type in ["_RGBIMAGE"]:
            self._build_RGB_image()
        elif self._Object_type in ["_RGBINTENSITYSURFACE"]:
            self._build_RGB_surface()
        elif self._Object_type in ["_BINARYIMAGE"]:
            self._build_surface()
        else:
            raise MountainsMapFileError(self._Object_type \
                + "is not a supported mountain object.")

        return self.signal_dict

    def _build_Xax(self,unpacked_dict,ind=0,nav=False):
        """Return X axis dictionary from an unpacked dict. index int and navigate
        boolean can be optionally passed. Default 0 and False respectively."""
        Xax = { 'name': unpacked_dict['_24_Name_of_X_Axis'],
                'size': unpacked_dict['_18_Number_of_Points'],
                'index_in_array': ind,
                'scale': unpacked_dict['_21_X_Spacing'],
                'offset': unpacked_dict['_53_X_Offset'],
                'units': unpacked_dict['_27_X_Step_Unit'],
                'navigate':nav,
                }
        return Xax

    def _build_Yax(self,unpacked_dict,ind=1,nav=False):
        """Return X axis dictionary from an unpacked dict. index int and navigate
        boolean can be optionally passed. Default 1 and False respectively."""
        Yax = { 'name': unpacked_dict['_25_Name_of_Y_Axis'],
                'size': unpacked_dict['_19_Number_of_Lines'],
                'index_in_array': ind,
                'scale': unpacked_dict['_22_Y_Spacing'],
                'offset': unpacked_dict['_54_Y_Offset'],
                'units': unpacked_dict['_28_Y_Step_Unit'],
                'navigate':nav,
                }
        return Yax

    def _build_Tax(self,unpacked_dict,size_key,ind=0,nav=True):
        """Return T axis dictionary from an unpacked surface object dict.
        Unlike x and y axes, the size key can be determined from various keys:
        _14_W_Size, _15_Size_of_Points or _03_Number_of_Objects. index int
        and navigate boolean can be optionally passed. Default 0 and
        True respectively."""

        #The T axis is somewhat special because it is only defined on series
        #and is thus only navigation. It is only defined on the first object
        #in a serie.
        #Here it needs to be checked that the T axis scale is not 0 Otherwise
        #it raises hyperspy errors
        scale = unpacked_dict['_56_T_Spacing']
        if scale == 0:
            scale =1

        Tax = { 'name': unpacked_dict['_58_T_Axis_Name'],
                'size': unpacked_dict[size_key],
                'index_in_array': ind,
                'scale': scale,
                'offset': unpacked_dict['_57_T_Offset'],
                'units': unpacked_dict['_59_T_Step_Unit'],
                'navigate':nav,
                }
        return Tax

    ### Build methods for individual surface objects
    def _build_hyperspectral_map(self,):
        """Build a hyperspectral map. Hyperspectral maps are single-object
        files with datapoints of _14_W_Size length"""

        #Check that the object contained only one object.
        #Probably overkill at this point but better safe than sorry
        if len(self._list_sur_file_content) != 1:
            raise MountainsMapFileError("Input {:s} File is not of Hyperspectral type".format(self._Object_type))

        #We get the dictionary with all the data
        hypdic = self._list_sur_file_content[0]

        #Add all the axes to the signal dict
        self.signal_dict['axes'].append(\
            self._build_Yax(hypdic,ind=0,nav=True))
        self.signal_dict['axes'].append(\
            self._build_Xax(hypdic,ind=1,nav=True))
        #Wavelength axis in hyperspectral surface files are stored as T Axis
        #with length set as _14_W_Size
        self.signal_dict['axes'].append(\
            self._build_Tax(hypdic,'_14_W_Size',ind=2,nav=False))

        #We reshape the data in the correct format
        self.signal_dict['data'] = hypdic['_62_points'].reshape(\
            hypdic['_19_Number_of_Lines'],
            hypdic['_18_Number_of_Points'],
            hypdic['_14_W_Size'],
            )

        self.signal_dict['metadata'] = self._build_metadata(hypdic)

        self.signal_dict['original_metadata'] = self._build_original_metadata()

    def _build_general_1D_data(self,):
        """Build general 1D Data objects. Currently work with spectra"""

        #Check that the object contained only one object.
        #Probably overkill at this point but better safe than sorry
        if len(self._list_sur_file_content) != 1:
            raise MountainsMapFileError("Corrupt file")

        #We get the dictionary with all the data
        hypdic = self._list_sur_file_content[0]

        #Add the axe to the signal dict
        self.signal_dict['axes'].append(\
            self._build_Xax(hypdic,ind=0,nav=False))

        #We reshape the data in the correct format
        self.signal_dict['data'] = hypdic['_62_points']

        #Build the metadata
        self._set_metadata_and_original_metadata(hypdic)

    def _build_spectrum(self,):
        """Build spectra objects. Spectra and 1D series of spectra are
        saved in the same object."""

        #We get the dictionary with all the data
        hypdic = self._list_sur_file_content[0]

        #Add the signal axis_src to the signal dict
        self.signal_dict['axes'].append(\
            self._build_Xax(hypdic,ind=1,nav=False))

        #If there is more than 1 spectrum also add the navigation axis
        if hypdic['_19_Number_of_Lines'] != 1:
            self.signal_dict['axes'].append(\
                self._build_Yax(hypdic,ind=0,nav=True))

        #We reshape the data in the correct format.
        #Edit: the data is now squeezed for unneeded dimensions
        self.signal_dict['data'] = np.squeeze(hypdic['_62_points'].reshape(\
            hypdic['_19_Number_of_Lines'],
            hypdic['_18_Number_of_Points'],
            ))

        self._set_metadata_and_original_metadata(hypdic)

    def _build_1D_series(self,):
        """Build a series of 1D objects. The T axis is navigation and set from
        the first object"""

        #First object dictionary
        hypdic = self._list_sur_file_content[0]

        #Metadata are set from first dictionary
        self._set_metadata_and_original_metadata(hypdic)

        #Add the series-axis to the signal dict
        self.signal_dict['axes'].append(\
            self._build_Tax(hypdic,'_03_Number_of_Objects',ind=0,nav=True))

        #All objects must share the same signal axis
        self.signal_dict['axes'].append(\
            self._build_Xax(hypdic,ind=1,nav=False))

        #We put all the data together
        data = []
        for obj in self._list_sur_file_content:
            data.append(obj['_62_points'])

        self.signal_dict['data'] = np.stack(data)

    def _build_surface(self,):
        """Build a surface"""

        #Check that the object contained only one object.
        #Probably overkill at this point but better safe than sorry
        if len(self._list_sur_file_content) != 1:
            raise MountainsMapFileError("CORRUPT {:s} FILE".format(self._Object_type))

        #We get the dictionary with all the data
        hypdic = self._list_sur_file_content[0]

        #Add all the axes to the signal dict
        self.signal_dict['axes'].append(\
            self._build_Yax(hypdic,ind=0,nav=False))
        self.signal_dict['axes'].append(\
            self._build_Xax(hypdic,ind=1,nav=False))



        #We reshape the data in the correct format
        shape = (hypdic['_19_Number_of_Lines'],hypdic['_18_Number_of_Points'])
        self.signal_dict['data'] = hypdic['_62_points'].reshape(shape)

        self._set_metadata_and_original_metadata(hypdic)

    def _build_surface_series(self,):
        """Build a series of surfaces. The T axis is navigation and set from
        the first object"""

        #First object dictionary
        hypdic = self._list_sur_file_content[0]

        #Metadata are set from first dictionary
        self._set_metadata_and_original_metadata(hypdic)

        #Add the series-axis to the signal dict
        self.signal_dict['axes'].append(\
            self._build_Tax(hypdic,'_03_Number_of_Objects',ind=0,nav=True))

        #All objects must share the same signal axes
        self.signal_dict['axes'].append(\
                self._build_Yax(hypdic,ind=1,nav=False))
        self.signal_dict['axes'].append(\
            self._build_Xax(hypdic,ind=2,nav=False))

        #shape of the surfaces in the series
        shape = (hypdic['_19_Number_of_Lines'],hypdic['_18_Number_of_Points'])
        #We put all the data together
        data = []
        for obj in self._list_sur_file_content:
            data.append(obj['_62_points'].reshape(shape))

        self.signal_dict['data'] = np.stack(data)

    def _build_RGB_surface(self,):
        """Build a series of surfaces. The T axis is navigation and set from
        P Size"""

        #First object dictionary
        hypdic = self._list_sur_file_content[0]

        #Metadata are set from first dictionary
        self._set_metadata_and_original_metadata(hypdic)

        #Add the series-axis to the signal dict
        self.signal_dict['axes'].append(\
            self._build_Tax(hypdic,'_08_P_Size',ind=0,nav=True))

        #All objects must share the same signal axes
        self.signal_dict['axes'].append(\
                self._build_Yax(hypdic,ind=1,nav=False))
        self.signal_dict['axes'].append(\
            self._build_Xax(hypdic,ind=2,nav=False))

        #shape of the surfaces in the series
        shape = (hypdic['_19_Number_of_Lines'],hypdic['_18_Number_of_Points'])
        #We put all the data together
        data = []
        for obj in self._list_sur_file_content:
            data.append(obj['_62_points'].reshape(shape))

        #Pushing data into the dictionary
        self.signal_dict['data'] = np.stack(data)

    def _build_RGB_image(self,):
        """Build an RGB image. The T axis is navigation and set from
        P Size"""

        #First object dictionary
        hypdic = self._list_sur_file_content[0]

        #Metadata are set from first dictionary
        self._set_metadata_and_original_metadata(hypdic)

        #Add the series-axis to the signal dict
        self.signal_dict['axes'].append(\
            self._build_Tax(hypdic,'_08_P_Size',ind=0,nav=True))

        #All objects must share the same signal axes
        self.signal_dict['axes'].append(\
                self._build_Yax(hypdic,ind=1,nav=False))
        self.signal_dict['axes'].append(\
            self._build_Xax(hypdic,ind=2,nav=False))

        #shape of the surfaces in the series
        shape = (hypdic['_19_Number_of_Lines'],hypdic['_18_Number_of_Points'])
        #We put all the data together
        data = []
        for obj in self._list_sur_file_content:
            data.append(obj['_62_points'].reshape(shape))

        #Pushing data into the dictionary
        self.signal_dict['data'] = np.stack(data)

        self.signal_dict.update({'post_process':[self.post_process_RGB]})

    ### Metadata utility methods
    def _build_metadata(self,unpacked_dict):
        """Return a minimalistic metadata dictionary according to hyperspy
        format. Accept a dictionary as an input because dictionary with the
        headers of a mountians object.

        Parameters
        ----------
        unpacked_dict: dictionary from the header of a surface file

        Returns
        -------
        metadict: dictionnary in the hyperspy metadata format

        """

        #Formatting for complicated strings. We add parentheses to units
        qty_unit = unpacked_dict['_29_Z_Step_Unit']
        #We strip unit from any character that might pre-format it
        qty_unit = qty_unit.strip(' \t\n()[]')
        #If unit string is still truthy after strip we add parentheses
        if qty_unit:
            qty_unit = "({:s})".format(qty_unit)


        quantity_str = " ".join([
            unpacked_dict['_26_Name_of_Z_Axis'],qty_unit]).strip()

        #Date and time are set in metadata only if all values are not set to 0

        date = [unpacked_dict['_45_Year'],
                unpacked_dict['_44_Month'],
                unpacked_dict['_43_Day']]
        if not all(v == 0 for v in date):
            date_str =  "{:4d}-{:2d}-{:2d}".format(date[0],date[1],date[2])
        else:
            date_str = ""

        time = [unpacked_dict['_42_Hours'],
                unpacked_dict['_41_Minutes'],
                unpacked_dict['_40_Seconds']]

        if not all(v == 0 for v in time):
            time_str = "{:d}:{:d}:{:d}".format(time[0],time[1],time[2])
        else:
            time_str = ""

        #Metadata dictionary initialization
        metadict = {
            "General":{
                "authors": unpacked_dict['_07_Operator_Name'],
                "date":date_str,
                "original_filename": os.path.split(self.filename)[1],
                "time": time_str,
                },
            "Signal": {
                "binned": False,
                "quantity": quantity_str,
                "signal_type": "",
                },
            }

        return metadict

    def _build_original_metadata(self,):
        """Builds a metadata dictionnary from the header"""
        original_metadata_dict = {}

        Ntot = (self._N_data_object+1)*(self._N_data_channels+1)

        #Iteration over Number of data objects
        for i in range(self._N_data_object):
            #Iteration over the Number of Data channels
            for j in range(self._N_data_channels):
                #Creating a dictionary key for each object
                k = (i+1)*(j+1)
                key = "Object_{:d}_Channel_{:d}".format(i,j)
                original_metadata_dict.update({key:{}})

                #We load one full object header
                a = self._list_sur_file_content[k-1]

                #Save it as original metadata dictionary
                headerdict = {"H"+l.lstrip('_'):a[l] for l in a if l not in \
                    ("_62_points",'_61_Private_zone')}
                original_metadata_dict[key].update({"Header" : headerdict})

                #The second dictionary might contain custom mountainsmap params
                parsedict = {}

                #Check if it is the case and append it to
                #original metadata if yes

                valid_comment = self._check_comments(a["_60_Comment"],'$','=')
                if valid_comment:
                    parsedict = self._MS_parse(a["_60_Comment"],'$','=')
                    parsedict = {l.lstrip('_'):m for l,m in parsedict.items()}
                    original_metadata_dict[key].update({"Parsed" : parsedict})

        return original_metadata_dict

    def _set_metadata_and_original_metadata(self,unpacked_dict):
        """Run successively _build_metadata and _build_original_metadata
        and set signal dictionary with results"""

        self.signal_dict['metadata'] = self._build_metadata(unpacked_dict)
        self.signal_dict['original_metadata'] = self._build_original_metadata()

    def _check_comments(self,commentsstr,prefix,delimiter):
        """Check if comment string is parsable into metadata dictionary.
        Some specific lines (empty or starting with @@) will be ignored,
        but any non-ignored line must conform to being a title line (beginning
        with the TITLESTART indicator) or being parsable (starting with Prefix
        and containing the key data delimiter). At the end, the comment is
        considered parsable if it contains minimum 1 parsable line and no
        non-ignorable non-parsable non-title line.

        Parameters
        ----------
        commentstr: string containing comments
        prefix: string (or char) character assumed to start each line.
        '$' if a .sur file.
        delimiter: string that delimits the keyword from value. always '='

        Returns
        -------
        valid: boolean
        """

        #Titlestart markers start with Prefix ($) followed by underscore
        TITLESTART = '{:s}_'.format(prefix)

        #We start by assuming that the comment string is valid
        #but contains 0 valid (= parsable) lines
        valid = True
        N_valid_lines = 0

        for line in commentsstr.splitlines():
            #Here we ignore any empty line or line starting with @@
            ignore = False
            if not line.strip() or line.startswith('@@'):
                ignore = True
            #If the line must not be ignored
            if not ignore:
                #If line starts with a titlestart marker we it counts as valid
                if line.startswith(TITLESTART):
                    N_valid_lines += 1
                # if it does not we check that it has the delimiter and
                # starts with prefix
                else:
                    #We check that line contains delimiter and prefix
                    #if it does the count of valid line is increased
                    if delimiter in line and line.startswith(prefix):
                        N_valid_lines += 1
                    #Otherwise the whole comment string is thrown out
                    else:
                        valid = False

        #finally, it total number of valid line is 0 we throw out this comments
        if N_valid_lines ==0:
            valid = False

        #return falsiness of the string.
        return valid

    def _MS_parse(self, strMS, prefix, delimiter):
        """ Parses a string containing metadata information. The string can be
        read from the comment section of a .sur file, or, alternatively, a file
        containing them with a similar formatting.

        Parameters
        ----------
        strMS: string containing metadata
        prefix: string (or char) character assumed to start each line.
        '$' if a .sur file.
        delimiter: string that delimits the keyword from value. always '='

        Returns
        -------
        dictMS: dictionnary in the correct hyperspy metadata format

        """
        #dictMS is created as an empty dictionnary
        dictMS = {}
        #Title lines start with an underscore
        TITLESTART = '{:s}_'.format(prefix)

        for line in strMS.splitlines() :
            #Here we ignore any empty line or line starting with @@
            ignore = False
            if not line.strip() or line.startswith('@@'):
                ignore = True
            #If the line must not be ignored
            if not ignore:
                if line.startswith(TITLESTART):
                    #We strip keys from whitespace at the end and beginning
                    keyMain = line[len(TITLESTART):].strip()
                    dictMS[keyMain] = {}
                elif line.startswith(prefix):
                    key, *liValue = line.split(delimiter)
                    #Key is also stripped from beginning or end whitespace
                    key = key[len(prefix):].strip()
                    strValue = liValue[0] if len(liValue)>0 else ""
                    # remove whitespace at the beginning of value
                    strValue = strValue.strip()
                    liValue = strValue.split(' ')
                    try :
                        if key == "Grating":
                            dictMS[keyMain][key] = liValue[0] # we don't want to eval this one
                        else :
                            dictMS[keyMain][key] = eval(liValue[0])
                    except :
                        dictMS[keyMain][key] = liValue[0]
                    if len(liValue) > 1:
                        dictMS[keyMain][key+'_units'] = liValue[1]
        return dictMS

    ### Post processing
    def post_process_RGB(self,signal):
        signal = signal.transpose()
        max_data = np.nanmax(signal.data)
        if max_data <=256:
            signal.change_dtype('uint8')
            signal.change_dtype('rgb8')
        elif max_data <=65536:
            signal.change_dtype('uint8')
            signal.change_dtype('rgb8')
        else:
            warnings.warn("""RGB-announced data could not be converted to
            uint8 or uint16 datatype""")
            pass

        return signal

    ### pack/unpack binary quantities
    def _get_int16(self,file, default=None, signed=True):
        """Read a 16-bits int with a user-definable default value if
        no file is given"""
        if file is None :
            return default
        b = file.read(2)
        if sys.byteorder == 'big' :
            return struct.unpack('>h', b)[0]
        else :
            return struct.unpack('<h', b)[0]

    def _set_int16(self, file, val):
	       file.write(struct.pack('<h', val))

    def _get_str(self, file, size, default=None, encoding='latin-1'):
        """Read a str of defined size in bytes with a user-definable default
        value if no file is given"""
        if file is None :
            return default
        read_str = file.read(size).decode(encoding)
        return read_str.strip(' \t\n')

    def _set_str(self, file, val, size, encoding='latin-1'):
        """Write a str of defined size in bytes to a file. struct.pack
        will automatically trim the string if it is too long"""
        file.write(struct.pack('<{:d}s'.format(size),
            '{{:<{:d}s}}'.format(size).format(val).encode(encoding)))

    def _get_int32(self,file, default=None):
        """Read a 32-bits int with a user-definable default value if no
        file is given"""
        if file is None :
            return default
        b = file.read(4)
        if sys.byteorder == 'big' :
            return struct.unpack('>i', b)[0]
        else :
            return struct.unpack('<i', b)[0]

    def _set_int32(self, file, val):
        """Write a 32-bits int in a file f """
        file.write(struct.pack('<i', val))

    def _get_float(self,file,default=None):
        """Read a 4-bytes (single precision) float from a binary file f with a
        default value if no file is given"""
        if file is None:
            return default
        return struct.unpack('<f', file.read(4))[0]

    def _set_float(file, val):
        """write a 4-bytes (single precision) float in a file"""
        file.write(struct.pack('<f', val))

    def _get_uint32(self, file, default=None):
    	if file is None :
    		return default
    	b = file.read(4)
    	if sys.byteorder == 'big' :
    		return struct.unpack('>I', b)[0]
    	else :
    		return struct.unpack('<I', b)[0]

    def _set_uint32(self, file, val):
	       file.write(struct.pack('<I', val))

    def _get_bytes(self, file, size, default=None):
        if file is None:
            return default
        else:
            return file.read(size)

    def _set_bytes(self, file, val, size):
    	file.write(struct.pack('<{:d}s'.format(size), val))

    def _unpack_comment(self,file,encoding='latin-1'):
        commentsize = self._get_work_dict_key_value("_50_Comment_size")
        return self._get_str(file,commentsize,encoding)

    def _pack_comment(self,file,val,encoding='latin-1'):
        commentsize = self._get_work_dict_key_value("_50_Comment_size")
        self._set_str(file,val,commentsize)

    def _unpack_private(self,file,encoding='latin-1'):
        privatesize = self._get_work_dict_key_value("_51_Private_size")
        return self._get_str(file,privatesize,encoding)

    def _pack_private(self,file,val,encoding='latin-1'):
        privatesize = self._get_work_dict_key_value("_51_Private_size")
        self._set_str(file,val,commentsize)

    def _unpack_data(self,file,encoding='latin-1'):
        """This needs to be special because it reads until the end of
        file. This causes an error in the series of data"""

        #Size of datapoints in bytes. Always int16 (==2) or 32 (==4)
        Psize = int(self._get_work_dict_key_value('_15_Size_of_Points')/8)
        dtype = np.int16 if Psize == 2 else np.int32

        if self._get_work_dict_key_value('_01_Signature') != 'DSCOMPRESSED' :
            #If the points are not compressed we need to read the exact
            #size occupied by datapoints

            #Datapoints in X and Y dimensions
            Npts_tot = self._get_work_dict_key_value('_20_Total_Nb_of_Pts')
            #Datasize in WL
            Wsize = self._get_work_dict_key_value('_14_W_Size')

            #We need to take into account the fact that Wsize is often
            #set to 0 instead of 1 in non-spectral data to compute the
            #space occupied by data in the file
            readsize = Npts_tot*Psize
            if Wsize != 0:
                readsize*=Wsize
            #if Npts_channel is not 0:
            #    readsize*=Npts_channel

            #Read the exact size of the data
            _points = np.frombuffer(file.read(readsize),dtype=dtype)
            #_points = np.fromstring(file.read(readsize),dtype=dtype)

        else:
            #If the points are compressed do the uncompress magic. There
            #the space occupied by datapoints is self-taken care of.
            #Number of streams
            _directoryCount = self._get_uint32(file)

            #empty lists to store the read sizes
            rawLengthData = []
            zipLengthData = []
            for i in range(_directoryCount):
                #Size of raw and compressed data sizes in each stream
                rawLengthData.append(self._get_uint32(file))
                zipLengthData.append(self._get_uint32(file))

            #We now initialize an empty binary string to store the results
            rawData = b''
            for i in range(_directoryCount):
                #And for each stream we uncompress using zip lib
                #and add it to raw string
                rawData += zlib.decompress(file.read(zipLengthData[i]))

            #Finally numpy converts it to a numeric object
            _points = np.frombuffer(rawData, dtype=dtype)
            #_points = np.fromstring(rawData, dtype=dtype)

        # rescale data
        #We set non measured points to nan according to .sur ways
        nm = []
        if self._get_work_dict_key_value("_11_Special_Points") == 1 :
            # has unmeasured points
            nm = _points == self._get_work_dict_key_value("_16_Zmin")-2

        #We set the point in the numeric scale
        _points = _points.astype(float) \
            * self._get_work_dict_key_value("_23_Z_Spacing") \
            * self._get_work_dict_key_value("_35_Z_Unit_Ratio") \
            + self._get_work_dict_key_value("_55_Z_Offset")

        _points[nm] = np.nan
        #Return the points, rescaled
        return _points

    def _pack_data(self,file,val,encoding='latin-1'):
        """This needs to be special because it writes until the end of
        file."""
        datasize = self._get_work_dict_key_value("_62_points")
        self._set_str(file,val,datasize)

def file_reader(filename,**kwds):
    """Read a mountainsmap .sur file and return a dictionnary containing the
    information necessary for creating the data object

    Parameters
    ----------
    filename: name of the .sur file to be read

    Returns
    -------
    signal_dict: dictionnary in the appropriate format. The dictionnary can
    contain several keys including 'data', 'axes', 'metadata', 'original_metadata',
    'post_process', 'mapping', 'attributes'.
    """

    ds = DigitalSurfHandler(filename)

    ds._read_sur_file()

    surdict = ds._build_sur_dict()

    return [surdict,]
