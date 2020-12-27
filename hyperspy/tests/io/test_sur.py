# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
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

import os

import numpy as np

from hyperspy.io import load

MY_PATH = os.path.dirname(__file__)

header_keys = ['H01_Signature',
                 'H02_Format',
                 'H03_Number_of_Objects',
                 'H04_Version',
                 'H05_Object_Type',
                 'H06_Object_Name',
                 'H07_Operator_Name',
                 'H08_P_Size',
                 'H09_Acquisition_Type',
                 'H10_Range_Type',
                 'H11_Special_Points',
                 'H12_Absolute',
                 'H13_Gauge_Resolution',
                 'H14_W_Size',
                 'H15_Size_of_Points',
                 'H16_Zmin',
                 'H17_Zmax',
                 'H18_Number_of_Points',
                 'H19_Number_of_Lines',
                 'H20_Total_Nb_of_Pts',
                 'H21_X_Spacing',
                 'H22_Y_Spacing',
                 'H23_Z_Spacing',
                 'H24_Name_of_X_Axis',
                 'H25_Name_of_Y_Axis',
                 'H26_Name_of_Z_Axis',
                 'H27_X_Step_Unit',
                 'H28_Y_Step_Unit',
                 'H29_Z_Step_Unit',
                 'H30_X_Length_Unit',
                 'H31_Y_Length_Unit',
                 'H32_Z_Length_Unit',
                 'H33_X_Unit_Ratio',
                 'H34_Y_Unit_Ratio',
                 'H35_Z_Unit_Ratio',
                 'H36_Imprint',
                 'H37_Inverted',
                 'H38_Levelled',
                 'H39_Obsolete',
                 'H40_Seconds',
                 'H41_Minutes',
                 'H42_Hours',
                 'H43_Day',
                 'H44_Month',
                 'H45_Year',
                 'H46_Day_of_week',
                 'H47_Measurement_duration',
                 'H48_Compressed_data_size',
                 'H49_Obsolete',
                 'H50_Comment_size',
                 'H51_Private_size',
                 'H52_Client_zone',
                 'H53_X_Offset',
                 'H54_Y_Offset',
                 'H55_Z_Offset',
                 'H56_T_Spacing',
                 'H57_T_Offset',
                 'H58_T_Axis_Name',
                 'H59_T_Step_Unit',
                 'H60_Comment',
                 ]

atto_head_keys = ['WAFER',
                            'SITE IMAGE',
                            'SEM',
                            'CHANNELS',
                            'SPECTROMETER',
                            'SCAN',
                            ]

atto_wafer_keys = ['Lot Number',
                     'ID',
                     'Type',
                     'Center Position X',
                     'Center Position X_units',
                     'Center Position Y',
                     'Center Position Y_units',
                     'Orientation',
                     'Orientation_units',
                     'Diameter',
                     'Diameter_units',
                     'Flat Length',
                     'Flat Length_units',
                     'Edge Exclusion',
                     'Edge Exclusion_units',
                     ]

atto_scan_keys = ['Mode',
                 'HYP Dwelltime',
                 'HYP Dwelltime_units',
                 'Resolution_X',
                 'Resolution_X_units',
                 'Resolution_Y',
                 'Resolution_Y_units',
                 'Reference_Size_X',
                 'Reference_Size_Y',
                 'Voltage Calibration Range_X',
                 'Voltage Calibration Range_X_units',
                 'Voltage Calibration Range_Y',
                 'Voltage Calibration Range_Y_units',
                 'Start_X',
                 'Size_X',
                 'Start_Y',
                 'Size_Y',
                 'Rotate',
                 'Rotate_units',
                 ]

def test_load_profile():
    #Signal loading
    fname = os.path.join(MY_PATH, "sur_data",
                         "test_profile.pro")
    s = load(fname)

    #Verifying signal shape and axes dimensions, navigation (not data themselves)
    assert s.data.shape == (128,)
    assert s.data.dtype == np.dtype(float)
    np.testing.assert_allclose(s.axes_manager[0].scale,8.252197e-05)
    np.testing.assert_allclose(s.axes_manager[0].offset,0.0)
    assert s.axes_manager[0].name == 'Width'
    assert s.axes_manager[0].units == 'mm'
    assert s.axes_manager[0].size == 128
    assert s.axes_manager[0].navigate == False

    #Metadata verification
    md = s.metadata
    assert md.Signal.quantity == 'CL Intensity (a.u.)'

    #Original metadata. We verify that the correct structure is given
    #and the right headers but not the values
    omd = s.original_metadata
    assert list(omd.as_dictionary().keys()) == ['Object_0_Channel_0']
    assert list(omd.Object_0_Channel_0.as_dictionary().keys()) == ['Header']
    assert list(omd.Object_0_Channel_0.Header.as_dictionary().keys()) \
                                                            == header_keys

def test_load_RGB():
    fname = os.path.join(MY_PATH, "sur_data",
                         "test_RGB.sur")
    s = load(fname)
    assert s.data.shape == (200, 200)
    assert s.data.dtype == np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])

    np.testing.assert_allclose(s.axes_manager[0].scale,0.35277777)
    np.testing.assert_allclose(s.axes_manager[0].offset,208.8444519)
    np.testing.assert_allclose(s.axes_manager[1].scale,0.35277777)
    np.testing.assert_allclose(s.axes_manager[1].offset,210.608337)
    assert s.axes_manager[0].name == 'X'
    assert s.axes_manager[0].units == 'mm'
    assert s.axes_manager[1].name == 'Y'
    assert s.axes_manager[1].units == 'mm'
    assert s.axes_manager[0].size == 200
    assert s.axes_manager[0].navigate == False
    assert s.axes_manager[1].size == 200
    assert s.axes_manager[1].navigate == False

    md = s.metadata
    assert md.Signal.quantity == 'Z'

    omd = s.original_metadata
    assert list(omd.as_dictionary().keys()) == ['Object_0_Channel_0',
                                                'Object_0_Channel_1',
                                                'Object_0_Channel_2']
    assert list(omd.Object_0_Channel_0.as_dictionary().keys()) == ['Header']
    assert list(omd.Object_0_Channel_0.Header.as_dictionary().keys()) == header_keys

def test_load_spectra():
    fname = os.path.join(MY_PATH, "sur_data",
                         "test_spectra.pro")
    s = load(fname)

    assert s.data.shape == (65, 512)
    assert s.data.dtype == np.dtype('float64')

    md = s.metadata
    assert md.Signal.quantity == 'CL Intensity (a.u.)'
    np.testing.assert_allclose(s.axes_manager[0].scale,0.00011458775406936184)
    np.testing.assert_allclose(s.axes_manager[0].offset,0.0)
    np.testing.assert_allclose(s.axes_manager[1].scale,1.084000246009964e-06)
    np.testing.assert_allclose(s.axes_manager[1].offset,0.00017284281784668565)
    assert s.axes_manager[0].name == 'Spectrum positi'
    assert s.axes_manager[0].units == 'mm'
    assert s.axes_manager[1].name == 'Wavelength'
    assert s.axes_manager[1].units == 'mm'
    assert s.axes_manager[0].size == 65
    assert s.axes_manager[0].navigate == True
    assert s.axes_manager[1].size == 512
    assert s.axes_manager[1].navigate == False

    omd = s.original_metadata
    assert list(omd.as_dictionary().keys()) == ['Object_0_Channel_0',]
    assert list(omd.Object_0_Channel_0.as_dictionary().keys()) == ['Header']
    assert list(omd.Object_0_Channel_0.Header.as_dictionary().keys()) \
                                                                == header_keys

def test_load_spectral_map_compressed():
    fname = os.path.join(MY_PATH, "sur_data",
                         "test_spectral_map_compressed.sur")
    s = load(fname)

    assert s.data.shape == (12, 10, 281)
    assert s.data.dtype == np.dtype('float64')

    md = s.metadata
    assert md.Signal.quantity == 'CL Intensity (a.u.)'
    np.testing.assert_allclose(s.axes_manager[0].scale,8.252198e-05)
    np.testing.assert_allclose(s.axes_manager[0].offset,0.005694016348570585)
    np.testing.assert_allclose(s.axes_manager[1].scale,8.252198e-05)
    np.testing.assert_allclose(s.axes_manager[1].offset,0.0054464503191411495)
    np.testing.assert_allclose(s.axes_manager[2].scale,1.084000246009964e-06)
    np.testing.assert_allclose(s.axes_manager[2].offset,0.00034411484375596046)
    assert s.axes_manager[0].name == 'Width'
    assert s.axes_manager[0].units == 'mm'
    assert s.axes_manager[1].name == 'Height'
    assert s.axes_manager[1].units == 'mm'
    assert s.axes_manager[2].name == 'Wavelength'
    assert s.axes_manager[2].units == 'mm'
    assert s.axes_manager[0].size == 10
    assert s.axes_manager[0].navigate == True
    assert s.axes_manager[1].size == 12
    assert s.axes_manager[1].navigate == True
    assert s.axes_manager[2].size == 281
    assert s.axes_manager[2].navigate == False

    omd = s.original_metadata
    assert list(omd.as_dictionary().keys()) == ['Object_0_Channel_0',]
    assert list(omd.Object_0_Channel_0.as_dictionary().keys()) \
                                                            == ['Header','Parsed']
    assert list(omd.Object_0_Channel_0.Header.as_dictionary().keys()) \
                                                            == header_keys

    assert list(omd.Object_0_Channel_0.Parsed.as_dictionary().keys()) \
                                                            == atto_head_keys

    assert list(omd.Object_0_Channel_0.Parsed.WAFER.as_dictionary().keys()) \
                                                            == atto_wafer_keys

    assert list(omd.Object_0_Channel_0.Parsed.SCAN.as_dictionary().keys()) \
                                                            == atto_scan_keys

def test_load_spectral_map():
    fname = os.path.join(MY_PATH, "sur_data",
                         "test_spectral_map.sur")
    s = load(fname)

    assert s.data.shape == (12, 10, 310)
    assert s.data.dtype == np.dtype('float64')

    md = s.metadata
    assert md.Signal.quantity == 'CL Intensity (a.u.)'
    np.testing.assert_allclose(s.axes_manager[0].scale,8.252197585534304e-05)
    np.testing.assert_allclose(s.axes_manager[0].offset,0.00701436772942543)
    np.testing.assert_allclose(s.axes_manager[1].scale,8.252197585534304e-05)
    np.testing.assert_allclose(s.axes_manager[1].offset,0.003053313121199608)
    np.testing.assert_allclose(s.axes_manager[2].scale,1.084000246009964e-6)
    np.testing.assert_allclose(s.axes_manager[2].offset,0.0003332748601678759)
    assert s.axes_manager[0].name == 'Width'
    assert s.axes_manager[0].units == 'mm'
    assert s.axes_manager[1].name == 'Height'
    assert s.axes_manager[1].units == 'mm'
    assert s.axes_manager[2].name == 'Wavelength'
    assert s.axes_manager[2].units == 'mm'
    assert s.axes_manager[0].size == 10
    assert s.axes_manager[0].navigate == True
    assert s.axes_manager[1].size == 12
    assert s.axes_manager[1].navigate == True
    assert s.axes_manager[2].size == 310
    assert s.axes_manager[2].navigate == False


    omd = s.original_metadata
    assert list(omd.as_dictionary().keys()) == ['Object_0_Channel_0',]
    assert list(omd.Object_0_Channel_0.as_dictionary().keys()) == ['Header','Parsed']
    assert list(omd.Object_0_Channel_0.Header.as_dictionary().keys()) \
                                                               == header_keys

    assert list(omd.Object_0_Channel_0.Parsed.as_dictionary().keys()) \
                                                            == atto_head_keys

    assert list(omd.Object_0_Channel_0.Parsed.WAFER.as_dictionary().keys()) \
                                                            == atto_wafer_keys

    assert list(omd.Object_0_Channel_0.Parsed.SCAN.as_dictionary().keys())  \
                                                            == atto_scan_keys

def test_load_spectrum_compressed():
    fname = os.path.join(MY_PATH, "sur_data",
                         "test_spectrum_compressed.pro")
    s = load(fname)
    md = s.metadata
    assert md.Signal.quantity == 'CL Intensity (a.u.)'
    assert s.data.shape == (512,)
    #np.testing.assert_allclose(s.axes_manager[0].scale,1.0)
    #np.testing.assert_allclose(s.axes_manager[0].offset,0.0)
    np.testing.assert_allclose(s.axes_manager[0].scale,1.084000246009964e-6)
    np.testing.assert_allclose(s.axes_manager[0].offset,172.84281784668565e-6)

    #assert s.axes_manager[0].name == 'T'
    #assert s.axes_manager[0].units == ''
    assert s.axes_manager[0].name == 'Wavelength'
    assert s.axes_manager[0].units == 'mm'
    #assert s.axes_manager[0].size == 1
    #assert s.axes_manager[0].navigate == True
    assert s.axes_manager[0].size == 512
    assert s.axes_manager[0].navigate == False

    omd = s.original_metadata
    assert list(omd.as_dictionary().keys()) == ['Object_0_Channel_0']
    assert list(omd.Object_0_Channel_0.as_dictionary().keys()) == ['Header']
    assert list(omd.Object_0_Channel_0.Header.as_dictionary().keys()) \
                                                                == header_keys

def test_load_spectrum():
    fname = os.path.join(MY_PATH, "sur_data",
                         "test_spectrum.pro")
    s = load(fname)
    assert s.data.shape == (512,)

    md = s.metadata
    assert md.Signal.quantity == 'CL Intensity (a.u.)'
    #np.testing.assert_allclose(s.axes_manager[0].scale,1.0)
    #np.testing.assert_allclose(s.axes_manager[0].offset,0.0)
    np.testing.assert_allclose(s.axes_manager[0].scale,1.084000246009964e-6)
    np.testing.assert_allclose(s.axes_manager[0].offset,172.84281784668565e-6)

    #assert s.axes_manager[0].name == 'T'
    #assert s.axes_manager[0].units == ''
    assert s.axes_manager[0].name == 'Wavelength'
    assert s.axes_manager[0].units == 'mm'
    #assert s.axes_manager[0].size == 1
    #assert s.axes_manager[0].navigate == True
    assert s.axes_manager[0].size == 512
    assert s.axes_manager[0].navigate == False

    omd = s.original_metadata
    assert list(omd.as_dictionary().keys()) == ['Object_0_Channel_0']
    assert list(omd.Object_0_Channel_0.as_dictionary().keys()) == ['Header']
    assert list(omd.Object_0_Channel_0.Header.as_dictionary().keys()) \
                                                                == header_keys

def test_load_surface():
    fname = os.path.join(MY_PATH, "sur_data",
                         "test_surface.sur")
    s = load(fname)
    md = s.metadata
    assert md.Signal.quantity == 'CL Intensity (a.u.)'
    assert s.data.shape == (128,128)
    np.testing.assert_allclose(s.axes_manager[0].scale,8.252198e-05)
    np.testing.assert_allclose(s.axes_manager[0].offset,0.0)
    np.testing.assert_allclose(s.axes_manager[1].scale,8.252198e-05)
    np.testing.assert_allclose(s.axes_manager[1].offset,0.0)

    assert s.axes_manager[0].name == 'Width'
    assert s.axes_manager[0].units == 'mm'
    assert s.axes_manager[1].name == 'Height'
    assert s.axes_manager[1].units == 'mm'
    assert s.axes_manager[0].size == 128
    assert s.axes_manager[0].navigate == False
    assert s.axes_manager[1].size == 128
    assert s.axes_manager[1].navigate == False

    omd = s.original_metadata
    assert list(omd.as_dictionary().keys()) == ['Object_0_Channel_0']
    assert list(omd.Object_0_Channel_0.as_dictionary().keys()) == ['Header']
    assert list(omd.Object_0_Channel_0.Header.as_dictionary().keys()) \
                                                            == header_keys
