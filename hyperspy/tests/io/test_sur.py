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
from numpy.testing import assert_allclose
import pytest

from hyperspy.tests.io.generate_dm_testing_files import (dm3_data_types,
                                                         dm4_data_types)
from hyperspy.io import load
from hyperspy.io_plugins.digital_micrograph import DigitalMicrographReader, ImageObject
from hyperspy.signals import Signal1D, Signal2D
import json

MY_PATH = os.path.dirname(__file__)

def test_load_profile():
    fname = os.path.join(MY_PATH, "sur_data",
                         "test_profile.pro")
    s = load(fname)
    md = s.metadata
    assert md.Signal.quantity == 'CL Intensity a.u.'
    assert s.data.shape == (128,)
    assert_allclose(s.axes_manager[0].scale,8.252197e-05)
    assert_allclose(s.axes_manager[0].offset,0.0)
    assert s.axes_manager[0].name == 'Width'
    assert s.axes_manager[0].units == 'mm'

    omd = s.original_metadata
    assert list(omd.as_dictionary().keys()) == ['Object_0_Channel_0']
    assert list(omd.Object_0_Channel_0.as_dictionary().keys()) == ['Header']
    assert list(omd.Object_0_Channel_0.Header.as_dictionary().keys()) == \
        ['H01_Signature',
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
