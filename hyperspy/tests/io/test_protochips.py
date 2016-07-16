# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 21:26:01 2015

@author: eric
"""
import os
import numpy as np
from datetime import datetime
import nose.tools as nt

import hyperspy.api as hs
from hyperspy.io_plugins.protochips import ProtochipsCSV

testdirpath = os.path.dirname(__file__)
dirpath = os.path.join(testdirpath, 'protochips_data')

def test_read():
    filename = os.path.join(dirpath, 'protochips_gas_cell.csv')
    s = hs.load(filename)
    nt.assert_equal(len(s), 5)
    nt.assert_equal(s[0].metadata.General.title, 'Holder Temperature (Degrees C)')
    nt.assert_equal(s[0].metadata.Signal.signal_type, 'Holder Temperature')
    nt.assert_equal(s[1].metadata.General.title, 'Holder Pressure (Torr)')
    nt.assert_equal(s[1].metadata.Signal.signal_type, 'Holder Pressure')
    nt.assert_equal(s[2].metadata.General.title, 'Tank1 Pressure (Torr)')
    nt.assert_equal(s[2].metadata.Signal.signal_type, 'Tank1 Pressure')
    nt.assert_equal(s[3].metadata.General.title, 'Tank2 Pressure (Torr)')
    nt.assert_equal(s[3].metadata.Signal.signal_type, 'Tank2 Pressure')
    nt.assert_equal(s[4].metadata.General.title, 'Vacuum Tank Pressure (Torr)')
    nt.assert_equal(s[4].metadata.Signal.signal_type, 'Vacuum Tank Pressure')

def datetime_gas_cell():
    date = '2014.12.15'
    time = '19:07:04.165'
    return datetime.strptime(date+time, "%Y.%m.%d%H:%M:%S.%f")
    
class test_ProtochipsGasCellCSV():
    def setUp(self):
        filename = os.path.join(dirpath, 'protochips_gas_cell.csv')
        self.s_list = hs.load(filename)
        
    def test_read_metadata(self):
        for s in self.s_list:
            nt.assert_equal(s.metadata.General.start_time, datetime_gas_cell())
            nt.assert_equal(s.axes_manager[0].units, 's')
            nt.assert_almost_equal(s.axes_manager[0].scale, 0.25994, places=5)
            nt.assert_equal(s.axes_manager[0].offset, 0)     
    
class test_ProtochipsGasCellCSVReader():
    def setUp(self):
        filename = os.path.join(dirpath, 'protochips_gas_cell.csv')
        self.pgc = ProtochipsCSV(filename)
        
    def test_read_column_name(self):
        assert self.pgc.column_name == ['Time', ' Notes', 'Holder Temperature',
                                        'Holder Pressure', 'Tank1 Pressure',
                                        'Tank2 Pressure', 'Vacuum Tank Pressure']
        
    def test_read_start_datetime(self):
        assert self.pgc.start_datetime == datetime_gas_cell()
         
    def test_read_data(self):
        gen = (self.pgc._data_dictionary[key] for key in self.pgc.logged_quantity_name_list)
        data = np.vstack(gen)
        expected_data = np.load(os.path.join(dirpath, 'protochips_data.npy'))
        assert (data == expected_data).all()
        
    def test_read_metadata_header(self):
        assert self.pgc.time_unit == 'Milliseconds'
        assert self.pgc.temperature_unit == 'Degrees C'
        assert self.pgc.pressure_unit == 'Torr'
        assert self.pgc.current_unit == 'Amps'
        assert self.pgc.voltage_unit == 'Volts'
        assert self.pgc.resistance_unit == 'Ohms'
        assert self.pgc.user == 'eric'