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

# To generate a new reference numpy file
generate_numpy_file = False
def create_numpy_file(filename, obj):
    gen = (obj._data_dictionary[key] for key in obj.logged_quantity_name_list)
    data = np.vstack(gen)   
    np.save(filename, data.T)

"""
Protochips gas cell
"""
def test_read_protochips_gas_cell():
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

    def test_read_original_metadata(self):
        om = self.s_list[0].original_metadata.Protochips_header
        nt.assert_equal(om.Calibration_file, 'The calibration files are saved'\
                        ' in metadata.General.notes')
        nt.assert_equal(om.Holder_Pressure_units, 'Torr')
        nt.assert_equal(om.Holder_Temperature_units, 'Degrees C')
        nt.assert_equal(om.Start_time, datetime_gas_cell())
        nt.assert_equal(om.Holder_Pressure_units, 'Torr')
        nt.assert_equal(om.Tank1_Pressure_units, 'Torr')
        nt.assert_equal(om.Tank2_Pressure_units, 'Torr')
        nt.assert_equal(om.Vacuum_Tank_Pressure_units, 'Torr')
        nt.assert_equal(om.Time_units, 'Milliseconds')
        nt.assert_equal(om.User, 'eric')
        
            
class test_ProtochipsGasCellCSVReader():
    def setUp(self):
        self.filename = os.path.join(dirpath, 'protochips_gas_cell.csv')
        self.pgc = ProtochipsCSV(self.filename)
        if generate_numpy_file:
            create_numpy_file(self.filename.replace('.csv', '.npy'), self.pgc)
            
    def test_read_column_name(self):
        nt.assert_equal(self.pgc.column_name, ['Time', 'Notes',
                                               'Holder Temperature',
                                               'Holder Pressure',
                                               'Tank1 Pressure',
                                               'Tank2 Pressure',
                                               'Vacuum Tank Pressure'])
        
    def test_read_start_datetime(self):
        nt.assert_equal(self.pgc.start_datetime, datetime_gas_cell())
         
    def test_read_data(self):
        gen = (self.pgc._data_dictionary[key] for key in self.pgc.logged_quantity_name_list)
        data = np.vstack(gen)
        expected_data = np.load(os.path.join(dirpath, 'protochips_gas_cell.npy'))
        np.testing.assert_allclose(data.T, expected_data)
        
    def test_read_metadata_header(self):
        nt.assert_equal(self.pgc.time_units, 'Milliseconds')
        nt.assert_equal(self.pgc.time_units, 'Milliseconds')
        nt.assert_equal(self.pgc.temperature_units, 'Degrees C')
        nt.assert_equal(self.pgc.pressure_units, 'Torr')
        nt.assert_equal(self.pgc.current_units, 'Amps')
        nt.assert_equal(self.pgc.voltage_units, 'Volts')
        nt.assert_equal(self.pgc.resistance_units, 'Ohms')
        nt.assert_equal(self.pgc.user, 'eric')
        
"""
Protochips electrical
"""
def test_read_protochips_electrical():
    filename = os.path.join(dirpath, 'protochips_electrical.csv')
    s = hs.load(filename)
    nt.assert_equal(len(s), 6)
    nt.assert_equal(s[0].metadata.General.title, 'Channel A Current (Amps)')
    nt.assert_equal(s[0].metadata.Signal.signal_type, 'Channel A Current')
    nt.assert_equal(s[1].metadata.General.title, 'Channel A Voltage (Volts)')
    nt.assert_equal(s[1].metadata.Signal.signal_type, 'Channel A Voltage')
    nt.assert_equal(s[2].metadata.General.title, 'Channel A Resistance (Ohms)')
    nt.assert_equal(s[2].metadata.Signal.signal_type, 'Channel A Resistance')
    nt.assert_equal(s[3].metadata.General.title, 'Channel B Current (Amps)')
    nt.assert_equal(s[3].metadata.Signal.signal_type, 'Channel B Current')
    nt.assert_equal(s[4].metadata.General.title, 'Channel B Voltage (Volts)')
    nt.assert_equal(s[4].metadata.Signal.signal_type, 'Channel B Voltage')        
    nt.assert_equal(s[5].metadata.General.title, 'Channel B Resistance (Ohms)')
    nt.assert_equal(s[5].metadata.Signal.signal_type, 'Channel B Resistance') 
    
class test_ProtochipsElectricalCSVReader():
    def setUp(self):
        self.filename = os.path.join(dirpath, 'protochips_electrical.csv')
        self.pa = ProtochipsCSV(self.filename)
        if generate_numpy_file:
            create_numpy_file(self.filename.replace('.csv', '.npy'), self.pa)
            
    def test_read_column_name(self):
        nt.assert_equal(self.pa.column_name, ['Time', 'Notes',
                                              'Channel A Current',
                                              'Channel A Voltage',
                                              'Channel A Resistance',
                                              'Channel B Current',
                                              'Channel B Voltage',
                                              'Channel B Resistance'])
                                       
    def test_read_start_datetime(self):
        dt = datetime.strptime('2014.10.0816:26:51.738', "%Y.%m.%d%H:%M:%S.%f")
        nt.assert_equal(self.pa.start_datetime, dt)
        
    def test_read_data(self):
        gen = (self.pa._data_dictionary[key] for key in self.pa.logged_quantity_name_list)
        data = np.vstack(gen)
        expected_data = np.load(os.path.join(dirpath, 'protochips_electrical.npy'))
        np.testing.assert_allclose(data.T, expected_data)

"""
Protochips thermal
"""     
def test_read_protochips_thermal():
    filename = os.path.join(dirpath, 'protochips_thermal.csv')
    s = hs.load(filename)
    nt.assert_equal(s.metadata.General.title, 'Channel A Temperature (Degrees C)')
    nt.assert_equal(s.metadata.Signal.signal_type, 'Channel A Temperature')
    
class test_ProtochipsThermallCSVReader():
    def setUp(self):
        self.filename = os.path.join(dirpath, 'protochips_thermal.csv')
        self.pt = ProtochipsCSV(self.filename)
        if generate_numpy_file:
            create_numpy_file(self.filename.replace('.csv', '.npy'), self.pt)
        
    def test_read_column_name(self):
        assert self.pt.column_name == ['Time', 'Notes',
                                       'Channel A Temperature']
                                       
    def test_read_start_datetime(self):
        dt = datetime.strptime('2014.12.0317:15:37.192', "%Y.%m.%d%H:%M:%S.%f")
        assert self.pt.start_datetime == dt

    def test_read_data(self):
        gen = (self.pt._data_dictionary[key] for key in self.pt.logged_quantity_name_list)
        data = np.vstack(gen)
        expected_data = np.load(os.path.join(dirpath, 'protochips_thermal.npy'))
        np.testing.assert_allclose(data.T, expected_data)

"""
Protochips electrothermal
"""
def test_read_protochips_electrothermal():
    filename = os.path.join(dirpath, 'protochips_electrothermal.csv')
    s = hs.load(filename)
    nt.assert_equal(len(s), 4)
    nt.assert_equal(s[0].metadata.General.title, 'Channel A Temperature (Degrees C)')
    nt.assert_equal(s[0].metadata.Signal.signal_type, 'Channel A Temperature')
    nt.assert_equal(s[1].metadata.General.title, 'Channel B Current (Amps)')
    nt.assert_equal(s[1].metadata.Signal.signal_type, 'Channel B Current')
    nt.assert_equal(s[2].metadata.General.title, 'Channel B Voltage (Volts)')
    nt.assert_equal(s[2].metadata.Signal.signal_type, 'Channel B Voltage')        
    nt.assert_equal(s[3].metadata.General.title, 'Channel B Resistance (Ohms)')
    nt.assert_equal(s[3].metadata.Signal.signal_type, 'Channel B Resistance') 
    
class test_ProtochipsElectrothermalCSVReader():
    def setUp(self):
        self.filename = os.path.join(dirpath, 'protochips_electrothermal.csv')
        self.pet = ProtochipsCSV(self.filename)
        if generate_numpy_file:
            create_numpy_file(self.filename.replace('.csv', '.npy'), self.pet)            
        
    def test_read_column_name(self):
        nt.assert_equal(self.pet.column_name, ['Time', 'Notes',
                                               'Channel A Temperature',
                                               'Channel B Current',
                                               'Channel B Voltage',
                                               'Channel B Resistance'])
                                       
    def test_read_start_datetime(self):
        dt = datetime.strptime('2014.11.0514:42:51.369', "%Y.%m.%d%H:%M:%S.%f")
        nt.assert_equal(self.pet.start_datetime, dt)
        
    def test_read_data(self):
        gen = (self.pet._data_dictionary[key] for key in self.pet.logged_quantity_name_list)
        data = np.vstack(gen)
        expected_data = np.load(os.path.join(dirpath,
                                             self.filename.replace('.csv', '.npy')))
        np.testing.assert_allclose(data.T, expected_data)
