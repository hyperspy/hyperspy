# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
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

import nose.tools as nt
import traits.api as t
import warnings

from hyperspy.axes import DataAxis, AxesManager, _get_appropriate_scale_units,\
                             _get_convert_units

def test_units_not_supported_by_pint_raising_warning():
    with warnings.catch_warnings(record=True) as w:
        scale, units = _get_convert_units(1.0, 'micron', 'nm')
        assert len(w) == 1 # catch one warning (there is no warning filtered)
        assert issubclass(w[-1].category, UserWarning)
    nt.assert_almost_equal(scale, 1.0, places=5)
    nt.assert_equal(units, 'micron')

def test_convert_to_units():
    scale, units = _get_convert_units(1.0, t.Undefined, 'nm')
    nt.assert_almost_equal(scale, 1.0, places=5)
    nt.assert_equal(units, t.Undefined)

    scale, units = _get_convert_units(0.5, 'µm', 'nm')
    nt.assert_almost_equal(scale, 500, places=5)
    nt.assert_equal(units, 'nm')

    scale, units = _get_convert_units(5, 'µm', 'cm')
    nt.assert_almost_equal(scale, 0.0005, places=5)
    nt.assert_equal(units, 'cm')
    
    scale, units = _get_convert_units(5, '1/µm', '1/nm')
    nt.assert_almost_equal(scale, 0.005, places=5)
    nt.assert_equal(units, '1/nm')

    scale, units = _get_convert_units(5, 'eV', 'keV')
    nt.assert_almost_equal(scale, 0.005, places=5)
    nt.assert_equal(units, 'keV')       
    
def test_get_appropriate_scale_unit():
    ##### Imaging #####
    # typical setting for high resolution image
    scale, unit = _get_appropriate_scale_units(12E-12, 'm', 2048)
    nt.assert_almost_equal(scale, 0.012, places=5)
    nt.assert_equal(unit, 'nm')

    # typical setting for nm resolution image
    scale, unit = _get_appropriate_scale_units(0.5E-9, 'm', 1024)
    nt.assert_almost_equal(scale, 0.5, places=5)
    nt.assert_equal(unit, 'nm')  

    # typical setting for nm resolution image
    scale, unit = _get_appropriate_scale_units(0.5E-9, 'm', 2048)
    nt.assert_almost_equal(scale, 0.5, places=5)
    nt.assert_equal(unit, 'nm')  
    
    # typical setting for nm resolution image
    scale, unit = _get_appropriate_scale_units(1E-9, 'm', 1024)
    nt.assert_almost_equal(scale, 1.0, places=5)
    nt.assert_equal(unit, 'nm')    

    # typical setting for µm resolution image
    scale, unit = _get_appropriate_scale_units(1E-6, 'm', 1024)
    nt.assert_almost_equal(scale, 1.0, places=5)
    nt.assert_equal(unit, 'µm') 

    # typical setting for a few tens of µm resolution image
    scale, unit = _get_appropriate_scale_units(10E-6, 'm', 1024)
    nt.assert_almost_equal(scale, 0.01, places=5)
    nt.assert_equal(unit, 'mm')

    ##### Diffraction #####    
    # typical TEM diffraction
    scale, unit = _get_appropriate_scale_units(0.1E9, '1/m', 1024)
    nt.assert_almost_equal(scale, 0.1, places=5)
    nt.assert_equal(unit, '1/nm')

    # typical TEM diffraction
    scale, unit = _get_appropriate_scale_units(0.1E9, '1/m', 256)
    nt.assert_almost_equal(scale, 0.1, places=5)
    nt.assert_equal(unit, '1/nm')
    
    # high camera length diffraction
    scale, unit = _get_appropriate_scale_units(1E6, '1/m', 4096)
    nt.assert_almost_equal(scale, 1.0, places=5)
    nt.assert_equal(unit, '1/µm')
    
    # typical EDS resolution
    scale, unit = _get_appropriate_scale_units(5E3, 'eV', 4096)
    nt.assert_almost_equal(scale, 5, places=5)
    nt.assert_equal(unit, 'keV')

    ##### Spectroscopy #####    
    # typical EELS resolution
    scale, unit = _get_appropriate_scale_units(0.2, 'eV', 2048)
    nt.assert_almost_equal(scale, 0.2, places=5)
    nt.assert_equal(unit, 'eV')

    # typical EELS resolution
    scale, unit = _get_appropriate_scale_units(1.0, 'eV', 2048)
    nt.assert_almost_equal(scale, 1.0, places=5)
    nt.assert_equal(unit, 'eV')
    
    # typical high resolution EELS resolution
    scale, unit = _get_appropriate_scale_units(0.05, 'eV', 100)
    nt.assert_almost_equal(scale, 0.05, places=5)
    nt.assert_equal(unit, 'eV')

    # typical high resolution EELS resolution
    scale, unit = _get_appropriate_scale_units(0.001, 'eV', 100)
    nt.assert_almost_equal(scale, 1.0, places=5)
    nt.assert_equal(unit, 'meV')
    
    # typical high resolution EELS resolution
    scale, unit = _get_appropriate_scale_units(0.001, 'eV', 2048)
    nt.assert_almost_equal(scale, 1.0, places=5)
    nt.assert_equal(unit, 'meV')
    
class TestDataAxis:

    def setUp(self):
        self.axis = DataAxis(size=2048, scale=12E-12, units='m')
        
    def test_convert_to_appropriate_scale_units(self):
        self.axis.convert_to_units(units=None)
        nt.assert_almost_equal(self.axis.scale, 0.012, places=5)
        nt.assert_equal(self.axis.units, 'nm')

    def test_convert_to_units(self):
        self.axis.convert_to_units(units='µm')
        nt.assert_almost_equal(self.axis.scale, 12E-6, places=8)
        nt.assert_equal(self.axis.units, 'µm')

    def test_units_not_supported_by_pint_no_warning_raised(self):
        # Suppose to do nothing, not raising a warning, not converting scale
        self.axis.units = 'micron'
        with warnings.catch_warnings(record=True) as w:
            self.axis.convert_to_units('m', filterwarning_action="ignore")
            assert len(w) == 0 # not catch warnings, they are ignored
        nt.assert_almost_equal(self.axis.scale, 12E-12, places=15)
        nt.assert_equal(self.axis.units, 'micron')

    def test_units_not_supported_by_pint_warning_raised(self):
        # raising a warning, not converting scale
        self.axis.units = 'micron'
        with warnings.catch_warnings(record=True) as w:
            self.axis.convert_to_units('m')
            assert len(w) == 1 # catch warnings
        nt.assert_almost_equal(self.axis.scale, 12E-12, places=15)
        nt.assert_equal(self.axis.units, 'micron')
                
class TestAxesManager:

    def setup(self):
        self.axes_list = [
            {'name': 'x',
             'navigate': True,
             'offset': 0.0,
             'scale': 1E-6,
             'size': 1024,
             'units': 'm'},
            {'name': 'y',
             'navigate': True,
             'offset': 0.0,
             'scale': 0.5E-9,
             'size': 1024,
             'units': 'm'},
            {'name': 'energy',
             'navigate': False,
             'offset': 0.0,
             'scale': 5.0,
             'size': 4096,
             'units': 'eV'}]

        self.am = AxesManager(self.axes_list)

    def test_appropriate_scale_unit(self):
        self.am.convert_units()
        nt.assert_almost_equal(self.am['x'].scale, 1.0, places=5)
        nt.assert_equal(self.am['x'].units, 'µm')
        nt.assert_almost_equal(self.am['y'].scale, 0.5, places=5)
        nt.assert_equal(self.am['y'].units, 'nm')
        nt.assert_almost_equal(self.am['energy'].scale, 0.005, places=5)
        nt.assert_equal(self.am['energy'].units, 'keV')
        
    def test_convert_to_navigation_units(self):
        self.am.convert_units(axes='navigation', units='µm')
        nt.assert_almost_equal(self.am['x'].scale, 1.0, places=5)
        nt.assert_equal(self.am['x'].units, 'µm')
        nt.assert_almost_equal(self.am['y'].scale, 0.5E-3, places=5)
        nt.assert_equal(self.am['y'].units, 'µm')
        nt.assert_almost_equal(self.am['energy'].scale,
                               self.axes_list[-1]['scale'], places=5)
        nt.assert_equal(self.am['energy'].units, self.axes_list[-1]['units'])

    def test_convert_to_navigation_units_list(self):
        self.am.convert_units(axes='navigation', units=['µm', 'nm'])
        nt.assert_almost_equal(self.am['x'].scale, 1000.0, places=5)
        nt.assert_equal(self.am['x'].units, 'nm')
        nt.assert_almost_equal(self.am['y'].scale, 0.5E-3, places=5)
        nt.assert_equal(self.am['y'].units, 'µm')
        nt.assert_almost_equal(self.am['energy'].scale,
                               self.axes_list[-1]['scale'], places=5)
        nt.assert_equal(self.am['energy'].units, self.axes_list[-1]['units'])
        
    def test_convert_to_signal_units(self):
        self.am.convert_units(axes='signal', units='keV')
        nt.assert_almost_equal(self.am['x'].scale, self.axes_list[0]['scale'],
                               places=5)
        nt.assert_equal(self.am['x'].units, self.axes_list[0]['units'])
        nt.assert_almost_equal(self.am['y'].scale, self.axes_list[1]['scale'],
                               places=5)
        nt.assert_equal(self.am['y'].units, self.axes_list[1]['units'])
        nt.assert_almost_equal(self.am['energy'].scale, 0.005, places=5)
        nt.assert_equal(self.am['energy'].units, 'keV')
        
    def test_convert_to_units_list(self):
        self.am.convert_units(units=['µm', 'nm', 'meV'])
        nt.assert_almost_equal(self.am['x'].scale, 1000.0, places=5)
        nt.assert_equal(self.am['x'].units, 'nm')
        nt.assert_almost_equal(self.am['y'].scale, 0.5E-3, places=5)
        nt.assert_equal(self.am['y'].units, 'µm')
        nt.assert_almost_equal(self.am['energy'].scale, 5E3, places=5)
        nt.assert_equal(self.am['energy'].units, 'meV')