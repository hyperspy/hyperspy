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

import numpy.testing as nt
import traits.api as t

from hyperspy.axes import DataAxis, AxesManager, UnitConversion
from hyperspy.misc.test_utils import assert_warns


class TestUnitConversion:

    def setup_method(self, method):
        self.uc = UnitConversion()
        self._set_units_scale_size(units='m', scale=1E-3)

    def _set_units_scale_size(self, units=t.Undefined, scale=1.0, size=100,
                              offset=0.0):
        self.uc.units = units
        self.uc.scale = scale
        self.uc.offset = offset
        self.uc.size = size

    def test_units_setter(self):
        self.uc.units = ' m'
        assert self.uc.units == 'm'
        self.uc.units = 'um'
        assert self.uc.units == 'µm'
        self.uc.units = ' µm'
        assert self.uc.units == 'µm'
        self.uc.units = ' km'
        assert self.uc.units == 'km'

    def test_ignore_conversion(self):
        assert self.uc._ignore_conversion(t.Undefined)
        with assert_warns(
                message="not supported for conversion.",
                category=UserWarning):
            assert self.uc._ignore_conversion('unit_not_supported')
        assert not self.uc._ignore_conversion('m')

    def test_converted_compact_scale_units(self):
        self.uc.units = 'micron'
        with assert_warns(
                message="not supported for conversion.",
                category=UserWarning):
            self.uc._convert_compact_scale_units()
        assert self.uc.units == 'micron'
        nt.assert_almost_equal(self.uc.scale, 1.0E-3)

    def test_convert_to_units(self):
        self._set_units_scale_size(t.Undefined, 1.0)
        self.uc._convert_scale_units('nm')
        assert self.uc.units == t.Undefined
        nt.assert_almost_equal(self.uc.scale, 1.0)

        self._set_units_scale_size('m', 1.0E-3)
        self.uc._convert_scale_units('µm')
        assert self.uc.units == 'µm'
        nt.assert_almost_equal(self.uc.scale, 1E3)

        self._set_units_scale_size('µm', 0.5)
        self.uc._convert_scale_units('nm')
        assert self.uc.units == 'nm'
        nt.assert_almost_equal(self.uc.scale, 500)

        self._set_units_scale_size('µm', 5)
        self.uc._convert_scale_units('cm')
        assert self.uc.units == 'cm'
        nt.assert_almost_equal(self.uc.scale, 0.0005)

        self._set_units_scale_size('1/µm', 5)
        self.uc._convert_scale_units('1/nm')
        assert self.uc.units == '1/nm'
        nt.assert_almost_equal(self.uc.scale, 0.005)

        self._set_units_scale_size('eV', 5)
        self.uc._convert_scale_units('keV')
        assert self.uc.units == 'keV'
        nt.assert_almost_equal(self.uc.scale, 0.005)

    def test_get_compact_unit(self):
        ##### Imaging #####
        # typical setting for high resolution image
        self._set_units_scale_size('m', 12E-12, 2048, 2E-9)
        self.uc._convert_compact_scale_units()
        assert self.uc.units == 'nm'
        nt.assert_almost_equal(self.uc.scale, 0.012)
        nt.assert_almost_equal(self.uc.offset, 2.0)

        # typical setting for nm resolution image
        self._set_units_scale_size('m', 0.5E-9, 1024)
        self.uc._convert_compact_scale_units()
        assert self.uc.units == 'nm'
        nt.assert_almost_equal(self.uc.scale, 0.5)
        nt.assert_almost_equal(self.uc.offset, 0.0)

        ##### Diffraction #####
        # typical TEM diffraction
        self._set_units_scale_size('1/m', 0.1E9, 1024)
        self.uc._convert_compact_scale_units()
        assert self.uc.units == '1/nm'
        nt.assert_almost_equal(self.uc.scale, 0.1)

        # typical TEM diffraction
        self._set_units_scale_size('1/m', 0.01E9, 256)
        self.uc._convert_compact_scale_units()
        assert self.uc.units == '1/nm'
        nt.assert_almost_equal(self.uc.scale, 0.01)

        # high camera length diffraction
        self._set_units_scale_size('1/m', 0.1E9, 4096)
        self.uc._convert_compact_scale_units()
        assert self.uc.units == '1/nm'
        nt.assert_almost_equal(self.uc.scale, 0.1)

        # typical EDS resolution
        self._set_units_scale_size('eV', 50, 4096, 0.0)
        self.uc._convert_compact_scale_units()
        assert self.uc.units == 'keV'
        nt.assert_almost_equal(self.uc.scale, 0.05)
        nt.assert_almost_equal(self.uc.offset, 0.0)

        ##### Spectroscopy #####
        # typical EELS resolution
        self._set_units_scale_size('eV', 0.2, 2048, 200.0)
        self.uc._convert_compact_scale_units()
        assert self.uc.units == 'eV'
        nt.assert_almost_equal(self.uc.scale, 0.2)
        nt.assert_almost_equal(self.uc.offset, 200.0)

        # typical EELS resolution
        self._set_units_scale_size('eV', 1.0, 2048, 500.0)
        self.uc._convert_compact_scale_units()
        assert self.uc.units == 'keV'
        nt.assert_almost_equal(self.uc.scale, 0.001)
        nt.assert_almost_equal(self.uc.offset, 0.5)

        # typical high resolution EELS resolution
        self._set_units_scale_size('eV', 0.05, 100)
        self.uc._convert_compact_scale_units()
        assert self.uc.units == 'eV'
        assert self.uc.scale == 0.05


class TestDataAxis:

    def setup_method(self, method):
        self.axis = DataAxis(size=2048, scale=12E-12, units='m', offset=5E-9)

    def test_convert_to_compact_units(self):
        self.axis.convert_to_units(units=None)
        nt.assert_almost_equal(self.axis.scale, 0.012)
        assert self.axis.units == 'nm'
        nt.assert_almost_equal(self.axis.offset, 5.0)

    def test_convert_to_units(self):
        self.axis.convert_to_units(units='µm')
        nt.assert_almost_equal(self.axis.scale, 12E-6)
        assert self.axis.units == 'µm'
        nt.assert_almost_equal(self.axis.offset, 0.005)

    def test_units_not_supported_by_pint_no_warning_raised(self):
        # Suppose to do nothing, not raising a warning, not converting scale
        self.axis.units = 'micron'
        self.axis.convert_to_units('m', filterwarning_action="ignore")
        nt.assert_almost_equal(self.axis.scale, 12E-12)
        assert self.axis.units == 'micron'
        nt.assert_almost_equal(self.axis.offset, 5E-09)

    def test_units_not_supported_by_pint_warning_raised(self):
        # raising a warning, not converting scale
        self.axis.units = 'micron'
        with assert_warns(

                message="not supported for conversion.",
                category=UserWarning):
            self.axis.convert_to_units('m')
        nt.assert_almost_equal(self.axis.scale, 12E-12)
        assert self.axis.units == 'micron'


class TestAxesManager:

    def setup_method(self, method):
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

    def test_compact_unit(self):
        self.am.convert_units()
        assert self.am['x'].units == 'µm'
        nt.assert_almost_equal(self.am['x'].scale, 1.0)
        assert self.am['y'].units == 'nm'
        nt.assert_almost_equal(self.am['y'].scale, 0.5)
        assert self.am['energy'].units == 'keV'
        nt.assert_almost_equal(self.am['energy'].scale, 0.005)

    def test_convert_to_navigation_units(self):
        self.am.convert_units(axes='navigation', units='mm')
        nt.assert_almost_equal(self.am['x'].scale, 1E-3)
        assert self.am['x'].units == 'mm'
        nt.assert_almost_equal(self.am['y'].scale, 0.5E-6)
        assert self.am['y'].units == 'mm'
        nt.assert_almost_equal(self.am['energy'].scale,
                               self.axes_list[-1]['scale'])

    def test_convert_to_navigation_units_list(self):
        self.am.convert_units(axes='navigation', units=['mm', 'nm'])
        nt.assert_almost_equal(self.am['x'].scale, 1000.0)
        assert self.am['x'].units == 'nm'
        nt.assert_almost_equal(self.am['y'].scale, 0.5E-6)
        assert self.am['y'].units == 'mm'
        nt.assert_almost_equal(self.am['energy'].scale,
                               self.axes_list[-1]['scale'])

    def test_convert_to_signal_units(self):
        self.am.convert_units(axes='signal', units='keV')
        nt.assert_almost_equal(self.am['x'].scale, self.axes_list[0]['scale'])
        assert self.am['x'].units == self.axes_list[0]['units']
        nt.assert_almost_equal(self.am['y'].scale, self.axes_list[1]['scale'])
        assert self.am['y'].units == self.axes_list[1]['units']
        nt.assert_almost_equal(self.am['energy'].scale, 0.005)
        assert self.am['energy'].units == 'keV'

    def test_convert_to_units_list(self):
        self.am.convert_units(units=['µm', 'nm', 'meV'])
        nt.assert_almost_equal(self.am['x'].scale, 1000.0)
        assert self.am['x'].units == 'nm'
        nt.assert_almost_equal(self.am['y'].scale, 0.5E-3)
        assert self.am['y'].units == 'µm'
        nt.assert_almost_equal(self.am['energy'].scale, 5E3)
        assert self.am['energy'].units == 'meV'
