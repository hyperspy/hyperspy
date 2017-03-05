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


import numpy as np
import hyperspy.misc.rgb_tools as rt


class TestRGBTools:

    def setup_method(self, method):
        self.data_c = np.ones((2, 2, 3), dtype=np.uint8, order='C')
        self.data_f = np.ones((2, 2, 3), dtype=np.uint8, order='F')
        mask = [[[0, 1, 1], [1, 0, 1]], [[1, 1, 0], [0, 0, 1]]]
        self.data_masked = np.ma.masked_array(self.data_c, mask,
                                              hard_mask=True)

    def test_rgbx2regular_array_corder_from_c(self):
        d = rt.rgbx2regular_array(self.data_c)
        assert d.flags['C_CONTIGUOUS']

    def test_rgbx2regular_array_corder_from_f(self):
        d = rt.rgbx2regular_array(self.data_f)
        assert d.flags['C_CONTIGUOUS']

    def test_rgbx2regular_array_corder_from_c_slices(self):
        d = rt.rgbx2regular_array(self.data_c[0:1, ...])
        assert d.flags['C_CONTIGUOUS']
        d = rt.rgbx2regular_array(self.data_c[:, 0:1, :])
        assert d.flags['C_CONTIGUOUS']

    def test_rgbx2regular_array_cordermask_from_cmasked(self):
        d = rt.rgbx2regular_array(self.data_masked)
        assert isinstance(d, np.ma.MaskedArray)
        assert d.flags['C_CONTIGUOUS']

    def test_rgbx2regular_array_cordermask_from_cmasked_slices(self):
        d = rt.rgbx2regular_array(self.data_masked[0:1, ...])
        assert d.flags['C_CONTIGUOUS']
        assert isinstance(d, np.ma.MaskedArray)
        d = rt.rgbx2regular_array(self.data_masked[:, 0:1, :])
        assert d.flags['C_CONTIGUOUS']
        assert isinstance(d, np.ma.MaskedArray)

    def test_regular_array2rgbx_corder_from_c(self):
        d = rt.regular_array2rgbx(self.data_c)
        assert d.flags['C_CONTIGUOUS']

    def test_regular_array2rgbx_corder_from_f(self):
        d = rt.regular_array2rgbx(self.data_f)
        assert d.flags['C_CONTIGUOUS']

    def test_regular_array2rgbx_corder_from_c_slices(self):
        d = rt.regular_array2rgbx(self.data_c[0:1, ...])
        assert d.flags['C_CONTIGUOUS']
        d = rt.regular_array2rgbx(self.data_c[:, 0:1, :])
        assert d.flags['C_CONTIGUOUS']

    def test_regular_array2rgbx_cordermask_from_cmasked(self):
        d = rt.regular_array2rgbx(self.data_masked)
        assert isinstance(d, np.ma.MaskedArray)
        assert d.flags['C_CONTIGUOUS']

    def test_regular_array2rgbx_cordermask_from_cmasked_slices(self):
        d = rt.regular_array2rgbx(self.data_masked[0:1, ...])
        assert isinstance(d, np.ma.MaskedArray)
        assert d.flags['C_CONTIGUOUS']
        d = rt.regular_array2rgbx(self.data_masked[:, 0:1, :])
        assert isinstance(d, np.ma.MaskedArray)
        assert d.flags['C_CONTIGUOUS']
