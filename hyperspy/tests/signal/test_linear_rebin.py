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


import numpy as np

from hyperspy.signals import EDSTEMSpectrum


class TestLinearRebin:
    def test_linear_downsize(self):
        spectrum = EDSTEMSpectrum(np.ones([3, 5, 1]))
        scale = (1.5, 2.5, 1)
        res = spectrum.rebin(scale=scale, crop=True)
        np.testing.assert_allclose(res.data, 3.75 * np.ones((1, 3, 1)))
        for axis in res.axes_manager._axes:
            assert scale[axis.index_in_axes_manager] == axis.scale
        res = spectrum.rebin(scale=scale, crop=False)
        np.testing.assert_allclose(res.data.sum(), spectrum.data.sum())

    def test_linear_upsize(self):
        spectrum = EDSTEMSpectrum(np.ones([4, 5, 10]))
        scale = [0.3, 0.2, 0.5]
        res = spectrum.rebin(scale=scale)
        np.testing.assert_allclose(res.data, 0.03 * np.ones((20, 16, 20)))
        for axis in res.axes_manager._axes:
            assert scale[axis.index_in_axes_manager] == axis.scale
        res = spectrum.rebin(scale=scale, crop=False)
        np.testing.assert_allclose(res.data.sum(), spectrum.data.sum())

    def test_linear_downscale_out(self):
        spectrum = EDSTEMSpectrum(np.ones([4, 1, 1]))
        scale = [1, 0.4, 1]
        res = spectrum.rebin(scale=scale)
        spectrum.data[2][0] = 5
        spectrum.rebin(scale=scale, out=res)
        np.testing.assert_allclose(
            res.data,
            [
                [[0.4]],
                [[0.4]],
                [[0.4]],
                [[0.4]],
                [[0.4]],
                [[2.0]],
                [[2.0]],
                [[1.2]],
                [[0.4]],
                [[0.4]],
            ],
        )
        for axis in res.axes_manager._axes:
            assert scale[axis.index_in_axes_manager] == axis.scale

    def test_linear_upscale_out(self):
        spectrum = EDSTEMSpectrum(np.ones([4, 1, 1]))
        scale = [1, 0.4, 1]
        res = spectrum.rebin(scale=scale)
        spectrum.data[2][0] = 5
        spectrum.rebin(scale=scale, out=res)
        np.testing.assert_allclose(
            res.data,
            [
                [[0.4]],
                [[0.4]],
                [[0.4]],
                [[0.4]],
                [[0.4]],
                [[2.0]],
                [[2.0]],
                [[1.2]],
                [[0.4]],
                [[0.4]],
            ],
            atol=1e-3,
        )
        for axis in res.axes_manager._axes:
            assert scale[axis.index_in_axes_manager] == axis.scale
