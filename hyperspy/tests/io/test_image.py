# -*- coding: utf-8 -*-
# Copyright 2007-2021 The HyperSpy developers
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
import tempfile

import numpy as np
import pytest

import hyperspy.api as hs


@pytest.mark.parametrize(("dtype"), ['uint8', 'uint32'])
@pytest.mark.parametrize(("ext"), ['png', 'bmp', 'gif', 'jpg'])
def test_save_load_cycle_grayscale(dtype, ext):
    s = hs.signals.Signal2D(np.arange(128*128).reshape(128, 128).astype(dtype))
    with tempfile.TemporaryDirectory() as tmpdir:
        print('Saving-loading cycle for the extension:', ext)
        filename = os.path.join(tmpdir, 'test_image.'+ext)
        s.save(filename)
        hs.load(filename)


@pytest.mark.parametrize(("color"), ['rgb8', 'rgba8', 'rgb16', 'rgba16'])
@pytest.mark.parametrize(("ext"), ['png', 'bmp', 'gif', 'jpeg'])
def test_save_load_cycle_color(color, ext):
    dim = 4 if "rgba" in color else 3
    dtype = 'uint8' if "8" in color else 'uint16'
    if dim == 4 and ext == 'jpeg':
        # JPEG does not support alpha channel.
        return
    print('color:', color, '; dim:', dim, '; dtype:', dtype)
    s = hs.signals.Signal1D(np.arange(128*128*dim).reshape(128, 128, dim).astype(dtype))
    s.change_dtype(color)
    with tempfile.TemporaryDirectory() as tmpdir:
        print('Saving-loading cycle for the extension:', ext)
        filename = os.path.join(tmpdir, 'test_image.'+ext)
        s.save(filename)
        hs.load(filename)


@pytest.mark.parametrize(("dtype"), ['uint8', 'uint32'])
@pytest.mark.parametrize(("ext"), ['png', 'bmp', 'gif', 'jpg'])
def test_save_load_cycle_kwds(dtype, ext):
    s = hs.signals.Signal2D(np.arange(128*128).reshape(128, 128).astype(dtype))
    with tempfile.TemporaryDirectory() as tmpdir:
        print('Saving-loading cycle for the extension:', ext)
        filename = os.path.join(tmpdir, 'test_image.'+ext)
        if ext == 'png':
            if dtype == 'uint32':
                kwds = {'bits': 32}
            else:
                kwds = {'optimize': True}
        elif ext == 'jpg':
            kwds = {'quality': 100, 'optimize': True}
        elif ext == 'gif':
            kwds = {'subrectangles': 'True', 'palettesize': 128}
        else:
            kwds = {}
        s.save(filename, **kwds)
        hs.load(filename, pilmode='L', as_grey=True)
