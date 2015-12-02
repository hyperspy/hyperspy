# -*- coding: utf-8 -*-
# Copyright 2007-2015 The HyperSpy developers
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
import nose.tools as nt
import hyperspy.api as hs
import numpy as np


dirpath = os.path.dirname(__file__)

file1 = os.path.join(dirpath, 'blockfile_data', 'test1.blo')
file2 = os.path.join(dirpath, 'blockfile_data', 'test2.blo')
save_path = os.path.join(dirpath, 'blockfile_data', 'save_temp.blo')

ref_data2 = np.array(
        [[[[20, 23, 25, 25, 27],
         [29, 23, 23,  0, 29],
         [24,  0,  0, 22, 18],
         [ 0, 14, 19, 17, 26],
         [19, 21, 22, 27, 20]],

        [[28, 25, 29, 15, 29],
         [12, 15, 12, 25, 24],
         [25, 26, 26, 18, 27],
         [19, 18, 20, 23, 28],
         [28, 18, 22, 25,  0]],

        [[21, 29, 25, 19, 18],
         [30, 15, 20, 22, 26],
         [23, 18, 26, 15, 25],
         [22, 25, 24, 15, 20],
         [22, 15, 15, 21, 23]]],


       [[[28, 25, 26, 24, 26],
         [26, 17,  0, 24, 12],
         [17, 18, 21, 19, 21],
         [21, 24, 19, 17,  0],
         [17, 14, 25, 15, 26]],

        [[25, 18, 20, 15, 24],
         [19, 13, 23, 18, 11],
         [ 0, 25,  0,  0, 14],
         [26, 22, 22, 11, 14],
         [21,  0, 15, 13, 19]],

        [[24, 18, 20, 22, 21],
         [13, 25, 20, 28, 29],
         [15, 17, 24, 23, 23],
         [22, 21, 21, 22, 18],
         [24, 25, 18, 18, 27]]]], dtype=np.uint8)

axes1 = {
    'axis-0': {
        'name': 'y', 'navigate': True, 'offset': 0.0,
        'scale': 12.8, 'size': 3, 'units': 'nm'},
    'axis-1': {
        'name': 'x', 'navigate': True, 'offset': 0.0,
        'scale': 12.8, 'size': 2, 'units': 'nm'},
    'axis-2': {
        'name': 'dy', 'navigate': False, 'offset': 0.0,
        'scale': 0.016061676839061997, 'size': 144, 'units': 'cm'},
    'axis-3': {
        'name': 'dx', 'navigate': False, 'offset': 0.0,
        'scale': 0.016061676839061997, 'size': 144, 'units': 'cm'}}

axes2 = {
    'axis-0': {
        'name': 'y', 'navigate': True, 'offset': 0.0,
        'scale': 64.0, 'size': 2, 'units': 'nm'},
    'axis-1': {
        'name': 'x', 'navigate': True, 'offset': 0.0,
        'scale': 64.0, 'size': 3, 'units': 'nm'},
    'axis-2': {
        'name': 'dy', 'navigate': False, 'offset': 0.0,
        'scale': 0.016061676839061997, 'size': 5, 'units': 'cm'},
    'axis-3': {
        'name': 'dx', 'navigate': False, 'offset': 0.0,
        'scale': 0.016061676839061997, 'size': 5, 'units': 'cm'}}


def test_load1():
    s = hs.load(file1)
    nt.assert_equal(s.data.shape, (3, 2, 144, 144))
    nt.assert_equal(s.axes_manager.as_dictionary(), axes1)


def test_load2():
    s = hs.load(file2)
    nt.assert_equal(s.data.shape, (2, 3, 5, 5))
    np.testing.assert_equal(s.axes_manager.as_dictionary(), axes2)
    np.testing.assert_allclose(s.data, ref_data2)


def test_save_load_cycle():
    signal = hs.load(file2)
    try:
        signal.save(save_path, overwrite=True)
        sig_reload = hs.load(save_path)
        np.testing.assert_equal(signal.data, sig_reload.data)
        nt.assert_equal(signal.axes_manager.as_dictionary(),
                        sig_reload.axes_manager.as_dictionary())
        nt.assert_equal(signal.original_metadata.as_dictionary(),
                        sig_reload.original_metadata.as_dictionary())
        nt.assert_is_instance(signal, hs.signals.Image)
    finally:
        os.remove(save_path)
