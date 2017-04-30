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


import os

import numpy as np
from numpy.testing import assert_allclose
import pytest


import hyperspy.api as hs


dirpath = os.path.dirname(__file__)

file1 = os.path.join(dirpath, 'dens_data', 'file1.dens')
file2 = os.path.join(dirpath, 'dens_data', 'file2.dens')
file3 = os.path.join(dirpath, 'dens_data', 'file3.dens')

ref_T = np.array([15.091, 16.828, 13.232, 50.117, 49.927, 49.986, 49.981])
ref_t = np.array([15.091, 16.828, 13.232, 50.117, 49.927, 49.986, 49.981])


def test_read1():
    s = hs.load(file1)
    np.testing.assert_allclose(s.data, ref_T)
    assert_allclose(s.axes_manager[0].scale, 0.33)
    assert_allclose(s.axes_manager[0].offset, 50077.68)
    ref_date, ref_time = "2015-04-16", "13:53:00"
    assert s.metadata.General.date == ref_date
    assert s.metadata.General.time == ref_time
    assert s.metadata.Signal.signal_type == ""
    assert s.metadata.Signal.quantity == "Temperature (Celsius)"


def test_read2():
    with pytest.raises(AssertionError):
        hs.load(file2)


def test_read3():
    with pytest.raises(AssertionError):
        hs.load(file3)
