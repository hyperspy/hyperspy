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
import pytest
from unittest.mock import patch

from hyperspy.signals import Signal1D
from hyperspy.io import load

dirpath = os.path.dirname(__file__)
filename = 'test_io_overwriting.hspy'
fullfilename = os.path.join(dirpath, filename)


class TestIOOverwriting:

    def setup_method(self, method):
        self.s = Signal1D(np.arange(10))
        self.new_s = Signal1D(np.ones(5))
        # make sure we start from a clean state
        self._clean_file()
        self.s.save(fullfilename)

    def _clean_file(self):
        if os.path.exists(fullfilename):
            os.remove(fullfilename)

    def _check_file_is_written(self, signal, filename):
        # check that the data have change
        s = load(filename)
        return np.array_equal(s.data, signal.data)

    def test_io_overwriting_True(self):
        # Overwrite is True, when file exists we overwrite
        self.new_s.save(fullfilename, overwrite=True)
        assert self._check_file_is_written(self.new_s, fullfilename)

    def test_io_overwriting_False(self):
        # Overwrite if False, file exists we don't overwrite
        self.new_s.save(fullfilename, overwrite=False)
        assert not self._check_file_is_written(self.new_s, fullfilename)

    @pytest.mark.parametrize("overwrite", [None, True, False])
    def test_io_overwriting_no_existing_file(self, overwrite):
        self._clean_file()  # remove the file
        self.new_s.save(fullfilename, overwrite=overwrite)
        assert self._check_file_is_written(self.new_s, fullfilename)

    def test_io_overwriting_None_existing_file_y(self):
        # Overwrite is None, when file exists we ask, mock `y` here
        with patch('builtins.input', return_value='y'):
            self.new_s.save(fullfilename)
            assert self._check_file_is_written(self.new_s, fullfilename)

    def test_io_overwriting_None_existing_file_n(self):        
        # Overwrite is None, when file exists we ask, mock `n` here
        with patch('builtins.input', return_value='n'):
            self.new_s.save(fullfilename)
            assert not self._check_file_is_written(self.new_s, fullfilename)

    def teardown_method(self, method):
        self._clean_file()
