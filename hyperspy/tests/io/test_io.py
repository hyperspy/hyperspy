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

import os
import hashlib
import numpy as np
import pytest
from unittest.mock import patch

from hyperspy.signals import Signal1D
from hyperspy.axes import DataAxis
from hyperspy.io_plugins import io_plugins


DIRPATH = os.path.dirname(__file__)
FILENAME = 'test_io_overwriting.hspy'
FULLFILENAME = os.path.join(DIRPATH, FILENAME)


class TestIOOverwriting:

    def setup_method(self, method):
        self.s = Signal1D(np.arange(10))
        self.new_s = Signal1D(np.ones(5))
        # make sure we start from a clean state
        self._clean_file()
        self.s.save(FULLFILENAME)
        self.s_file_hashed = self._hash_file(FULLFILENAME)

    def _hash_file(self, filename):
        with open(filename, 'rb') as file:
            md5_hash = hashlib.md5(file.read())
            file_hashed = md5_hash.hexdigest()
        return file_hashed

    def _clean_file(self):
        if os.path.exists(FULLFILENAME):
            os.remove(FULLFILENAME)

    def _check_file_is_written(self, filename):
        # Check that we have a different hash, in case the file have different
        # content from the original, the hash will be different.
        return not self.s_file_hashed == self._hash_file(filename)

    def test_io_overwriting_True(self):
        # Overwrite is True, when file exists we overwrite
        self.new_s.save(FULLFILENAME, overwrite=True)
        assert self._check_file_is_written(FULLFILENAME)

    def test_io_overwriting_False(self):
        # Overwrite if False, file exists we don't overwrite
        self.new_s.save(FULLFILENAME, overwrite=False)
        assert not self._check_file_is_written(FULLFILENAME)

    @pytest.mark.parametrize("overwrite", [None, True, False])
    def test_io_overwriting_no_existing_file(self, overwrite):
        self._clean_file()  # remove the file
        self.new_s.save(FULLFILENAME, overwrite=overwrite)
        assert self._check_file_is_written(FULLFILENAME)

    def test_io_overwriting_None_existing_file_y(self):
        # Overwrite is None, when file exists we ask, mock `y` here
        with patch('builtins.input', return_value='y'):
            self.new_s.save(FULLFILENAME)
            assert self._check_file_is_written(FULLFILENAME)

    def test_io_overwriting_None_existing_file_n(self):
        # Overwrite is None, when file exists we ask, mock `n` here
        with patch('builtins.input', return_value='n'):
            self.new_s.save(FULLFILENAME)
            assert not self._check_file_is_written(FULLFILENAME)

    def teardown_method(self, method):
        self._clean_file()

class TestNonUniformAxisCheck:

    def setup_method(self, method):
        axis = DataAxis(axis = 1/np.arange(10), navigate = False)
        self.s = Signal1D(np.arange(10), axes=(axis.get_axis_dictionary(), ))
        # make sure we start from a clean state
    
    def test_io_nonuniform(self):
        assert(self.s.axes_manager[0].is_uniform == False)
        self.s.save('tmp.hspy', overwrite = True)
        with pytest.raises(OSError):
            self.s.save('tmp.msa', overwrite = True)

    def test_nonuniform_writer_characteristic(self):
        for plugin in io_plugins:
            try:
                plugin.non_uniform_axis is True
            except AttributeError:
                print(plugin.format_name + ' IO-plugin is missing the '
                      'characteristic `non_uniform_axis`')

    def teardown_method(self):
        if os.path.exists('tmp.hspy'):
            os.remove('tmp.hspy')
        if os.path.exists('tmp.msa'):
            os.remove('tmp.msa')
            
