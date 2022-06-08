# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import hashlib
import os
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import hyperspy.api as hs
from hyperspy.signals import Signal1D
from hyperspy.axes import DataAxis
from hyperspy.io_plugins import io_plugins
from hyperspy import __version__ as hs_version
from hyperspy.misc.test_utils import assert_deep_almost_equal


FULLFILENAME = Path(__file__).resolve().parent.joinpath("test_io_overwriting.hspy")


class TestIOOverwriting:
    def setup_method(self, method):
        self.s = Signal1D(np.arange(10))
        self.new_s = Signal1D(np.ones(5))
        # make sure we start from a clean state
        self._clean_file()
        self.s.save(FULLFILENAME)
        self.s_file_hashed = self._hash_file(FULLFILENAME)

    def _hash_file(self, filename):
        with open(filename, "rb") as file:
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
        with patch("builtins.input", return_value="y"):
            self.new_s.save(FULLFILENAME)
            assert self._check_file_is_written(FULLFILENAME)

    def test_io_overwriting_None_existing_file_n(self):
        # Overwrite is None, when file exists we ask, mock `n` here
        with patch("builtins.input", return_value="n"):
            self.new_s.save(FULLFILENAME)
            assert not self._check_file_is_written(FULLFILENAME)

    def test_io_overwriting_invalid_parameter(self):
        with pytest.raises(ValueError, match="parameter can only be"):
            self.new_s.save(FULLFILENAME, overwrite="spam")

    def teardown_method(self, method):
        self._clean_file()

class TestNonUniformAxisCheck:

    def setup_method(self, method):
        axis = DataAxis(axis = 1/(np.arange(10)+1), navigate = False)
        self.s = Signal1D(np.arange(10), axes=(axis.get_axis_dictionary(), ))
        # make sure we start from a clean state
    
    def test_io_nonuniform(self):
        assert(self.s.axes_manager[0].is_uniform == False)
        self.s.save('tmp.hspy', overwrite = True)
        with pytest.raises(TypeError, match = "not supported for non-uniform"):
            self.s.save('tmp.msa', overwrite = True)

    def test_nonuniform_writer_characteristic(self):
        for plugin in io_plugins:
            try:
                plugin.non_uniform_axis is True
            except AttributeError:
                print(plugin.format_name + ' IO-plugin is missing the '
                      'characteristic `non_uniform_axis`')

    def test_nonuniform_error(self):
        assert(self.s.axes_manager[0].is_uniform == False)
        incompatible_writers = [plugin.file_extensions[plugin.default_extension]
                      for plugin in io_plugins if (plugin.writes is True or
                      plugin.writes is not False and (1, 0) in plugin.writes)
                      and plugin.non_uniform_axis is False]
        for ext in incompatible_writers:
            with pytest.raises(TypeError, match = "not supported for non-uniform"):
                filename = 'tmp.' + ext
                self.s.save(filename, overwrite = True)

    def teardown_method(self):
        if os.path.exists('tmp.hspy'):
            os.remove('tmp.hspy')
        if os.path.exists('tmp.msa'):
            os.remove('tmp.msa')
            

def test_glob_wildcards():
    s = Signal1D(np.arange(10))

    with tempfile.TemporaryDirectory() as dirpath:
        fnames = [os.path.join(dirpath, f"temp[1x{x}].hspy") for x in range(2)]

        for f in fnames:
            s.save(f)

        with pytest.raises(ValueError, match="No filename matches the pattern"):
            _ = hs.load(fnames[0])

        t = hs.load([fnames[0]])
        assert len(t) == 1

        t = hs.load(fnames)
        assert len(t) == 2

        t = hs.load(os.path.join(dirpath, "temp*.hspy"))
        assert len(t) == 2

        t = hs.load(os.path.join(dirpath, "temp[*].hspy"), escape_square_brackets=True,)
        assert len(t) == 2

        with pytest.raises(ValueError, match="No filename matches the pattern"):
            _ = hs.load(os.path.join(dirpath, "temp[*].hspy"))

        # Test pathlib.Path
        t = hs.load(Path(dirpath, "temp[1x0].hspy"))
        assert len(t) == 1

        t = hs.load([Path(dirpath, "temp[1x0].hspy"), Path(dirpath, "temp[1x1].hspy")])
        assert len(t) == 2

        t = hs.load(list(Path(dirpath).glob("temp*.hspy")))
        assert len(t) == 2

        t = hs.load(Path(dirpath).glob("temp*.hspy"))
        assert len(t) == 2


def test_file_not_found_error():
    with tempfile.TemporaryDirectory() as dirpath:
        temp_fname = os.path.join(dirpath, "temp.hspy")

        if os.path.exists(temp_fname):
            os.remove(temp_fname)

        with pytest.raises(ValueError, match='No filename matches the pattern'):
            _ = hs.load(temp_fname)

        with pytest.raises(FileNotFoundError):
            _ = hs.load([temp_fname])


def test_file_reader_error():
    # Only None, str or objects with attr "file_reader" are supported
    s = Signal1D(np.arange(10))

    with tempfile.TemporaryDirectory() as dirpath:
        f = os.path.join(dirpath, "temp.hspy")
        s.save(f)

        with pytest.raises(ValueError, match="reader"):
            _ = hs.load(f, reader=123)


def test_file_reader_warning(caplog):
    # Test fallback to Pillow imaging library
    s = Signal1D(np.arange(10))

    with tempfile.TemporaryDirectory() as dirpath:
        f = os.path.join(dirpath, "temp.hspy")
        s.save(f)

        with pytest.raises(ValueError, match="Could not load"):
            with caplog.at_level(logging.WARNING):
                _ = hs.load(f, reader="some_unknown_file_extension")

            assert "Unable to infer file type from extension" in caplog.text


def test_file_reader_options():
    s = Signal1D(np.arange(10))

    with tempfile.TemporaryDirectory() as dirpath:
        f = os.path.join(dirpath, "temp.hspy")
        s.save(f)

        # Test string reader
        t = hs.load(Path(dirpath, "temp.hspy"), reader="hspy")
        assert len(t) == 1
        np.testing.assert_allclose(t.data, np.arange(10))

        # Test object reader
        from hyperspy.io_plugins import hspy

        t = hs.load(Path(dirpath, "temp.hspy"), reader=hspy)
        assert len(t) == 1
        np.testing.assert_allclose(t.data, np.arange(10))


def test_save_default_format():
    s = Signal1D(np.arange(10))

    with tempfile.TemporaryDirectory() as dirpath:
        f = os.path.join(dirpath, "temp")
        s.save(f)

        t = hs.load(Path(dirpath, "temp.hspy"))
        assert len(t) == 1


def test_load_original_metadata():
    s = Signal1D(np.arange(10))
    s.original_metadata.a = 0

    with tempfile.TemporaryDirectory() as dirpath:
        f = os.path.join(dirpath, "temp")
        s.save(f)
        assert s.original_metadata.as_dictionary() != {}

        t = hs.load(Path(dirpath, "temp.hspy"))
        assert t.original_metadata.as_dictionary() == s.original_metadata.as_dictionary()

        t = hs.load(Path(dirpath, "temp.hspy"), load_original_metadata=False)
        assert t.original_metadata.as_dictionary() == {}


def test_load_save_filereader_metadata():
    # tests that original FileReader metadata is correctly persisted and
    # appended through a save and load cycle

    my_path = os.path.dirname(__file__)
    s = hs.load(os.path.join(my_path, "msa_files", "example1.msa"))
    assert s.metadata.General.FileIO.Number_0.io_plugin == \
           'hyperspy.io_plugins.msa'
    assert s.metadata.General.FileIO.Number_0.operation == 'load'
    assert s.metadata.General.FileIO.Number_0.hyperspy_version == hs_version

    with tempfile.TemporaryDirectory() as dirpath:
        f = os.path.join(dirpath, "temp")
        s.save(f)
        expected = {
            '0': {
                'io_plugin': 'hyperspy.io_plugins.msa',
                'operation': 'load',
                'hyperspy_version': hs_version
            },
            '1': {
                'io_plugin': 'hyperspy.io_plugins.hspy',
                'operation': 'save',
                'hyperspy_version': hs_version
            },
            '2': {
                'io_plugin': 'hyperspy.io_plugins.hspy',
                'operation': 'load',
                'hyperspy_version': hs_version
            },
        }
        del s.metadata.General.FileIO.Number_0.timestamp  # runtime dependent
        del s.metadata.General.FileIO.Number_1.timestamp  # runtime dependent
        assert \
            s.metadata.General.FileIO.Number_0.as_dictionary() == expected['0']
        assert \
            s.metadata.General.FileIO.Number_1.as_dictionary() == expected['1']

        t = hs.load(Path(dirpath, "temp.hspy"))
        del t.metadata.General.FileIO.Number_0.timestamp  # runtime dependent
        del t.metadata.General.FileIO.Number_1.timestamp  # runtime dependent
        del t.metadata.General.FileIO.Number_2.timestamp  # runtime dependent
        assert t.metadata.General.FileIO.as_dictionary() == expected
