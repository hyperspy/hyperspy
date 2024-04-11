# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
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
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from rsciio import IO_PLUGINS

import hyperspy.api as hs
from hyperspy import __version__ as hs_version
from hyperspy.axes import DataAxis
from hyperspy.exceptions import VisibleDeprecationWarning
from hyperspy.signals import Signal1D

PATH = Path(__file__).resolve()
FULLFILENAME = PATH.parent.joinpath("test_io_overwriting.hspy")


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
        axis = DataAxis(axis=1 / (np.arange(10) + 1), navigate=False)
        self.s = Signal1D(np.arange(10), axes=(axis.get_axis_dictionary(),))
        # make sure we start from a clean state

    def test_io_nonuniform(self):
        assert self.s.axes_manager[0].is_uniform is False
        self.s.save("tmp.hspy", overwrite=True)
        with pytest.raises(TypeError, match="not supported for non-uniform"):
            self.s.save("tmp.msa", overwrite=True)

    def test_nonuniform_writer_characteristic(self):
        for plugin in IO_PLUGINS:
            if "non_uniform_axis" not in plugin:
                print(
                    plugin.name + " IO-plugin is missing the "
                    "characteristic `non_uniform_axis`"
                )

    def test_nonuniform_error(self):
        assert self.s.axes_manager[0].is_uniform is False
        incompatible_writers = [
            plugin["file_extensions"][plugin["default_extension"]]
            for plugin in IO_PLUGINS
            if (
                plugin["writes"] is True
                or plugin["writes"] is not False
                and [1, 0] in plugin["writes"]
            )
            and not plugin["non_uniform_axis"]
        ]
        for ext in incompatible_writers:
            with pytest.raises(TypeError, match="not supported for non-uniform"):
                filename = "tmp." + ext
                self.s.save(filename, overwrite=True)

    def teardown_method(self):
        if os.path.exists("tmp.hspy"):
            os.remove("tmp.hspy")
        if os.path.exists("tmp.msa"):
            os.remove("tmp.msa")


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

        t = hs.load(
            os.path.join(dirpath, "temp[*].hspy"),
            escape_square_brackets=True,
        )
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

        with pytest.raises(ValueError, match="No filename matches the pattern"):
            _ = hs.load(temp_fname)

        with pytest.raises(FileNotFoundError):
            _ = hs.load([temp_fname])


def test_file_reader_error(tmp_path):
    # Only None, str or objects with attr "file_reader" are supported
    s = Signal1D(np.arange(10))

    f = tmp_path / "temp.hspy"
    s.save(f)

    with pytest.raises(ValueError, match="reader"):
        _ = hs.load(f, reader=123)


def test_file_reader_warning(caplog, tmp_path):
    # Test fallback to Pillow imaging library
    s = Signal1D(np.arange(10))

    f = tmp_path / "temp.hspy"
    s.save(f)

    try:
        with caplog.at_level(logging.WARNING):
            _ = hs.load(f, reader="some_unknown_file_extension")

        assert "Unable to infer file type from extension" in caplog.text
    except (ValueError, OSError):
        # Test fallback to Pillow imaging library
        pass


def test_file_reader_options(tmp_path):
    # Remove when fixed in rosettasciio
    # it should be possible to read emd file without having to install sparse
    pytest.importorskip("sparse")
    s = Signal1D(np.arange(10))

    s.save(Path(tmp_path, "temp.hspy"))
    s.save(Path(tmp_path, "temp.emd"))

    # Test string reader
    t = hs.load(Path(tmp_path, "temp.hspy"), reader="hspy")
    assert len(t) == 1
    np.testing.assert_allclose(t.data, np.arange(10))

    # Test string reader uppercase
    t = hs.load(Path(tmp_path, "temp.hspy"), reader="HSpy")
    assert len(t) == 1
    np.testing.assert_allclose(t.data, np.arange(10))

    # Test string reader alias
    t = hs.load(Path(tmp_path, "temp.hspy"), reader="hyperspy")
    assert len(t) == 1
    np.testing.assert_allclose(t.data, np.arange(10))

    # Test string reader name
    t = hs.load(Path(tmp_path, "temp.emd"), reader="emd")
    assert len(t) == 1
    np.testing.assert_allclose(t.data, np.arange(10))

    # Test string reader aliases
    t = hs.load(Path(tmp_path, "temp.emd"), reader="Electron Microscopy Data (EMD)")
    assert len(t) == 1
    np.testing.assert_allclose(t.data, np.arange(10))
    t = hs.load(Path(tmp_path, "temp.emd"), reader="Electron Microscopy Data")
    assert len(t) == 1
    np.testing.assert_allclose(t.data, np.arange(10))

    # Test object reader
    from rsciio import hspy

    t = hs.load(tmp_path / "temp.hspy", reader=hspy)
    assert len(t) == 1
    np.testing.assert_allclose(t.data, np.arange(10))


def test_save_default_format(tmp_path):
    s = Signal1D(np.arange(10))

    s.save(tmp_path / "temp")

    t = hs.load(tmp_path / "temp.hspy")
    assert len(t) == 1


def test_load_original_metadata(tmp_path):
    s = Signal1D(np.arange(10))
    s.original_metadata.a = 0

    s.save(tmp_path / "temp")
    assert s.original_metadata.as_dictionary() != {}

    t = hs.load(tmp_path / "temp.hspy")
    assert t.original_metadata.as_dictionary() == s.original_metadata.as_dictionary()

    t = hs.load(tmp_path / "temp.hspy", load_original_metadata=False)
    assert t.original_metadata.as_dictionary() == {}


def test_marker_save_load(tmp_path):
    s = hs.signals.Signal1D(np.arange(10))
    m = hs.plot.markers.Points(offsets=np.array([[2, 2], [3, 3]]), sizes=10)
    s.add_marker(m, permanent=True)
    fname = tmp_path / "test.hspy"
    s.save(fname)
    s2 = hs.load(fname)
    print(s.metadata.Markers, s2.metadata.Markers)
    assert str(s.metadata.Markers) == str(s2.metadata.Markers)
    assert s2.metadata.Markers["Points"]._signal is s2

    s2.plot()


def test_load_save_filereader_metadata(tmp_path):
    # tests that original FileReader metadata is correctly persisted and
    # appended through a save and load cycle

    fname = PATH.parent / "drawing" / "data" / "Cr_L_cl.hspy"
    with pytest.warns(VisibleDeprecationWarning):
        s = hs.load(fname)
    assert s.metadata.General.FileIO.Number_0.io_plugin == "rsciio.hspy"
    assert s.metadata.General.FileIO.Number_0.operation == "load"
    assert s.metadata.General.FileIO.Number_0.hyperspy_version == hs_version

    f = tmp_path / "temp"
    s.save(f)
    expected = {
        "0": {
            "io_plugin": "rsciio.hspy",
            "operation": "load",
            "hyperspy_version": hs_version,
        },
        "1": {
            "io_plugin": "rsciio.hspy",
            "operation": "save",
            "hyperspy_version": hs_version,
        },
        "2": {
            "io_plugin": "rsciio.hspy",
            "operation": "load",
            "hyperspy_version": hs_version,
        },
    }
    del s.metadata.General.FileIO.Number_0.timestamp  # runtime dependent
    del s.metadata.General.FileIO.Number_1.timestamp  # runtime dependent
    assert s.metadata.General.FileIO.Number_0.as_dictionary() == expected["0"]
    assert s.metadata.General.FileIO.Number_1.as_dictionary() == expected["1"]

    t = hs.load(tmp_path / "temp.hspy")
    del t.metadata.General.FileIO.Number_0.timestamp  # runtime dependent
    del t.metadata.General.FileIO.Number_1.timestamp  # runtime dependent
    del t.metadata.General.FileIO.Number_2.timestamp  # runtime dependent
    assert t.metadata.General.FileIO.as_dictionary() == expected
