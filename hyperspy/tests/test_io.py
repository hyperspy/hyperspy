# -*- coding: utf-8 -*-
# Copyright 2007-2025 The HyperSpy developers
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
import warnings
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
    except (ValueError, OSError, IndexError):
        # Test fallback to Pillow imaging library
        # IndexError is for oldest supported version build on Github CI
        pass

    assert "Unable to infer file type from extension" in caplog.text


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


def test_save_extension_parameter_deprecation_warning(tmp_path):
    """Test that using the 'extension' parameter raises a deprecation warning."""
    s = Signal1D(np.arange(10))

    with pytest.warns(FutureWarning, match="The 'extension' parameter is deprecated"):
        s.save(tmp_path / "test", extension="hspy", overwrite=True)

    # Verify the file was saved correctly despite the deprecation
    assert (tmp_path / "test.hspy").exists()


def test_save_extension_and_file_format_conflict_error(tmp_path):
    """Test that providing both 'extension' and 'file_format' raises a ValueError."""
    s = Signal1D(np.arange(10))

    with pytest.raises(
        ValueError, match="Cannot specify both 'extension' and 'file_format'"
    ):
        s.save(tmp_path / "test", extension="hspy", file_format="msa")


def test_save_extension_parameter_backward_compatibility(tmp_path):
    """Test that extension parameter still works for backward compatibility."""
    s = Signal1D(np.arange(10))

    # Test with different extensions
    test_cases = [
        ("hspy", "test.hspy"),
        ("msa", "test.msa"),
    ]

    for ext, expected_file in test_cases:
        with pytest.warns(FutureWarning):
            s.save(tmp_path / "test", extension=ext, overwrite=True)
        assert (tmp_path / expected_file).exists()
        (tmp_path / expected_file).unlink()  # Clean up


def test_save_extension_parameter_with_directory_path(tmp_path):
    """Test extension parameter works with directory paths (backward compatibility)."""
    s = Signal1D(np.arange(10))

    # Create a source file to get tmp_parameters
    source_file = tmp_path / "source.hspy"
    s.save(source_file)
    s_loaded = hs.load(source_file)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Test with extension parameter and directory path
    with pytest.warns(FutureWarning):
        s_loaded.save(output_dir, extension="msa", overwrite=True)

    assert (output_dir / "source.msa").exists()


def test_save_file_format_parameter_no_warning(tmp_path):
    """Test that using 'file_format' parameter does not raise any warning."""
    s = Signal1D(np.arange(10))

    # This should not raise any warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        s.save(tmp_path / "test", file_format="hspy", overwrite=True)

    assert (tmp_path / "test.hspy").exists()


def test_save_file_format_parameter_with_directory_path(tmp_path):
    """Test file_format parameter works correctly with directory paths."""
    s = Signal1D(np.arange(10))

    # Create a source file to get tmp_parameters
    source_file = tmp_path / "source.hspy"
    s.save(source_file)
    s_loaded = hs.load(source_file)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Test with file_format parameter and directory path
    s_loaded.save(output_dir, file_format="msa", overwrite=True)

    assert (output_dir / "source.msa").exists()


def test_save_extension_precedence_with_file_format_fallback(tmp_path):
    """Test the precedence order when extension is deprecated."""
    s = Signal1D(np.arange(10))

    # Create a source file to get tmp_parameters
    source_file = tmp_path / "source.hspy"
    s.save(source_file)
    s_loaded = hs.load(source_file)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # When only file_format is provided (no extension), it should use file_format
    s_loaded.save(output_dir, file_format="msa", overwrite=True)
    assert (output_dir / "source.msa").exists()

    # Clean up
    (output_dir / "source.msa").unlink()

    # When neither extension nor file_format is provided, should fall back to current tmp_parameters
    # Note: tmp_parameters are updated after each save, so this will use .msa format
    s_loaded.save(output_dir, overwrite=True)
    assert (output_dir / "source.msa").exists()


def test_save_extension_parameter_maps_to_file_format(tmp_path):
    """Test that the deprecated extension parameter correctly determines the output file extension."""
    s = Signal1D(np.arange(10))

    # Create a source file to get tmp_parameters
    source_file = tmp_path / "source.hspy"
    s.save(source_file)
    s_loaded = hs.load(source_file)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Test that extension="msa" creates a .msa file
    with pytest.warns(FutureWarning):
        s_loaded.save(output_dir, extension="msa", overwrite=True)

    # Should create a .msa file, not .msa.msa or similar
    assert (output_dir / "source.msa").exists()
    assert not (output_dir / "source.msa.msa").exists()


def test_save_extension_parameter_strips_leading_dot(tmp_path):
    """Test that extension parameter correctly handles extensions with leading dots."""
    s = Signal1D(np.arange(10))

    # Test with leading dot
    with pytest.warns(FutureWarning):
        s.save(tmp_path / "test", extension=".hspy", overwrite=True)

    assert (tmp_path / "test.hspy").exists()

    # Test without leading dot (should work the same)
    with pytest.warns(FutureWarning):
        s.save(tmp_path / "test2", extension="hspy", overwrite=True)

    assert (tmp_path / "test2.hspy").exists()


def test_save_file_format_unknown_format_error(tmp_path):
    """Test that unknown file_format raises a ValueError."""
    s = Signal1D(np.arange(10))

    # Create a source file to get tmp_parameters
    source_file = tmp_path / "source.hspy"
    s.save(source_file)
    s_loaded = hs.load(source_file)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Test with an unknown/invalid file format - should raise ValueError
    with pytest.raises(ValueError, match="does not match any format available"):
        s_loaded.save(output_dir, file_format="unknown_format", overwrite=True)


def test_save_extension_parameter_current_directory_path(tmp_path):
    """Test extension parameter when filename parent is current directory."""
    s = Signal1D(np.arange(10))

    # Change to the tmp directory to test current directory behavior
    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        # Test with extension parameter on a filename with no parent path
        with pytest.warns(FutureWarning):
            s.save("test", extension="msa", overwrite=True)

        assert Path("test.msa").exists()

    finally:
        os.chdir(original_cwd)


def test_save_base_filename_already_has_extension(tmp_path):
    """Test filename construction when base filename already has the target extension."""
    s = Signal1D(np.arange(10))

    # Create a source file with a specific name that already includes the target extension
    source_file = tmp_path / "data.msa"  # Note: saving as .msa but with .msa name
    s.save(
        source_file.with_suffix(".hspy")
    )  # First save as .hspy to get tmp_parameters
    s_loaded = hs.load(source_file.with_suffix(".hspy"))

    # Manually set the tmp_parameters filename to include the extension
    s_loaded.tmp_parameters.filename = "data.msa"

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # When we save with file_format="msa", it should not double the extension
    s_loaded.save(output_dir, file_format="msa", overwrite=True)

    # Should create "data.msa", not "data.msa.msa"
    assert (output_dir / "data.msa").exists()
    assert not (output_dir / "data.msa.msa").exists()


def test_save_stacklevel_in_deprecation_warning():
    """Test that the deprecation warning points to the correct stack level."""
    s = Signal1D(np.arange(10))

    # Capture the warning and check that stacklevel is set correctly
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s.save("test_stacklevel", extension="hspy", overwrite=True)

        assert len(w) == 1
        assert issubclass(w[0].category, FutureWarning)
        # The warning should point to this test file, not to signal.py
        assert "test_io.py" in str(w[0].filename)
        assert "signal.py" not in str(w[0].filename)

    # Clean up
    Path("test_stacklevel.hspy").unlink(missing_ok=True)


def test_save_extension_parameter_none_handling(tmp_path):
    """Test the extension=None handling logic in filename construction."""
    s = Signal1D(np.arange(10))

    # Create a source file to get tmp_parameters
    source_file = tmp_path / "source.hspy"
    s.save(source_file)
    s_loaded = hs.load(source_file)

    # Save to None filename with no extension parameter (extension=None)
    # This should use tmp_parameters for everything
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # This tests the path where extension=None and we fall back to tmp_parameters.extension
    s_loaded.save(output_dir, overwrite=True)

    # Should use the original extension from tmp_parameters
    assert (output_dir / "source.hspy").exists()


def test_save_file_format_with_directory_ending_slash(tmp_path):
    """Test that directory path detection works with explicit trailing slash."""
    s = Signal1D(np.arange(10))

    # Create a source file to get tmp_parameters
    source_file = tmp_path / "source.hspy"
    s.save(source_file)
    s_loaded = hs.load(source_file)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Test with explicit trailing slash to ensure directory detection works
    output_dir_with_slash = str(output_dir) + "/"
    s_loaded.save(output_dir_with_slash, file_format="msa", overwrite=True)

    assert (output_dir / "source.msa").exists()


def test_save_extension_parameter_overrides_tmp_parameters_extension(tmp_path):
    """Test that explicit extension parameter overrides tmp_parameters.extension."""
    s = Signal1D(np.arange(10))

    # Create a source file to get tmp_parameters with .hspy extension
    source_file = tmp_path / "source.hspy"
    s.save(source_file)
    s_loaded = hs.load(source_file)

    # Use extension parameter to override the tmp_parameters extension
    with pytest.warns(FutureWarning):
        s_loaded.save(tmp_path / "test", extension="msa", overwrite=True)

    # Should create .msa file, overriding the .hspy from tmp_parameters
    assert (tmp_path / "test.msa").exists()
    assert not (tmp_path / "test.hspy").exists()
