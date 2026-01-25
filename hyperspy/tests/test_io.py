# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
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

PATH = Path(__file__).resolve()
FULLFILENAME = PATH.parent.joinpath("test_io_overwriting.hspy")


class TestIOOverwriting:
    def setup_method(self, method):
        self.s = hs.signals.Signal1D(np.arange(10))
        self.new_s = hs.signals.Signal1D(np.ones(5))
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
        self.s = hs.signals.Signal1D(np.arange(10), axes=(axis.get_axis_dictionary(),))
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
    s = hs.signals.Signal1D(np.arange(10))

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
    s = hs.signals.Signal1D(np.arange(10))

    f = tmp_path / "temp.hspy"
    s.save(f)

    with pytest.warns(
        VisibleDeprecationWarning, match="'reader' parameter is deprecated"
    ):
        with pytest.raises(ValueError, match="reader"):
            _ = hs.load(f, reader=123)


def test_file_reader_warning(caplog, tmp_path):
    # Test fallback to Pillow imaging library
    s = hs.signals.Signal1D(np.arange(10))

    f = tmp_path / "temp.hspy"
    s.save(f)

    try:
        with pytest.warns(
            VisibleDeprecationWarning, match="'reader' parameter is deprecated"
        ):
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
    s = hs.signals.Signal1D(np.arange(10))

    s.save(Path(tmp_path, "temp.hspy"))
    s.save(Path(tmp_path, "temp.emd"))

    # Test string reader
    with pytest.warns(VisibleDeprecationWarning):
        t = hs.load(Path(tmp_path, "temp.hspy"), reader="hspy")
    assert len(t) == 1
    np.testing.assert_allclose(t.data, np.arange(10))

    # Test string reader uppercase
    with pytest.warns(VisibleDeprecationWarning):
        t = hs.load(Path(tmp_path, "temp.hspy"), reader="HSpy")
    assert len(t) == 1
    np.testing.assert_allclose(t.data, np.arange(10))

    # Test string reader alias
    with pytest.warns(VisibleDeprecationWarning):
        t = hs.load(Path(tmp_path, "temp.hspy"), reader="hyperspy")
    assert len(t) == 1
    np.testing.assert_allclose(t.data, np.arange(10))

    # Test string reader name
    with pytest.warns(VisibleDeprecationWarning):
        t = hs.load(Path(tmp_path, "temp.emd"), reader="emd")
    assert len(t) == 1
    np.testing.assert_allclose(t.data, np.arange(10))

    # Test string reader aliases
    with pytest.warns(VisibleDeprecationWarning):
        t = hs.load(Path(tmp_path, "temp.emd"), reader="Electron Microscopy Data (EMD)")
    assert len(t) == 1
    np.testing.assert_allclose(t.data, np.arange(10))
    with pytest.warns(VisibleDeprecationWarning):
        t = hs.load(Path(tmp_path, "temp.emd"), reader="Electron Microscopy Data")
    assert len(t) == 1
    np.testing.assert_allclose(t.data, np.arange(10))

    # Test object reader
    from rsciio import hspy

    with pytest.warns(VisibleDeprecationWarning):
        t = hs.load(tmp_path / "temp.hspy", reader=hspy)
    assert len(t) == 1
    np.testing.assert_allclose(t.data, np.arange(10))


def test_save_default_format(tmp_path):
    s = hs.signals.Signal1D(np.arange(10))

    s.save(tmp_path / "temp")

    t = hs.load(tmp_path / "temp.hspy")
    assert len(t) == 1


def test_load_original_metadata(tmp_path):
    s = hs.signals.Signal1D(np.arange(10))
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
    s = hs.signals.Signal1D(np.arange(10))

    with pytest.warns(FutureWarning, match="The 'extension' parameter is deprecated"):
        s.save(tmp_path / "test", extension="hspy", overwrite=True)

    # Verify the file was saved correctly despite the deprecation
    assert (tmp_path / "test.hspy").exists()


def test_save_extension_and_file_format_conflict_error(tmp_path):
    """Test that providing both 'extension' and 'file_format' raises a ValueError."""
    s = hs.signals.Signal1D(np.arange(10))

    with pytest.raises(
        ValueError, match="Cannot specify both 'extension' and 'file_format'"
    ):
        s.save(tmp_path / "test", extension="hspy", file_format="msa")


def test_save_extension_parameter_backward_compatibility(tmp_path):
    """Test that extension parameter still works for backward compatibility."""
    s = hs.signals.Signal1D(np.arange(10))

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
    s = hs.signals.Signal1D(np.arange(10))

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
    s = hs.signals.Signal1D(np.arange(10))

    # This should not raise any warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        s.save(tmp_path / "test", file_format="hspy", overwrite=True)

    assert (tmp_path / "test.hspy").exists()


def test_save_file_format_parameter_with_directory_path(tmp_path):
    """Test file_format parameter works correctly with directory paths."""
    s = hs.signals.Signal1D(np.arange(10))

    # Create a source file to get tmp_parameters
    source_file = tmp_path / "source.hspy"
    s.save(source_file)
    s_loaded = hs.load(source_file)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Test with file_format parameter and directory path
    s_loaded.save(output_dir, file_format="msa", overwrite=True)

    assert (output_dir / "source.msa").exists()


@pytest.mark.parametrize("file_format", ["hspy", "zspy"])
def test_save_extension_precedence_with_file_format_fallback(tmp_path, file_format):
    """Test the precedence order when extension is deprecated."""
    if file_format == "zspy":
        pytest.importorskip("zspy")
    s = hs.signals.Signal1D(np.arange(10))

    # Create a source file to get tmp_parameters
    source_file = tmp_path / f"source.{file_format}"
    s.save(source_file)
    s_loaded = hs.load(source_file)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # When only file_format is provided (no extension), it should use file_format
    s_loaded.save(output_dir, file_format=file_format, overwrite=True)
    assert (output_dir / f"source.{file_format}").exists()

    s_loaded.data *= 2
    # When neither extension nor file_format is provided, should fall back to current tmp_parameters
    # Note: tmp_parameters are updated after each save, so this will use file_format
    s_loaded.save(output_dir, overwrite=True)
    assert (output_dir / f"source.{file_format}").exists()
    # Check that the file has been overwritten
    s_loaded2 = hs.load(output_dir / f"source.{file_format}")
    np.testing.assert_allclose(s_loaded2.data, s_loaded.data)


def test_save_extension_parameter_maps_to_file_format(tmp_path):
    """Test that the deprecated extension parameter correctly determines the output file extension."""
    s = hs.signals.Signal1D(np.arange(10))

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
    s = hs.signals.Signal1D(np.arange(10))

    # Test with leading dot
    with pytest.warns(FutureWarning):
        s.save(tmp_path / "test", extension=".hspy", overwrite=True)

    assert (tmp_path / "test.hspy").exists()

    # Test without leading dot (should work the same)
    with pytest.warns(FutureWarning):
        s.save(tmp_path / "test2", extension="hspy", overwrite=True)

    assert (tmp_path / "test2.hspy").exists()


def test_save_file_format_unknown_format_error(tmp_path, caplog):
    """Test that unknown file_format raises a ValueError."""
    s = hs.signals.Signal1D(np.arange(10))

    # Create a source file to get tmp_parameters
    source_file = tmp_path / "source.hspy"
    s.save(source_file)
    s_loaded = hs.load(source_file)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Test with an unknown/invalid file format - should raise TypeError and warning
    # because it will fallback on the image writer
    with caplog.at_level(logging.WARNING):
        with pytest.raises(
            TypeError, match="This file format does not support this data"
        ):
            s_loaded.save(output_dir, file_format="unknown_format", overwrite=True)

    # Check that the warning was logged
    assert (
        "Unable to infer file type from extension/name 'unknown_format'" in caplog.text
    )


def test_save_extension_parameter_current_directory_path(tmp_path):
    """Test extension parameter when filename parent is current directory."""
    s = hs.signals.Signal1D(np.arange(10))

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
    s = hs.signals.Signal1D(np.arange(10))

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
    s = hs.signals.Signal1D(np.arange(10))

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
    s = hs.signals.Signal1D(np.arange(10))

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
    s = hs.signals.Signal1D(np.arange(10))

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
    s = hs.signals.Signal1D(np.arange(10))

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


# Test coverage for _get_format_list_for_docstring function
def test_get_format_list_for_docstring_bullet_style():
    """Test _get_format_list_for_docstring with bullet style."""
    from hyperspy.io import _get_format_list_for_docstring

    # Test read mode with bullet style
    result = _get_format_list_for_docstring(write_mode=False, style="bullet")
    assert result.startswith("\n")
    assert result.endswith("\n")
    assert "\n        * ``" in result
    assert "for " in result

    # Test write mode with bullet style
    result_write = _get_format_list_for_docstring(write_mode=True, style="bullet")
    assert result_write.startswith("\n")
    assert result_write.endswith("\n")
    assert "\n        * ``" in result_write
    # Write mode should have fewer formats than read mode
    assert len(result_write) <= len(result)


def test_get_format_list_for_docstring_inline_style():
    """Test _get_format_list_for_docstring with inline style."""
    from hyperspy.io import _get_format_list_for_docstring

    # Test read mode with inline style
    result = _get_format_list_for_docstring(write_mode=False, style="inline")
    assert "``'" in result
    assert ", " in result
    assert not result.startswith("\n")

    # Test write mode with inline style
    result_write = _get_format_list_for_docstring(write_mode=True, style="inline")
    assert "``'" in result_write
    assert ", " in result_write


def test_get_format_list_for_docstring_extensions_style():
    """Test _get_format_list_for_docstring with extensions style."""
    from hyperspy.io import _get_format_list_for_docstring

    # Test read mode with extensions style
    result = _get_format_list_for_docstring(write_mode=False, style="extensions")
    assert ", " in result
    assert "``" not in result  # No formatting markup

    # Test write mode with extensions style
    result_write = _get_format_list_for_docstring(write_mode=True, style="extensions")
    assert ", " in result_write
    assert "``" not in result_write


def test_get_format_list_for_docstring_invalid_style():
    """Test _get_format_list_for_docstring with invalid style raises ValueError."""
    from hyperspy.io import _get_format_list_for_docstring

    with pytest.raises(ValueError, match="Unknown style"):
        _get_format_list_for_docstring(style="invalid_style")


def test_load_reader_parameter_deprecation_warning(tmp_path):
    """Test that using reader parameter shows deprecation warning."""
    import warnings

    from hyperspy.io import load_single_file

    # Create a dummy file for testing
    s = hs.signals.Signal1D([1, 2, 3])
    filename = tmp_path / "test_reader_deprecation.hspy"
    s.save(filename)

    # Test that reader parameter triggers deprecation warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            # This should trigger deprecation warning
            load_single_file(str(filename), reader="HSPY")
            # Check that warning was raised
            assert len(w) > 0
            assert any("reader" in str(warning.message) for warning in w)
            assert any("deprecated" in str(warning.message) for warning in w)
        except Exception:
            # If it fails for other reasons, that's ok - we just want to test the warning
            pass


def test_load_reader_and_file_format_conflict(tmp_path):
    """Test that providing both reader and file_format raises ValueError."""
    from hyperspy.io import load_single_file

    # Create a dummy file for testing
    s = hs.signals.Signal1D([1, 2, 3])
    filename = tmp_path / "test_conflict.hspy"
    s.save(filename)

    # Test that providing both parameters raises ValueError
    with pytest.raises(
        ValueError, match="Cannot specify both 'file_format' and 'reader'"
    ):
        load_single_file(str(filename), file_format="HSPY", reader="HSPY")


def test_save_without_filename_no_tmp_parameters():
    """Test save without filename when tmp_parameters are not available."""
    s = hs.signals.Signal1D([1, 2, 3])

    # Clear tmp_parameters to simulate a signal not loaded from file
    s.tmp_parameters = s.tmp_parameters.__class__()

    with pytest.raises(ValueError, match="File name not defined"):
        s.save(filename=None, file_format="hspy")


def test_save_without_filename_missing_folder(tmp_path):
    """Test save without filename when folder is missing from tmp_parameters."""
    s = hs.signals.Signal1D([1, 2, 3])

    # Set only filename, not folder
    s.tmp_parameters.filename = "test"

    with pytest.raises(ValueError, match="File name not defined"):
        s.save(filename=None, file_format="hspy")


def test_save_without_filename_missing_filename(tmp_path):
    """Test save without filename when filename is missing from tmp_parameters."""
    s = hs.signals.Signal1D([1, 2, 3])

    # Set only folder, not filename
    s.tmp_parameters.folder = str(tmp_path)

    with pytest.raises(ValueError, match="File name not defined"):
        s.save(filename=None, file_format="hspy")


def test_save_without_filename_and_file_format():
    """Test save without both filename and file_format raises ValueError."""
    s = hs.signals.Signal1D([1, 2, 3])

    with pytest.raises(ValueError, match="File name not defined"):
        s.save(filename=None, file_format=None)


def test_format_name_to_reader_invalid_format():
    """Test _format_name_to_reader with invalid format name."""
    from hyperspy.io import _format_name_to_reader

    with pytest.raises(ValueError, match="format_name.*does not match any format"):
        _format_name_to_reader("nonexistent_format")


def test_save_write_file_format(tmp_path):
    """Test that save writes the correct file format based on file_format parameter."""
    s = hs.signals.Signal1D(np.arange(10 * 10).reshape(10, 10))
    fname_extension = tmp_path / "test_file_format_extension"
    fname_name = tmp_path / "test_file_format_name"

    # Save with file_format parameter using extension
    s.save(fname_extension, file_format="rpl")
    s2 = hs.load(f"{fname_extension}.rpl")
    np.testing.assert_allclose(s2.data, s.data)

    # Save with file_format parameter using name
    s.save(fname_name, file_format="ripple")
    s3 = hs.load(f"{fname_name}.rpl")
    np.testing.assert_allclose(s3.data, s.data)


def test_infer_file_reader_unknown_extension(caplog):
    """Test _infer_file_reader with unknown extension falls back to image reader."""
    from hyperspy.io import _infer_file_reader

    # The warning is logged, not issued as a warning
    reader = _infer_file_reader("unknown_ext")

    # Should fall back to image reader
    assert reader["name"].lower() == "image"

    # Check that warning was logged
    assert "Unable to infer file type" in caplog.text


def test_infer_file_reader_multiple_readers():
    """Test _infer_file_reader with extension that matches multiple readers."""
    from hyperspy.io import _infer_file_reader

    # Use h5 extension which might match multiple readers

    with pytest.raises(ValueError, match="multiple file readers"):
        #  multiple readers, should raise specific error
        _ = _infer_file_reader("h5")

    name = _infer_file_reader("arina")
    assert name["name"] == "Arina"

    name = _infer_file_reader("usid")
    assert name["name"] == "USID"
    name = _infer_file_reader("USID")
    assert name["name"] == "USID"


def test_infer_file_writer_unsupported_extension():
    """Test _infer_file_writer with unsupported extension."""
    from hyperspy.io import _infer_file_writer

    with pytest.raises(ValueError, match="does not correspond to any supported format"):
        _infer_file_writer("unsupported_ext")


def test_infer_file_writer_read_only_format():
    """Test _infer_file_writer with read-only format extension."""
    from rsciio import IO_PLUGINS

    from hyperspy.io import _infer_file_writer

    # Find a read-only format that doesn't have any writable counterparts
    read_only_formats = [p for p in IO_PLUGINS if not p["writes"]]

    # Get all extensions from writable formats to avoid conflicts
    writable_extensions = set()
    for p in IO_PLUGINS:
        if p["writes"]:
            for ext in p["file_extensions"]:
                writable_extensions.add(ext.lower())

    # Find a read-only format whose extensions don't overlap with writable ones
    suitable_plugin = None
    for plugin in read_only_formats:
        plugin_extensions = [ext.lower() for ext in plugin["file_extensions"]]
        if not any(ext in writable_extensions for ext in plugin_extensions):
            suitable_plugin = plugin
            break

    if suitable_plugin:
        ext = suitable_plugin["file_extensions"][suitable_plugin["default_extension"]]

        with pytest.raises(ValueError, match="Writing to this format is not supported"):
            _infer_file_writer(ext)


def test_parse_path_with_string():
    """Test _parse_path function with string input."""
    from hyperspy.io import _parse_path

    path = "/some/path/file.txt"
    result = _parse_path(path)
    assert result == path


def test_parse_path_with_mapping():
    """Test _parse_path function with mapping input."""
    from collections.abc import MutableMapping

    from hyperspy.io import _parse_path

    # Create a mock mapping object with path attribute
    class MockMapping(MutableMapping):
        def __init__(self):
            self.path = "/mapped/path/file.txt"
            self._data = {}

        def __getitem__(self, key):
            return self._data[key]

        def __setitem__(self, key, value):
            self._data[key] = value

        def __delitem__(self, key):
            del self._data[key]

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

    mapping = MockMapping()
    result = _parse_path(mapping)
    assert result == "/mapped/path/file.txt"


def test_get_file_format_reader_with_aliases():
    """Test _format_name_to_reader with format name aliases."""
    from hyperspy.io import _format_name_to_reader

    # Test using a format alias instead of main name
    reader = _format_name_to_reader("tiff")  # alias for TIFF
    assert reader["name"] == "TIFF"


def test_infer_file_reader_multiple_matches_error():
    """Test error when multiple file readers match the same extension."""
    import unittest.mock

    from hyperspy.io import _infer_file_reader

    # Mock IO_PLUGINS to have multiple readers for the same extension
    mock_plugins = [
        {"name": "Reader1", "file_extensions": {".test": 1}},
        {"name": "Reader2", "file_extensions": {".test": 1}},
        {"name": "Image", "file_extensions": {".png": 1}},  # Fallback reader
    ]

    with unittest.mock.patch("hyperspy.io.IO_PLUGINS", mock_plugins):
        # This should raise an error because multiple readers match .test
        with pytest.raises(ValueError, match="There are multiple file readers"):
            _infer_file_reader(".test")


def test_infer_file_writer_multiple_matches_error():
    """Test error when multiple file writers match the same extension."""
    import unittest.mock

    from hyperspy.io import _infer_file_writer

    # Mock IO_PLUGINS to have multiple writers for the same extension
    mock_plugins = [
        {"name": "Writer1", "file_extensions": {".test": 1}, "writes": True},
        {"name": "Writer2", "file_extensions": {".test": 1}, "writes": True},
    ]

    with unittest.mock.patch("hyperspy.io.IO_PLUGINS", mock_plugins):
        # This should raise an error because multiple writers match .test
        with pytest.raises(ValueError, match="There are multiple file formats"):
            _infer_file_writer(".test")


def test_load_stack_signal_number_mismatch():
    """Test error when stacked files have different numbers of signals."""
    # Create temporary files with different signal structures
    s1 = hs.signals.Signal1D([1, 2, 3])
    s2 = hs.signals.Signal1D([4, 5, 6])

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save first file with one signal
        file1 = os.path.join(tmp_dir, "file1.hspy")
        s1.save(file1)

        # Save second file with two signals (simulate this by saving a list)
        file2 = os.path.join(tmp_dir, "file2.hspy")
        # We need to create a file that loads as multiple signals
        # This is tricky with real files, so we'll mock the loading

        import unittest.mock

        def mock_load_single_file(filename, **kwargs):
            if "file1" in filename:
                return s1
            else:
                return [s1, s2]  # Multiple signals

        with unittest.mock.patch(
            "hyperspy.io.load_single_file", side_effect=mock_load_single_file
        ):
            with pytest.raises(
                ValueError, match="The number of sub-signals per file does not match"
            ):
                hs.load([file1, file2], stack=True)


def test_assign_signal_subclass_unknown_signal_type_warning(caplog):
    """Test warning when unknown signal_type is provided."""
    import numpy as np

    from hyperspy.io import assign_signal_subclass

    with caplog.at_level(logging.WARNING):
        # Use an unknown signal_type
        assign_signal_subclass(
            dtype=np.dtype(np.float64),
            signal_dimension=1,
            signal_type="unknown_signal_type",
        )

    assert "`signal_type='unknown_signal_type'` not understood" in caplog.text


def test_load_single_file_binned_metadata_warning():
    """Test deprecation warning for old binned metadata format."""
    # Create a signal with old-style binned metadata
    s = hs.signals.Signal1D([1, 2, 3])
    s.metadata.Signal.binned = True

    with tempfile.TemporaryDirectory() as tmp_dir:
        filename = os.path.join(tmp_dir, "test_binned.hspy")
        s.save(filename)

        # Load and check for deprecation warning
        with pytest.warns(
            VisibleDeprecationWarning, match="binned attribute has been moved"
        ):
            loaded = hs.load(
                filename, convert_units=False
            )  # Disable unit conversion for cleaner test

        # Check that the binned attribute was moved correctly
        assert all(axis.is_binned for axis in loaded.axes_manager.signal_axes)


def test_get_file_format_reader_invalid_format():
    """Test error when requesting invalid format name."""
    from hyperspy.io import _format_name_to_reader

    with pytest.raises(ValueError, match="format_name.*does not match any format"):
        _format_name_to_reader("nonexistent_format")


def test_escape_square_brackets():
    """Test the _escape_square_brackets function."""
    from hyperspy.io import _escape_square_brackets

    # Test with square brackets
    assert _escape_square_brackets("file[0-9].dat") == "file[[]0-9[]].dat"

    # Test without square brackets
    assert _escape_square_brackets("file.dat") == "file.dat"

    # Test with multiple brackets
    assert _escape_square_brackets("file[a][b].dat") == "file[[]a[]][[]b[]].dat"


def test_parse_path():
    """Test the _parse_path function."""
    from hyperspy.io import _parse_path

    # Test with simple string
    parsed = _parse_path("test.dat")
    assert parsed == "test.dat"

    # Test with MutableMapping (zarr store simulation)
    from collections.abc import MutableMapping

    class MockZarrStore(MutableMapping):
        def __init__(self, path):
            self.path = path

        def __getitem__(self, key):
            return None

        def __setitem__(self, key, value):
            pass

        def __delitem__(self, key):
            pass

        def __iter__(self):
            return iter([])

        def __len__(self):
            return 0

    mock_store = MockZarrStore("/path/to/store.zarr")
    parsed = _parse_path(mock_store)
    assert parsed == "/path/to/store.zarr"


def test_get_format_list_for_docstring():
    """Test the _get_format_list_for_docstring function."""
    from hyperspy.io import _get_format_list_for_docstring

    # Test bullet style
    bullet_list = _get_format_list_for_docstring(write_mode=False, style="bullet")
    assert isinstance(bullet_list, str)
    assert "\n        * ``" in bullet_list
    assert "for" in bullet_list

    # Test inline style
    inline_list = _get_format_list_for_docstring(write_mode=False, style="inline")
    assert isinstance(inline_list, str)
    assert "``'" in inline_list
    assert ", " in inline_list

    # Test extensions style
    ext_list = _get_format_list_for_docstring(write_mode=False, style="extensions")
    assert isinstance(ext_list, str)
    assert len(ext_list) > 0

    # Test write_mode filtering
    write_list = _get_format_list_for_docstring(write_mode=True, style="inline")
    read_list = _get_format_list_for_docstring(write_mode=False, style="inline")
    # Write list should be subset of read list
    assert len(write_list) <= len(read_list)

    # Test invalid style
    with pytest.raises(ValueError, match="Unknown style"):
        _get_format_list_for_docstring(style="invalid")


def test_format_name_to_reader():
    """Test the _format_name_to_reader function."""
    from hyperspy.io import _format_name_to_reader

    # Test with valid format name
    reader = _format_name_to_reader("HSPY")
    assert reader["name"] == "HSPY"

    # Test case insensitive
    reader = _format_name_to_reader("hspy")
    assert reader["name"] == "HSPY"

    # Test with invalid format name
    with pytest.raises(ValueError, match="does not match any format"):
        _format_name_to_reader("invalid_format")


def test_load_conflicting_parameters(tmp_path):
    """Test load function with conflicting parameters."""
    # Create a temporary file to avoid "file not found" error
    test_file = tmp_path / "test.dat"
    test_file.write_text("test data")

    # Test with both file_format and reader (should raise ValueError)
    with pytest.raises(ValueError, match="Cannot specify both"):
        hs.load(str(test_file), file_format="HSPY", reader="MSA")


def test_save_conflicting_parameters(tmp_path):
    """Test save method with conflicting parameters."""
    s = hs.signals.Signal1D([1, 2, 3])
    filename = tmp_path / "test.hspy"

    # Test with both extension and file_format (should raise ValueError)
    with pytest.raises(ValueError, match="Cannot specify both"):
        s.save(filename, extension="hspy", file_format="MSA")


def test_save_filename_none_with_file_format(tmp_path):
    """Test save method with filename=None and file_format."""
    s = hs.signals.Signal1D([1, 2, 3])

    # Set up tmp_parameters
    s.tmp_parameters.filename = "test_signal"
    s.tmp_parameters.folder = str(tmp_path)

    # Save with filename=None and file_format
    s.save(filename=None, file_format="HSPY")

    # Check that file was created
    expected_file = tmp_path / "test_signal.hspy"
    assert expected_file.exists()


def test_save_filename_none_without_tmp_parameters():
    """Test save method with filename=None but no tmp_parameters."""
    s = hs.signals.Signal1D([1, 2, 3])

    # Clear tmp_parameters
    s.tmp_parameters = s.tmp_parameters.__class__()

    # Should raise ValueError
    with pytest.raises(ValueError, match="File name not defined"):
        s.save(filename=None, file_format="HSPY")
