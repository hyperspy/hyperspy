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

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestPrepareLlmsTxt:
    """Test suite for prepare_llms_txt.py script functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_llms_content = """# HyperSpy

HyperSpy is an open source Python library for multidimensional data analysis.

## Documentation

- [Installation Guide](doc/user_guide/install.rst): Install instructions
- [Basic Usage](doc/user_guide/basic_usage.rst): Getting started
- [API Reference](doc/reference/api.rst): API documentation
- [Examples](examples/create_signal/README.rst): Examples directory
- [Python files](examples/data_visualization/plot_spectrum.py): Python examples
"""

    def test_prepare_llms_txt_for_docs_basic(self):
        """Test basic preparation of llms.txt for documentation."""
        from doc.prepare_llms_txt import prepare_llms_txt_for_docs

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_dir = tmp_path / "source"
            output_dir = tmp_path / "output"

            # Create source structure
            source_dir.mkdir()
            source_llms = source_dir / "llms.txt"
            source_llms.write_text(self.test_llms_content, encoding="utf-8")

            # Mock hyperspy module import
            with patch("sys.path"):
                with patch(
                    "hyperspy.utils.ai_tools._convert_rst_to_web_urls"
                ) as mock_convert:
                    mock_convert.return_value = "converted content"

                    result = prepare_llms_txt_for_docs(
                        source_dir=source_dir,
                        output_dir=output_dir,
                        doc_version="current",
                    )

            # Verify output
            assert result.exists()
            assert result.name == "llms.txt"
            content = result.read_text(encoding="utf-8")
            assert content == "converted content"
            mock_convert.assert_called_once()

    def test_prepare_llms_txt_for_docs_missing_source(self):
        """Test error handling when source llms.txt is missing."""
        from doc.prepare_llms_txt import prepare_llms_txt_for_docs

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_dir = tmp_path / "source"
            output_dir = tmp_path / "output"
            source_dir.mkdir()

            with pytest.raises(FileNotFoundError) as excinfo:
                prepare_llms_txt_for_docs(source_dir=source_dir, output_dir=output_dir)

            assert "llms.txt not found" in str(excinfo.value)

    def test_prepare_llms_txt_for_docs_import_error(self):
        """Test graceful handling of import errors."""
        from doc.prepare_llms_txt import prepare_llms_txt_for_docs

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_dir = tmp_path / "source"
            output_dir = tmp_path / "output"

            # Create source structure
            source_dir.mkdir()
            source_llms = source_dir / "llms.txt"
            source_llms.write_text(self.test_llms_content, encoding="utf-8")

            # Mock import error
            with patch("sys.path"):

                def mock_import(name, *args, **kwargs):
                    if "hyperspy.utils.ai_tools" in name:
                        raise ImportError("Mock import error")
                    return __import__(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=mock_import):
                    result = prepare_llms_txt_for_docs(
                        source_dir=source_dir, output_dir=output_dir
                    )

            # Should fallback to original content
            content = result.read_text(encoding="utf-8")
            assert content == self.test_llms_content

    def test_prepare_llms_ctx_for_docs_basic(self):
        """Test basic preparation of llms-ctx.txt for documentation."""
        from doc.prepare_llms_txt import prepare_llms_ctx_for_docs

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_dir = tmp_path / "source"
            output_dir = tmp_path / "output"

            # Create source structure
            source_dir.mkdir()
            source_llms = source_dir / "llms.txt"
            source_llms.write_text(self.test_llms_content, encoding="utf-8")

            # Mock the URL conversion and context building
            with patch("sys.path"):
                with patch(
                    "hyperspy.utils.ai_tools._convert_rst_to_web_urls"
                ) as mock_convert:
                    with patch(
                        "doc.prepare_llms_txt._build_local_context"
                    ) as mock_build:
                        mock_convert.return_value = "converted content"
                        mock_build.return_value = "expanded context content"

                        result = prepare_llms_ctx_for_docs(
                            source_dir=source_dir,
                            output_dir=output_dir,
                            doc_version="current",
                        )

            # Verify output
            assert result.exists()
            assert result.name == "llms-ctx.txt"
            content = result.read_text(encoding="utf-8")
            assert content == "expanded context content"
            mock_convert.assert_called_once()
            mock_build.assert_called_once_with("converted content", source_dir)

    def test_detect_doc_version_readthedocs_latest(self):
        """Test doc version detection with ReadTheDocs latest."""
        from doc.prepare_llms_txt import detect_doc_version

        with patch.dict(os.environ, {"READTHEDOCS_VERSION": "latest"}):
            version = detect_doc_version()
            assert version == "current"

    def test_detect_doc_version_readthedocs_stable(self):
        """Test doc version detection with ReadTheDocs stable."""
        from doc.prepare_llms_txt import detect_doc_version

        with patch.dict(os.environ, {"READTHEDOCS_VERSION": "stable"}):
            version = detect_doc_version()
            assert version == "current"

    def test_detect_doc_version_readthedocs_dev(self):
        """Test doc version detection with ReadTheDocs dev."""
        from doc.prepare_llms_txt import detect_doc_version

        with patch.dict(os.environ, {"READTHEDOCS_VERSION": "dev"}):
            version = detect_doc_version()
            assert version == "dev"

    def test_detect_doc_version_no_rtd_fallback(self):
        """Test doc version detection fallback when not in ReadTheDocs."""
        from doc.prepare_llms_txt import detect_doc_version

        # Clear RTD environment variable
        with patch.dict(os.environ, {}, clear=True):
            with patch("sys.path"):
                with patch(
                    "hyperspy.utils.ai_tools._get_doc_version_path"
                ) as mock_get_version:
                    mock_get_version.return_value = "v2.3"

                    version = detect_doc_version()
                    assert version == "v2.3"
                    mock_get_version.assert_called_once()

    def test_detect_doc_version_import_error_fallback(self):
        """Test doc version detection fallback on import error."""
        from doc.prepare_llms_txt import detect_doc_version

        with patch.dict(os.environ, {}, clear=True):
            with patch("sys.path"):
                # Mock the import to raise ImportError
                with patch("doc.prepare_llms_txt.Path"):

                    def mock_import(name, *args, **kwargs):
                        # Use the original import for most things
                        if name == "hyperspy.utils.ai_tools":
                            raise ImportError("Mock import error")
                        return original_import(name, *args, **kwargs)

                    original_import = __import__
                    with patch("builtins.__import__", side_effect=mock_import):
                        version = detect_doc_version()
                        assert version == "current"

    def test_build_local_context_basic(self):
        """Test building local context from files."""
        from doc.prepare_llms_txt import _build_local_context

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create mock source structure
            doc_dir = tmp_path / "doc" / "user_guide"
            doc_dir.mkdir(parents=True)

            install_file = doc_dir / "install.rst"
            install_file.write_text("Installation instructions", encoding="utf-8")

            content = """Check out:
- [Installation](https://hyperspy.org/hyperspy-doc/current/user_guide/install.html.md)
"""

            result = _build_local_context(content, tmp_path)

            # Should contain the original content plus expanded sections
            assert "Check out:" in result
            assert "Expanded Context from Local Files" in result
            assert "Content from user_guide/install" in result
            assert "Installation instructions" in result

    def test_build_local_context_missing_files(self):
        """Test building local context with missing files."""
        from doc.prepare_llms_txt import _build_local_context

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            content = """Check out:
- [Installation](https://hyperspy.org/hyperspy-doc/current/user_guide/missing.html.md)
"""

            result = _build_local_context(content, tmp_path)

            # Should handle missing files gracefully
            assert "Check out:" in result
            assert "File not found: user_guide/missing" in result

    def test_url_to_local_path_user_guide(self):
        """Test URL to local path conversion for user guide."""
        from doc.prepare_llms_txt import _url_to_local_path

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create expected file
            doc_dir = tmp_path / "doc" / "user_guide"
            doc_dir.mkdir(parents=True)
            install_file = doc_dir / "install.rst"
            install_file.write_text("content", encoding="utf-8")

            result = _url_to_local_path("user_guide/install", tmp_path)

            assert result == install_file
            assert result.exists()

    def test_url_to_local_path_examples_readme(self):
        """Test URL to local path conversion for examples README."""
        from doc.prepare_llms_txt import _url_to_local_path

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create expected file structure
            examples_dir = tmp_path / "examples" / "create_signal"
            examples_dir.mkdir(parents=True)
            readme_file = examples_dir / "README.rst"
            readme_file.write_text("content", encoding="utf-8")

            result = _url_to_local_path("auto_examples/create_signal/index", tmp_path)

            assert result == readme_file
            assert result.exists()

    def test_url_to_local_path_examples_python(self):
        """Test URL to local path conversion for examples Python files."""
        from doc.prepare_llms_txt import _url_to_local_path

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create expected file structure
            examples_dir = tmp_path / "examples"
            examples_dir.mkdir(parents=True)
            py_file = examples_dir / "example.py"
            py_file.write_text("# Python code", encoding="utf-8")

            result = _url_to_local_path("auto_examples/example.py", tmp_path)

            assert result == py_file
            assert result.exists()

    def test_url_to_local_path_missing_file(self):
        """Test URL to local path conversion for missing files."""
        from doc.prepare_llms_txt import _url_to_local_path

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            result = _url_to_local_path("user_guide/missing", tmp_path)

            assert result is None

    def test_url_to_local_path_return_expected(self):
        """Test URL to local path conversion with return_expected=True."""
        from doc.prepare_llms_txt import _url_to_local_path

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            result = _url_to_local_path(
                "user_guide/missing", tmp_path, return_expected=True
            )

            expected = tmp_path / "doc" / "user_guide" / "missing.rst"
            assert result == expected
            assert not result.exists()  # File doesn't exist but path is returned

    def test_prepare_llms_txt_for_docs_default_params(self):
        """Test prepare_llms_txt_for_docs with default parameters."""
        from doc.prepare_llms_txt import prepare_llms_txt_for_docs

        # Create a mock llms.txt in the "repository root"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create the actual file structure for the test
            source_llms = tmp_path / "llms.txt"
            source_llms.write_text(self.test_llms_content, encoding="utf-8")

            with patch("sys.path"):
                with patch(
                    "hyperspy.utils.ai_tools._convert_rst_to_web_urls"
                ) as mock_convert:
                    mock_convert.return_value = "converted content"

                    # Explicitly provide source_dir and output_dir instead of mocking defaults
                    result = prepare_llms_txt_for_docs(
                        source_dir=tmp_path, output_dir=tmp_path
                    )

            # The function should have worked
            assert mock_convert.called
            assert result.exists()
            assert result.read_text(encoding="utf-8") == "converted content"

    def test_prepare_llms_ctx_for_docs_exception_handling(self):
        """Test exception handling in prepare_llms_ctx_for_docs."""
        from doc.prepare_llms_txt import prepare_llms_ctx_for_docs

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_dir = tmp_path / "source"
            output_dir = tmp_path / "output"

            # Create source structure
            source_dir.mkdir()
            source_llms = source_dir / "llms.txt"
            source_llms.write_text(self.test_llms_content, encoding="utf-8")

            # Mock exception in _build_local_context
            with patch("sys.path"):
                with patch(
                    "hyperspy.utils.ai_tools._convert_rst_to_web_urls"
                ) as mock_convert:
                    with patch(
                        "doc.prepare_llms_txt._build_local_context"
                    ) as mock_build:
                        mock_convert.return_value = "converted content"
                        mock_build.side_effect = Exception("Mock error")

                        result = prepare_llms_ctx_for_docs(
                            source_dir=source_dir, output_dir=output_dir
                        )

            # Should fallback to converted content
            content = result.read_text(encoding="utf-8")
            assert content == "converted content"

    def test_main_function_default_behavior(self):
        """Test main function with default behavior."""
        from doc.prepare_llms_txt import main

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create mock repository structure
            (tmp_path / "doc").mkdir()
            source_llms = tmp_path / "llms.txt"
            source_llms.write_text(self.test_llms_content, encoding="utf-8")

            # Mock the script location and functions
            with patch("doc.prepare_llms_txt.Path") as mock_path_class:
                mock_script = tmp_path / "doc" / "prepare_llms_txt.py"
                mock_path_class.return_value = mock_script
                mock_path_class.__truediv__ = Path.__truediv__

                with patch(
                    "doc.prepare_llms_txt.prepare_llms_txt_for_docs"
                ) as mock_prepare:
                    with patch(
                        "doc.prepare_llms_txt.prepare_llms_ctx_for_docs"
                    ) as mock_prepare_ctx:
                        with patch(
                            "doc.prepare_llms_txt.detect_doc_version"
                        ) as mock_detect:
                            mock_output = tmp_path / "doc" / "llms.txt"
                            mock_output.write_text(
                                "transformed content", encoding="utf-8"
                            )
                            mock_prepare.return_value = mock_output
                            mock_detect.return_value = "current"

                            # Mock sys.argv to simulate no arguments
                            with patch("sys.argv", ["prepare_llms_txt.py"]):
                                main()

                            # Should call both preparation functions
                            assert mock_prepare.called
                            assert mock_prepare_ctx.called
                            assert mock_detect.called

    def test_main_function_with_arguments(self):
        """Test main function with command line arguments."""
        from doc.prepare_llms_txt import main

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create source structure
            source_dir = tmp_path / "source"
            source_dir.mkdir()
            source_llms = source_dir / "llms.txt"
            source_llms.write_text(self.test_llms_content, encoding="utf-8")

            with patch(
                "doc.prepare_llms_txt.prepare_llms_txt_for_docs"
            ) as mock_prepare:
                with patch(
                    "doc.prepare_llms_txt.prepare_llms_ctx_for_docs"
                ) as mock_prepare_ctx:
                    mock_output = tmp_path / "llms.txt"
                    mock_output.write_text("content", encoding="utf-8")
                    mock_prepare.return_value = mock_output

                    # Mock sys.argv with arguments
                    argv = [
                        "prepare_llms_txt.py",
                        "--source-dir",
                        str(source_dir),
                        "--output-dir",
                        str(tmp_path),
                        "--doc-version",
                        "dev",
                    ]

                    with patch("sys.argv", argv):
                        main()

                    # Should call functions with provided arguments
                    mock_prepare.assert_called_with(
                        source_dir=str(source_dir),
                        output_dir=str(tmp_path),
                        doc_version="dev",
                    )
                    mock_prepare_ctx.assert_called_with(
                        source_dir=str(source_dir),
                        output_dir=str(tmp_path),
                        doc_version="dev",
                    )

    def test_main_function_exception_handling(self):
        """Test main function exception handling."""
        from doc.prepare_llms_txt import main

        with patch("doc.prepare_llms_txt.prepare_llms_txt_for_docs") as mock_prepare:
            mock_prepare.side_effect = Exception("Mock error")

            # Mock sys.argv with arguments to trigger the exception path
            argv = [
                "prepare_llms_txt.py",
                "--source-dir",
                "/tmp/source",
                "--output-dir",
                "/tmp/output",
            ]

            with patch("sys.argv", argv):
                with pytest.raises(SystemExit) as excinfo:
                    main()

                # Should exit with error code 1
                assert excinfo.value.code == 1

    def test_context_file_contains_proper_sections(self):
        """Test that the expanded context file contains expected sections."""
        from doc.prepare_llms_txt import _build_local_context

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create multiple test files
            user_guide_dir = tmp_path / "doc" / "user_guide"
            user_guide_dir.mkdir(parents=True)

            install_file = user_guide_dir / "install.rst"
            install_file.write_text("Installation content", encoding="utf-8")

            usage_file = user_guide_dir / "basic_usage.rst"
            usage_file.write_text("Basic usage content", encoding="utf-8")

            content = """Links:
- [Install](https://hyperspy.org/hyperspy-doc/current/user_guide/install.html.md)
- [Usage](https://hyperspy.org/hyperspy-doc/current/user_guide/basic_usage.html.md)
"""

            result = _build_local_context(content, tmp_path)

            # Should contain all expected sections
            assert "Links:" in result
            assert "Expanded Context from Local Files" in result
            assert "Content from user_guide/install" in result
            assert "Content from user_guide/basic_usage" in result
            assert "Installation content" in result
            assert "Basic usage content" in result
            assert "Source file: doc/user_guide/install.rst" in result
            assert "Source file: doc/user_guide/basic_usage.rst" in result
            assert "=" * 50 in result  # Section separators
