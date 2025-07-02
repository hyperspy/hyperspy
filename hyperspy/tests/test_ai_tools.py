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

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("llms_txt", reason="llms_txt is required for AI context generation")


class TestGenerateAIContext:
    """Test suite for generate_ai_context function."""

    def test_function_available_in_api(self):
        """Test that the function is available in hyperspy.api."""
        import hyperspy.api as hs

        assert hasattr(hs, "generate_ai_context")
        assert callable(hs.generate_ai_context)

    def test_generate_basic_context(self):
        """Test generating basic context without optional sections."""
        import hyperspy.api as hs

        context = hs.generate_ai_context(include_optional=False)

        assert isinstance(context, str)
        assert len(context) > 0
        assert "HyperSpy" in context
        # Basic content should be present
        assert "multidimensional" in context.lower()

    def test_generate_full_context(self):
        """Test generating full context with optional sections."""
        import hyperspy.api as hs

        context = hs.generate_ai_context(include_optional=True)

        assert isinstance(context, str)
        assert len(context) > 0
        assert "HyperSpy" in context
        # Full context should be larger than basic context
        basic_context = hs.generate_ai_context(include_optional=False)
        assert len(context) >= len(basic_context)

    def test_save_to_file(self):
        """Test saving context to a file."""
        import hyperspy.api as hs

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / "test_context.txt"

            result = hs.generate_ai_context(
                include_optional=False, output_file=output_file
            )

            # Function should return None when saving to file
            assert result is None

            # File should exist and contain content
            assert output_file.exists()
            content = output_file.read_text(encoding="utf-8")
            assert len(content) > 0
            assert "HyperSpy" in content

    def test_save_to_file_with_string_path(self):
        """Test saving context to a file using string path instead of Path object."""
        import hyperspy.api as hs

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = str(Path(tmp_dir) / "test_context_str.txt")

            result = hs.generate_ai_context(
                include_optional=False, output_file=output_file
            )

            # Function should return None when saving to file
            assert result is None

            # File should exist and contain content
            assert Path(output_file).exists()
            content = Path(output_file).read_text(encoding="utf-8")
            assert len(content) > 0
            assert "HyperSpy" in content

    def test_save_to_nested_directory(self):
        """Test saving context to a file in a nested directory that doesn't exist yet."""
        import hyperspy.api as hs

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / "nested" / "dirs" / "context.txt"

            result = hs.generate_ai_context(
                include_optional=False, output_file=output_file
            )

            # Function should return None when saving to file
            assert result is None

            # File should exist and contain content
            assert output_file.exists()
            content = output_file.read_text(encoding="utf-8")
            assert len(content) > 0
            assert "HyperSpy" in content

    def test_import_error_without_llms_txt(self, monkeypatch):
        """Test that ImportError is raised when llms_txt is not available."""
        import hyperspy.api as hs

        # Mock the import to raise ImportError
        def mock_import(name, *args, **kwargs):
            if name == "llms_txt.core":
                raise ImportError("No module named 'llms_txt'")
            return __import__(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)

        with pytest.raises(ImportError) as excinfo:
            hs.generate_ai_context()

        assert "llms_txt" in str(excinfo.value)
        assert "pip install llms_txt" in str(excinfo.value)
        assert "llmstxt.org" in str(excinfo.value)

    def test_file_not_found_error(self, monkeypatch):
        """Test that FileNotFoundError is raised when llms.txt is missing."""
        from unittest.mock import patch

        import hyperspy.api as hs

        # Mock Path.exists to return False
        with patch.object(Path, "exists", return_value=False):
            with pytest.raises(FileNotFoundError) as excinfo:
                hs.generate_ai_context()

            assert "llms.txt" in str(excinfo.value)
            assert "incomplete HyperSpy installation" in str(excinfo.value)

    def test_runtime_error_on_generation_failure(self, monkeypatch):
        """Test that RuntimeError is raised when context generation fails."""
        from unittest.mock import patch

        import hyperspy.api as hs

        # Mock create_ctx to raise an exception
        def mock_create_ctx(*args, **kwargs):
            raise ValueError("Mocked generation failure")

        with patch("llms_txt.core.create_ctx", side_effect=mock_create_ctx):
            with pytest.raises(RuntimeError) as excinfo:
                hs.generate_ai_context()

            assert "Failed to generate context" in str(excinfo.value)

    def test_function_available_in_utils(self):
        """Test that the function is also available in hyperspy.utils."""
        from hyperspy.utils import generate_ai_context

        assert callable(generate_ai_context)

        # Test that it works the same way
        context = generate_ai_context(include_optional=False)
        assert isinstance(context, str)
        assert len(context) > 0
        assert "HyperSpy" in context

    def test_save_to_file_prints_message(self, capsys):
        """Test that saving to file prints a confirmation message."""
        from hyperspy.utils import ai_tools

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / "test_context.txt"

            ai_tools.generate_ai_context(
                include_optional=False, output_file=output_file
            )

            # Check that a message was printed
            captured = capsys.readouterr()
            assert "AI context saved to:" in captured.out
            assert str(output_file) in captured.out

    def test_utf8_encoding_handling(self):
        """Test that UTF-8 encoding is properly handled for file operations."""
        from hyperspy.utils import ai_tools

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / "test_utf8.txt"

            # Generate and save context
            ai_tools.generate_ai_context(
                include_optional=False, output_file=output_file
            )

            # Read the file back and ensure it's valid UTF-8
            content = output_file.read_text(encoding="utf-8")
            assert isinstance(content, str)
            assert len(content) > 0
            assert "HyperSpy" in content

            # Ensure the file can be read with explicit UTF-8 encoding
            with open(output_file, "r", encoding="utf-8") as f:
                file_content = f.read()
                assert file_content == content

    @patch("pathlib.Path.read_text")
    def test_file_reading_error_handling(self, mock_read_text):
        """Test error handling when reading llms.txt file fails."""
        from hyperspy.utils import ai_tools

        # Mock read_text to raise an exception
        mock_read_text.side_effect = UnicodeDecodeError(
            "utf-8", b"", 0, 1, "invalid start byte"
        )

        with pytest.raises(UnicodeDecodeError):
            ai_tools.generate_ai_context(include_optional=False)

    @patch("pathlib.Path.write_text")
    def test_file_writing_error_handling(self, mock_write_text):
        """Test error handling when writing output file fails."""
        from hyperspy.utils import ai_tools

        # Mock write_text to raise an exception
        mock_write_text.side_effect = PermissionError("Permission denied")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / "test_context.txt"

            with pytest.raises(PermissionError):
                ai_tools.generate_ai_context(
                    include_optional=False, output_file=output_file
                )

    @patch("llms_txt.core.create_ctx")
    def test_optional_parameter_passing(self, mock_create_ctx):
        """Test that the optional parameter is correctly passed to create_ctx."""
        from hyperspy.utils import ai_tools

        mock_create_ctx.return_value = "mocked context"

        # Test with include_optional=False
        ai_tools.generate_ai_context(include_optional=False)
        args, kwargs = mock_create_ctx.call_args
        assert kwargs["optional"] is False

        # Test with include_optional=True
        ai_tools.generate_ai_context(include_optional=True)
        args, kwargs = mock_create_ctx.call_args
        assert kwargs["optional"] is True

    @patch("llms_txt.core.create_ctx")
    def test_content_conversion_applied(self, mock_create_ctx):
        """Test that RST to Markdown URL conversion is applied to llms.txt content."""
        from hyperspy.utils import ai_tools

        mock_create_ctx.return_value = "mocked context"

        # Call the function
        ai_tools.generate_ai_context(include_optional=False)

        # Check that create_ctx was called with converted content
        args, kwargs = mock_create_ctx.call_args
        content = args[0]

        # Should contain Markdown URLs, not RST paths
        assert "hyperspy.org/hyperspy-doc/" in content
        assert ".html.md" in content
        # Should not contain unconverted RST paths (allowing for some occurrences in headers)
        rst_path_count = content.count("doc/user_guide/")
        assert rst_path_count <= 2, f"Found {rst_path_count} unconverted RST paths"

    @patch("llms_txt.core.create_ctx")
    @patch("pathlib.Path.exists")
    def test_hyperspy_installation_path_resolution(self, mock_exists, mock_create_ctx):
        """Test that the llms.txt file path is correctly resolved relative to HyperSpy installation."""
        import hyperspy
        from hyperspy.utils import ai_tools

        mock_create_ctx.return_value = "mocked context"
        mock_exists.return_value = True

        # Use a list to store the accessed path (mutable object)
        accessed_paths = []

        def mock_read_text_with_tracking(self, encoding=None):
            # Store the path that was accessed
            accessed_paths.append(self)
            return "# Mock llms.txt content\n[Test](doc/user_guide/test.rst): Test"

        with patch.object(Path, "read_text", mock_read_text_with_tracking):
            ai_tools.generate_ai_context(include_optional=False)

            # Check that the correct path was accessed
            expected_path = Path(hyperspy.__file__).parent.parent / "llms.txt"
            assert len(accessed_paths) == 1
            assert accessed_paths[0] == expected_path

    def test_string_conversion_of_context_object(self):
        """Test that the context object is properly converted to string."""
        from unittest.mock import MagicMock

        from hyperspy.utils import ai_tools

        # Create a mock context object that has a __str__ method
        mock_context = MagicMock()
        mock_context.__str__ = MagicMock(return_value="stringified context")

        with patch("llms_txt.core.create_ctx", return_value=mock_context):
            result = ai_tools.generate_ai_context(include_optional=False)

            # Verify that str() was called on the context object
            mock_context.__str__.assert_called_once()
            assert result == "stringified context"

    @patch("llms_txt.core.create_ctx")
    def test_both_include_optional_values(self, mock_create_ctx):
        """Test that both True and False values for include_optional work correctly."""
        from hyperspy.utils import ai_tools

        mock_create_ctx.return_value = "mocked context"

        # Test default behavior (should be False)
        ai_tools.generate_ai_context()
        args, kwargs = mock_create_ctx.call_args
        assert kwargs["optional"] is False

        # Test explicit False
        ai_tools.generate_ai_context(include_optional=False)
        args, kwargs = mock_create_ctx.call_args
        assert kwargs["optional"] is False

        # Test explicit True
        ai_tools.generate_ai_context(include_optional=True)
        args, kwargs = mock_create_ctx.call_args
        assert kwargs["optional"] is True


class TestVersionDetection:
    """Test suite for version detection functionality."""

    def test_get_doc_version_path_dev_versions(self):
        """Test version detection for development versions."""
        import hyperspy
        from hyperspy.utils.ai_tools import _get_doc_version_path

        # Test various development version patterns
        dev_versions = [
            "2.4.0.dev30+ga416ec60b.d20250702",
            "2.3.0.dev1",
            "2.3.0+untagged.123.g1234567",
            "2.3.0rc1",
            "2.3.0a1",
            "2.3.0b2",
        ]

        for version in dev_versions:
            with patch.object(hyperspy, "__version__", version):
                result = _get_doc_version_path()
                assert result == "dev", f"Failed for version {version}"

    def test_get_doc_version_path_stable_versions(self):
        """Test version detection for stable versions."""
        import hyperspy
        from hyperspy.utils.ai_tools import _get_doc_version_path

        # Test stable version patterns
        stable_versions = {
            "2.3.1": "current",
            "2.3.0": "current",
            "3.0.0": "current",
            "2.2.0": "v2.2",
            "2.1.0": "v2.1",
            "1.7.0": "v1.7",
        }

        for version, expected in stable_versions.items():
            with patch.object(hyperspy, "__version__", version):
                result = _get_doc_version_path()
                assert result == expected, (
                    f"Failed for version {version}, got {result}, expected {expected}"
                )

    def test_get_doc_version_path_malformed_versions(self):
        """Test version detection for malformed version strings."""
        import hyperspy
        from hyperspy.utils.ai_tools import _get_doc_version_path

        # Test malformed versions that should fallback to 'current'
        malformed_versions = [
            "unknown",
            "",
            "2",
            "not.a.version",
        ]

        for version in malformed_versions:
            with patch.object(hyperspy, "__version__", version):
                result = _get_doc_version_path()
                assert result == "current", (
                    f"Failed to fallback to 'current' for version {version}"
                )

    def test_get_doc_version_path_single_part_version(self):
        """Test version detection for single-part version strings."""
        import hyperspy
        from hyperspy.utils.ai_tools import _get_doc_version_path

        # Test versions with only one part (should fallback to current)
        single_part_versions = ["2", "3", "1"]

        for version in single_part_versions:
            with patch.object(hyperspy, "__version__", version):
                result = _get_doc_version_path()
                assert result == "current", (
                    f"Failed to fallback to 'current' for single-part version {version}"
                )

    def test_get_doc_version_path_exception_handling(self):
        """Test version detection when int conversion fails."""
        import hyperspy
        from hyperspy.utils.ai_tools import _get_doc_version_path

        # Test versions that will cause ValueError in int() conversion
        problematic_versions = [
            "2.3.alpha",  # non-numeric minor
            "two.three.one",  # non-numeric major
            "2.3beta.1",  # non-numeric minor with beta
            "v2.3.1",  # prefixed version
        ]

        for version in problematic_versions:
            with patch.object(hyperspy, "__version__", version):
                result = _get_doc_version_path()
                assert result == "current", (
                    f"Failed to fallback to 'current' for problematic version {version}"
                )

    def test_get_doc_version_path_extra_version_parts(self):
        """Test version detection with extra version parts."""
        import hyperspy
        from hyperspy.utils.ai_tools import _get_doc_version_path

        # Test versions with more than 3 parts
        extra_part_versions = {
            "2.3.1.4": "current",  # Should still use major.minor logic
            "2.2.0.1": "v2.2",  # Older version with extra part
            "1.7.5.2": "v1.7",  # Even older version
            "3.0.0.0": "current",  # Future version
        }

        for version, expected in extra_part_versions.items():
            with patch.object(hyperspy, "__version__", version):
                result = _get_doc_version_path()
                assert result == expected, (
                    f"Failed for version {version}, got {result}, expected {expected}"
                )

    def test_get_doc_version_path_edge_case_versions(self):
        """Test version detection for edge case version patterns."""
        import hyperspy
        from hyperspy.utils.ai_tools import _get_doc_version_path

        # Test edge cases around the boundary conditions
        edge_cases = {
            "2.2.9": "v2.2",  # Just below current threshold
            "2.3.0": "current",  # At the threshold
            "2.99.0": "current",  # High minor version
            "1.0.0": "v1.0",  # Low version
        }

        for version, expected in edge_cases.items():
            with patch.object(hyperspy, "__version__", version):
                result = _get_doc_version_path()
                assert result == expected, (
                    f"Failed for edge case version {version}, got {result}, expected {expected}"
                )


class TestRSTToMarkdownConversion:
    """Test suite for RST to Markdown URL conversion."""

    def test_convert_rst_to_markdown_urls_basic(self):
        """Test basic RST to Markdown URL conversion."""
        from hyperspy.utils.ai_tools import (
            _convert_rst_to_markdown_urls,
            _get_doc_version_path,
        )

        # Mock content with RST paths
        rst_content = """
# HyperSpy

## Getting Started

- [Installation Guide](doc/user_guide/install.rst): Complete installation instructions
- [Basic Usage](doc/user_guide/basic_usage.rst): Fundamental concepts

## API Reference

- [Main API](doc/reference/api.rst): Complete API reference
- [Signal Classes](doc/reference/api.signals/): All signal types
        """

        # Convert to Markdown URLs
        markdown_content = _convert_rst_to_markdown_urls(rst_content)

        # Check that conversions happened
        doc_version = _get_doc_version_path()
        base_url = f"https://hyperspy.org/hyperspy-doc/{doc_version}"

        assert f"{base_url}/user_guide/install.html.md" in markdown_content
        assert f"{base_url}/user_guide/basic_usage.html.md" in markdown_content
        assert f"{base_url}/reference/api.html.md" in markdown_content
        assert f"{base_url}/reference/api.signals.html.md" in markdown_content

        # Check that original RST paths are gone
        assert "doc/user_guide/install.rst" not in markdown_content
        assert "doc/user_guide/basic_usage.rst" not in markdown_content
        assert "doc/reference/api.rst" not in markdown_content
        assert "doc/reference/api.signals/" not in markdown_content

    def test_convert_rst_to_markdown_urls_with_dev_version(self):
        """Test RST to Markdown conversion with development version."""
        import hyperspy
        from hyperspy.utils.ai_tools import _convert_rst_to_markdown_urls

        rst_content = "- [Test](doc/user_guide/test.rst): Test link"

        with patch.object(hyperspy, "__version__", "2.4.0.dev30+abc123"):
            markdown_content = _convert_rst_to_markdown_urls(rst_content)
            assert (
                "https://hyperspy.org/hyperspy-doc/dev/user_guide/test.html.md"
                in markdown_content
            )

    def test_convert_rst_to_markdown_urls_with_stable_version(self):
        """Test RST to Markdown conversion with stable version."""
        import hyperspy
        from hyperspy.utils.ai_tools import _convert_rst_to_markdown_urls

        rst_content = "- [Test](doc/user_guide/test.rst): Test link"

        with patch.object(hyperspy, "__version__", "2.3.1"):
            markdown_content = _convert_rst_to_markdown_urls(rst_content)
            assert (
                "https://hyperspy.org/hyperspy-doc/current/user_guide/test.html.md"
                in markdown_content
            )

    def test_convert_rst_to_markdown_urls_preserves_non_doc_links(self):
        """Test that non-doc links are preserved unchanged."""
        from hyperspy.utils.ai_tools import _convert_rst_to_markdown_urls

        rst_content = """
- [GitHub Examples](https://github.com/hyperspy/hyperspy/tree/main/examples): Examples
- [External Link](https://example.com): External link
- [Installation Guide](doc/user_guide/install.rst): This should be converted
        """

        markdown_content = _convert_rst_to_markdown_urls(rst_content)

        # External links should be preserved
        assert (
            "https://github.com/hyperspy/hyperspy/tree/main/examples"
            in markdown_content
        )
        assert "https://example.com" in markdown_content

        # Doc links should be converted
        assert "doc/user_guide/install.rst" not in markdown_content
        assert "hyperspy.org/hyperspy-doc/" in markdown_content

    def test_convert_rst_to_markdown_urls_all_patterns(self):
        """Test conversion of all supported RST path patterns."""
        from hyperspy.utils.ai_tools import (
            _convert_rst_to_markdown_urls,
            _get_doc_version_path,
        )

        rst_content = """
- [User Guide](doc/user_guide/basic_usage.rst): User guide
- [Dev Guide](doc/dev_guide/intro.rst): Development guide
- [API Reference](doc/reference/api.rst): API reference
- [Signal API](doc/reference/api.signals/): Signal API directory
- [Signal Processing](doc/user_guide/signal/index.rst): Signal processing guide
- [MVA Guide](doc/user_guide/mva/index.rst): Multivariate analysis guide
        """

        markdown_content = _convert_rst_to_markdown_urls(rst_content)
        doc_version = _get_doc_version_path()
        base_url = f"https://hyperspy.org/hyperspy-doc/{doc_version}"

        # Check all pattern types are converted
        assert f"{base_url}/user_guide/basic_usage.html.md" in markdown_content
        assert f"{base_url}/dev_guide/intro.html.md" in markdown_content
        assert f"{base_url}/reference/api.html.md" in markdown_content
        assert f"{base_url}/reference/api.signals.html.md" in markdown_content
        assert f"{base_url}/user_guide/signal/index.html.md" in markdown_content
        assert f"{base_url}/user_guide/mva/index.html.md" in markdown_content

    def test_convert_rst_to_markdown_urls_no_rst_paths(self):
        """Test conversion when content has no RST paths."""
        from hyperspy.utils.ai_tools import _convert_rst_to_markdown_urls

        rst_content = """
# HyperSpy

This is some content without any documentation links.

- [External Link](https://example.com): External link
- [GitHub](https://github.com/hyperspy/hyperspy): GitHub repo
        """

        markdown_content = _convert_rst_to_markdown_urls(rst_content)

        # Content should remain unchanged
        assert markdown_content == rst_content
        # External links should be preserved
        assert "https://example.com" in markdown_content
        assert "https://github.com/hyperspy/hyperspy" in markdown_content

    def test_convert_rst_to_markdown_urls_nested_subdirectories(self):
        """Test conversion of nested subdirectory paths."""
        from hyperspy.utils.ai_tools import (
            _convert_rst_to_markdown_urls,
            _get_doc_version_path,
        )

        rst_content = """
- [Deep Signal Guide](doc/user_guide/signal/advanced/fitting.rst): Advanced fitting
- [Deep Dev Guide](doc/dev_guide/testing/unit_tests.rst): Unit testing
- [Deep Reference](doc/reference/api/signals/eels.rst): EELS API
        """

        markdown_content = _convert_rst_to_markdown_urls(rst_content)
        doc_version = _get_doc_version_path()
        base_url = f"https://hyperspy.org/hyperspy-doc/{doc_version}"

        # Check that nested paths are correctly converted
        assert (
            f"{base_url}/user_guide/signal/advanced/fitting.html.md" in markdown_content
        )
        assert f"{base_url}/dev_guide/testing/unit_tests.html.md" in markdown_content
        assert f"{base_url}/reference/api/signals/eels.html.md" in markdown_content

        # Check that original paths are gone
        assert "doc/user_guide/signal/advanced/fitting.rst" not in markdown_content
        assert "doc/dev_guide/testing/unit_tests.rst" not in markdown_content
        assert "doc/reference/api/signals/eels.rst" not in markdown_content

    def test_convert_rst_to_markdown_urls_malformed_paths(self):
        """Test conversion handles malformed or partial RST paths gracefully."""
        from hyperspy.utils.ai_tools import _convert_rst_to_markdown_urls

        rst_content = """
- [Good Path](doc/user_guide/install.rst): This should be converted
- [Partial Path](doc/user_guide/): No .rst extension
- [Wrong Path](docs/user_guide/install.rst): Wrong directory name
- [No Extension](doc/user_guide/install): No .rst extension
- [Empty](doc//install.rst): Double slash
        """

        markdown_content = _convert_rst_to_markdown_urls(rst_content)

        # Only the good path should be converted
        assert "hyperspy.org/hyperspy-doc/" in markdown_content
        assert "install.html.md" in markdown_content

        # Malformed paths should remain unchanged
        assert "doc/user_guide/):" in markdown_content  # Partial path preserved
        assert "docs/user_guide/install.rst" in markdown_content  # Wrong dir preserved
        assert "doc/user_guide/install):" in markdown_content  # No extension preserved
        assert "doc//install.rst" in markdown_content  # Double slash preserved

    def test_convert_rst_to_markdown_urls_multiple_occurrences(self):
        """Test conversion of multiple occurrences of the same path."""
        from hyperspy.utils.ai_tools import (
            _convert_rst_to_markdown_urls,
            _get_doc_version_path,
        )

        rst_content = """
- [Installation](doc/user_guide/install.rst): Install guide
- [Install Again](doc/user_guide/install.rst): Same link again
- [Basic Usage](doc/user_guide/basic_usage.rst): Basic guide
- [Basic Again](doc/user_guide/basic_usage.rst): Same basic guide
        """

        markdown_content = _convert_rst_to_markdown_urls(rst_content)
        doc_version = _get_doc_version_path()
        base_url = f"https://hyperspy.org/hyperspy-doc/{doc_version}"

        # All occurrences should be converted
        assert markdown_content.count(f"{base_url}/user_guide/install.html.md") == 2
        assert markdown_content.count(f"{base_url}/user_guide/basic_usage.html.md") == 2

        # No original paths should remain
        assert "doc/user_guide/install.rst" not in markdown_content
        assert "doc/user_guide/basic_usage.rst" not in markdown_content

    def test_convert_rst_to_markdown_urls_empty_content(self):
        """Test conversion with empty content."""
        from hyperspy.utils.ai_tools import _convert_rst_to_markdown_urls

        empty_content = ""
        result = _convert_rst_to_markdown_urls(empty_content)
        assert result == ""

    def test_convert_rst_to_markdown_urls_only_whitespace(self):
        """Test conversion with only whitespace content."""
        from hyperspy.utils.ai_tools import _convert_rst_to_markdown_urls

        whitespace_content = "   \n\n  \t  \n  "
        result = _convert_rst_to_markdown_urls(whitespace_content)
        assert result == whitespace_content


class TestGenerateAIContextIntegration:
    """Test integration of new functionality with generate_ai_context."""

    @patch("llms_txt.core.create_ctx")
    def test_url_conversion_integration(self, mock_create_ctx):
        """Test that generate_ai_context correctly uses URL conversion logic."""
        import hyperspy
        from hyperspy.utils import ai_tools

        mock_create_ctx.return_value = "mocked context"

        # Test with development version
        with patch.object(hyperspy, "__version__", "2.4.0.dev30+abc123"):
            ai_tools.generate_ai_context(include_optional=True)
            args, kwargs = mock_create_ctx.call_args
            content = args[0]
            assert "hyperspy.org/hyperspy-doc/dev/" in content
            assert ".html.md" in content
            assert "doc/user_guide/basic_usage.rst" not in content
            assert kwargs["optional"] is True

        # Test with stable version
        with patch.object(hyperspy, "__version__", "2.3.1"):
            ai_tools.generate_ai_context(include_optional=True)
            args, kwargs = mock_create_ctx.call_args
            content = args[0]
            assert "hyperspy.org/hyperspy-doc/current/" in content
            assert ".html.md" in content
            assert "doc/user_guide/basic_usage.rst" not in content
            assert kwargs["optional"] is True


# Skip all tests if llms_txt is not installed
pytestmark = pytest.mark.skipif(
    pytest.importorskip is None, reason="llms_txt is not installed"
)
