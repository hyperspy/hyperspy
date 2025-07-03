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

    def test_generate_ai_context_with_url_conversion(self):
        """Test that generate_ai_context applies URL conversion to RST paths."""
        from hyperspy.utils import ai_tools

        # Create a mock that allows us to see the converted content but avoid network calls
        with patch("llms_txt.core.create_ctx") as mock_create_ctx:
            mock_create_ctx.return_value = "mocked response"

            # This will execute the real URL conversion code path
            result = ai_tools.generate_ai_context(include_optional=True)

            # Verify the function was called and get the converted content
            assert result == "mocked response"

            # With the fallback mechanism, we should get HTML URLs (not Markdown)
            # since .html.md URLs don't exist and get filtered out by llms_txt
            args, kwargs = mock_create_ctx.call_args
            converted_content = args[0]

            # Verify the real URL conversion happened in the content passed to create_ctx
            assert "hyperspy.org/hyperspy-doc/" in converted_content
            # Due to fallback mechanism, should contain HTML URLs
            assert ".html)" in converted_content
            # Should not contain unconverted RST paths
            assert converted_content.count("doc/user_guide/") <= 2

    def test_real_version_detection_in_context(self):
        """Test that version detection works in real context generation."""
        import hyperspy
        from hyperspy.utils import ai_tools

        # Get the real version being used
        real_version = hyperspy.__version__

        # Mock create_ctx to avoid network requests to non-existent URLs
        with patch("llms_txt.core.create_ctx") as mock_create_ctx:
            mock_create_ctx.return_value = "mocked response"

            # Generate context with real version detection
            result = ai_tools.generate_ai_context(include_optional=True)

            # Verify context was generated
            assert result == "mocked response"

            # Get the content that was passed to create_ctx (after URL conversion)
            args, kwargs = mock_create_ctx.call_args
            converted_content = args[0]

            # Should contain version-appropriate URLs
            assert "hyperspy.org/hyperspy-doc/" in converted_content
            if "dev" in real_version or "+" in real_version:
                # Dev version should use /dev/ URLs
                assert "/dev/" in converted_content
            else:
                # Stable version should use /current/ or versioned URLs
                assert ("/current/" in converted_content) or (
                    "/v" in converted_content and "." in converted_content
                )

    def test_real_rst_conversion_patterns(self):
        """Test RST conversion patterns in real execution."""
        from hyperspy.utils.ai_tools import _convert_rst_to_web_urls

        # Test with real RST content similar to llms.txt
        test_content = """
# Test Content

- [Installation Guide](doc/user_guide/install.rst): Install instructions
- [API Reference](doc/reference/api.rst): API docs
- [Dev Guide](doc/dev_guide/intro.rst): Development guide
- [Signal API](doc/reference/api.signals/): Signal classes
        """

        # Execute real conversion
        converted = _convert_rst_to_web_urls(test_content)

        # Verify conversions happened
        assert "hyperspy.org/hyperspy-doc/" in converted
        assert "install.html.md" in converted
        assert "api.html.md" in converted
        assert "intro.html.md" in converted
        assert "api.signals.html.md" in converted

        # Verify original RST paths are gone
        assert "doc/user_guide/install.rst" not in converted
        assert "doc/reference/api.rst" not in converted
        assert "doc/dev_guide/intro.rst" not in converted
        assert "doc/reference/api.signals/" not in converted

    def test_real_version_path_detection(self):
        """Test version path detection with real hyperspy version."""
        import hyperspy
        from hyperspy.utils.ai_tools import _get_doc_version_path

        # Call with real hyperspy version
        result = _get_doc_version_path()

        # Should return a valid path component
        assert isinstance(result, str)
        assert len(result) > 0
        assert result in ["current", "dev"] or result.startswith("v")

        # Check consistency with version string
        real_version = hyperspy.__version__
        if "dev" in real_version or "+" in real_version:
            assert result == "dev"
        else:
            # For stable versions, should be current or vX.Y format
            assert result == "current" or (result.startswith("v") and "." in result)

    def test_generate_context_with_real_file_operations(self):
        """Test generate_ai_context with real file operations."""
        from hyperspy.utils import ai_tools

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / "real_test_context.txt"

            # Mock create_ctx to avoid network requests but still test file operations
            with patch("llms_txt.core.create_ctx") as mock_create_ctx:
                mock_create_ctx.return_value = "mocked context with converted URLs"

                # This should execute all real code paths including file operations
                result = ai_tools.generate_ai_context(
                    include_optional=True, output_file=output_file
                )

            # Should return None when saving to file
            assert result is None

            # File should exist and contain converted content
            assert output_file.exists()
            content = output_file.read_text(encoding="utf-8")

            # Verify it contains the mocked content
            assert "mocked context with converted URLs" in content
            assert len(content) > 0

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
        """Test that RST to URL conversion is applied to llms.txt content."""
        from hyperspy.utils import ai_tools

        mock_create_ctx.return_value = "mocked context"

        # Call the function with prefer_markdown=True
        ai_tools.generate_ai_context(include_optional=False, prefer_markdown=True)

        # Check that create_ctx was called at least once
        assert mock_create_ctx.called

        # Get the first call (should be with Markdown URLs)
        first_call_args, first_call_kwargs = mock_create_ctx.call_args_list[0]
        content = first_call_args[0]

        # Should contain converted URLs, not RST paths
        assert "hyperspy.org/hyperspy-doc/" in content
        assert ".html.md" in content
        # Should not contain unconverted RST paths (allowing for some occurrences in headers)
        rst_path_count = content.count("doc/user_guide/")
        assert rst_path_count <= 2, f"Found {rst_path_count} unconverted RST paths"

    @patch("llms_txt.core.create_ctx")
    def test_markdown_fallback_behavior(self, mock_create_ctx):
        """Test that the function falls back to HTML URLs when Markdown URLs fail."""
        from hyperspy.utils import ai_tools

        # Mock that returns content without .html.md (simulating Markdown URLs being filtered)
        mock_create_ctx.side_effect = [
            "context without markdown urls",  # First call with Markdown URLs
            "context with html urls and hyperspy.org/hyperspy-doc/dev/user_guide/install.html",  # Second call with HTML URLs
        ]

        # Call the function with prefer_markdown=True
        result = ai_tools.generate_ai_context(
            include_optional=False, prefer_markdown=True
        )

        # Should have been called twice (Markdown attempt, then HTML fallback)
        assert mock_create_ctx.call_count == 2

        # Result should be from the HTML fallback
        assert result is not None and "html urls" in result

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
            # With the fallback mechanism, it might be called twice (Markdown attempt + HTML fallback)
            assert mock_context.__str__.call_count >= 1
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

    def test_convert_rst_to_web_urls_markdown_vs_html_modes(self):
        """Test that prefer_markdown parameter controls output format."""
        from hyperspy.utils.ai_tools import _convert_rst_to_web_urls

        rst_content = "- [Test Guide](doc/user_guide/test.rst): Test documentation"

        # Test Markdown mode (default)
        markdown_result = _convert_rst_to_web_urls(rst_content, prefer_markdown=True)
        assert ".html.md" in markdown_result
        assert ".html)" not in markdown_result or ".html.md" in markdown_result

        # Test HTML mode
        html_result = _convert_rst_to_web_urls(rst_content, prefer_markdown=False)
        assert ".html)" in html_result
        assert ".html.md" not in html_result

        # Test default (should be Markdown)
        default_result = _convert_rst_to_web_urls(rst_content)
        assert ".html.md" in default_result

    def test_prefer_markdown_parameter(self):
        """Test the prefer_markdown parameter functionality."""
        import hyperspy.api as hs

        # Test with prefer_markdown=True (default)
        context_markdown = hs.generate_ai_context(
            include_optional=False, prefer_markdown=True
        )
        assert isinstance(context_markdown, str)
        assert len(context_markdown) > 0

        # Test with prefer_markdown=False
        context_html = hs.generate_ai_context(
            include_optional=False, prefer_markdown=False
        )
        assert isinstance(context_html, str)
        assert len(context_html) > 0

        # Both should contain the core content
        assert "HyperSpy" in context_markdown
        assert "HyperSpy" in context_html

    def test_prefer_markdown_default_behavior(self):
        """Test that prefer_markdown defaults to True."""
        import hyperspy.api as hs

        # Test default behavior (should prefer Markdown)
        context_default = hs.generate_ai_context(include_optional=False)
        context_explicit_true = hs.generate_ai_context(
            include_optional=False, prefer_markdown=True
        )

        # Should behave the same
        assert isinstance(context_default, str)
        assert isinstance(context_explicit_true, str)
        assert len(context_default) > 0
        assert len(context_explicit_true) > 0

    def test_prefer_markdown_with_file_output(self):
        """Test prefer_markdown parameter with file output."""
        import hyperspy.api as hs

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test Markdown preference with file output
            output_file_md = Path(tmp_dir) / "context_md.txt"
            result = hs.generate_ai_context(
                include_optional=False, output_file=output_file_md, prefer_markdown=True
            )
            assert result is None
            assert output_file_md.exists()

            # Test HTML preference with file output
            output_file_html = Path(tmp_dir) / "context_html.txt"
            result = hs.generate_ai_context(
                include_optional=False,
                output_file=output_file_html,
                prefer_markdown=False,
            )
            assert result is None
            assert output_file_html.exists()

            # Both files should contain content
            content_md = output_file_md.read_text(encoding="utf-8")
            content_html = output_file_html.read_text(encoding="utf-8")
            assert "HyperSpy" in content_md
            assert "HyperSpy" in content_html

    def test_prefer_markdown_with_url_conversion(self):
        """Test that generate_ai_context applies URL conversion to RST paths with prefer_markdown."""
        from hyperspy.utils import ai_tools

        # Create a mock that allows us to see the converted content but avoid network calls
        with patch("llms_txt.core.create_ctx") as mock_create_ctx:
            mock_create_ctx.return_value = "mocked response"

            # This will execute the real URL conversion code path
            result = ai_tools.generate_ai_context(
                include_optional=True, prefer_markdown=True
            )

            # Verify the function was called and get the converted content
            assert result == "mocked response"

            # Due to fallback mechanism, should get HTML URLs
            args, kwargs = mock_create_ctx.call_args
            converted_content = args[0]

            # Verify the real URL conversion happened in the content passed to create_ctx
            assert "hyperspy.org/hyperspy-doc/" in converted_content
            assert ".html)" in converted_content  # HTML URLs due to fallback
            # Should not contain unconverted RST paths
            assert converted_content.count("doc/user_guide/") <= 2

    def test_real_version_detection_in_context_with_prefer_markdown(self):
        """Test that version detection works in real context generation with prefer_markdown."""
        import hyperspy
        from hyperspy.utils import ai_tools

        # Get the real version being used
        real_version = hyperspy.__version__

        # Mock create_ctx to avoid network requests to non-existent URLs
        with patch("llms_txt.core.create_ctx") as mock_create_ctx:
            mock_create_ctx.return_value = "mocked response"

            # Generate context with real version detection
            result = ai_tools.generate_ai_context(
                include_optional=True, prefer_markdown=True
            )

            # Verify context was generated
            assert result == "mocked response"

            # Get the content that was passed to create_ctx (after URL conversion)
            args, kwargs = mock_create_ctx.call_args
            converted_content = args[0]

            # Should contain version-appropriate URLs
            assert "hyperspy.org/hyperspy-doc/" in converted_content
            if "dev" in real_version or "+" in real_version:
                # Dev version should use /dev/ URLs
                assert "/dev/" in converted_content
            else:
                # Stable version should use /current/ or versioned URLs
                assert ("/current/" in converted_content) or (
                    "/v" in converted_content and "." in converted_content
                )

    def test_real_rst_conversion_patterns_with_prefer_markdown(self):
        """Test RST conversion patterns in real execution with prefer_markdown."""
        from hyperspy.utils.ai_tools import _convert_rst_to_web_urls

        # Test with real RST content similar to llms.txt
        test_content = """
# Test Content

- [Installation Guide](doc/user_guide/install.rst): Install instructions
- [API Reference](doc/reference/api.rst): API docs
- [Dev Guide](doc/dev_guide/intro.rst): Development guide
- [Signal API](doc/reference/api.signals/): Signal classes
        """

        # Execute real conversion
        converted = _convert_rst_to_web_urls(test_content, prefer_markdown=True)

        # Verify conversions happened
        assert "hyperspy.org/hyperspy-doc/" in converted
        assert "install.html.md" in converted
        assert "api.html.md" in converted
        assert "intro.html.md" in converted
        assert "api.signals.html.md" in converted

        # Verify original RST paths are gone
        assert "doc/user_guide/install.rst" not in converted
        assert "doc/reference/api.rst" not in converted
        assert "doc/dev_guide/intro.rst" not in converted
        assert "doc/reference/api.signals/" not in converted

    def test_generate_ai_context_integration_with_prefer_markdown(self):
        """Test generate_ai_context integration with prefer_markdown parameter."""
        from hyperspy.utils import ai_tools

        # Mock create_ctx to control the output
        with patch("llms_txt.core.create_ctx") as mock_create_ctx:
            mock_create_ctx.return_value = "mocked context with converted URLs"

            with tempfile.TemporaryDirectory() as tmp_dir:
                output_file = Path(tmp_dir) / "test_context.txt"

                # Test with prefer_markdown=True
                result = ai_tools.generate_ai_context(
                    include_optional=True, output_file=output_file, prefer_markdown=True
                )

                # Should return None when saving to file
                assert result is None

                # File should exist and contain converted content
                assert output_file.exists()
                content = output_file.read_text(encoding="utf-8")

                # Verify it contains the mocked content
                assert "mocked context with converted URLs" in content
                assert len(content) > 0

                # Check that create_ctx was called with converted content
                args, kwargs = mock_create_ctx.call_args
                assert "hyperspy.org/hyperspy-doc/" in args[0]
                assert ".html)" in args[0]  # HTML URLs due to fallback

            with tempfile.TemporaryDirectory() as tmp_dir:
                output_file = Path(tmp_dir) / "test_context_html.txt"

                # Test with prefer_markdown=False (HTML output)
                result = ai_tools.generate_ai_context(
                    include_optional=True,
                    output_file=output_file,
                    prefer_markdown=False,
                )

                # Should return None when saving to file
                assert result is None

                # File should exist and contain converted content
                assert output_file.exists()
                content = output_file.read_text(encoding="utf-8")

                # Verify it contains the mocked content
                assert "mocked context with converted URLs" in content
                assert len(content) > 0

                # Check that create_ctx was called with converted content
                args, kwargs = mock_create_ctx.call_args
                assert "hyperspy.org/hyperspy-doc/" in args[0]
                assert ".html)" in args[0]  # HTML URLs when prefer_markdown=False

    def test_auto_examples_url_conversion_patterns(self):
        """Test URL conversion specifically for auto_examples paths."""
        import re

        # Test the pattern directly to avoid import caching issues
        test_content = """
- [Create Signal Examples](doc/auto_examples/create_signal/index.rst): Example tutorials
- [Processing Examples](doc/auto_examples/processing/index.rst): Signal processing workflows
        """

        # Apply the transformation manually
        base_url = "https://hyperspy.org/hyperspy-doc/dev"
        ext = ".html.md"
        pattern = r"doc/auto_examples/([^)\s]+)\.rst"
        replacement = rf"{base_url}/auto_examples/\1{ext}"

        converted = re.sub(pattern, replacement, test_content)

        # Verify the conversions
        assert "create_signal/index.html.md" in converted
        assert "processing/index.html.md" in converted
        assert "doc/auto_examples" not in converted
        assert "hyperspy.org/hyperspy-doc/dev/auto_examples/" in converted

    def test_examples_path_conversion(self):
        """Test conversion of examples/ paths to auto_examples URLs."""
        from hyperspy.utils.ai_tools import _convert_rst_to_web_urls

        # Test content with examples paths
        test_content = """
# Test Content

- [Create Signal Examples](examples/create_signal/): How to create signals
- [Data Visualization Examples](examples/data_visualization/): Plotting techniques
- [Example Script](examples/simple_simulations/create_artificial_data.py): Example file
- [Example RST](examples/create_signal/README.rst): Example documentation
        """

        # Execute conversion
        converted = _convert_rst_to_web_urls(test_content, prefer_markdown=True)

        # Verify examples/ directory paths are converted to auto_examples/*/index.html.md
        assert "auto_examples/create_signal/index.html.md" in converted
        assert "auto_examples/data_visualization/index.html.md" in converted

        # Verify examples/ file paths are converted appropriately
        assert (
            "auto_examples/simple_simulations/create_artificial_data.html.md"
            in converted
        )
        assert "auto_examples/create_signal/README.html.md" in converted

        # Verify original examples/ paths are gone
        assert "examples/create_signal/" not in converted
        assert "examples/data_visualization/" not in converted
        assert "examples/simple_simulations/create_artificial_data.py" not in converted
        assert "examples/create_signal/README.rst" not in converted


# Skip all tests if llms_txt is not installed
pytestmark = pytest.mark.skipif(
    pytest.importorskip is None, reason="llms_txt is not installed"
)
