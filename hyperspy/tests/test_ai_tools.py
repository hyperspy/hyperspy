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


# Skip all tests if llms_txt is not installed
pytestmark = pytest.mark.skipif(
    pytest.importorskip is None, reason="llms_txt is not installed"
)
