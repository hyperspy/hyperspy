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
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestCopyMarkdownDocs:
    """Test suite for copy_markdown_docs.py script functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_llms_content = """# HyperSpy

Basic content for testing.

Documentation:
- [Installation](https://hyperspy.org/hyperspy-doc/current/user_guide/install.html.md)
"""

        self.test_llms_ctx_content = """# HyperSpy Context

Expanded context content.

Multiple sections with local file content included.
"""

    def test_copy_llms_txt_both_files_exist(self, capsys):
        """Test copy_llms_txt when both files exist."""
        from scripts.copy_markdown_docs import copy_llms_txt

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create source files
            doc_dir = tmp_path / "doc"
            doc_dir.mkdir()
            llms_source = doc_dir / "llms.txt"
            ctx_source = doc_dir / "llms-ctx.txt"

            llms_source.write_text(self.test_llms_content, encoding="utf-8")
            ctx_source.write_text(self.test_llms_ctx_content, encoding="utf-8")

            # Create HTML directory
            html_dir = tmp_path / "html"
            html_dir.mkdir()

            # Change to test directory to simulate script execution context
            original_cwd = os.getcwd()
            try:
                os.chdir(tmp_path)
                copy_llms_txt(html_dir)
            finally:
                os.chdir(original_cwd)

            # Verify files were copied
            llms_target = html_dir / "llms.txt"
            ctx_target = html_dir / "llms-ctx.txt"

            assert llms_target.exists()
            assert ctx_target.exists()

            # Verify content
            assert llms_target.read_text(encoding="utf-8") == self.test_llms_content
            assert ctx_target.read_text(encoding="utf-8") == self.test_llms_ctx_content

            # Check printed messages
            captured = capsys.readouterr()
            assert "Copying llms.txt" in captured.out
            assert "Copying llms-ctx.txt" in captured.out
            assert "Successfully copied llms.txt" in captured.out
            assert "Successfully copied llms-ctx.txt" in captured.out
            assert "web-compatible URLs" in captured.out
            assert "expanded context" in captured.out

    def test_copy_llms_txt_missing_source_files(self, capsys):
        """Test copy_llms_txt when source files are missing."""
        from scripts.copy_markdown_docs import copy_llms_txt

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create HTML directory but no source files
            html_dir = tmp_path / "html"
            html_dir.mkdir()

            # Change to test directory
            original_cwd = os.getcwd()
            try:
                os.chdir(tmp_path)
                copy_llms_txt(html_dir)
            finally:
                os.chdir(original_cwd)

            # Verify files were not created
            llms_target = html_dir / "llms.txt"
            ctx_target = html_dir / "llms-ctx.txt"

            assert not llms_target.exists()
            assert not ctx_target.exists()

            # Check warning messages
            captured = capsys.readouterr()
            assert "WARNING: llms.txt not found" in captured.out
            assert "WARNING: llms-ctx.txt not found" in captured.out

    def test_copy_llms_txt_only_basic_file_exists(self, capsys):
        """Test copy_llms_txt when only llms.txt exists."""
        from scripts.copy_markdown_docs import copy_llms_txt

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create only basic llms.txt
            doc_dir = tmp_path / "doc"
            doc_dir.mkdir()
            llms_source = doc_dir / "llms.txt"
            llms_source.write_text(self.test_llms_content, encoding="utf-8")

            # Create HTML directory
            html_dir = tmp_path / "html"
            html_dir.mkdir()

            # Change to test directory
            original_cwd = os.getcwd()
            try:
                os.chdir(tmp_path)
                copy_llms_txt(html_dir)
            finally:
                os.chdir(original_cwd)

            # Verify only basic file was copied
            llms_target = html_dir / "llms.txt"
            ctx_target = html_dir / "llms-ctx.txt"

            assert llms_target.exists()
            assert not ctx_target.exists()

            # Check messages
            captured = capsys.readouterr()
            assert "Successfully copied llms.txt" in captured.out
            assert "WARNING: llms-ctx.txt not found" in captured.out

    def test_copy_llms_txt_url_counting(self, capsys):
        """Test that URL counting works correctly."""
        from scripts.copy_markdown_docs import copy_llms_txt

        content_with_urls = """# Test
- [Link 1](https://example.com/file1.html.md)
- [Link 2](https://example.com/file2.html.md)
- [Link 3](https://example.com/file3.html.md)
"""

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create source file with URLs
            doc_dir = tmp_path / "doc"
            doc_dir.mkdir()
            llms_source = doc_dir / "llms.txt"
            llms_source.write_text(content_with_urls, encoding="utf-8")

            # Create HTML directory
            html_dir = tmp_path / "html"
            html_dir.mkdir()

            # Change to test directory
            original_cwd = os.getcwd()
            try:
                os.chdir(tmp_path)
                copy_llms_txt(html_dir)
            finally:
                os.chdir(original_cwd)

            # Check URL count in output
            captured = capsys.readouterr()
            assert "3 web-compatible URLs" in captured.out

    def test_main_function_success(self, capsys):
        """Test main function successful execution."""
        from scripts.copy_markdown_docs import main

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Mock environment variable
            with patch.dict(os.environ, {"READTHEDOCS_OUTPUT": str(tmp_path)}):
                # Create required directories
                html_dir = tmp_path / "html"
                html_dir.mkdir()
                temp_md_dir = tmp_path / "temp_markdown"
                temp_md_dir.mkdir()

                # Create some test markdown files
                (temp_md_dir / "index.md").write_text("# Index", encoding="utf-8")
                subdir = temp_md_dir / "user_guide"
                subdir.mkdir()
                (subdir / "install.md").write_text("# Install", encoding="utf-8")

                # Create llms.txt files for copy function
                doc_dir = tmp_path / "doc"
                doc_dir.mkdir()
                (doc_dir / "llms.txt").write_text(
                    self.test_llms_content, encoding="utf-8"
                )

                # Mock subprocess.run for sphinx build
                mock_result = MagicMock()
                mock_result.stdout = "Sphinx build output"

                with patch("subprocess.run", return_value=mock_result) as mock_run:
                    # Change to temp directory to simulate execution context
                    original_cwd = os.getcwd()
                    try:
                        os.chdir(tmp_path)
                        main()
                    finally:
                        os.chdir(original_cwd)

                # Verify subprocess was called correctly
                mock_run.assert_called_once()
                args = mock_run.call_args[0][0]
                assert sys.executable in args
                assert "-m" in args
                assert "sphinx" in args
                assert "-b" in args
                assert "markdown" in args

                # Verify markdown files were copied with correct extensions
                assert (html_dir / "index.html.md").exists()
                assert (html_dir / "user_guide" / "install.html.md").exists()

                # Verify llms.txt was copied
                assert (html_dir / "llms.txt").exists()

                # Check output messages
                captured = capsys.readouterr()
                assert "Building Markdown documentation" in captured.out
                assert "Sphinx Markdown build completed" in captured.out
                assert "Successfully copied Markdown documentation" in captured.out

    def test_main_function_missing_readthedocs_output(self):
        """Test main function when READTHEDOCS_OUTPUT is not set."""
        from scripts.copy_markdown_docs import main

        # Clear the environment variable
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(SystemExit) as excinfo:
                main()

            assert excinfo.value.code == 1

    def test_main_function_sphinx_build_failure(self, capsys):
        """Test main function when Sphinx build fails."""
        from scripts.copy_markdown_docs import main

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with patch.dict(os.environ, {"READTHEDOCS_OUTPUT": str(tmp_path)}):
                # Mock subprocess.run to raise CalledProcessError
                error = subprocess.CalledProcessError(1, "sphinx")
                error.stdout = "Build stdout"
                error.stderr = "Build stderr"

                with patch("subprocess.run", side_effect=error):
                    with pytest.raises(SystemExit) as excinfo:
                        main()

                    assert excinfo.value.code == 1

                # Check error output
                captured = capsys.readouterr()
                assert "ERROR: Sphinx Markdown build failed" in captured.out
                assert "Return code: 1" in captured.out

    def test_main_function_general_exception(self, capsys):
        """Test main function when a general exception occurs."""
        from scripts.copy_markdown_docs import main

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with patch.dict(os.environ, {"READTHEDOCS_OUTPUT": str(tmp_path)}):
                # Mock subprocess.run to raise a general exception
                with patch("subprocess.run", side_effect=Exception("General error")):
                    with pytest.raises(SystemExit) as excinfo:
                        main()

                    assert excinfo.value.code == 1

                # Check error output
                captured = capsys.readouterr()
                assert "ERROR: Failed to copy Markdown documentation" in captured.out
                assert "General error" in captured.out

    def test_main_function_cleanup_temp_directory(self):
        """Test that temporary directory is cleaned up."""
        from scripts.copy_markdown_docs import main

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with patch.dict(os.environ, {"READTHEDOCS_OUTPUT": str(tmp_path)}):
                # Create required directories
                html_dir = tmp_path / "html"
                html_dir.mkdir()

                # Create the temporary markdown directory that would be created during execution
                temp_markdown_dir = tmp_path / "temp_markdown"
                temp_markdown_dir.mkdir()

                # Mock subprocess.run
                mock_result = MagicMock()
                mock_result.stdout = "Build output"

                with patch("subprocess.run", return_value=mock_result):
                    with patch("shutil.rmtree") as mock_rmtree:
                        original_cwd = os.getcwd()
                        try:
                            os.chdir(tmp_path)
                            main()
                        finally:
                            os.chdir(original_cwd)

                        # Verify cleanup was called
                        assert mock_rmtree.called
                        # The temp directory path should be in the call
                        call_args = mock_rmtree.call_args[0][0]
                        assert "temp_markdown" in str(call_args)

    def test_main_function_no_markdown_files(self, capsys):
        """Test main function when no markdown files are generated."""
        from scripts.copy_markdown_docs import main

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with patch.dict(os.environ, {"READTHEDOCS_OUTPUT": str(tmp_path)}):
                # Create required directories but no markdown files
                html_dir = tmp_path / "html"
                html_dir.mkdir()

                # Mock subprocess.run
                mock_result = MagicMock()
                mock_result.stdout = "Build output"

                with patch("subprocess.run", return_value=mock_result):
                    original_cwd = os.getcwd()
                    try:
                        os.chdir(tmp_path)
                        main()
                    finally:
                        os.chdir(original_cwd)

                # Check warning message
                captured = capsys.readouterr()
                assert "WARNING: Temporary Markdown directory" in captured.out
                assert "does not exist" in captured.out

    def test_markdown_file_extension_conversion(self):
        """Test that .md files are correctly renamed to .html.md."""
        from scripts.copy_markdown_docs import main

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with patch.dict(os.environ, {"READTHEDOCS_OUTPUT": str(tmp_path)}):
                # Create required directories
                html_dir = tmp_path / "html"
                html_dir.mkdir()
                temp_md_dir = tmp_path / "temp_markdown"
                temp_md_dir.mkdir()

                # Create test files with various structures
                files_to_create = [
                    "index.md",
                    "user_guide/install.md",
                    "user_guide/basic_usage.md",
                    "reference/api.md",
                    "dev_guide/contributing.md",
                ]

                for file_path in files_to_create:
                    full_path = temp_md_dir / file_path
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    full_path.write_text(f"Content of {file_path}", encoding="utf-8")

                # Mock subprocess.run
                mock_result = MagicMock()
                mock_result.stdout = "Build output"

                with patch("subprocess.run", return_value=mock_result):
                    original_cwd = os.getcwd()
                    try:
                        os.chdir(tmp_path)
                        main()
                    finally:
                        os.chdir(original_cwd)

                # Verify all files were converted correctly
                expected_files = [
                    "index.html.md",
                    "user_guide/install.html.md",
                    "user_guide/basic_usage.html.md",
                    "reference/api.html.md",
                    "dev_guide/contributing.html.md",
                ]

                for expected_file in expected_files:
                    target_path = html_dir / expected_file
                    assert target_path.exists(), (
                        f"Expected file {expected_file} was not created"
                    )

                    # Verify content was preserved
                    original_name = expected_file.replace(".html.md", ".md")
                    expected_content = f"Content of {original_name}"
                    assert expected_content in target_path.read_text(encoding="utf-8")

    def test_copy_llms_txt_character_count(self, capsys):
        """Test that character count is correctly reported for llms-ctx.txt."""
        from scripts.copy_markdown_docs import copy_llms_txt

        large_context = "A" * 1000 + "\n" + "B" * 500  # 1501 characters

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create source files
            doc_dir = tmp_path / "doc"
            doc_dir.mkdir()
            ctx_source = doc_dir / "llms-ctx.txt"
            ctx_source.write_text(large_context, encoding="utf-8")

            # Create HTML directory
            html_dir = tmp_path / "html"
            html_dir.mkdir()

            # Change to test directory
            original_cwd = os.getcwd()
            try:
                os.chdir(tmp_path)
                copy_llms_txt(html_dir)
            finally:
                os.chdir(original_cwd)

            # Check character count in output
            captured = capsys.readouterr()
            assert "1501 characters of expanded context" in captured.out

    def test_integration_with_real_directory_structure(self):
        """Integration test with more realistic directory structure."""
        from scripts.copy_markdown_docs import main

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with patch.dict(os.environ, {"READTHEDOCS_OUTPUT": str(tmp_path)}):
                # Create realistic directory structure
                html_dir = tmp_path / "html"
                html_dir.mkdir()
                temp_md_dir = tmp_path / "temp_markdown"
                temp_md_dir.mkdir()

                # Create doc structure for llms files
                doc_dir = tmp_path / "doc"
                doc_dir.mkdir()

                # Create realistic llms.txt content
                llms_content = """# HyperSpy

Documentation links:
- [Install](https://hyperspy.org/hyperspy-doc/current/user_guide/install.html.md)
- [API](https://hyperspy.org/hyperspy-doc/current/reference/api.html.md)
"""
                (doc_dir / "llms.txt").write_text(llms_content, encoding="utf-8")
                (doc_dir / "llms-ctx.txt").write_text(
                    "Extended context", encoding="utf-8"
                )

                # Create markdown files similar to real Sphinx output
                md_files = {
                    "index.md": "# HyperSpy Documentation",
                    "user_guide/index.md": "# User Guide",
                    "user_guide/install.md": "# Installation",
                    "reference/index.md": "# API Reference",
                    "dev_guide/index.md": "# Developer Guide",
                }

                for file_path, content in md_files.items():
                    full_path = temp_md_dir / file_path
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    full_path.write_text(content, encoding="utf-8")

                # Mock successful Sphinx build
                mock_result = MagicMock()
                mock_result.stdout = "Sphinx build completed successfully"

                with patch("subprocess.run", return_value=mock_result):
                    original_cwd = os.getcwd()
                    try:
                        os.chdir(tmp_path)
                        main()
                    finally:
                        os.chdir(original_cwd)

                # Verify complete integration
                # 1. Markdown files converted
                for file_path in md_files.keys():
                    html_md_path = html_dir / file_path.replace(".md", ".html.md")
                    assert html_md_path.exists()

                # 2. LLMS files copied
                assert (html_dir / "llms.txt").exists()
                assert (html_dir / "llms-ctx.txt").exists()

                # 3. Content preserved
                llms_content_copied = (html_dir / "llms.txt").read_text(
                    encoding="utf-8"
                )
                assert "Documentation links:" in llms_content_copied
                assert "hyperspy.org" in llms_content_copied
