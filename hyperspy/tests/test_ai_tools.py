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

from unittest.mock import patch


class TestVersionDetection:
    """Test suite for version detection and path resolution functions."""

    def test_get_doc_version_path_dev_version(self):
        """Test version path detection for development versions."""
        from hyperspy.utils.ai_tools import _get_doc_version_path

        with patch("hyperspy.__version__", "2.0.0.dev123"):
            result = _get_doc_version_path()
            assert result == "dev"

    def test_get_doc_version_path_dev_plus_version(self):
        """Test version path detection for versions with + suffix."""
        from hyperspy.utils.ai_tools import _get_doc_version_path

        with patch("hyperspy.__version__", "2.0.0+abc123"):
            result = _get_doc_version_path()
            assert result == "dev"

    def test_get_doc_version_path_alpha_version(self):
        """Test version path detection for alpha versions."""
        from hyperspy.utils.ai_tools import _get_doc_version_path

        with patch("hyperspy.__version__", "2.0.0a1"):
            result = _get_doc_version_path()
            assert result == "dev"

    def test_get_doc_version_path_beta_version(self):
        """Test version path detection for beta versions."""
        from hyperspy.utils.ai_tools import _get_doc_version_path

        with patch("hyperspy.__version__", "2.0.0b2"):
            result = _get_doc_version_path()
            assert result == "dev"

    def test_get_doc_version_path_rc_version(self):
        """Test version path detection for release candidate versions."""
        from hyperspy.utils.ai_tools import _get_doc_version_path

        with patch("hyperspy.__version__", "2.0.0rc1"):
            result = _get_doc_version_path()
            assert result == "dev"

    def test_get_doc_version_path_current_stable(self):
        """Test version path detection for current stable versions."""
        from hyperspy.utils.ai_tools import _get_doc_version_path

        with patch("hyperspy.__version__", "2.3.1"):
            result = _get_doc_version_path()
            assert result == "current"

    def test_get_doc_version_path_older_stable(self):
        """Test version path detection for older stable versions."""
        from hyperspy.utils.ai_tools import _get_doc_version_path

        with patch("hyperspy.__version__", "2.1.0"):
            result = _get_doc_version_path()
            assert result == "v2.1"

    def test_get_doc_version_path_very_old_stable(self):
        """Test version path detection for very old stable versions."""
        from hyperspy.utils.ai_tools import _get_doc_version_path

        with patch("hyperspy.__version__", "1.9.0"):
            result = _get_doc_version_path()
            assert result == "v1.9"

    def test_get_doc_version_path_future_version(self):
        """Test version path detection for future major versions."""
        from hyperspy.utils.ai_tools import _get_doc_version_path

        with patch("hyperspy.__version__", "3.0.0"):
            result = _get_doc_version_path()
            assert result == "current"

    def test_get_doc_version_path_malformed_version(self):
        """Test version path detection with malformed version string."""
        from hyperspy.utils.ai_tools import _get_doc_version_path

        with patch("hyperspy.__version__", "not.a.version"):
            result = _get_doc_version_path()
            assert result == "current"  # Falls back to current

    def test_get_doc_version_path_empty_version(self):
        """Test version path detection with empty version string."""
        from hyperspy.utils.ai_tools import _get_doc_version_path

        with patch("hyperspy.__version__", ""):
            result = _get_doc_version_path()
            assert result == "current"  # Falls back to current

    def test_get_doc_version_path_single_number(self):
        """Test version path detection with single number version."""
        from hyperspy.utils.ai_tools import _get_doc_version_path

        with patch("hyperspy.__version__", "2"):
            result = _get_doc_version_path()
            assert (
                result == "current"
            )  # Falls back to current due to insufficient parts

    def test_get_doc_version_path_major_only(self):
        """Test version path detection with major version only."""
        from hyperspy.utils.ai_tools import _get_doc_version_path

        with patch("hyperspy.__version__", "2."):
            result = _get_doc_version_path()
            assert result == "current"  # Falls back to current due to parsing error

    def test_get_doc_version_path_non_numeric_major(self):
        """Test version path detection with non-numeric major version."""
        from hyperspy.utils.ai_tools import _get_doc_version_path

        with patch("hyperspy.__version__", "v2.3.0"):
            result = _get_doc_version_path()
            assert result == "current"  # Falls back to current due to parsing error

    def test_get_doc_version_path_non_numeric_minor(self):
        """Test version path detection with non-numeric minor version."""
        from hyperspy.utils.ai_tools import _get_doc_version_path

        with patch("hyperspy.__version__", "2.x.0"):
            result = _get_doc_version_path()
            assert result == "current"  # Falls back to current due to parsing error

    def test_get_doc_version_path_complex_dev_version(self):
        """Test version path detection with complex development version strings."""
        from hyperspy.utils.ai_tools import _get_doc_version_path

        with patch("hyperspy.__version__", "2.0.0.dev123+abc456"):
            result = _get_doc_version_path()
            assert result == "dev"


class TestUrlConversion:
    """Test suite for URL conversion functionality."""

    def test_convert_rst_to_web_urls_examples_directories(self):
        """Test conversion of examples directory paths."""
        from hyperspy.utils.ai_tools import _convert_rst_to_web_urls

        test_content = """
- [Create Signal Examples](examples/create_signal/): How to create signals
- [Data Visualization](examples/data_visualization/): Plotting techniques
        """

        result = _convert_rst_to_web_urls(test_content, prefer_markdown=True)

        assert "auto_examples/create_signal/index.html.md" in result
        assert "auto_examples/data_visualization/index.html.md" in result
        assert "](examples/create_signal/)" not in result
        assert "](examples/data_visualization/)" not in result

    def test_convert_rst_to_web_urls_examples_python_files(self):
        """Test conversion of examples Python file paths."""
        from hyperspy.utils.ai_tools import _convert_rst_to_web_urls

        test_content = """
- [Example Script](examples/simple_simulations/create_artificial_data.py): Example file
- [Another Script](examples/plotting/plot_basic.py): Plotting example
        """

        result = _convert_rst_to_web_urls(test_content, prefer_markdown=True)

        assert (
            "auto_examples/simple_simulations/create_artificial_data.py.html.md"
            in result
        )
        assert "auto_examples/plotting/plot_basic.py.html.md" in result
        assert "](examples/simple_simulations/create_artificial_data.py)" not in result
        assert "](examples/plotting/plot_basic.py)" not in result

    def test_convert_rst_to_web_urls_examples_rst_files(self):
        """Test conversion of examples RST file paths."""
        from hyperspy.utils.ai_tools import _convert_rst_to_web_urls

        test_content = """
- [Example Documentation](examples/create_signal/README.rst): Example docs
        """

        result = _convert_rst_to_web_urls(test_content, prefer_markdown=True)

        # README.rst files in directories should become index.html.md
        assert "auto_examples/create_signal/index.html.md" in result
        assert "](examples/create_signal/README.rst)" not in result

    def test_convert_rst_to_web_urls_html_mode_examples(self):
        """Test examples conversion in HTML mode."""
        from hyperspy.utils.ai_tools import _convert_rst_to_web_urls

        test_content = """
- [Examples Directory](examples/create_signal/): Examples
- [Example Script](examples/test.py): Script
        """

        result = _convert_rst_to_web_urls(test_content, prefer_markdown=False)

        assert "auto_examples/create_signal/index.html" in result
        assert "auto_examples/test.py.html" in result
        assert ".html.md" not in result

    def test_convert_rst_to_web_urls_mixed_paths(self):
        """Test conversion of mixed doc and examples paths."""
        from hyperspy.utils.ai_tools import _convert_rst_to_web_urls

        test_content = """
- [User Guide](doc/user_guide/install.rst): Installation
- [Examples](examples/create_signal/): Signal creation
- [API Reference](doc/reference/api.rst): API docs
- [Example Script](examples/test.py): Test script
        """

        result = _convert_rst_to_web_urls(test_content, prefer_markdown=True)

        # Check doc conversions
        assert "user_guide/install.html.md" in result
        assert "reference/api.html.md" in result
        # Check examples conversions
        assert "auto_examples/create_signal/index.html.md" in result
        assert "auto_examples/test.py.html.md" in result
        # Check originals are gone
        assert "](doc/user_guide/install.rst)" not in result
        assert "](examples/create_signal/)" not in result

    def test_convert_rst_to_web_urls_doc_version_none_default(self):
        """Test URL conversion with doc_version=None to trigger auto-detection."""
        from hyperspy.utils.ai_tools import _convert_rst_to_web_urls

        test_content = "- [User Guide](doc/user_guide/install.rst): Installation"

        # This should trigger the internal call to _get_doc_version_path()
        result = _convert_rst_to_web_urls(
            test_content, prefer_markdown=True, doc_version=None
        )

        # Should contain the converted URL
        assert "user_guide/install.html.md" in result
        assert "](doc/user_guide/install.rst)" not in result

    def test_convert_rst_to_web_urls_dev_guide_files(self):
        """Test conversion of dev guide RST files."""
        from hyperspy.utils.ai_tools import _convert_rst_to_web_urls

        test_content = """
- [Contributing](doc/dev_guide/contributing.rst): How to contribute
- [Building Docs](doc/dev_guide/documentation.rst): Documentation guide
        """

        result = _convert_rst_to_web_urls(test_content, prefer_markdown=True)

        assert "dev_guide/contributing.html.md" in result
        assert "dev_guide/documentation.html.md" in result
        assert "](doc/dev_guide/contributing.rst)" not in result
        assert "](doc/dev_guide/documentation.rst)" not in result

    def test_convert_rst_to_web_urls_examples_standalone_rst_files(self):
        """Test conversion of standalone RST files in examples."""
        from hyperspy.utils.ai_tools import _convert_rst_to_web_urls

        test_content = """
- [Example Docs](examples/signals/example_guide.rst): Example documentation
        """

        result = _convert_rst_to_web_urls(test_content, prefer_markdown=True)

        assert "auto_examples/signals/example_guide.rst.html.md" in result
        assert "](examples/signals/example_guide.rst)" not in result

    def test_convert_rst_to_web_urls_directory_paths(self):
        """Test conversion of directory paths for doc sections."""
        from hyperspy.utils.ai_tools import _convert_rst_to_web_urls

        test_content = """
- [Reference Directory](doc/reference/api.signals/): API signals
- [User Guide Section](doc/user_guide/getting_started/): Getting started
- [Dev Guide Section](doc/dev_guide/tools/): Development tools
        """

        result = _convert_rst_to_web_urls(test_content, prefer_markdown=True)

        # Check reference directory conversion
        assert "reference/api.signals.html.md" in result
        # Check user guide directory conversion
        assert "user_guide/getting_started/index.html.md" in result
        # Check dev guide directory conversion
        assert "dev_guide/tools/index.html.md" in result

        # Check originals are gone
        assert "](doc/reference/api.signals/)" not in result
        assert "](doc/user_guide/getting_started/)" not in result
        assert "](doc/dev_guide/tools/)" not in result

    def test_convert_rst_to_web_urls_custom_doc_version(self):
        """Test URL conversion with custom doc_version parameter."""
        from hyperspy.utils.ai_tools import _convert_rst_to_web_urls

        test_content = "- [User Guide](doc/user_guide/install.rst): Installation"

        result = _convert_rst_to_web_urls(
            test_content, prefer_markdown=True, doc_version="v2.5"
        )

        # Should use the custom version
        assert "hyperspy.org/hyperspy-doc/v2.5/user_guide/install.html.md" in result
        assert "](doc/user_guide/install.rst)" not in result

    def test_convert_rst_to_web_urls_no_matches(self):
        """Test URL conversion with content that has no matching patterns."""
        from hyperspy.utils.ai_tools import _convert_rst_to_web_urls

        test_content = """
- [External Link](https://example.com): External resource
- [Simple Text]: No parentheses here
- Some text without any links
        """

        result = _convert_rst_to_web_urls(test_content, prefer_markdown=True)

        # Content should remain unchanged since no patterns match
        assert result == test_content
