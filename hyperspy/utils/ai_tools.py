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

"""
Internal tools for AI integration and LLM context generation.

This module contains internal utilities used by documentation build scripts
to generate llms.txt and llms-ctx.txt files for AI/LLM integration.
"""

import re


def _get_doc_version_path():
    """
    Determine the appropriate documentation version path based on HyperSpy version.

    Returns
    -------
    str
        The version path component for hyperspy.org documentation URLs.
        Examples: 'current', 'dev', 'v2.2', etc.
    """
    import hyperspy

    version = hyperspy.__version__

    # Check if it's a development version (contains 'dev', '+', or 'a'/'b'/'rc')
    # For 'a', 'b', 'rc' patterns, make sure they're followed by numbers (e.g., 'a1', 'b2', 'rc1')
    import re

    if (
        "dev" in version
        or "+" in version
        or re.search(r"[ab]\d+", version)
        or re.search(r"rc\d+", version)
    ):
        return "dev"

    # For stable releases, parse the version
    # Version format is typically: "2.3.1" or "2.3.0"
    try:
        # Remove any extra suffixes and parse
        clean_version = version.split("+")[0].split("dev")[0]
        version_parts = clean_version.split(".")

        if len(version_parts) >= 2:
            major = int(version_parts[0])
            minor = int(version_parts[1])

            # For now, map based on major.minor
            # This logic can be enhanced to check what's actually the latest
            version_string = f"v{major}.{minor}"

            # For the current latest stable versions, use 'current'
            # Update this logic as new versions are released
            if (major == 2 and minor >= 3) or major > 2:
                return "current"
            else:
                return version_string

    except (ValueError, IndexError):
        pass

    # Fallback to current for any parsing errors
    return "current"


def _convert_rst_to_web_urls(llms_content, prefer_markdown=True, doc_version=None):
    """
    Convert local RST file paths in llms.txt content to appropriate web URLs.

    Prefers Markdown format for better AI readability, with HTML fallback.

    Parameters
    ----------
    llms_content : str
        The content of the llms.txt file with local RST paths.
    prefer_markdown : bool, optional
        If True, generates Markdown URLs (.md). If False, generates HTML URLs (.html).
        Default is True for better AI readability.
    doc_version : str, optional
        The documentation version to use. If None, auto-detects from HyperSpy version.

    Returns
    -------
    str
        The content with RST paths converted to web URLs.
    """
    if doc_version is None:
        doc_version = _get_doc_version_path()

    base_url = f"https://hyperspy.org/hyperspy-doc/{doc_version}"

    # Choose extension based on preference
    ext = ".html.md" if prefer_markdown else ".html"

    # Transform local paths to web URLs
    transformations = [
        # Examples paths - specific file extensions first (most specific patterns first)
        (r"examples/([^)]+\.py)\)", rf"{base_url}/auto_examples/\1{ext})"),
        # Examples paths - README.rst files in directories (examples/create_signal/README.rst -> auto_examples/create_signal/index.ext)
        (
            r"examples/([^)]+)/README\.rst\)",
            rf"{base_url}/auto_examples/\1/index{ext})",
        ),
        (r"examples/([^)]+\.rst)\)", rf"{base_url}/auto_examples/\1{ext})"),
        # Examples paths - directories ending with / (examples/create_signal/ -> base_url/auto_examples/create_signal/index.ext)
        (r"examples/([^)]+)/\)", rf"{base_url}/auto_examples/\1/index{ext})"),
        # Documentation files (doc/path/file.rst -> base_url/path/file.ext)
        (r"doc/user_guide/([^)]+)\.rst\)", rf"{base_url}/user_guide/\1{ext})"),
        (r"doc/dev_guide/([^)]+)\.rst\)", rf"{base_url}/dev_guide/\1{ext})"),
        (r"doc/reference/([^)]+)\.rst\)", rf"{base_url}/reference/\1{ext})"),
        # Handle directory paths without .rst extension (doc/reference/api.signals/ -> base_url/reference/api.signals.ext)
        (r"doc/reference/([^)]+)/\)", rf"{base_url}/reference/\1{ext})"),
        # Handle user_guide and dev_guide directory paths
        (r"doc/user_guide/([^)]+)/\)", rf"{base_url}/user_guide/\1/index{ext})"),
        (r"doc/dev_guide/([^)]+)/\)", rf"{base_url}/dev_guide/\1/index{ext})"),
    ]

    # Apply transformations
    converted_content = llms_content
    for pattern, replacement in transformations:
        converted_content = re.sub(pattern, replacement, converted_content)

    return converted_content
