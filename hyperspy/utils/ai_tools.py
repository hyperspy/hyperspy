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
Tools for AI integration and LLM context generation.
"""

import re
from pathlib import Path


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


def _convert_rst_to_web_urls(llms_content, prefer_markdown=True):
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

    Returns
    -------
    str
        The content with RST paths converted to web URLs.
    """
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
    ]

    # Apply transformations
    converted_content = llms_content
    for pattern, replacement in transformations:
        converted_content = re.sub(pattern, replacement, converted_content)

    return converted_content


def generate_ai_context(include_optional=False, output_file=None, prefer_markdown=True):
    """
    Generate AI context from HyperSpy's llms.txt file.

    This function uses the `llms_txt` package to expand HyperSpy's llms.txt file
    into a comprehensive context file suitable for AI/LLM interactions. The function
    automatically converts local RST file paths to version-appropriate web URLs
    based on the current HyperSpy installation.

    Parameters
    ----------
    include_optional : bool, default False
        If True, include optional sections and web content in the generated context.
        This creates a more comprehensive but larger context file.
    output_file : str or Path, optional
        Path where to save the generated context. If None, returns the context as a string.
    prefer_markdown : bool, default True
        If True, prefer Markdown format URLs (.md) for better AI readability.
        If False, use HTML URLs (.html). If Markdown URLs are not available,
        automatically falls back to HTML.

    Returns
    -------
    str or None
        If output_file is None, returns the generated context as a string.
        If output_file is provided, saves to file and returns None.

    Raises
    ------
    ImportError
        If the llms_txt package is not installed.
    FileNotFoundError
        If the llms.txt file cannot be found in the HyperSpy installation.

    Examples
    --------
    >>> import hyperspy.api as hs
    >>> # Generate basic context as string with Markdown URLs
    >>> context = hs.generate_ai_context()
    >>>
    >>> # Generate context with HTML URLs
    >>> context = hs.generate_ai_context(prefer_markdown=False)
    >>>
    >>> # Generate full context with web content and save to file
    >>> hs.generate_ai_context(include_optional=True, output_file="hyperspy_context.txt")

    Notes
    -----
    This function requires the `llms_txt` package to be installed:

        pip install llms_txt

    The function automatically detects the HyperSpy version and converts local
    RST file paths in llms.txt to appropriate web URLs:

    - Development versions (containing 'dev', '+', or pre-release markers) → `/dev/` docs
    - Stable releases → `/current/` docs (for latest) or `/vX.Y/` docs (for older versions)

    URL Format Selection:
    - prefer_markdown=True: URLs end with .md for better AI readability
    - prefer_markdown=False: URLs end with .html for standard web access
    - Automatic fallback: If Markdown URLs aren't accessible, falls back to HTML

    The generated context includes:
    - Project overview and key concepts
    - Documentation links (expanded if include_optional=True)
    - API references and examples
    - Extension ecosystem information
    - Usage patterns and best practices
    """
    try:
        from llms_txt.core import create_ctx
    except ImportError as e:
        raise ImportError(
            "The 'llms_txt' package is required to generate AI context files. "
            "Please install it using:\n\n"
            "    pip install llms_txt\n\n"
            "For more information, see: https://llmstxt.org/"
        ) from e

    # Find the llms.txt file in the HyperSpy installation
    import hyperspy

    hyperspy_dir = Path(hyperspy.__file__).parent.parent
    llms_file = hyperspy_dir / "llms.txt"

    if not llms_file.exists():
        raise FileNotFoundError(
            f"Could not find llms.txt file at {llms_file}. "
            "This might indicate an incomplete HyperSpy installation."
        )

    # Read the llms.txt content
    llms_content = llms_file.read_text(encoding="utf-8")

    # Try Markdown URLs first if preferred
    context_str = None
    if prefer_markdown:
        llms_content_markdown = _convert_rst_to_web_urls(
            llms_content, prefer_markdown=True
        )
        try:
            context = create_ctx(llms_content_markdown, optional=include_optional)
            context_str = str(context)

            # Check if Markdown URLs are actually present in the output
            if ".md" in context_str:
                # Success with Markdown URLs
                pass
            else:
                # Markdown URLs were filtered out, try HTML fallback
                context_str = None
        except Exception:
            # Error with Markdown URLs, try HTML fallback
            context_str = None

    # If Markdown failed or wasn't preferred, use HTML URLs
    if context_str is None:
        llms_content_html = _convert_rst_to_web_urls(
            llms_content, prefer_markdown=False
        )
        try:
            context = create_ctx(llms_content_html, optional=include_optional)
            context_str = str(context)
        except Exception as e:
            raise RuntimeError(f"Failed to generate context from llms.txt: {e}") from e

    # Handle output
    if output_file is not None:
        output_path = Path(output_file)
        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(context_str, encoding="utf-8")
        print(f"AI context saved to: {output_path}")
        return None
    else:
        return context_str
