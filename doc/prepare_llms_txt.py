#!/usr/bin/env python3
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
Script to prepare llms.txt and llms-ctx.txt for documentation build.

This script copies the root llms.txt file and transforms it for web documentation
by converting RST paths to .html.md URLs using the existing ai_tools.py logic.
It also generates an expanded context file (llms-ctx.txt) with full content.

Usage:
------
1. Automatic (default): python prepare_llms_txt.py
   - Copies from repository root to doc/ directory
   - Uses "current" documentation version
   - Used automatically during Sphinx documentation builds
   - Generates both llms.txt and llms-ctx.txt

2. Manual with options: python prepare_llms_txt.py --doc-version dev --output-dir /tmp
   - Allows customization of source, output, and documentation version

Integration:
-----------
This script is automatically called during Sphinx documentation builds via
the setup() function in conf.py, ensuring that the online documentation
always has up-to-date llms.txt and llms-ctx.txt files with web-compatible URLs.

The transformation ensures that:
- Local RST paths (doc/user_guide/install.rst) become web URLs
- URLs point to the correct documentation version (current, dev, v2.x)
- URLs use .html.md extension for better AI/LLM readability
- Files are placed at the documentation root for web crawler access
- llms.txt contains the basic transformed content
- llms-ctx.txt contains the expanded full context with fetched content
"""

import re
import sys
from pathlib import Path


def prepare_llms_txt_for_docs(source_dir=None, output_dir=None, doc_version="current"):
    """
    Prepare llms.txt for documentation build.

    Parameters
    ----------
    source_dir : str or Path, optional
        Directory containing the source llms.txt file.
        If None, uses the repository root.
    output_dir : str or Path, optional
        Directory where to place the transformed llms.txt file.
        If None, uses the current directory.
    doc_version : str, optional
        Documentation version for URL generation. Default is "current".
    """
    # Determine source directory (repository root)
    if source_dir is None:
        # This script is in doc/, so the repo root is one level up
        script_dir = Path(__file__).parent
        source_dir = script_dir.parent
    source_dir = Path(source_dir)

    # Determine output directory
    if output_dir is None:
        output_dir = Path.cwd()
    output_dir = Path(output_dir)

    # Source llms.txt file
    source_file = source_dir / "llms.txt"
    if not source_file.exists():
        raise FileNotFoundError(f"Source llms.txt not found at: {source_file}")

    # Read the source file
    llms_content = source_file.read_text(encoding="utf-8")

    # Import the transformation function
    # Add the source directory to path to import hyperspy modules
    sys.path.insert(0, str(source_dir))
    try:
        from hyperspy.utils.ai_tools import _convert_rst_to_web_urls

        # Transform the content for web documentation
        # Use prefer_markdown=True to generate .html.md URLs for better AI readability
        transformed_content = _convert_rst_to_web_urls(
            llms_content, prefer_markdown=True, doc_version=doc_version
        )

    except ImportError as e:
        print(f"Warning: Could not import transformation function: {e}")
        print("Using source content without transformation.")
        transformed_content = llms_content
    finally:
        # Remove the added path
        sys.path.pop(0)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write the transformed file
    output_file = output_dir / "llms.txt"
    output_file.write_text(transformed_content, encoding="utf-8")

    print(f"Transformed llms.txt written to: {output_file}")
    return output_file


def prepare_llms_ctx_for_docs(source_dir=None, output_dir=None, doc_version="current"):
    """
    Prepare llms-ctx.txt (expanded context) for documentation build.

    Parameters
    ----------
    source_dir : str or Path, optional
        Directory containing the source llms.txt file.
        If None, uses the repository root.
    output_dir : str or Path, optional
        Directory where to place the expanded context file.
        If None, uses the current directory.
    doc_version : str, optional
        Documentation version for URL generation. Default is "current".
    """
    # Determine source directory (repository root)
    if source_dir is None:
        # This script is in doc/, so the repo root is one level up
        script_dir = Path(__file__).parent
        source_dir = script_dir.parent
    source_dir = Path(source_dir)

    # Determine output directory
    if output_dir is None:
        output_dir = Path.cwd()
    output_dir = Path(output_dir)

    # Source llms.txt file
    source_file = source_dir / "llms.txt"
    if not source_file.exists():
        raise FileNotFoundError(f"Source llms.txt not found at: {source_file}")

    # Read the source file
    llms_content = source_file.read_text(encoding="utf-8")

    # Import the transformation function and context generation
    # Add the source directory to path to import hyperspy modules
    sys.path.insert(0, str(source_dir))
    try:
        from hyperspy.utils.ai_tools import _convert_rst_to_web_urls

        # Transform the content for web documentation
        # Use prefer_markdown=True to generate .html.md URLs for better AI readability
        transformed_content = _convert_rst_to_web_urls(
            llms_content, prefer_markdown=True, doc_version=doc_version
        )

        # Generate the expanded context
        try:
            # For documentation builds, we want to use local files
            # instead of fetching from URLs that don't exist yet
            print("Building expanded context using local files...")

            # Build the context using local files directly
            expanded_content = _build_local_context(transformed_content, source_dir)

            print(f"Generated expanded context with {len(expanded_content)} characters")

        except Exception as e:
            print(f"Warning: Failed to generate expanded context: {e}")
            print("Using basic transformed content instead")
            expanded_content = transformed_content

    except ImportError as e:
        print(f"Warning: Could not import transformation function: {e}")
        print("Using source content without transformation.")
        expanded_content = llms_content
    finally:
        # Remove the added path
        sys.path.pop(0)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write the expanded context file
    output_file = output_dir / "llms-ctx.txt"
    output_file.write_text(expanded_content, encoding="utf-8")

    print(f"Expanded context written to: {output_file}")
    return output_file


def detect_doc_version():
    """
    Detect the appropriate documentation version for URL generation.

    Returns
    -------
    str
        The documentation version (e.g., 'current', 'dev', 'v2.3').
    """
    import os

    # Check if we're in a ReadTheDocs environment
    rtd_version = os.environ.get("READTHEDOCS_VERSION")
    if rtd_version:
        print(f"ReadTheDocs environment detected, version: {rtd_version}")
        # ReadTheDocs version mapping
        if rtd_version == "latest":
            return "current"
        elif rtd_version == "stable":
            return "current"
        else:
            # For other versions, use as-is (dev, v2.3, etc.)
            return rtd_version

    # Fallback to auto-detection from HyperSpy version
    try:
        # Add current directory to path to import hyperspy modules
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))

        from hyperspy.utils.ai_tools import _get_doc_version_path

        version = _get_doc_version_path()
        print(f"Auto-detected documentation version: {version}")
        return version
    except ImportError:
        print("Could not auto-detect version, using 'current'")
        return "current"


def _build_local_context(content, source_dir):
    """
    Build expanded context by reading local files instead of fetching URLs.

    This function is used specifically during documentation builds to include
    content from local files instead of trying to fetch from web URLs.

    Parameters
    ----------
    content : str
        Content with web URLs (e.g., https://hyperspy.org/hyperspy-doc/current/...)
    source_dir : Path
        Source directory (repository root)

    Returns
    -------
    str
        Expanded content with local file contents included
    """
    import re

    # Start with the basic content
    expanded_parts = [content]
    expanded_parts.append("\n\n# Expanded Context from Local Files\n")

    # Find all .html.md URLs in the content
    url_pattern = r"https://hyperspy\.org/hyperspy-doc/[^/]+/([^)]+)\.html\.md"
    urls = re.findall(url_pattern, content)

    for relative_url in urls:
        # Convert URL path to local file path
        local_path = _url_to_local_path(relative_url, source_dir)

        if local_path and local_path.exists():
            try:
                # Read the local file content
                file_content = local_path.read_text(encoding="utf-8")

                # Add a section header
                expanded_parts.append(f"\n## Content from {relative_url}\n")
                expanded_parts.append(
                    f"Source file: {local_path.relative_to(source_dir)}\n\n"
                )
                expanded_parts.append(file_content)
                expanded_parts.append("\n" + "=" * 50 + "\n")

            except Exception as e:
                expanded_parts.append(f"\n## Error reading {relative_url}\n")
                expanded_parts.append(f"Could not read {local_path}: {e}\n\n")
        else:
            expanded_parts.append(f"\n## File not found: {relative_url}\n")
            expected_path = _url_to_local_path(
                relative_url, source_dir, return_expected=True
            )
            expanded_parts.append(f"Expected at: {expected_path}\n\n")

    return "".join(expanded_parts)


def _url_to_local_path(relative_url, source_dir, return_expected=False):
    """
    Convert a relative URL to a local file path.

    Parameters
    ----------
    relative_url : str
        Relative URL path (e.g., "user_guide/install")
    source_dir : Path
        Source directory (repository root)
    return_expected : bool
        If True, always return a path even if file doesn't exist

    Returns
    -------
    Path or None
        Local file path, or None if file doesn't exist and return_expected=False
    """
    source_dir = Path(source_dir)

    # Map URL patterns to local file paths
    if relative_url.startswith("user_guide/"):
        local_path = source_dir / "doc" / f"{relative_url}.rst"
    elif relative_url.startswith("dev_guide/"):
        local_path = source_dir / "doc" / f"{relative_url}.rst"
    elif relative_url.startswith("reference/"):
        local_path = source_dir / "doc" / f"{relative_url}.rst"
    elif relative_url.startswith("auto_examples/"):
        if relative_url.endswith("/index"):
            # auto_examples/create_signal/index -> examples/create_signal/README.rst
            dir_name = relative_url.replace("auto_examples/", "").replace("/index", "")
            local_path = source_dir / "examples" / dir_name / "README.rst"
        elif relative_url.endswith(".py"):
            # auto_examples/file.py -> examples/file.py
            file_name = relative_url.replace("auto_examples/", "")
            local_path = source_dir / "examples" / file_name
        else:
            # auto_examples/something.rst -> examples/something.rst
            file_name = relative_url.replace("auto_examples/", "")
            local_path = source_dir / "examples" / f"{file_name}.rst"
    else:
        # Default mapping
        local_path = source_dir / "doc" / f"{relative_url}.rst"

    if return_expected or local_path.exists():
        return local_path
    else:
        return None


def main():
    """Command-line interface for the script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare llms.txt for documentation build"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        help="Source directory containing llms.txt (default: repository root)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for transformed llms.txt (default: current directory)",
    )
    parser.add_argument(
        "--doc-version",
        type=str,
        default="current",
        help="Documentation version for URL generation (default: current)",
    )

    args = parser.parse_args()

    # If no arguments are provided, use the default behavior
    if not any([args.source_dir, args.output_dir, args.doc_version != "current"]):
        # Default behavior: transform from repo root to doc directory
        repo_root = Path(__file__).parent.parent
        source_file = repo_root / "llms.txt"
        doc_dir = Path(__file__).parent

        print("Preparing llms.txt for documentation build...")
        print(f"Source: {source_file}")
        print(f"Target directory: {doc_dir}")

        if not source_file.exists():
            raise FileNotFoundError(f"Source llms.txt not found at {source_file}")

        # Auto-detect documentation version
        doc_version = detect_doc_version()

        # Use the main function to prepare the file
        output_file = prepare_llms_txt_for_docs(
            source_dir=repo_root, output_dir=doc_dir, doc_version=doc_version
        )

        # Also generate the expanded context file
        print("\nPreparing llms-ctx.txt (expanded context) for documentation build...")
        prepare_llms_ctx_for_docs(
            source_dir=repo_root, output_dir=doc_dir, doc_version=doc_version
        )
    else:
        # Use provided arguments
        try:
            output_file = prepare_llms_txt_for_docs(
                source_dir=args.source_dir,
                output_dir=args.output_dir,
                doc_version=args.doc_version,
            )

            # Also generate the expanded context file
            print(
                "\nPreparing llms-ctx.txt (expanded context) for documentation build..."
            )
            prepare_llms_ctx_for_docs(
                source_dir=args.source_dir,
                output_dir=args.output_dir,
                doc_version=args.doc_version,
            )
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Count the number of transformed URLs for feedback
    content = output_file.read_text(encoding="utf-8")
    html_md_count = len(re.findall(r"\.html\.md", content))

    print("Successfully prepared llms.txt for documentation build")
    print(f"Transformed {html_md_count} URLs to .html.md format")


if __name__ == "__main__":
    main()
