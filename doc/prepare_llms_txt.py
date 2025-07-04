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
Script to prepare llms.txt for documentation build.

This script copies the root llms.txt file and transforms it for web documentation
by converting RST paths to .html.md URLs using the existing ai_tools.py logic.

Usage:
------
1. Automatic (default): python prepare_llms_txt.py
   - Copies from repository root to doc/ directory
   - Uses "current" documentation version
   - Used automatically during Sphinx documentation builds

2. Manual with options: python prepare_llms_txt.py --doc-version dev --output-dir /tmp
   - Allows customization of source, output, and documentation version

Integration:
-----------
This script is automatically called during Sphinx documentation builds via
the setup() function in conf.py, ensuring that the online documentation
always has an up-to-date llms.txt file with web-compatible URLs.

The transformation ensures that:
- Local RST paths (doc/user_guide/install.rst) become web URLs
- URLs point to the correct documentation version (current, dev, v2.x)
- URLs use .html.md extension for better AI/LLM readability
- The file is placed at the documentation root for web crawler access
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
    else:
        # Use provided arguments
        try:
            output_file = prepare_llms_txt_for_docs(
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
