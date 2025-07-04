#!/usr/bin/env python3
"""
Script to copy Markdown documentation files to the HTML output directory.

This script is used by ReadTheDocs to make Markdown files available at predictable URLs
alongside the HTML documentation. It builds the Markdown docs using Sphinx and then
copies them directly alongside the HTML files with .html.md extension in the output directory.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    """Copy Markdown documentation to HTML output directory."""
    # Get the ReadTheDocs output directory
    output_dir = os.environ.get("READTHEDOCS_OUTPUT")
    if not output_dir:
        print("ERROR: READTHEDOCS_OUTPUT environment variable not set")
        sys.exit(1)

    output_path = Path(output_dir)
    html_dir = output_path / "html"

    # Create a temporary directory for building Markdown docs
    temp_markdown_dir = output_path / "temp_markdown"

    print("Building Markdown documentation...")
    print(f"HTML output directory: {html_dir}")
    print(
        "Target: Copy Markdown files directly alongside HTML files with .html.md extension"
    )

    try:
        # Build Markdown documentation using Sphinx
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sphinx",
                "-b",
                "markdown",
                "doc/",  # Source directory
                str(temp_markdown_dir),  # Temporary output directory
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        print("Sphinx Markdown build completed successfully")
        print(f"Build output:\n{result.stdout}")

        # Copy all Markdown files from temp directly to HTML directory
        if temp_markdown_dir.exists():
            print(f"Copying Markdown files from {temp_markdown_dir} to {html_dir}")

            # Copy all .md files, preserving directory structure but changing extension to .html.md
            for md_file in temp_markdown_dir.rglob("*.md"):
                # Calculate relative path from temp_markdown_dir
                rel_path = md_file.relative_to(temp_markdown_dir)

                # Change extension from .md to .html.md
                target_filename = rel_path.stem + ".html.md"
                target_path = html_dir / rel_path.parent / target_filename

                # Create parent directories if needed
                target_path.parent.mkdir(parents=True, exist_ok=True)

                # Copy the file with new extension
                shutil.copy2(md_file, target_path)
                print(f"Copied: {rel_path} -> {rel_path.parent / target_filename}")

            print(f"Successfully copied Markdown documentation to {html_dir}")

            # List some of the created files for verification
            md_files = list(html_dir.rglob("*.html.md"))
            print(f"Created {len(md_files)} Markdown files:")
            for md_file in sorted(md_files)[:10]:  # Show first 10 files
                rel_path = md_file.relative_to(html_dir)
                print(f"  - {rel_path}")
            if len(md_files) > 10:
                print(f"  ... and {len(md_files) - 10} more files")
        else:
            print(
                f"WARNING: Temporary Markdown directory {temp_markdown_dir} does not exist"
            )

        # Copy llms.txt to HTML root directory
        copy_llms_txt(html_dir)

    except subprocess.CalledProcessError as e:
        print("ERROR: Sphinx Markdown build failed")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to copy Markdown documentation: {e}")
        sys.exit(1)
    finally:
        # Clean up temporary directory
        if temp_markdown_dir.exists():
            print(f"Cleaning up temporary directory: {temp_markdown_dir}")
            shutil.rmtree(temp_markdown_dir)


def copy_llms_txt(html_dir):
    """Copy llms.txt to the root of the HTML output directory."""
    # The llms.txt should have been prepared by the pre_build step
    llms_source = Path("doc/llms.txt")
    llms_target = html_dir / "llms.txt"

    if llms_source.exists():
        print(f"Copying llms.txt from {llms_source} to {llms_target}")
        shutil.copy2(llms_source, llms_target)
        print("Successfully copied llms.txt to HTML root directory")

        # Verify the content
        if llms_target.exists():
            content = llms_target.read_text(encoding="utf-8")
            html_md_count = len(
                [line for line in content.split("\n") if ".html.md" in line]
            )
            print(f"llms.txt contains {html_md_count} web-compatible URLs")
    else:
        print(f"WARNING: llms.txt not found at {llms_source}")
        print("The llms.txt file should have been prepared by the pre_build step")


if __name__ == "__main__":
    main()
