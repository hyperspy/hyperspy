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

from pathlib import Path


def generate_ai_context(include_optional=False, output_file=None):
    """
    Generate AI context from HyperSpy's llms.txt file.

    This function uses the `llms_txt` package to expand HyperSpy's llms.txt file
    into a comprehensive context file suitable for AI/LLM interactions.

    Parameters
    ----------
    include_optional : bool, default False
        If True, include optional sections and web content in the generated context.
        This creates a more comprehensive but larger context file.
    output_file : str or Path, optional
        Path where to save the generated context. If None, returns the context as a string.

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
    >>> # Generate basic context as string
    >>> context = hs.generate_ai_context()
    >>>
    >>> # Generate full context with web content and save to file
    >>> hs.generate_ai_context(include_optional=True, output_file="hyperspy_context.txt")
    >>>
    >>> # Generate context for offline use
    >>> context = hs.generate_ai_context(include_optional=True)
    >>> with open("my_context.txt", "w") as f:
    ...     f.write(context)

    Notes
    -----
    This function requires the `llms_txt` package to be installed:

        pip install llms_txt

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

    # Generate the context using llms_txt
    try:
        context = create_ctx(llms_content, optional=include_optional)
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
