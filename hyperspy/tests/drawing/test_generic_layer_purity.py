# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
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

"""Static checks that the generic drawing layer does not import matplotlib.

Files that route all drawing through ``PlottingBackend`` must contain no
``import matplotlib`` / ``from matplotlib`` statement at any nesting level.
Files with deliberately matplotlib-specific methods are excluded.
"""

import ast
from pathlib import Path

import pytest

# Files that must stay free of matplotlib imports.
GENERIC_FILES = [
    "hyperspy/drawing/figure.py",
    "hyperspy/drawing/widget.py",
    "hyperspy/drawing/he.py",
    "hyperspy/drawing/hie.py",
    "hyperspy/drawing/hse.py",
    "hyperspy/drawing/_widgets/vertical_line.py",
    "hyperspy/drawing/_widgets/horizontal_line.py",
    "hyperspy/drawing/_widgets/scalebar.py",
]


@pytest.mark.parametrize("filepath", GENERIC_FILES)
def test_no_direct_matplotlib_import(filepath):
    """Generic-layer files must not import matplotlib directly."""
    repo_root = Path(__file__).parents[3]
    source = (repo_root / filepath).read_text()
    tree = ast.parse(source, filename=filepath)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("matplotlib"), (
                        f"{filepath}: direct `import matplotlib` at line {node.lineno}"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("matplotlib"):
                    raise AssertionError(
                        f"{filepath}: `from matplotlib` import at line {node.lineno}"
                    )
