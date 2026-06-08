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

"""Static purity tests: generic-layer files must not import matplotlib directly (Phase 4).

Files that route all drawing through PlottingBackend must contain zero
``import matplotlib`` or ``from matplotlib`` statements at any nesting level.

NOTE: signal.py and several _widgets/ files contain matplotlib imports in
methods that are explicitly MPL-specific (e.g. plot_decomposition_results,
SpanSelector, PolygonSelector).  Those files are excluded here and are
tracked as Phase-5 work.
"""

import ast
from pathlib import Path

import pytest

# Files that have been cleaned in Phase 4 and must stay clean.
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
