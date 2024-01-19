# -*- coding: utf-8 -*-
# Copyright 2007-2023 The HyperSpy developers
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

import importlib
import logging
from pathlib import Path
from hyperspy import docstrings

_logger = logging.getLogger(__name__)


if Path(__file__).parent.parent.name == "site-packages":  # pragma: no cover
    # Tested in the "build" workflow on GitHub CI
    from importlib.metadata import version

    __version__ = version("hyperspy")
else:
    # Editable install
    from setuptools_scm import get_version

    __version__ = get_version(Path(__file__).parent.parent)


__doc__ = (
    """
HyperSpy: a multi-dimensional data analysis package for Python
==============================================================

Documentation is available in the docstrings and online at
https://hyperspy.org/hyperspy-doc/current/index.html.

All public packages, functions and classes are in :mod:`~hyperspy.api`. All
other packages and modules are for internal consumption and should not be
needed for data analysis.

%s

More details in the :mod:`~hyperspy.api` docstring.

"""
    % docstrings.START_HSPY
)


__all__ = ["api", "__version__"]


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    if name in __all__:  # pragma: no cover
        # We can't get this block covered in the test suite because it is
        # already imported, when running the test suite.
        # If this is broken, a lot of things will be broken!
        return importlib.import_module("." + name, "hyperspy")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
