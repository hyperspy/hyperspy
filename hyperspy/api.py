# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
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
import sys

from hyperspy.defaults_parser import preferences
from hyperspy.logger import set_log_level

# Need to run before other import to use the logger during import
_logger = logging.getLogger(__name__)
set_log_level(preferences.General.logging_level)

from hyperspy import __version__  # noqa: E402
from hyperspy.docstrings import START_HSPY as _START_HSPY_DOCSTRING  # noqa: E402

__doc__ = (
    """

All public packages, functions and classes are available in this module.

%s

Functions:

    :func:`~.api.get_configuration_directory_path`
        Return the configuration directory path.

    :func:`~.api.interactive`
        Define operations that are automatically recomputed on event changes.

    :func:`~.api.load`
        Load data into BaseSignal instances from supported files.

    :data:`~.api.preferences`
        Preferences class instance to configure the default value of different
        parameters. It has a CLI and a GUI that can be started by execting its
        `gui` method i.e. `preferences.gui()`.

    :func:`~.api.print_known_signal_types`
        Print all known `signal_type`.

    :func:`~.api.set_log_level`
        Convenience function to set HyperSpy's the log level.

    :func:`~.api.stack`
        Stack several signals.

    :func:`~.api.transpose`
        Transpose a signal.

The :mod:`~.api` package contains the following submodules/packages:

    :mod:`~.api.signals`
        `Signal` classes which are the core of HyperSpy. Use this modules to
        create `Signal` instances manually from numpy arrays. Note that to
        load data from supported file formats is more convenient to use the
        `load` function.
    :mod:`~.api.model`
        Components that can be used to create a model for curve fitting.
    :mod:`~.api.plot`
        Plotting functions that operate on multiple signals.
    :mod:`~.api.data`
        Synthetic datasets.
    :mod:`~.api.roi`
        Region of interests (ROIs) that operate on `BaseSignal` instances and
        include widgets for interactive operation.
    :mod:`~.api.samfire`
        SAMFire utilities (strategies, Pool, fit convergence tests)


For more details see their doctrings.

"""
    % _START_HSPY_DOCSTRING
)


def get_configuration_directory_path():
    """Return configuration path"""
    from hyperspy.defaults_parser import config_path

    return config_path


# ruff: noqa: F822

__all__ = [
    "data",
    "get_configuration_directory_path",
    "interactive",
    "load",
    "model",
    "plot",
    "preferences",
    "print_known_signal_types",
    "roi",
    "samfire",
    "set_log_level",
    "signals",
    "stack",
    "transpose",
    "__version__",
]


# mapping following the pattern: from value import key
_import_mapping = {
    "interactive": ".utils",
    "load": ".io",
    "markers": ".utils",
    "model": ".utils",
    "plot": ".utils",
    "print_known_signal_types": ".utils",
    "roi": ".utils",
    "samfire": ".utils",
    "stack": ".utils",
    "transpose": ".utils",
}


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    if name in __all__:
        if name in _import_mapping.keys():
            import_path = "hyperspy" + _import_mapping.get(name)
            return getattr(importlib.import_module(import_path), name)
        else:
            return importlib.import_module("." + name, "hyperspy")
    # Special case _ureg to use it as a singleton
    elif name == "_ureg":
        if "_ureg" not in globals():
            import pint

            setattr(sys.modules[__name__], "_ureg", pint.get_application_registry())
        return getattr(sys.modules[__name__], "_ureg")

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
