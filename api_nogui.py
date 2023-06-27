# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
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

# Set the PyQt API to 2 to avoid incompatibilities between matplotlib
# traitsui

import logging
import importlib
import sys

_logger = logging.getLogger(__name__)
from hyperspy.logger import set_log_level
from hyperspy.defaults_parser import preferences
set_log_level(preferences.General.logging_level)

from hyperspy import docstrings
from hyperspy.Release import version as __version__


__doc__ = """

All public packages, functions and classes are available in this module.

%s

Functions:

    :py:func:`~.api_nogui.get_configuration_directory_path`
        Return the configuration directory path.

    :py:func:`~.interactive.interactive`
        Define operations that are automatically recomputed on event changes.

    :py:func:`~.io.load`
        Load data into BaseSignal instances from supported files.

    :py:data:`~.defaults_parser.preferences`
        Preferences class instance to configure the default value of different
        parameters. It has a CLI and a GUI that can be started by execting its
        `gui` method i.e. `preferences.gui()`.

    :py:func:`~.utils.print_known_signal_types`
        Print all known `signal_type`.

    :py:func:`~.logger.set_log_level`
        Convenience function to set HyperSpy's the log level.

    :py:func:`~.utils.stack`
        Stack several signals.

    :py:func:`~.utils.transpose`
        Transpose a signal.

The :mod:`~hyperspy.api` package contains the following submodules/packages:

    :mod:`~.signals`
        `Signal` classes which are the core of HyperSpy. Use this modules to
        create `Signal` instances manually from numpy arrays. Note that to
        load data from supported file formats is more convenient to use the
        `load` function.
    :mod:`~.utils.model`
        Contains the :mod:`~utils.model` module with
        components that can be used to create a model for curve fitting.
    :mod:`~.utils.eds`
        Functions for energy dispersive X-rays data analysis.
    :mod:`~.utils.material`
        Useful functions for materials properties and elements database that
        includes physical properties and X-rays and EELS energies.
    :mod:`~.utils.plot`
        Plotting functions that operate on multiple signals.
    :mod:`~.datasets`
        Example datasets.
    :mod:`~.utils.roi`
        Region of interests (ROIs) that operate on `BaseSignal` instances and
        include widgets for interactive operation.
    :mod:`~.utils.samfire`
        SAMFire utilities (strategies, Pool, fit convergence tests)


For more details see their doctrings.

""" % docstrings.START_HSPY


del docstrings


def get_configuration_directory_path():
    import hyperspy.misc.config_dir
    return hyperspy.misc.config_dir.config_path


__all__ = [
    'datasets',
    'eds',
    'get_configuration_directory_path',
    'interactive',
    'load',
    'markers',
    'material',
    'model',
    'plot',
    'preferences',
    'print_known_signal_types',
    'roi',
    'samfire',
    'set_log_level',
    'signals',
    'stack',
    'transpose',
    '__version__',
    ]


# mapping following the pattern: from value import key
_import_mapping = {
    'eds':'.utils',
    'interactive': '.utils',
    'load': '.io',
    'markers': '.utils',
    'material': '.utils',
    'model': '.utils',
    'plot': '.utils',
    'print_known_signal_types': '.utils',
    'roi': '.utils',
    'samfire': '.utils',
    'stack': '.utils',
    'transpose': '.utils',
    }


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    if name in __all__:
        if name in _import_mapping.keys():
            import_path = 'hyperspy' + _import_mapping.get(name)
            return getattr(importlib.import_module(import_path), name)
        else:
            return importlib.import_module("." + name, 'hyperspy')
    # Special case _ureg to use it as a singleton
    elif name == '_ureg':
        if '_ureg' not in globals():
            from pint import UnitRegistry
            setattr(sys.modules[__name__], '_ureg', UnitRegistry())
        return getattr(sys.modules[__name__], '_ureg')

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
