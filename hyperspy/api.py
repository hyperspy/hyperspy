# -*- coding: utf-8 -*-

# Set the PyQt API to 2 to avoid incompatibilities between matplotlib
# traitsui

from hyperspy import docstrings

__doc__ = """

All public packages, functions and classes are available in this module.

%s

Functions:

    create_model
        Create a model for curve fitting.

    get_configuration_directory_path
        Return the configuration directory path.

    load
        Load data into BaseSignal instances from supported files.

    preferences
        Preferences class instance to configure the default value of different
        parameters. It has a CLI and a GUI that can be started by execting its
        `gui` method i.e. `preferences.gui()`.

    stack
        Stack several signals.

    interactive
        Define operations that are automatically recomputed on event changes.

    set_log_level
        Convenience function to set HyperSpy's the log level.


The :mod:`~hyperspy.api` package contains the following submodules/packages:

    :mod:`~hyperspy.api.signals`
        `Signal` classes which are the core of HyperSpy. Use this modules to
        create `Signal` instances manually from numpy arrays. Note that to
        load data from supported file formats is more convenient to use the
        `load` function.
    :mod:`~hyperspy.api.model`
        Contains the :mod:`~hyperspy.api.model.components` module with
        components that can be used to create a model for curve fitting.
    :mod:`~hyperspy.api.eds`
        Functions for energy dispersive X-rays data analysis.
    :mod:`~hyperspy.api.material`
        Useful functions for materials properties and elements database that
        includes physical properties and X-rays and EELS energies.
    :mod:`~hyperspy.api.plot`
        Plotting functions that operate on multiple signals.
    :mod:`~hyperspy.api.datasets`
        Example datasets.
    :mod:`~hyperspy.api.roi`
        Region of interests (ROIs) that operate on `BaseSignal` instances and
        include widgets for interactive operation.
    :mod:`~hyperspy.api.samfire`
        SAMFire utilities (strategies, Pool, fit convergence tests)


For more details see their doctrings.

""" % docstrings.START_HSPY

# Remove the module to avoid polluting the namespace
del docstrings


from hyperspy.Release import version as __version__
from hyperspy import signals
from hyperspy.io import load
from hyperspy.defaults_parser import preferences
from hyperspy.utils import *
from hyperspy import datasets
from hyperspy.logger import set_log_level

set_log_level(preferences.General.logging_level)



def get_configuration_directory_path():
    import hyperspy.misc.config_dir
    return hyperspy.misc.config_dir.config_path
