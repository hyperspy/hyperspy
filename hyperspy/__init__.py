"""
HyperSpy: a multi-dimensional data analysis package for Python
==============================================================

Documentation is available in the docstrings and online at
http://hyperspy.org/hyperspy-doc/current/index.html.

All public packages, functions and classes are in :mod:`~hyperspy.hspy`. All
other packages are for internal consumption.

When starting HyperSpy using the starting script e.g. by typing ``hyperspy`` in
a console, using the context menu entries or using the links in the
``Start Menu``, the :mod:`~hyperspy.hspy` package is imported in the user
namespace. When using HyperSpy as a library, it is reccommended to import
the :mod:`~hyperspy.hspy` package as follows:

    from hyperspy import hspy as hs

The :mod:`~hyperspy.hspy` package contains the following subpackages:

    :mod:`~hyperspy.hspy.signals`
        Specialized Signal instances.

    :mod:`~hyperspy.hspy.utils`
        Functions that operate of Signal instances and other goodies.

    :mod:`~hyperspy.hspy.components`
        Components that can be used to create a model for curve fitting.

More details in the :mod:`~hyperspy.hspy` docstring.

"""
# -*- coding: utf-8 -*-


import Release

__version__ = Release.version
