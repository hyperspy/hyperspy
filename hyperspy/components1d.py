# -*- coding: utf-8 -*-
# Copyright 2007-2021 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.


__doc__ = """

Components that can be used to define a 1D model for e.g. curve fitting.

There are some components that are only useful for one particular kind of signal
and therefore their name are preceded by the signal name: eg. eels_cl_edge.

Writing a new template is easy: see the user guide documentation on creating
components.

For more details see each component docstring.
====================================================================
"""


# -*- coding: utf-8 -*-
from hyperspy.extensions import EXTENSIONS as _EXTENSIONS
import importlib

_g = globals()
for _component, _specs in _EXTENSIONS["components1D"].items():
    # Don't add the new Polynomial to the API.
    # To use it the old `Polynomial` has a `legacy` keyword.
    # TODO: remove in HyperSpy v2.0
    if _component == "eab91275-88db-4855-917a-cdcbe7209592":
        continue
    _g[_component] = getattr(
        importlib.import_module(
            _specs["module"]), _component)

del importlib, _component, _specs, _g

# Generating the documentation

# Grab all the currently defined globals and make a copy of the keys
# (can't use it directly, as the size changes)
_keys = [key for key in globals().keys()]

# For every key in alphabetically sorted order
for key in sorted(_keys):
    # if it does not start with a "_"
    if not key.startswith('_'):
        # get the component class (or function)
        component = eval(key)
        # If the component has documentation, grab the first 43 characters of
        # the first line of the documentation. Else just use two dots ("..")
        second_part = '..' if component.__doc__ is None else \
            component.__doc__.split('\n')[0][:43] + '..'
        # append the component name (up to 25 characters + one space) and the
        # start of the documentation as one line to the current doc
        __doc__ += key[:25] + ' ' * (26 - len(key)) + second_part + '\n'

# delete all the temporary things from the namespace once done
# so that they don't show up in the auto-complete
del key, _keys, component, second_part
