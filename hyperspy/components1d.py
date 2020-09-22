# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
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

Writing a new template is really easy, just edit _template.py and maybe take a
look to the other components.

For more details see each component docstring.
====================================================================
"""

from hyperspy._components.arctan import Arctan
from hyperspy._components.bleasdale import Bleasdale
from hyperspy._components.heaviside import HeavisideStep
from hyperspy._components.eels_double_power_law import DoublePowerLaw
from hyperspy._components.eels_cl_edge import EELSCLEdge
from hyperspy._components.error_function import Erf
from hyperspy._components.exponential import Exponential
from hyperspy._components.gaussian import Gaussian
from hyperspy._components.gaussianhf import GaussianHF
from hyperspy._components.logistic import Logistic
from hyperspy._components.lorentzian import Lorentzian
from hyperspy._components.offset import Offset
from hyperspy._components.power_law import PowerLaw
from hyperspy._components.pes_see import SEE
from hyperspy._components.rc import RC
from hyperspy._components.eels_vignetting import Vignetting
from hyperspy._components.voigt import Voigt
from hyperspy._components.scalable_fixed_pattern import ScalableFixedPattern
from hyperspy._components.polynomial import Polynomial
from hyperspy._components.pes_core_line_shape import PESCoreLineShape
from hyperspy._components.volume_plasmon_drude import VolumePlasmonDrude
from hyperspy._components.expression import Expression

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
