# -*- coding: utf-8 -*-
# Copyright 2007-2011 The HyperSpy developers
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


'''

Components that can be used to define a model for e.g. curve fitting.

There are some components that are only useful for one particular kind of signal
and therefore their name are preceded by the signal name: eg. eels_cl_edge.

Writing a new template is really easy, just edit _template.py and maybe take a
look to the other components.

For more details see each component docstring.

'''

from hyperspy._components.arctan import Arctan
from hyperspy._components.bleasdale import Bleasdale
from hyperspy._components.eels_double_offset import DoubleOffset
from hyperspy._components.eels_double_power_law import DoublePowerLaw
from hyperspy._components.eels_cl_edge import EELSCLEdge
from hyperspy._components.error_function import Erf
from hyperspy._components.exponential import Exponential
from hyperspy._components.gaussian import Gaussian
from hyperspy._components.logistic import Logistic
from hyperspy._components.lorentzian import Lorentzian
from hyperspy._components.offset import Offset
from hyperspy._components.power_law import PowerLaw
from hyperspy._components.pes_see import SEE
from hyperspy._components.rc import RC
from hyperspy._components.spline import Spline
from hyperspy._components.eels_vignetting import Vignetting
from hyperspy._components.voigt import Voigt
from hyperspy._components.scalable_fixed_pattern import ScalableFixedPattern
from hyperspy._components.polynomial import Polynomial
from hyperspy._components.pes_core_line_shape import PESCoreLineShape
from hyperspy._components.volume_plasmon_drude import VolumePlasmonDrude
