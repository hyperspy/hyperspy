# -*- coding: utf-8 -*-
# Copyright © 2007 Francisco Javier de la Peña
#
# This file is part of EELSLab.
#
# EELSLab is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# EELSLab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EELSLab; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  
# USA
'''Components that can be used to define a model

There are some components that are only useful for one particular kind of signal
and therefore their name are preceded by the signal name: eg. eels_cl_edge.

Writing a new template is really easy, just edit _template.py and maybe take a 
look to the other components.
'''
from eelslab.components.bleasdale import Bleasdale
from eelslab.components.eels_double_offset import DoubleOffset
from eelslab.components.eels_double_power_law import DoublePowerLaw
from eelslab.components.eels_cl_edge import EELSCLEdge
from eelslab.components.error_function import Erf
from eelslab.components.exponential import Exponential
from eelslab.components.fixed_pattern import FixedPattern
from eelslab.components.gaussian import Gaussian
from eelslab.components.line import Line
from eelslab.components.logistic import Logistic
from eelslab.components.lorentzian import Lorentzian
from eelslab.components.offset import Offset
from eelslab.components.parabole import Parabole
from eelslab.components.power_law import PowerLaw
from eelslab.components.pes_see import SEE
from eelslab.components.rc import RC
from eelslab.components.spline import Spline
from eelslab.components.eels_vignetting import Vignetting
from eelslab.components.voigt import Voigt
from eelslab.components.resizeble_fixed_pattern import ResizebleFixedPattern


