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

from __future__ import division

# import numpy as np
# import math

from hyperspy.model import Model
from hyperspy._signals.eds import EDSSpectrum
# from hyperspy.misc.elements import elements as elements_db
# from hyperspy.misc.eds import utils as utils_eds
# import hyperspy.components as create_component
# from hyperspy import utils


class EDSModel(Model):

    """Build a fit a model

    Parameters
    ----------
    spectrum : an Spectrum (or any Spectrum subclass) instance
    auto_background : boolean
        If True, and if spectrum is an EELS instance adds automatically
        a powerlaw to the model and estimate the parameters by the
        two-area method.

    """

    def __init__(self, spectrum,
                 auto_add_lines=True,
                 *args, **kwargs):
        Model.__init__(self, spectrum, *args, **kwargs)

    @property
    def spectrum(self):
        return self._spectrum

    @spectrum.setter
    def spectrum(self, value):
        if isinstance(value, EDSSpectrum):
            self._spectrum = value
            # self.spectrum._are_microscope_parameters_missing()
        else:
            raise ValueError(
                "This attribute can only contain an EDSSpectrum "
                "but an object of type %s was provided" %
                str(type(value)))
