# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
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

from hyperspy.models.edsmodel import EDSModel


class EDSSEMModel(EDSModel):

    """Build and fit a model to EDS data acquired in the SEM.

    Parameters
    ----------
    spectrum : EDSSEMSpectrum

    auto_add_lines : bool
        If True, automatically add Gaussians for all X-rays generated
        in the energy range by an element, using the edsmodel.add_family_lines
        method.
    auto_background : bool
        If True, adds automatically a polynomial order 6 to the model,
        using the edsmodel.add_polynomial_background method.

    Any extra arguments are passed to the Model constructor.
    """

    def __init__(self, spectrum,
                 auto_background=True,
                 auto_add_lines=True,
                 *args, **kwargs):
        EDSModel.__init__(self, spectrum, auto_background, auto_add_lines,
                          *args, **kwargs)
