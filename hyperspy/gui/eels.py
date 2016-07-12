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

import traits.api as t
import traitsui.api as tu
from traitsui.menu import OKButton, CancelButton


class TEMParametersUI(t.HasTraits):
    convergence_angle = t.Float(t.Undefined,
                                label='Convergence semi-angle (mrad)')
    beam_energy = t.Float(t.Undefined,
                          label='Beam energy (keV)')
    collection_angle = t.Float(t.Undefined,
                               label='Collection semi-angle (mrad)')

    traits_view = tu.View(
        tu.Group('beam_energy',
                 'convergence_angle',
                 label='TEM', show_border=True),
        tu.Group('collection_angle',
                 label='EELS', show_border=True),
        kind='modal', buttons=[OKButton, CancelButton],
        title='TEM parameters definition wizard')
