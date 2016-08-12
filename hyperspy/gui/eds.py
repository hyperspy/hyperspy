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


class SEMParametersUI(t.HasTraits):

    beam_energy = t.Float(t.Undefined,
                          label='Beam energy (keV)')
    live_time = t.Float(t.Undefined,
                        label='Live time (s)')
    tilt_stage = t.Float(t.Undefined,
                         label='Stage tilt (degree)')
    azimuth_angle = t.Float(t.Undefined,
                            label='Azimuth angle (degree)')
    elevation_angle = t.Float(t.Undefined,
                              label='Elevation angle (degree)')
    energy_resolution_MnKa = t.Float(t.Undefined,
                                     label='Energy resolution MnKa (eV)')

    traits_view = tu.View(
        tu.Group('beam_energy',
                 'tilt_stage',
                 label='SEM', show_border=True),
        tu.Group('live_time', 'azimuth_angle',
                 'elevation_angle', 'energy_resolution_MnKa',
                 label='EDS', show_border=True),
        kind='modal', buttons=[OKButton, CancelButton],
        title='SEM parameters definition wizard')


class TEMParametersUI(t.HasTraits):

    beam_energy = t.Float(t.Undefined,
                          label='Beam energy (keV)')
    real_time = t.Float(t.Undefined,
                        label='Real time (s)')
    tilt_stage = t.Float(t.Undefined,
                         label='Stage tilt (degree)')
    live_time = t.Float(t.Undefined,
                        label='Live time (s)')
    probe_area = t.Float(t.Undefined,
                         label='Beam/probe area (nm^2)')
    azimuth_angle = t.Float(t.Undefined,
                            label='Azimuth angle (degree)')
    elevation_angle = t.Float(t.Undefined,
                              label='Elevation angle (degree)')
    energy_resolution_MnKa = t.Float(t.Undefined,
                                     label='Energy resolution MnKa (eV)')
    beam_current = t.Float(t.Undefined,
                           label='Beam current (nA)')

    traits_view = tu.View(
        tu.Group('beam_energy',
                 'tilt_stage',
                 'probe_area',
                 'beam_current',
                 label='TEM', show_border=True),
        tu.Group('real_time', 'live_time', 'azimuth_angle',
                 'elevation_angle', 'energy_resolution_MnKa',
                 label='EDS', show_border=True),
        kind='modal', buttons=[OKButton, CancelButton],
        title='TEM parameters definition wizard')
