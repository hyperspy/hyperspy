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


class General(t.HasTraits):
    title = t.Str(t.Undefined)
    original_filename = t.File(t.Undefined)
    signal_kind = t.Str(t.Undefined)
    record_by = t.Enum('spectrum', 'image', default=t.Undefined)


class SEDParametersUI(t.HasTraits):

    beam_energy = t.Float(t.Undefined,
                          label='Beam energy (keV)')
    camera_length = t.Float(t.Undefined,
                            label='Camera length (m)')
    scan_rotation = t.Float(t.Undefined,
                            label='Scan rotation (degrees)')
    convergence_angle = t.Float(t.Undefined,
                                label='Convergence angle (mrad)')
    precession_angle = t.Float(t.Undefined,
                               label='Precession angle (mrad)')
    precession_frequency = t.Float(t.Undefined,
                                   label='Precession frequency (Hz)')
    exposure_time = t.Float(t.Undefined,
                            label='Exposure time (ms)')
    traits_view = tu.View(
        tu.Group('beam_energy', 'camera_length', 'scan_rotation',
                 'convergence_angle', 'precession_angle',
                 'precession_frequency', 'exposure_time',
                 label='SED', show_border=True),
        kind='modal', buttons=[OKButton, CancelButton],
title='SED parameters definition wizard')
