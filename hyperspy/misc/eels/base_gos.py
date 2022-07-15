# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import numpy as np

from hyperspy.misc.math_tools import get_linear_interpolation
from hyperspy.misc.elements import elements


class GOSBase:

    def read_elements(self):
        element = self.element
        subshell = self.subshell
        # Convert to the "GATAN" nomenclature
        if (element in elements) is not True:
            raise ValueError("The given element " + element +
                             " is not in the database.")
        elif subshell not in elements[element]['Atomic_properties']['Binding_energies']:
            raise ValueError(
                "The given subshell " + subshell +
                " is not in the database.\n" +
                "The available subshells are:\n" +
                str(list(elements[element]['Atomic_properties']['subshells'].keys())))

        self.onset_energy = \
            elements[
                element][
                'Atomic_properties'][
                'Binding_energies'][
                subshell][
                'onset_energy (eV)']
        self.Z = elements[element]['General_properties']['Z']
        self.element_dict = elements[element]

    def get_parametrized_qaxis(self, k1, k2, n):
        return k1 * (np.exp(np.arange(n) * k2) - 1) * 1e10

    def get_parametrized_energy_axis(self, k1, k2, n):
        return k1 * (np.exp(np.arange(n) * k2 / k1) - 1)

    def get_qaxis_and_gos(self, ienergy, qmin, qmax):
        qgosi = self.gos_array[ienergy, :]
        if qmax > self.qaxis[-1]:
            # Linear extrapolation
            g1, g2 = qgosi[-2:]
            q1, q2 = self.qaxis[-2:]
            gosqmax = get_linear_interpolation((q1, g1), (q2, g2), qmax)
            qaxis = np.hstack((self.qaxis, qmax))
            qgosi = np.hstack((qgosi, gosqmax))
        else:
            index = self.qaxis.searchsorted(qmax)
            g1, g2 = qgosi[index - 1:index + 1]
            q1, q2 = self.qaxis[index - 1: index + 1]
            gosqmax = get_linear_interpolation((q1, g1), (q2, g2), qmax)
            qaxis = np.hstack((self.qaxis[:index], qmax))
            qgosi = np.hstack((qgosi[:index, ], gosqmax))

        if qmin > 0:
            index = self.qaxis.searchsorted(qmin)
            g1, g2 = qgosi[index - 1:index + 1]
            q1, q2 = qaxis[index - 1:index + 1]
            gosqmin = get_linear_interpolation((q1, g1), (q2, g2), qmin)
            qaxis = np.hstack((qmin, qaxis[index:]))
            qgosi = np.hstack((gosqmin, qgosi[index:],))
        return qaxis, qgosi.clip(0)
