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


def bragg_scattering_angle(d, E0=100):
    """Calculate the first order bragg diffraction semiangle.

    Parameters
    ----------
    d : float
        interplanar distance in m.
    E0 : float
        Incident energy in keV

    Returns
    -------
    float : Semiangle of scattering of the first order difracted beam. This is
    two times the bragg angle.

    """

    gamma = 1 + E0 / 511.0
    v_rel = np.sqrt(1 - 1 / gamma ** 2)
    e_lambda = 2 * np.pi / (2590e9 * (gamma * v_rel))  # m

    return e_lambda / d


def effective_Z(Z_list, exponent=2.94):
    """Effective atomic number of a compound or mixture.

    Exponent = 2.94 for X-ray absorption.

    Parameters
    ----------
    Z_list : list of tuples
        A list of tuples (f,Z) where f is the number of atoms of the element in
        the molecule and Z its atomic number

    Returns
    -------
    float

    """
    if not np.iterable(Z_list) or not np.iterable(Z_list[0]):
        raise ValueError(
            "Z_list should be a list of tuples (f,Z) "
            "where f is the number of atoms of the element"
            "in the molecule and Z its atomic number"
        )

    exponent = float(exponent)
    temp = 0
    total_e = 0
    for Z in Z_list:
        temp += Z[1] * Z[1] ** exponent
        total_e += Z[0] * Z[1]
    return (temp / total_e) ** (1 / exponent)
