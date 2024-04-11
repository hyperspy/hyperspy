# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
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

import hyperspy.api as hs


def two_gaussians(add_noise=True, return_model=False):
    """
    Create synthetic data consisting of two Gaussian functions with
    random centers and area

    Parameters
    ----------
    add_noise : bool
        If True, add noise to the signal.
    return_model : bool
        If True, returns the model in addition to the signal.

    Returns
    -------
    :class:`~.api.signals.BaseSignal` or tuple
        Returns tuple when ``return_model=True``.
    """

    domain = 32  # size of the square domain
    hfactor = 600
    cent = (domain // 2, domain // 2)
    y, x = np.ogrid[-cent[0] : domain - cent[0], -cent[1] : domain - cent[1]]

    def gaussian2d(x, y, A=1, x0=0, y0=0, sigmax=20, sigmay=10):
        return A * np.exp(
            -((x - x0) ** 2 / 2 / sigmax**2 + (y - y0) ** 2 / 2 / sigmay**2)
        )

    center_narrow = 50 + 10 * np.sin(3 * np.pi * x / domain) * np.cos(
        4 * np.pi * y / domain
    )
    center_wide = 50 + 10 * (
        -0.1 * np.sin(3 * np.pi * x / domain) * np.cos(4 * np.pi * y / domain)
    )

    r = np.sqrt(x**2 + y**2)
    h_narrow = 0.5 * (0.5 + np.sin(r) ** 2) * gaussian2d(x, y) * hfactor
    h_wide = (0.5 + np.cos(r) ** 2) * gaussian2d(x, y) * hfactor

    s = hs.signals.Signal1D(np.ones((domain, domain, 1024)))
    s.metadata.General.title = "Two Gaussians"
    s.axes_manager[0].name = "x"
    s.axes_manager[0].units = "nm"
    s.axes_manager[1].name = "y"
    s.axes_manager[1].units = "nm"

    s.axes_manager[2].name = "Energy"
    s.axes_manager[2].name = "eV"
    s.axes_manager[2].scale = 0.1
    m = s.create_model()

    gs01 = hs.model.components1D.GaussianHF()
    gs01.name = "wide"
    m.append(gs01)
    gs01.fwhm.value = 60
    gs01.centre.map["values"][:] = center_wide
    gs01.centre.map["is_set"][:] = True
    gs01.height.map["values"][:] = h_wide
    gs01.height.map["is_set"][:] = True

    gs02 = hs.model.components1D.GaussianHF()
    gs02.name = "narrow"
    m.append(gs02)
    gs02.fwhm.value = 6
    gs02.centre.map["values"][:] = center_narrow
    gs02.centre.map["is_set"][:] = True
    gs02.height.map["values"][:] = h_narrow
    gs02.height.map["is_set"][:] = True
    s.data = m.as_signal(show_progressbar=False).data
    s.change_dtype(np.int64)
    s.add_poissonian_noise(random_state=0)
    m.store("ground truth")

    if return_model:
        return s, m
    else:
        return s
