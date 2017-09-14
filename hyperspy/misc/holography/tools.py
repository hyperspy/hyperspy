# -*- coding: utf-8 -*-
# Copyright 2007-2017 The HyperSpy developers
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

import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
import logging

_logger = logging.getLogger(__name__)


def calculate_carrier_frequency(holo_data, sb_position, scale):
    """
    Calculates fringe carrier frequency of a hologram

    Parameters
    ----------
    holo_data: ndarray
        The data of the hologram.
    sb_position: tuple
        Position of the sideband with the reference to non-shifted FFT
    scale: tuple
        Scale of the axes that will be used for the calculation.

    Returns
    -------
    Carrier frequency
    """

    shape = holo_data.shape
    origins = [np.array((0, 0)),
               np.array((0, shape[1])),
               np.array((shape[0], shape[1])),
               np.array((shape[0], 0))]
    origin_index = np.argmin([np.linalg.norm(origin-sb_position) for origin in origins])
    return np.linalg.norm(np.multiply(origins[origin_index]-sb_position, scale))
