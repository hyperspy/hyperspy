# Copyright 2007-2016 The Hyperspy developers
#
# This file is part of  Hyperspy.
#
#  Hyperspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  Hyperspy.  If not, see <http://www.gnu.org/licenses/>.

import pytest
import numpy as np
from numpy.testing import assert_allclose

from hyperspy.signals import Signal1D, Signal2D, ComplexSignal1D, ComplexSignal2D, BaseSignal


def _generate_parameters():
    parameters = []
    for lazy in [False, True]:
        for window_type in ['hann', 'hamming']:
            parameters.append([lazy,
                               window_type])
    return parameters


@pytest.mark.parametrize('lazy, window_type', _generate_parameters())
def test_apodization(lazy, window_type):
    SIZE_NAV0 = 2
    SIZE_NAV1 = 3
    SIZE_NAV2 = 4
    SIZE_SIG0 = 50

    # ax_dict0 = {'size': SIZE_NAV0, 'navigate': True}
    # ax_dict1 = {'size': SIZE_NAV1, 'navigate': True}
    # ax_dict2 = {'size': SIZE_SIG0, 'navigate': False}
    # ax_dict3 = {'size': SIZE_NAV2, 'navigate': True}
    # data = np.random.rand(SIZE_NAV0 * SIZE_NAV1 * SIZE_SIG0 * SIZE_NAV2).reshape(
    #     (SIZE_NAV0, SIZE_NAV1, SIZE_SIG0, SIZE_NAV2))
    # signal1d = Signal1D(data,
    #                     axes=[ax_dict0, ax_dict1, ax_dict2, ax_dict3])
    data = np.random.rand(SIZE_NAV0 * SIZE_NAV1 * SIZE_SIG0 * SIZE_NAV2).reshape(
        (SIZE_NAV0, SIZE_NAV1, SIZE_NAV2, SIZE_SIG0))
    signal1d = Signal1D(data)
    signal1d_a = signal1d.apply_apodization(type=window_type)
    if window_type == 'hann':
        window = np.hanning(SIZE_SIG0)
    elif window_type == 'hamming':
        window = np.hamming(SIZE_SIG0)
    # data_a = data * window[np.newaxis, np.newaxis, :, np.newaxis]
    data_a = data * window[np.newaxis, np.newaxis, np.newaxis, :]

    assert np.alltrue(signal1d_a.data == data_a)



