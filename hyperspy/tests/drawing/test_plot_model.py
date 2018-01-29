# -*- coding: utf-8 -*-
# Copyright 2007-2018 The HyperSpy developers
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

import os
import numpy as np
import pytest

import hyperspy.api as hs
from hyperspy.signals import Signal1D, EELSSpectrum
from hyperspy.components1d import Gaussian


my_path = os.path.dirname(__file__)
baseline_dir = 'plot_model'
default_tol = 2.0


def create_ll_signal(signal_shape=1000):
    offset = 0
    zlp_param = {'A': 10000.0, 'centre': 0.0 + offset, 'sigma': 15}
    zlp = Gaussian(**zlp_param)
    plasmon_param = {'A': 2000.0, 'centre': 200 + offset, 'sigma': 75}
    plasmon = Gaussian(**plasmon_param)
    axis = np.arange(signal_shape)
    data = zlp.function(axis) + plasmon.function(axis)
    ll = EELSSpectrum(data)
    ll.axes_manager[-1].offset = -offset
    return ll


def create_sum_of_gaussians(convolved=False):
    param1 = {'A': 1000.0, 'centre': 500.0, 'sigma': 50.0}
    gs1 = Gaussian(**param1)
    param2 = {'A': 600.0, 'centre': 200.0, 'sigma': 30.0}
    gs2 = Gaussian(**param2)
    param3 = {'A': 2000.0, 'centre': 600.0, 'sigma': 10.0}
    gs3 = Gaussian(**param3)

    axis = np.arange(1000)
    data = gs1.function(axis) + gs2.function(axis) + gs3.function(axis)

    if convolved:
        to_convolved = create_ll_signal(data.shape[0]).data
        data = np.convolve(data, to_convolved) / sum(to_convolved)

    return Signal1D(data[:1000])


def _generate_parameters():
    parameters = []
    for convolved in [True, False]:
        for plot_component in [True, False]:
            parameters.append([convolved, plot_component])
    return parameters


@pytest.mark.parametrize(("convolved", "plot_component"),
                         _generate_parameters())
@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol)
def test_plot_gaussian_eelsmodel(convolved, plot_component):
    s = create_sum_of_gaussians(convolved)
    s = EELSSpectrum(s.data)

    ll = create_ll_signal(1000) if convolved else None
    s.set_microscope_parameters(200, 20, 50)
    m = s.create_model(auto_background=False, ll=ll)

    m.extend([Gaussian(), Gaussian(), Gaussian()])
    m[0].centre.value = 200
    m[0].centre.free = False
    m[1].centre.value = 500
    m[1].centre.free = False
    m[2].centre.value = 600
    m[2].centre.free = False

    m.fit()
    m.plot(plot_components=plot_component)
    return m._plot.signal_plot.figure


@pytest.mark.parametrize(("convolved"), [False, True])
@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol)
def test_fit_EELS_convolved(convolved):
    dname = os.path.join(my_path, 'data')
    cl = hs.load(os.path.join(dname, 'Cr_L_cl.hspy'))
    ll = hs.load(os.path.join(dname, 'Cr_L_ll.hspy')) if convolved else None
    m = cl.create_model(auto_background=False, ll=ll)
    m.fit()
    m.plot(plot_components=True)
    return m._plot.signal_plot.figure


if __name__ == '__main__':
    test_plot_gaussian_eelsmodel(False, True)
    test_plot_gaussian_eelsmodel(True, True)
