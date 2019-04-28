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
import numpy.testing as nt

import hyperspy.api as hs
from hyperspy.signals import Signal1D, EELSSpectrum
from hyperspy.components1d import Gaussian


my_path = os.path.dirname(__file__)
baseline_dir = 'plot_model'
default_tol = 2.0


def create_ll_signal(signal_shape=1000):
    offset = 0
    zlp_param = {'A': 10000.0, 'centre': 0.0 + offset, 'sigma': 15.0}
    zlp = Gaussian(**zlp_param)
    plasmon_param = {'A': 2000.0, 'centre': 200.0 + offset, 'sigma': 75.0}
    plasmon = Gaussian(**plasmon_param)
    axis = np.arange(signal_shape)
    data = zlp.function(axis) + plasmon.function(axis)
    ll = EELSSpectrum(data)
    ll.axes_manager[-1].offset = -offset
    ll.axes_manager[-1].scale = 0.1
    return ll


A_value_gaussian = [1000.0, 600.0, 2000.0]
centre_value_gaussian = [50.0, 20.0, 60.0]
sigma_value_gaussian = [5.0, 3.0, 1.0]
scale = 0.1


def create_sum_of_gaussians(convolved=False):
    param1 = {'A': A_value_gaussian[0],
              'centre': centre_value_gaussian[0] / scale,
              'sigma': sigma_value_gaussian[0] / scale}
    gs1 = Gaussian(**param1)
    param2 = {'A': A_value_gaussian[1],
              'centre': centre_value_gaussian[1] / scale,
              'sigma': sigma_value_gaussian[1] / scale}
    gs2 = Gaussian(**param2)
    param3 = {'A': A_value_gaussian[2],
              'centre': centre_value_gaussian[2] / scale,
              'sigma': sigma_value_gaussian[2] / scale}
    gs3 = Gaussian(**param3)

    axis = np.arange(1000)
    data = gs1.function(axis) + gs2.function(axis) + gs3.function(axis)

    if convolved:
        to_convolved = create_ll_signal(data.shape[0]).data
        data = np.convolve(data, to_convolved) / sum(to_convolved)

    s = Signal1D(data[:1000])
    s.axes_manager[-1].scale = scale
    return s


def _generate_parameters():
    parameters = []
    for convolved in [True, False]:
        for plot_component in [True, False]:
            for binned in [True, False]:
                parameters.append([convolved, plot_component, binned])
    return parameters


@pytest.mark.parametrize(("convolved", "plot_component", "binned"),
                         _generate_parameters())
@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol)
def test_plot_gaussian_eelsmodel(convolved, plot_component, binned):
    s = create_sum_of_gaussians(convolved)
    s.set_signal_type('EELS')
    s.metadata.General.title = 'Convolved: {}, plot_component: {}, binned: {}'.format(
        convolved, plot_component, binned)

    ll = create_ll_signal(1000) if convolved else None

    s.set_microscope_parameters(200, 20, 50)
    s.metadata.Signal.binned = binned
    m = s.create_model(auto_background=False, ll=ll)

    m.extend([Gaussian(), Gaussian(), Gaussian()])

    def set_gaussian(gaussian, centre, sigma):
        gaussian.centre.value = centre
        gaussian.centre.free = False
        gaussian.sigma.value = sigma
        gaussian.sigma.free = False

    for gaussian, centre, sigma in zip(m, centre_value_gaussian,
                                       sigma_value_gaussian):
        set_gaussian(gaussian, centre, sigma)

    m.fit()
    m.plot(plot_components=plot_component)

    def A_value(s, component, binned):
        if binned:
            return component.A.value * scale
        else:
            return component.A.value

    if convolved:
        nt.assert_almost_equal(A_value(s, m[0], binned), 0.014034, decimal=5)
        nt.assert_almost_equal(A_value(s, m[1], binned), 0.008420, decimal=5)
        nt.assert_almost_equal(A_value(s, m[2], binned), 0.028068, decimal=5)
    else:
        nt.assert_almost_equal(A_value(s, m[0], binned), 100.0)
        nt.assert_almost_equal(A_value(s, m[1], binned), 60.0)
        nt.assert_almost_equal(A_value(s, m[2], binned), 200.0)

    return m._plot.signal_plot.figure


@pytest.mark.parametrize(("convolved"), [False, True])
@pytest.mark.mpl_image_compare(
    baseline_dir=baseline_dir, tolerance=default_tol)
def test_fit_EELS_convolved(convolved):
    dname = os.path.join(my_path, 'data')
    cl = hs.load(os.path.join(dname, 'Cr_L_cl.hspy'))
    cl.metadata.Signal.binned = False
    cl.metadata.General.title = 'Convolved: {}'.format(convolved)
    ll = hs.load(os.path.join(dname, 'Cr_L_ll.hspy')) if convolved else None
    m = cl.create_model(auto_background=False, ll=ll, GOS='hydrogenic')
    m.fit(kind='smart')
    m.plot(plot_components=True)
    return m._plot.signal_plot.figure
