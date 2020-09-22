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

import numpy as np
from hyperspy import model
from hyperspy.exceptions import NavigationSizeError


def AIC(model):
    """Calculates the Akaike information criterion

    AIC = 2 k - 2 ln(L)

    where
    L is the maximum likelihood function value,
    k is the number of free parameters.
    """

    # maybe should not have any Offset components?
    # more than a single pixel, needs iterating, don't do that for now
    if model.axes_manager.navigation_size:
        raise NavigationSizeError(model.axes_manager.navigation_size, 0)

    model._set_p0()  # correctly set the parameters (numbers / values)
    lnL = model._poisson_likelihood_function(
        model.p0,
        model.axis.axis[
            model.channel_switches])
    k = len(model.p0) + 1  # +1 for the variance
    return 2 * k - 2 * lnL


def AICc(model):
    _aic = AIC(model)
    n = model.axes_manager.signal_size
    k = len(model.p0) + 1
    return _aic + (2. * k * (k + 1)) / (n - k - 1)


def BIC(model):
    """Calculates the Bayesian information criterion

    BIC = -2 * ln(L) + k * ln(n)

    where
    L is the maximum likelihood function,
    k is the number of free parameters, and
    n is the number of data points (observations) / sample size.
    """
    # maybe should not have any Offset components?
    # more than a single pixel, needs iterating, don't do that for now
    if model.axes_manager.navigation_size:
        raise NavigationSizeError(model.axes_manager.navigation_size, 0)

    model._set_p0()  # correctly set the parameters (numbers / values)
    lnL = model._poisson_likelihood_function(
        model.p0,
        model.axis.axis[
            model.channel_switches])
    n = model.axes_manager.signal_size
    k = len(model.p0) + 1
    return k * np.log(n) - 2. * lnL
