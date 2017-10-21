# -*- coding: utf-8 -*-
# Copyright 2007-2011 The HyperSpy developers
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

from hyperspy.utils.model_selection import AIC, AICc, BIC
import numpy as np


def notexp_o(x):
    if x > 1:
        return np.exp(1.) * (x * x + 1.) * 0.5

    elif x <= 1 and x > -1:
        return np.exp(x)

    else:
        return 2. / ((x * x + 1.) * np.exp(1.))

notexp = np.vectorize(notexp_o)


class AIC_test(object):

    def __init__(self, tolerance):
        self.name = 'Akaike information criterion test'
        self.tolerance = tolerance
        self.expected = 0.

    def test(self, model, ind):
        m = model.inav[ind[::-1]]
        m.fetch_stored_values()
        _aic = AIC(m)
        return np.abs(
            notexp(_aic) - self.expected) < notexp(self.tolerance)

    def map(self, model, mask):
        ind_list = np.where(mask)
        ans = mask.copy()
        for i in range(ind_list[0].size):
            ind = tuple([lst[i] for lst in ind_list])
            ans[ind] = self.test(model, ind)
        return ans


class AICc_test(object):

    def __init__(self, tolerance):
        self.name = 'Akaike information criterion (with a correction) test'
        self.tolerance = tolerance
        self.expected = 0.

    def test(self, model, ind):
        m = model.inav[ind[::-1]]
        m.fetch_stored_values()
        _aicc = AICc(m)
        return np.abs(
            notexp(_aicc) - self.expected) < notexp(self.tolerance)

    def map(self, model, mask):
        ind_list = np.where(mask)
        ans = mask.copy()
        for i in range(ind_list[0].size):
            ind = tuple([lst[i] for lst in ind_list])
            ans[ind] = self.test(model, ind)
        return ans


class BIC_test(object):

    def __init__(self, tolerance):
        self.name = 'Bayesian information criterion test'
        self.tolerance = tolerance
        self.expected = 0.

    def test(self, model, ind):
        m = model.inav[ind[::-1]]
        m.fetch_stored_values()
        _bic = BIC(m)
        return np.abs(
            notexp(_bic) - self.expected) < notexp(self.tolerance)

    def map(self, model, mask):
        ind_list = np.where(mask)
        ans = mask.copy()
        for i in range(ind_list[0].size):
            ind = tuple([lst[i] for lst in ind_list])
            ans[ind] = self.test(model, ind)
        return ans
