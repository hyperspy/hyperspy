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

import numpy as np

from numpy.testing import assert_allclose

from hyperspy.utils.model_selection import AIC, AICc, BIC
from hyperspy.signals import Signal1D
from hyperspy.components1d import Gaussian, Lorentzian


class TestModelSelection:

    def setup_method(self, method):
        s = Signal1D(range(10))
        m1 = s.create_model()
        m2 = s.create_model()
        m1.append(Gaussian())
        m2.append(Lorentzian())
        m1.fit()
        m2.fit()
        self.m1 = m1
        self.m2 = m2

    def test_AIC(self):
        _aic1 = AIC(self.m1)
        _aic2 = AIC(self.m2)
        assert_allclose(_aic1, 74.477061729373233)
        assert_allclose(_aic2, 72.749265802224159)

    def test_AICc(self):
        _aicc1 = AICc(self.m1)
        _aicc2 = AICc(self.m2)
        assert_allclose(_aicc1, 82.477061729373233)
        assert_allclose(_aicc2, 80.749265802224159)

    def test_BIC(self):
        _bic1 = BIC(self.m1)
        _bic2 = BIC(self.m2)
        assert_allclose(_bic1, 75.68740210134942)
        assert_allclose(_bic2, 73.959606174200346)
