# Copyright 2007-2016 The HyperSpy developers
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
# along with HyperSpy. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import nose.tools as nt
from hyperspy.misc.utils import DictionaryTreeBrowser
from hyperspy.signals import Signal1D
from hyperspy.components1d import Lorentzian


class TestRedChisq:

    def setUp(self):
        self.shape = (7, 15)
        art_model = DictionaryTreeBrowser()
        art_model.set_item('red_chisq.data', np.ones(self.shape))
        art_model.red_chisq.data[3, 5] = 0.8
        art_model.red_chisq.data[2, 5] = 2.
        self.m = art_model
        # have to be imported here, as otherwise crashes nosetools
        from hyperspy.samfire_utils.goodness_of_fit_tests.red_chisq import \
            red_chisq_test as rct
        self.t = rct(0.9)

    def test_changing_tolerance(self):
        t = self.t
        t.tolerance = 1.0
        nt.assert_equal(t.tolerance, 1.0)
        t.tolerance = -3.
        nt.assert_equal(t.tolerance, 3)

    def test_index(self):
        t = self.t
        ind = (0, 0)
        nt.assert_true(t.test(self.m, ind))
        ind = (2, 5)
        nt.assert_false(t.test(self.m, ind))
        ind = (3, 5)
        nt.assert_true(t.test(self.m, ind))
        t.tolerance = 0.1
        nt.assert_false(t.test(self.m, ind))

    def test_map(self):
        t = self.t
        mask = np.ones(self.shape, dtype='bool')
        mask[2, 5] = False
        ans = t.map(self.m, mask)
        nt.assert_true(np.all(ans == mask))
        mask[0, 0] = False
        ans = t.map(self.m, mask)
        nt.assert_true(np.all(ans == mask))


class TestInformationCriteria:

    def setUp(self):
        m = Signal1D(np.arange(30).reshape((3, 10))).create_model()
        m.append(Lorentzian())
        m.multifit(show_progressbar=False)
        self.m = m
        # have to be imported here, as otherwise crashes nosetools
        from hyperspy.samfire_utils.goodness_of_fit_tests.information_theory \
            import (AIC_test, AICc_test, BIC_test)
        self.aic = AIC_test(0.)
        self.aicc = AICc_test(0.)
        self.bic = BIC_test(0.)

    def test_index(self):
        ind = (0,)
        nt.assert_false(self.aic.test(self.m, ind))
        nt.assert_false(self.aicc.test(self.m, ind))
        nt.assert_false(self.bic.test(self.m, ind))
        ind = (1,)
        nt.assert_true(self.aic.test(self.m, ind))
        nt.assert_true(self.aicc.test(self.m, ind))
        nt.assert_true(self.bic.test(self.m, ind))
        ind = (2,)
        nt.assert_true(self.aic.test(self.m, ind))
        nt.assert_true(self.aicc.test(self.m, ind))
        nt.assert_true(self.bic.test(self.m, ind))

        self.aic.tolerance = -50
        self.aicc.tolerance = -50
        self.bic.tolerance = -50
        ind = (1,)
        nt.assert_false(self.aic.test(self.m, ind))
        nt.assert_false(self.aicc.test(self.m, ind))
        nt.assert_false(self.bic.test(self.m, ind))
        ind = (2,)
        nt.assert_true(self.aic.test(self.m, ind))
        nt.assert_true(self.aicc.test(self.m, ind))
        nt.assert_true(self.bic.test(self.m, ind))

    def test_map(self):
        mask = np.array([True, True, False])
        nt.assert_true(np.all(self.aic.map(self.m, mask) == [0, 1, 0]))
        nt.assert_true(np.all(self.aicc.map(self.m, mask) == [0, 1, 0]))
        nt.assert_true(np.all(self.bic.map(self.m, mask) == [0, 1, 0]))
        self.aic.tolerance = -50
        self.aicc.tolerance = -50
        self.bic.tolerance = -50
        nt.assert_true(np.all(self.aic.map(self.m, mask) == [0, 0, 0]))
        nt.assert_true(np.all(self.aicc.map(self.m, mask) == [0, 0, 0]))
        nt.assert_true(np.all(self.bic.map(self.m, mask) == [0, 0, 0]))
