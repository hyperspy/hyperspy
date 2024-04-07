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

from hyperspy.samfire_utils.goodness_of_fit_tests.information_theory import (
    AIC_test,
    AICc_test,
    BIC_test,
)
from hyperspy.samfire_utils.goodness_of_fit_tests.red_chisq import red_chisq_test

__all__ = [
    "AIC_test",
    "AICc_test",
    "BIC_test",
    "red_chisq_test",
]


def __dir__():
    return sorted(__all__)
