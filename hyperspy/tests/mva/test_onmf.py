# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
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
from hyperspy.learn.onmf import onmf


def compare(a, b):
    n1, n2 = list(map(np.linalg.norm, [a, b]))
    return np.linalg.norm((a / n1) - (b / n2))


m = 512
n = 256
r = 3
rng = np.random.RandomState(101)
U = rng.uniform(0, 1, (m, r))
V = rng.uniform(0, 1, (r, n))
X = np.dot(U, V)

sparse = 0.1  # Fraction of corrupted pixels
E = 1000 * rng.binomial(1, sparse, X.shape)
Y = X + E


def test_default():
    W, H = onmf(X, r)
    res = compare(np.dot(W, H), X.T)
    print(res)
    assert res < 0.06


def test_corrupted_default():
    W, H = onmf(Y, r)
    res = compare(np.dot(W, H), X.T)
    print(res)
    assert res < 0.13


def test_robust():
    W, H = onmf(X, r, robust=True)
    res = compare(np.dot(W, H), X.T)
    print(res)
    assert res < 0.05


def test_corrupted_robust():
    W, H = onmf(Y, r, robust=True)
    res = compare(np.dot(W, H), X.T)
    print(res)
    assert res < 0.1
