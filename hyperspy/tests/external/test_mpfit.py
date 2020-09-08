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

from copy import deepcopy

import numpy as np
from scipy.optimize import OptimizeResult

from hyperspy.external.mpfit.mpfit import mpfit


def Flin(x, p):
    y = p[0] - p[1] * x
    return y


def myfunctlin(p, fjac=None, x=None, y=None, err=None):
    """Linear test function.

    Parameter values are passed in "p"
    If fjac==None then partial derivatives should not be computed.
    It will always be None if MPFIT is called with default flag.
    Non-negative status value means MPFIT should continue,
    while negative means stop the calculation.
    """
    model = Flin(x, p)
    status = 0

    return [status, (y - model) / err]


def generate_toy_model():
    x = np.array(
        [
            -1.7237128e00,
            1.8712276e00,
            -9.6608055e-01,
            -2.8394297e-01,
            1.3416969e00,
            1.3757038e00,
            -1.3703436e00,
            4.2581975e-02,
            -1.4970151e-01,
            8.2065094e-01,
        ]
    )
    y = np.array(
        [
            1.9000429e-01,
            6.5807428e00,
            1.4582725e00,
            2.7270851e00,
            5.5969253e00,
            5.6249280e00,
            0.787615,
            3.2599759e00,
            2.9771762e00,
            4.5936475e00,
        ]
    )
    ey = 0.07 * np.ones(y.shape)
    p0 = np.array([1.0, 1.0])  # initial conditions
    pactual = np.array([3.2, 1.78])  # actual values used to make data
    parbase = {"value": 0.0, "fixed": 0, "limited": [0, 0], "limits": [0.0, 0.0]}

    parinfo = []
    for _ in range(len(pactual)):
        parinfo.append(deepcopy(parbase))
    for i in range(len(pactual)):
        parinfo[i]["value"] = p0[i]

    fa = {"x": x, "y": y, "err": ey}

    return p0, parinfo, fa


def test_linfit():
    p0, parinfo, fa = generate_toy_model()
    m = mpfit(myfunctlin, p0, parinfo=parinfo, functkw=fa)
    res = m.optimize_result

    assert res.success
    assert res.dof == 8
    assert res.status == 1

    np.testing.assert_allclose(m.params, np.array([3.20996572, -1.7709542]), rtol=5e-7)
    np.testing.assert_allclose(m.perror, np.array([0.02221018, 0.01893756]), rtol=5e-7)

    chisq = np.sum(myfunctlin(m.params, x=fa["x"], y=fa["y"], err=fa["err"])[1] ** 2)
    np.testing.assert_allclose(np.array([chisq]), np.array([2.756284983]), rtol=5e-7)


def test_linfit_bounds():
    p0, parinfo, fa = generate_toy_model()

    # This is a bad bound, but its to test it works
    parinfo[0]["limits"] = [1.5, 1.8]
    parinfo[0]["limited"] = [0, 1]

    m = mpfit(myfunctlin, p0, parinfo=parinfo, functkw=fa)
    res = m.optimize_result

    assert res.success
    assert res.dof == 8
    assert res.status == 1

    assert m.params[0] >= 1.5 and m.params[0] <= 1.8
    np.testing.assert_allclose(m.params, np.array([1.8, -1.86916384]), rtol=5e-7)
    np.testing.assert_allclose(m.perror, np.array([0.0, 0.01887426]), rtol=5e-7)

    chisq = np.sum(myfunctlin(m.params, x=fa["x"], y=fa["y"], err=fa["err"])[1] ** 2)
    print(chisq)
    np.testing.assert_allclose(np.array([chisq]), np.array([4032.830936495]), rtol=5e-7)


def test_linfit_tied():
    p0, parinfo, fa = generate_toy_model()

    # This is a bad tie, but its to test it works
    parinfo[0]["tied"] = "2 * p[1]"

    m = mpfit(myfunctlin, p0, parinfo=parinfo, functkw=fa)
    res = m.optimize_result

    assert res.success
    assert res.dof == 9
    assert res.status == 1

    np.testing.assert_allclose(m.params[0] * 0.5, m.params[1], rtol=5e-7)
    np.testing.assert_allclose(m.params, np.array([1.60881708, 0.80440854]), rtol=5e-7)
    np.testing.assert_allclose(m.perror, np.array([0.0, 0.00990717]), rtol=5e-7)

    chisq = np.sum(myfunctlin(m.params, x=fa["x"], y=fa["y"], err=fa["err"])[1] ** 2)
    np.testing.assert_allclose(np.array([chisq]), np.array([25465.436783]), rtol=5e-7)


def myfunctrosenbrock(p, fjac=None):
    """Rosenbrock test function"""
    res = np.array(
        [1 - p[0], -(1 - p[0]), 10 * (p[1] - p[0] ** 2), -10 * (p[1] - p[0] ** 2)]
    )
    status = 0
    return [status, res]


def test_rosenbrock():
    p0 = np.array([-1.0, 1.0])
    m = mpfit(myfunctrosenbrock, p0)
    res = m.optimize_result
    assert isinstance(res, OptimizeResult)
    assert res.success

    exp_param = np.array([1.0, 1.0])
    exp_fnorm = 0.0
    exp_error = np.array([0.70710678, 1.41598024])
    np.testing.assert_allclose(res.x, exp_param, rtol=5e-7)
    np.testing.assert_allclose(res.fnorm, exp_fnorm, rtol=5e-7)
    np.testing.assert_allclose(res.perror, exp_error, rtol=5e-7)


def test_rosenbrock_error():
    # Initial conditions
    p0 = np.array([-1.0, 1.0])
    m = mpfit(myfunctrosenbrock, p0, maxiter=1)
    assert m.status == 5
    np.testing.assert_allclose(m.params, p0, rtol=5e-7)

