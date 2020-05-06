import numpy as N
import copy

from hyperspy.external.mpfit.mpfit import mpfit


def Flin(x, p):
    y = p[0] - p[1] * x
    return y


def myfunctlin(p, fjac=None, x=None, y=None, err=None):
    # Parameter values are passed in "p"
    # If fjac==None then partial derivatives should not be
    # computed.  It will always be None if MPFIT is called with default
    # flag.
    model = Flin(x, p)
    # Non-negative status value means MPFIT should continue, negative means
    # stop the calculation.
    status = 0
    return [status, (y - model) / err]


def test_linfit():
    x = N.array([-1.7237128E+00, 1.8712276E+00, -9.6608055E-01,
                 -2.8394297E-01, 1.3416969E+00, 1.3757038E+00,
                 -1.3703436E+00, 4.2581975E-02, -1.4970151E-01,
                 8.2065094E-01])
    y = N.array([1.9000429E-01, 6.5807428E+00, 1.4582725E+00,
                 2.7270851E+00, 5.5969253E+00, 5.6249280E+00,
                 0.787615, 3.2599759E+00, 2.9771762E+00,
                 4.5936475E+00])
    ey = 0.07 * N.ones(y.shape, dtype='float64')
    p0 = N.array([1.0, 1.0], dtype='float64')  # initial conditions
    pactual = N.array([3.2, 1.78])  # actual values used to make data
    parbase = {'value': 0., 'fixed': 0, 'limited': [0, 0], 'limits': [0., 0.]}
    parinfo = []
    for i in range(len(pactual)):
        parinfo.append(copy.deepcopy(parbase))
    for i in range(len(pactual)):
        parinfo[i]['value'] = p0[i]
    fa = {'x': x, 'y': y, 'err': ey}
    m = mpfit(myfunctlin, p0, parinfo=parinfo, functkw=fa)
    if m.status <= 0:
        print('error message = ', m.errmsg)
    assert N.allclose(
        m.params, N.array([3.20996572, -1.7709542], dtype='float64'))
    assert N.allclose(
        m.perror, N.array([0.02221018, 0.01893756], dtype='float64'))
    chisq = (myfunctlin(m.params, x=x, y=y, err=ey)[1] ** 2).sum()

    assert N.allclose(
        N.array(
            [chisq], dtype='float64'), N.array(
            [2.756284983], dtype='float64'))
    assert m.dof == 8
    return


def myfunctrosenbrock(p, fjac=None):
    # rosenbrock function
    res = N.array(
        [1 - p[0], -(1 - p[0]), 10 * (p[1] - p[0] ** 2), -10 * (p[1] - p[0] ** 2)])
    status = 0
    return [status, res]


def test_rosenbrock():
    p0 = N.array([-1, 1.], dtype='float64')  # initial conditions
    pactual = N.array([1., 1.])  # actual minimum of the rosenbrock function
    m = mpfit(myfunctrosenbrock, p0)
    if m.status <= 0:
        print('error message = ', m.errmsg)
    assert m.status > 0
    assert N.allclose(m.params, pactual)
    assert N.allclose(m.fnorm, 0)
    return

if __name__ == "__main__":
    run_module_suite()
