"""
Perform Levenberg-Marquardt least-squares minimization, based on MINPACK-1.

AUTHORS
=======
The original version of this software, called LMFIT, was written in FORTRAN
as part of the MINPACK-1 package by Jorge More, Danny Sorenson, Burton Garbow
and Kenneth Hillstrom.

Craig Markwardt converted the FORTRAN code to IDL.
    Craig B. Markwardt, NASA/GSFC Code 662, Greenbelt, MD 20770
    craigm@lheamail.gsfc.nasa.gov
    UPDATED VERSIONs can be found on my WEB PAGE:
    https://cow.physics.wisc.edu/~craigm/idl/idl.html

Mark Rivers created this Python version from Craig's IDL version.
    Mark Rivers, University of Chicago
    Building 434A, Argonne National Laboratory
    9700 South Cass Avenue, Argonne, IL 60439
    rivers@cars.uchicago.edu
    Updated versions can be found at https://cars.uchicago.edu/software

Sergey Koposov converted the Mark's Python version from Numeric to numpy
    Sergey Koposov, University of Cambridge, Institute of Astronomy,
    Madingley road, CB3 0HA, Cambridge, UK
    koposov@ast.cam.ac.uk
    Updated versions can be found at https://code.google.com/p/astrolibpy/source/browse/trunk/


MODIFICATION HISTORY
====================
- Translated from MINPACK-1 in FORTRAN, Apr-Jul 1998, CM
  Copyright (C) 1997-2002, Craig Markwardt
  This software is provided as is without any warranty whatsoever.
  Permission to use, copy, modify, and distribute modified or
  unmodified copies is granted, provided this copyright and disclaimer
  are included unchanged.
- Translated from MPFIT (Craig Markwardt's IDL package) to Python,
  August, 2002. Mark Rivers
- Converted from Numeric to numpy (Sergey Koposov, July 2008)
- Fixed the analytic derivatives features (The HyperSpy Developers, 2011)
- Reformatted and updated docstrings, and return results as a
  scipy.optimize.OptimizeResult object (The HyperSpy Developers, 2020)


DESCRIPTION
===========
MPFIT uses the Levenberg-Marquardt technique to solve the
least-squares problem. In its typical use, MPFIT will be used to
fit a user-supplied function (the "model") to user-supplied data
points (the "data") by adjusting a set of parameters. MPFIT is
based upon MINPACK-1 (LMDIF.F) by More' and collaborators.

For example, a researcher may think that a set of observed data
points is best modelled with a Gaussian curve. A Gaussian curve is
parameterized by its mean, standard deviation and normalization.
MPFIT will, within certain constraints, find the set of parameters
which best fits the data. The fit is "best" in the least-squares
sense; that is, the sum of the weighted squared differences between
the model and data is minimized.

The Levenberg-Marquardt technique is a particular strategy for
iteratively searching for the best fit. This particular
implementation is drawn from MINPACK-1 (see NETLIB), and is much faster
and more accurate than the version provided in the Scientific Python package
in Scientific.Functions.LeastSquares.
This version allows upper and lower bounding constraints to be placed on each
parameter, or the parameter can be held fixed.

The user-supplied Python function should return an array of weighted
deviations between model and data. In a typical scientific problem
the residuals should be weighted so that each deviate has a
gaussian sigma of 1.0. If X represents values of the independent
variable, Y represents a measurement for each value of X, and ERR
represents the error in the measurements, then the deviates could
be calculated as follows:

    DEVIATES = (Y - F(X)) / ERR

where F is the analytical function representing the model. You are
recommended to use the convenience functions MPFITFUN and
MPFITEXPR, which are driver functions that calculate the deviates
for you. If ERR are the 1-sigma uncertainties in Y, then

    TOTAL( DEVIATES^2 )

will be the total chi-squared value. MPFIT will minimize the
chi-square value. The values of X, Y and ERR are passed through
MPFIT to the user-supplied function via the FUNCTKW keyword.

Simple constraints can be placed on parameter values by using the
PARINFO keyword to MPFIT. See below for a description of this
keyword.

MPFIT does not perform more general optimization tasks. See TNMIN
instead. MPFIT is customized, based on MINPACK-1, to the
least-squares minimization problem.


USER FUNCTION
=============
The user must define a function which returns the appropriate
values as specified above. The function should return the weighted
deviations between the model and the data. It should also return a status
flag and an optional partial derivative array. For applications which
use finite-difference derivatives -- the default -- the user
function should be declared in the following way:

    def myfunct(p, fjac=None, x=None, y=None, err=None):
        # Parameter values are passed in "p"
        # If fjac==None then partial derivatives should not be
        # computed. It will always be None if MPFIT is called with default
        # flag.
        model = F(x, p)
        # Non-negative status value means MPFIT should continue,
        # negative means stop the calculation.
        status = 0
        return status, (y - model) / err

See below for applications with analytical derivatives.

The keyword parameters X, Y, and ERR in the example above are
suggestive but not required. Any parameters can be passed to
MYFUNCT by using the functkw keyword to MPFIT. Use MPFITFUN and
MPFITEXPR if you need ideas on how to do that. The function *must*
accept a parameter list, P.

In general there are no restrictions on the number of dimensions in
X, Y or ERR. However the deviates *must* be returned in a
one-dimensional Numeric array of type Float.

User functions may also indicate a fatal error condition using the
status return described above. If status is set to a number between
-15 and -1 then MPFIT will stop the calculation and return to the caller.


ANALYTIC DERIVATIVES
====================
In the search for the best-fit solution, MPFIT by default
calculates derivatives numerically via a finite difference
approximation. The user-supplied function need not calculate the
derivatives explicitly. However, if you desire to compute them
analytically, then the AUTODERIVATIVE=0 keyword must be passed to MPFIT.
As a practical matter, it is often sufficient and even faster to allow
MPFIT to calculate the derivatives numerically, and so
AUTODERIVATIVE=0 is not necessary.

If AUTODERIVATIVE=0 is used then the user function must check the parameter
FJAC, and if FJAC!=None then return the partial derivative array in the
return list.

    def myfunct(p, fjac=None, x=None, y=None, err=None):
        # Parameter values are passed in "p"
        # If FJAC!=None then partial derivatives must be computed.
        # FJAC contains an array of len(p), where each entry
        # is 1 if that parameter is free and 0 if it is fixed.
        model = F(x, p)
        # Non-negative status value means MPFIT should continue,
        # negative means stop the calculation.
        status = 0
        if dojac:
            pderiv = np.zeros((len(x), len(p)))
            for j in range(len(p)):
                pderiv[:, j] = FGRAD(x, p, j)
        else:
            pderiv = None
        return status, (y - model) / err, pderiv

where FGRAD(x, p, i) is a user function which must compute the
derivative of the model with respect to parameter P[i] at X. When
finite differencing is used for computing derivatives (ie, when
AUTODERIVATIVE=1), or when MPFIT needs only the errors but not the
derivatives the parameter FJAC=None.

Derivatives should be returned in the PDERIV array. PDERIV should be an m x
n array, where m is the number of data points and n is the number
of parameters. dp[i,j] is the derivative at the ith point with
respect to the jth parameter.

The derivatives with respect to fixed parameters are ignored; zero
is an appropriate value to insert for those derivatives. Upon
input to the user function, FJAC is set to a vector with the same
length as P, with a value of 1 for a parameter which is free, and a
value of zero for a parameter which is fixed (and hence no
derivative needs to be calculated).

If the data is higher than one dimensional, then the *last*
dimension should be the parameter dimension. Example: fitting a
50x50 image, "dp" should be 50x50xNPAR.


CONSTRAINING PARAMETER VALUES WITH THE PARINFO KEYWORD
======================================================
The behavior of MPFIT can be modified with respect to each
parameter to be fitted. A parameter value can be fixed; simple
boundary constraints can be imposed; limitations on the parameter
changes can be imposed; properties of the automatic derivative can
be modified; and parameters can be tied to one another.

These properties are governed by the PARINFO structure, which is
passed as a keyword parameter to MPFIT.

PARINFO should be a list of dictionaries, one list entry for each parameter.
Each parameter is associated with one element of the array, in
numerical order. The dictionary can have the following keys
(none are required, keys are case insensitive):

    'value' :
        the starting parameter value (but see the START_PARAMS
        parameter for more information).
    'fixed' :
        a boolean value, whether the parameter is to be held
        fixed or not. Fixed parameters are not varied by
        MPFIT, but are passed on to MYFUNCT for evaluation.
    'limited' :
        a two-element boolean array. If the first/second
        element is set, then the parameter is bounded on the
        lower/upper side. A parameter can be bounded on both
        sides. Both LIMITED and LIMITS must be given
        together.
    'limits' :
        a two-element float array. Gives the
        parameter limits on the lower and upper sides,
        respectively. Zero, one or two of these values can be
        set, depending on the values of LIMITED. Both LIMITED
        and LIMITS must be given together.
    'parname' :
        a string, giving the name of the parameter. The
        fitting code of MPFIT does not use this tag in any
        way. However, the default iterfunct will print the
        parameter name if available.
    'step' :
        the step size to be used in calculating the numerical
        derivatives. If set to zero, then the step size is
        computed automatically. Ignored when AUTODERIVATIVE=0.
    'mpside' :
        the sidedness of the finite difference when computing
        numerical derivatives. This field can take four
        values:

            0 - one-sided derivative computed automatically
            1 - one-sided derivative (f(x+h) - f(x)  )/h
            -1 - one-sided derivative (f(x)   - f(x-h))/h
            2 - two-sided derivative (f(x+h) - f(x-h))/(2*h)

        Where H is the STEP parameter described above. The
        "automatic" one-sided derivative method will chose a
        direction for the finite difference which does not
        violate any constraints. The other methods do not
        perform this check. The two-sided method is in
        principle more precise, but requires twice as many
        function evaluations. Default: 0.
    'mpmaxstep' :
        the maximum change to be made in the parameter
        value. During the fitting process, the parameter
        will never be changed by more than this value in
        one iteration. A value of 0 indicates no maximum.
        Default: 0.
    'tied' :
        a string expression which "ties" the parameter to other
        free or fixed parameters. Any expression involving
        constants and the parameter array P are permitted.
        Example: if parameter 2 is always to be twice parameter
        1 then use the following: parinfo(2).tied = '2 * p[1]'.
        Since they are totally constrained, tied parameters are
        considered to be fixed; no errors are computed for them.
        [ NOTE: the PARNAME can't be used in expressions. ]
    'mpprint' :
        if set to 1, then the default iterfunct will print the
        parameter value. If set to 0, the parameter value
        will not be printed. This tag can be used to
        selectively print only a few parameter values out of
        many. Default: 1 (all parameters printed)

Future modifications to the PARINFO structure, if any, will involve
adding dictionary tags beginning with the two letters "MP".
Therefore programmers are urged to avoid using tags starting with
the same letters; otherwise they are free to include their own
fields within the PARINFO structure, and they will be ignored.

For example:

    parinfo = [
        {"value": 0.0, "fixed": 0, "limited": [0, 0], "limits": [0.0, 0.0]}
        for i in range(5)
    ]
    parinfo[0]["fixed"] = 1
    parinfo[4]["limited"][0] = 1
    parinfo[4]["limits"][0] = 50.0
    values = [5.7, 2.2, 500.0, 1.5, 2000.0]
    for i in range(5):
        parinfo[i]["value"] = values[i]

A total of 5 parameters, with starting values of 5.7,
2.2, 500, 1.5, and 2000 are given. The first parameter
is fixed at a value of 5.7, and the last parameter is
constrained to be above 50.


EXAMPLE
=======
Minimizes sum of squares of MYFUNCT. MYFUNCT is called with the X,
Y, and ERR keyword parameters that are given by FUNCTKW. The
results can be obtained from the returned object m.

    import mpfit

    x = np.arange(100)
    p0 = [5.7, 2.2, 500.0, 1.5, 2000.0]
    y = p[0] + p[1] * [x] + p[2] * [x ** 2] + p[3] * sqrt(x) + p[4] * log(x)
    fa = {"x": x, "y": y, "err": err}
    m = mpfit("myfunct", p0, functkw=fa)
    print(f"status = {m.status}")
    if m.status <= 0:
        print(f"error message = {m.errmsg}")
    print(f"parameters = {m.params}")


THEORY OF OPERATION
===================
There are many specific strategies for function minimization. One
very popular technique is to use function gradient information to
realize the local structure of the function. Near a local minimum
the function value can be taylor expanded about x0 as follows:

    f(x) = f(x0) + f'(x0) . (x-x0) + (1/2) (x-x0) . f''(x0) . (x-x0)
           -----   ---------------   -------------------------------  (1)
    Order   0th          1st                      2nd

Here f'(x) is the gradient vector of f at x, and f''(x) is the
Hessian matrix of second derivatives of f at x. The vector x is
the set of function parameters, not the measured data vector. One
can find the minimum of f, f(xm) using Newton's method, and
arrives at the following linear equation:

    f''(x0) . (xm-x0) = - f'(x0)                            (2)

If an inverse can be found for f''(x0) then one can solve for
(xm-x0), the step vector from the current position x0 to the new
projected minimum. Here the problem has been linearized (ie, the
gradient information is known to first order). f''(x0) is
symmetric n x n matrix, and should be positive definite.

The Levenberg - Marquardt technique is a variation on this theme.
It adds an additional diagonal term to the equation which may aid the
convergence properties:

    (f''(x0) + nu I) . (xm-x0) = -f'(x0)                  (2a)

where I is the identity matrix. When nu is large, the overall
matrix is diagonally dominant, and the iterations follow steepest
descent. When nu is small, the iterations are quadratically
convergent.

In principle, if f''(x0) and f'(x0) are known then xm-x0 can be
determined. However the Hessian matrix is often difficult or
impossible to compute. The gradient f'(x0) may be easier to
compute, if even by finite difference techniques. So-called
quasi-Newton techniques attempt to successively estimate f''(x0)
by building up gradient information as the iterations proceed.

In the least squares problem there are further simplifications
which assist in solving eqn (2). The function to be minimized is
a sum of squares:

    f = Sum(hi^2)                                         (3)

where hi is the ith residual out of m residuals as described
above. This can be substituted back into eqn (2) after computing
the derivatives:

    f'  = 2 Sum(hi  hi')
    f'' = 2 Sum(hi' hj') + 2 Sum(hi hi'')                (4)

If one assumes that the parameters are already close enough to a
minimum, then one typically finds that the second term in f'' is
negligible [or, in any case, is too difficult to compute]. Thus,
equation (2) can be solved, at least approximately, using only
gradient information.

In matrix notation, the combination of eqns (2) and (4) becomes:

    hT' . h' . dx = - hT' . h                          (5)

Where h is the residual vector (length m), hT is its transpose, h'
is the Jacobian matrix (dimensions n x m), and dx is (xm-x0). The
user function supplies the residual vector h, and in some cases h'
when it is not found by finite differences (see MPFIT_FDJAC2,
which finds h and hT'). Even if dx is not the best absolute step
to take, it does provide a good estimate of the best *direction*,
so often a line minimization will occur along the dx vector
direction.

The method of solution employed by MINPACK is to form the Q . R
factorization of h', where Q is an orthogonal matrix such that QT .
Q = I, and R is upper right triangular. Using h' = Q . R and the
ortogonality of Q, eqn (5) becomes

    (RT . QT) . (Q . R) . dx = - (RT . QT) . h
                 RT . R . dx = - RT . QT . h         (6)
                      R . dx = - QT . h

where the last statement follows because R is upper triangular.
Here, R, QT and h are known so this is a matter of solving for dx.
The routine MPFIT_QRFAC provides the QR factorization of h, with
pivoting, and MPFIT_QRSOLV provides the solution for dx.


REFERENCES
==========
- MINPACK-1, Jorge More', available from netlib (www.netlib.org).
  "Optimization Software Guide," Jorge More' and Stephen Wright,
  SIAM, *Frontiers in Applied Mathematics*, Number 14.
- More', Jorge J., "The Levenberg-Marquardt Algorithm:
  Implementation and Theory," in *Numerical Analysis*, ed. Watson,
  G. A., Lecture Notes in Mathematics 630, Springer-Verlag, 1977.

"""

import logging
import numpy as np
from scipy.linalg import get_blas_funcs
from scipy.optimize import OptimizeResult


_logger = logging.getLogger(__name__)


class machar:
    def __init__(self, double=True):
        info = np.finfo(float) if double else np.finfo(np.float32)

        self.machep = info.eps
        self.maxnum = info.max
        self.minnum = info.tiny

        self.maxlog = np.log(self.maxnum)
        self.minlog = np.log(self.minnum)
        self.rdwarf = np.sqrt(self.minnum * 1.5) * 10
        self.rgiant = np.sqrt(self.maxnum) * 0.1


class mpfit:
    """Perform Levenberg-Marquardt least-squares minimization.

    Minimize the sum of the squares of m nonlinear functions in
    n variables by a modification of the Levenberg-Marquardt algorithm.

    Parameters
    ----------
    fcn : callable
        The function to be minimized. The function should return the weighted
        deviations between the model and the data, as described above.
    xall : np.ndarray
        An array of starting values for each of the parameters of the model.
        The number of parameters should be fewer than the number of measurements.
        This parameter is optional if the parinfo keyword is used (but see
        parinfo). The parinfo keyword provides a mechanism to fix or constrain
        individual parameters.
    autoderivative : bool, default True
        If this is set, derivatives of the function will be computed
        automatically via a finite differencing procedure. If not set, then
        fcn must provide the (analytical) derivatives. To supply your own
        analytical derivatives, explicitly pass autoderivative=False
    functkw : None or dict, default None
        If None, No extra parameters are passed to the user-supplied function.
        A dictionary which contains the parameters to be passed to the
        user-supplied function specified by fcn via the standard Python
        keyword dictionary mechanism. This is the way you can pass additional
        data to your user-supplied function without using global variables.
    ftol : float, default 1e-10
        Termination occurs when both the actual and predicted relative
        reductions in the sum of squares are at most ftol (and status
        is accordingly set to 1 or 3). Therefore, ftol measures the
        relative error desired in the sum of squares.
    gtol : float, default 1e-10
        A nonnegative input variable. Termination occurs when the cosine of
        the angle between fvec and any column of the jacobian is at most gtol
        in absolute value (and status is accordingly set to 4). Therefore,
        gtol measures the orthogonality desired between the function vector
        and the columns of the jacobian.
    xtol : float, default 1e-10
        A nonnegative input variable. Termination occurs when the relative error
        between two consecutive iterates is at most xtol (and status is
        accordingly set to 2 or 3). Therefore, xtol measures the relative error
        desired in the approximate solution.
    iterkw : None or dict, default None
        The keyword arguments to be passed to iterfunct via the dictionary
        keyword mechanism. This should be a dictionary and is similar in
        operation to FUNCTKW.
    iterfunct : None or callable, default None
        The name of a function to be called upon each NPRINT iteration of the
        MPFIT routine. It should be declared in the following way:
        def iterfunct(myfunct, p, iter, fnorm, functkw=None,
                                        parinfo=None, quiet=False, dof=None, [iterkw keywords here])
        # perform custom iteration update

        iterfunct must accept all three keyword parameters (FUNCTKW, PARINFO
        and QUIET).

        myfunct:  The user-supplied function to be minimized,
        p:        The current set of model parameters
        niter:    The iteration number
        functkw:  The arguments to be passed to myfunct.
        fnorm:    The chi-squared value.
        quiet:    Set when no textual output should be printed.
        dof:      The number of degrees of freedom, normally the number of points
                        less the number of free parameters.
        See below for documentation of parinfo.

        In implementation, iterfunct can perform updates to the terminal or
        graphical user interface, to provide feedback while the fit proceeds.
        If the fit is to be stopped for any reason, then iterfunct should return a
        a status value between -15 and -1. Otherwise it should return None
        (e.g. no return statement) or 0.
        In principle, iterfunct should probably not modify the parameter values,
        because it may interfere with the algorithm's stability. In practice it
        is allowed.

        Default: an internal routine is used to print the parameter values.

        Set iterfunct=None if there is no user-defined routine and you don't
        want the internal default routine be called.
    maxiter : int, default 200
        The maximum number of iterations to perform. If the number is exceeded,
        then the status value is set to 5 and MPFIT returns.
    nocovar : bool, default False
        If True, does not estimate the parameter covariance matrix.
    nprint : int, default 1
        The frequency with which iterfunct is called. A value of 1 indicates
        that iterfunct is called with every iteration, while 2 indicates every
        other iteration, etc. Note that several Levenberg-Marquardt attempts
        can be made in a single iteration.
    parinfo : None or ..., default None
        Provides a mechanism for more sophisticated constraints to be placed on
        parameter values. When parinfo is None, then it is assumed that
        all parameters are free and unconstrained. Values in parinfo are never
        modified during a call to MPFIT.
    quiet : bool, default False
        Set this keyword when no textual output should be printed by MPFIT
    damp : float, default 0
        A scalar number, indicating the cut-off value of residuals where
        "damping" will occur. Residuals with magnitudes greater than this
        number will be replaced by their hyperbolic tangent. This partially
        mitigates the so-called large residual problem inherent in
        least-squares solvers. A value of 0 indicates no damping.
        Note: DAMP doesn't work with autoderivative=False


    Returns
    -------
    self : object
        The object itself.

    Attributes
    ----------
    status : int
        An integer status code is returned. All values greater than zero can
        represent success (however .status == 5 may indicate failure to
        converge). It can have one of the following values:

            * -16: A parameter or function value has become infinite or an undefined
            number. This is usually a consequence of numerical overflow in the
            user's model function, which must be avoided.
            * -15 to -1: These are error codes that either MYFUNCT or iterfunct may
            return to terminate the fitting process. Values from -15 to -1 are
            reserved for the user functions and will not clash with MPFIT.
            * 0: Improper input parameters.
            * 1: Both actual and predicted relative reductions in the sum of
            squares are at most ftol.
            * 2: Relative error between two consecutive iterates is at most xtol
            * 3: Conditions for status = 1 and status = 2 both hold.
            * 4: The cosine of the angle between fvec and any column of the jacobian
            is at most gtol in absolute value.
            * 5: The maximum number of iterations has been reached.
            * 6: ftol is too small. No further reduction in the sum of
            squares is possible.
            * 7: xtol is too small. No further improvement in the approximate
            solution x is possible.
            * 8: gtol is too small. fvec is orthogonal to the columns of
            the jacobian to machine precision.

    fnorm : float
        The summed squared residuals for the returned parameter values.
    covar : np.ndarray
        The covariance matrix for the set of parameters returned by MPFIT.
        The matrix is NxN where N is the number of  parameters. The square root
        of the diagonal elements gives the formal 1-sigma statistical errors on
        the parameters if errors were treated "properly" in fcn.
        Parameter errors are also returned in .perror.

        To compute the correlation matrix, pcor, use this example:
        cov = mpfit.covar
        pcor = cov * 0.
        for i in range(n):
                for j in range(n):
                        pcor[i,j] = cov[i,j]/sqrt(cov[i,i]*cov[j,j])

        If nocovar is set or MPFIT terminated abnormally, then .covar is set to
        a scalar with value None.
    errmsg : str
        A string error or warning message is returned.
    nfev : int
        The number of calls to MYFUNCT performed.
    niter : int
        The number of iterations completed.
    perror : np.ndarray
        The formal 1-sigma errors in each parameter, computed from the
        covariance matrix. If a parameter is held fixed, or if it touches a
        boundary, then the error is reported as zero.

        If the fit is unweighted (i.e. no errors were given, or the weights
        were uniformly set to unity), then .perror will probably not represent
        the true parameter uncertainties.

        *If* you can assume that the true reduced chi-squared value is unity --
        meaning that the fit is implicitly assumed to be of good quality --
        then the estimated parameter uncertainties can be computed by scaling
        .perror by the measured chi-squared value.

        dof = len(x) - len(mpfit.params) # deg of freedom
        # scaled uncertainties
        pcerror = mpfit.perror * sqrt(mpfit.fnorm / dof)

    """

    (blas_enorm32,) = get_blas_funcs(["nrm2"], np.array([0], dtype=np.float32))
    (blas_enorm64,) = get_blas_funcs(["nrm2"], np.array([0], dtype=float))

    def __init__(
        self,
        fcn,
        xall=None,
        functkw=None,
        parinfo=None,
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
        damp=0.0,
        maxiter=200,
        factor=100.0,
        nprint=1,
        iterfunct="default",
        iterkw=None,
        nocovar=False,
        rescale=0,
        autoderivative=True,
        quiet=False,
        diag=None,
        epsfcn=None,
        debug=0,
    ):
        if iterkw is None:
            iterkw = {}
        if functkw is None:
            functkw = {}
        self.niter = 0
        self.params = None
        self.covar = None
        self.perror = None
        self.status = 0  # Invalid input flag set while we check inputs
        self.errmsg = ""
        self.nfev = 0
        self.damp = damp
        self.dof = 0

        if fcn is None:
            self.errmsg = "Usage: parms = mpfit('myfunt', ... )"
            return

        if iterfunct == "default":
            iterfunct = self.defiter

        # Parameter damping doesn't work when user is providing their own
        # gradients.
        if (self.damp != 0) and not autoderivative:
            self.errmsg = (
                "ERROR: keywords DAMP and AUTODERIVATIVE are mutually exclusive"
            )
            return

        # Parameters can either be stored in parinfo, or x. x takes precedence
        # if it exists
        if (xall is None) and (parinfo is None):
            self.errmsg = "ERROR: must pass parameters in P or PARINFO"
            return

        # Be sure that PARINFO is of the right type
        if parinfo is not None:
            if not isinstance(parinfo, list):
                self.errmsg = "ERROR: PARINFO must be a list of dictionaries."
                return
            else:
                if not isinstance(parinfo[0], dict):
                    self.errmsg = "ERROR: PARINFO must be a list of dictionaries."
                    return
            if (xall is not None) and (len(xall) != len(parinfo)):
                self.errmsg = "ERROR: number of elements in PARINFO and P must agree"
                return

        # If the parameters were not specified at the command line, then
        # extract them from PARINFO
        if xall is None:
            xall = self.parinfo(parinfo, "value")
            if xall is None:
                self.errmsg = 'ERROR: either P or PARINFO(*)["value"] must be supplied.'
                return

        # Make sure parameters are numpy arrays
        xall = np.asarray(xall)
        # In the case if the xall is not float or if is float but has less
        # than 64 bits we do convert it into double
        if xall.dtype.kind != "f" or xall.dtype.itemsize <= 4:
            xall = xall.astype(float)

        npar = len(xall)
        self.fnorm = -1.0
        fnorm1 = -1.0

        # TIED parameters?
        ptied = self.parinfo(parinfo, "tied", default="", n=npar)
        self.qanytied = 0
        for i in range(npar):
            ptied[i] = ptied[i].strip()
            if ptied[i] != "":
                self.qanytied = 1
        self.ptied = ptied

        # FIXED parameters ?
        pfixed = self.parinfo(parinfo, "fixed", default=0, n=npar)
        pfixed = pfixed == 1
        for i in range(npar):
            # Tied parameters are also effectively fixed
            pfixed[i] = pfixed[i] or (ptied[i] != "")

        # Finite differencing step, absolute and relative, and sidedness of
        # deriv.
        step = self.parinfo(parinfo, "step", default=0.0, n=npar)
        dstep = self.parinfo(parinfo, "relstep", default=0.0, n=npar)
        dside = self.parinfo(parinfo, "mpside", default=0, n=npar)

        # Maximum and minimum steps allowed to be taken in one iteration
        maxstep = self.parinfo(parinfo, "mpmaxstep", default=0.0, n=npar)
        minstep = self.parinfo(parinfo, "mpminstep", default=0.0, n=npar)
        qmin = minstep != 0
        qmin[:] = False  # Remove minstep for now!!
        qmax = maxstep != 0
        if np.any(qmin & qmax & (maxstep < minstep)):
            self.errmsg = "ERROR: MPMINSTEP is greater than MPMAXSTEP"
            return
        wh = (np.nonzero((qmin != 0.0) | (qmax != 0.0)))[0]
        qminmax = len(wh > 0)

        # Finish up the free parameters
        ifree = (np.nonzero(pfixed != 1))[0]
        nfree = len(ifree)
        if nfree == 0:
            self.errmsg = "ERROR: no free parameters"
            return

        # Compose only VARYING parameters
        # self.params is the set of parameters to be returned
        self.params = xall.copy()
        x = self.params[ifree]  # x is the set of free parameters

        # LIMITED parameters ?
        limited = self.parinfo(parinfo, "limited", default=[0, 0], n=npar)
        limits = self.parinfo(parinfo, "limits", default=[0.0, 0.0], n=npar)
        if (limited is not None) and (limits is not None):
            # Error checking on limits in parinfo
            if np.any(
                (limited[:, 0] & (xall < limits[:, 0]))
                | (limited[:, 1] & (xall > limits[:, 1]))
            ):
                self.errmsg = "ERROR: parameters are not within PARINFO limits"
                return
            if np.any(
                (limited[:, 0] & limited[:, 1])
                & (limits[:, 0] >= limits[:, 1])
                & (pfixed == 0)
            ):
                self.errmsg = "ERROR: PARINFO parameter limits are not consistent"
                return

            # Transfer structure values to local variables
            qulim = (limited[:, 1])[ifree]
            ulim = (limits[:, 1])[ifree]
            qllim = (limited[:, 0])[ifree]
            llim = (limits[:, 0])[ifree]

            if np.any((qulim != 0.0) | (qllim != 0.0)):
                qanylim = 1
            else:
                qanylim = 0
        else:
            # Fill in local variables with dummy values
            qulim = np.zeros(nfree)
            ulim = x * 0.0
            qllim = qulim
            llim = x * 0.0
            qanylim = 0

        n = len(x)
        # Check input parameters for errors
        if (
            (n < 0)
            or (ftol <= 0)
            or (xtol <= 0)
            or (gtol <= 0)
            or (maxiter < 0)
            or (factor <= 0)
        ):
            self.errmsg = "ERROR: input keywords are inconsistent"
            return

        if rescale != 0:
            self.errmsg = "ERROR: DIAG parameter scales are inconsistent"
            if len(diag) < n:
                return
            if np.any(diag <= 0):
                return
            self.errmsg = ""

        [self.status, fvec] = self.call(fcn, self.params, functkw)

        if self.status < 0:
            self.errmsg = f"ERROR: first call to {fcn} failed"
            return
        # If the returned fvec has more than four bits I assume that we have
        # double precision
        # It is important that the machar is determined by the precision of
        # the returned value, not by the precision of the input array
        if np.array([fvec]).dtype.itemsize > 4:
            self.machar = machar(double=True)
            self.blas_enorm = mpfit.blas_enorm64
        else:
            self.machar = machar(double=False)
            self.blas_enorm = mpfit.blas_enorm32
        machep = self.machar.machep

        m = len(fvec)
        if m < n:
            self.errmsg = "ERROR: number of parameters must not exceed data"
            return
        self.dof = m - nfree
        self.fnorm = self.enorm(fvec)

        # Initialize Levelberg-Marquardt parameter and iteration counter

        par = 0.0
        self.niter = 1
        qtf = x * 0.0
        self.status = 0

        # Beginning of the outer loop

        while True:
            # If requested, call fcn to enable printing of iterates
            self.params[ifree] = x
            if self.qanytied:
                self.params = self.tie(self.params, ptied)

            if (nprint > 0) and (iterfunct is not None):
                if ((self.niter - 1) % nprint) == 0:
                    xnew0 = self.params.copy()

                    dof = np.max([len(fvec) - len(x), 0])
                    status = iterfunct(
                        fcn,
                        self.params,
                        self.niter,
                        self.fnorm ** 2,
                        functkw=functkw,
                        parinfo=parinfo,
                        quiet=quiet,
                        dof=dof,
                        **iterkw,
                    )
                    if status is not None:
                        self.status = status

                    # Check for user termination
                    if self.status < 0:
                        self.errmsg = f"WARNING: premature termination by {iterfunct}"
                        return

                    # If parameters were changed (grrr..) then re-tie
                    if np.max(abs(xnew0 - self.params)) > 0:
                        if self.qanytied:
                            self.params = self.tie(self.params, ptied)
                        x = self.params[ifree]

            # Calculate the jacobian matrix
            self.status = 2
            _logger.debug("calling MPFIT_FDJAC2")
            fjac = self.fdjac2(
                fcn,
                x,
                fvec,
                step,
                qulim,
                ulim,
                dside,
                epsfcn=epsfcn,
                autoderivative=autoderivative,
                dstep=dstep,
                functkw=functkw,
                ifree=ifree,
                xall=self.params,
            )
            if fjac is None:
                self.errmsg = "WARNING: premature termination by FDJAC2"
                return

            # Determine if any of the parameters are pegged at the limits
            if qanylim:
                _logger.debug("zeroing derivatives of pegged parameters")
                whlpeg = (np.nonzero(qllim & (x == llim)))[0]
                nlpeg = len(whlpeg)
                whupeg = (np.nonzero(qulim & (x == ulim)))[0]
                nupeg = len(whupeg)
                # See if any "pegged" values should keep their derivatives
                if nlpeg > 0:
                    # Total derivative of sum wrt lower pegged parameters
                    for i in range(nlpeg):
                        sum0 = sum(fvec * fjac[:, whlpeg[i]])
                        if sum0 > 0:
                            fjac[:, whlpeg[i]] = 0
                if nupeg > 0:
                    # Total derivative of sum wrt upper pegged parameters
                    for i in range(nupeg):
                        sum0 = sum(fvec * fjac[:, whupeg[i]])
                        if sum0 < 0:
                            fjac[:, whupeg[i]] = 0

            # Compute the QR factorization of the jacobian
            [fjac, ipvt, wa1, wa2] = self.qrfac(fjac, pivot=True)

            # On the first iteration if "diag" is unspecified, scale
            # according to the norms of the columns of the initial jacobian
            _logger.debug("rescaling diagonal elements")
            if self.niter == 1:
                if (rescale == 0) or (len(diag) < n):
                    diag = wa2.copy()
                    diag[diag == 0] = 1.0

                # On the first iteration, calculate the norm of the scaled x
                # and initialize the step bound delta
                wa3 = diag * x
                xnorm = self.enorm(wa3)
                delta = factor * xnorm
                if delta == 0.0:
                    delta = factor

            # Form (q transpose)*fvec and store the first n components in qtf
            _logger.debug("forming (q transpose)*fvec")
            wa4 = fvec.copy()
            for j in range(n):
                lj = ipvt[j]
                temp3 = fjac[j, lj]
                if temp3 != 0:
                    fj = fjac[j:, lj]
                    wj = wa4[j:]
                    # *** optimization wa4(j:*)
                    wa4[j:] = wj - fj * sum(fj * wj) / temp3
                fjac[j, lj] = wa1[j]
                qtf[j] = wa4[j]
            # From this point on, only the square matrix, consisting of the
            # triangle of R, is needed.
            fjac = fjac[0:n, 0:n]
            fjac.shape = [n, n]
            temp = fjac.copy()
            for i in range(n):
                temp[:, i] = fjac[:, ipvt[i]]
            fjac = temp.copy()

            # Check for overflow. This should be a cheap test here since FJAC
            # has been reduced to a (small) square matrix, and the test is
            # O(N^2).
            # wh = where(finite(fjac) EQ 0, ct)
            # if ct GT 0 then goto, FAIL_OVERFLOW

            # Compute the norm of the scaled gradient
            _logger.debug("computing the scaled gradient")
            gnorm = 0.0
            if self.fnorm != 0:
                for j in range(n):
                    l_ = ipvt[j]
                    if wa2[l_] != 0:
                        sum0 = sum(fjac[0 : j + 1, j] * qtf[0 : j + 1]) / self.fnorm
                        gnorm = np.max([gnorm, abs(sum0 / wa2[l_])])

            # Test for convergence of the gradient norm
            if gnorm <= gtol:
                self.status = 4
                break
            if maxiter == 0:
                self.status = 5
                break

            # Rescale if necessary
            if rescale == 0:
                diag = np.choose(diag > wa2, (wa2, diag))

            # Beginning of the inner loop
            while True:
                # Determine the levenberg-marquardt parameter
                _logger.debug("calculating LM parameter (MPFIT_)")
                [fjac, par, wa1, wa2] = self.lmpar(
                    fjac, ipvt, diag, qtf, delta, wa1, wa2, par=par
                )
                # Store the direction p and x+p. Calculate the norm of p
                wa1 = -wa1

                if (qanylim == 0) and (qminmax == 0):
                    # No parameter limits, so just move to new position WA2
                    alpha = 1.0
                    wa2 = x + wa1

                else:
                    # Respect the limits. If a step were to go out of bounds, then
                    # we should take a step in the same direction but shorter distance.
                    # The step should take us right to the limit in that case.
                    alpha = 1.0

                    if qanylim:
                        # Do not allow any steps out of bounds
                        _logger.debug("checking for a step out of bounds")
                        if nlpeg > 0:
                            wa1[whlpeg] = np.clip(wa1[whlpeg], 0.0, np.max(wa1))
                        if nupeg > 0:
                            wa1[whupeg] = np.clip(wa1[whupeg], np.min(wa1), 0.0)

                        dwa1 = abs(wa1) > machep
                        whl = (
                            np.nonzero(((dwa1 != 0.0) & qllim) & ((x + wa1) < llim))
                        )[0]
                        if len(whl) > 0:
                            t = (llim[whl] - x[whl]) / wa1[whl]
                            alpha = np.min([alpha, np.min(t)])
                        whu = (
                            np.nonzero(((dwa1 != 0.0) & qulim) & ((x + wa1) > ulim))
                        )[0]
                        if len(whu) > 0:
                            t = (ulim[whu] - x[whu]) / wa1[whu]
                            alpha = np.min([alpha, np.min(t)])

                    # Obey any max step values.
                    if qminmax:
                        nwa1 = wa1 * alpha
                        whmax = (
                            np.nonzero((qmax[ifree] != 0.0) & (maxstep[ifree] > 0))
                        )[0]
                        if len(whmax) > 0:
                            mrat = np.max(
                                abs(nwa1[whmax]) / abs(maxstep[ifree[whmax]])
                            )
                            if mrat > 1:
                                alpha /= mrat

                    # Scale the resulting vector
                    wa1 *= alpha
                    wa2 = x + wa1

                    # Adjust the final output values. If the step put us exactly
                    # on a boundary, make sure it is exact.
                    sgnu = (ulim >= 0) * 2.0 - 1.0
                    sgnl = (llim >= 0) * 2.0 - 1.0
                    # Handles case of
                    #        ... nonzero *LIM ... ...zero * LIM
                    ulim1 = ulim * (1 - sgnu * machep) - (ulim == 0) * machep
                    llim1 = llim * (1 + sgnl * machep) + (llim == 0) * machep
                    wh = (np.nonzero((qulim != 0) & (wa2 >= ulim1)))[0]
                    if len(wh) > 0:
                        wa2[wh] = ulim[wh]
                    wh = (np.nonzero((qllim != 0.0) & (wa2 <= llim1)))[0]
                    if len(wh) > 0:
                        wa2[wh] = llim[wh]

                wa3 = diag * wa1
                pnorm = self.enorm(wa3)

                # On the first iteration, adjust the initial step bound
                if self.niter == 1:
                    delta = np.min([delta, pnorm])

                self.params[ifree] = wa2

                # Evaluate the function at x+p and calculate its norm
                _logger.debug(f"calling {fcn}")
                [self.status, wa4] = self.call(fcn, self.params, functkw)
                if self.status < 0:
                    self.errmsg = f"WARNING: premature termination by {fcn}"
                    return
                fnorm1 = self.enorm(wa4)

                # Compute the scaled actual reduction
                _logger.debug("computing convergence criteria")
                actred = -1.0
                if (0.1 * fnorm1) < self.fnorm:
                    actred = -((fnorm1 / self.fnorm) ** 2) + 1.0

                # Compute the scaled predicted reduction and the scaled directional
                # derivative
                for j in range(n):
                    wa3[j] = 0
                    wa3[0 : j + 1] = wa3[0 : j + 1] + fjac[0 : j + 1, j] * wa1[ipvt[j]]

                # Remember, alpha is the fraction of the full LM step actually
                # taken
                temp1 = self.enorm(alpha * wa3) / self.fnorm
                temp2 = (np.sqrt(alpha * par) * pnorm) / self.fnorm
                prered = temp1 * temp1 + (temp2 * temp2) / 0.5
                dirder = -(temp1 * temp1 + temp2 * temp2)

                # Compute the ratio of the actual to the predicted reduction.
                ratio = 0.0
                if prered != 0:
                    ratio = actred / prered

                # Update the step bound
                if ratio <= 0.25:
                    if actred >= 0:
                        temp = 0.5
                    else:
                        temp = 0.5 * dirder / (dirder + 0.5 * actred)
                    if ((0.1 * fnorm1) >= self.fnorm) or (temp < 0.1):
                        temp = 0.1
                    delta = temp * np.min([delta, pnorm / 0.1])
                    par /= temp
                else:
                    if (par == 0) or (ratio >= 0.75):
                        delta = pnorm / 0.5
                        par *= 0.5

                # Test for successful iteration
                if ratio >= 0.0001:
                    # Successful iteration. Update x, fvec, and their norms
                    x = wa2
                    wa2 = diag * x
                    fvec = wa4
                    xnorm = self.enorm(wa2)
                    self.fnorm = fnorm1
                    self.niter += 1

                # Tests for convergence
                if (abs(actred) <= ftol) and (prered <= ftol) and (0.5 * ratio <= 1):
                    self.status = 1
                if delta <= xtol * xnorm:
                    self.status = 2
                if (
                    (abs(actred) <= ftol)
                    and (prered <= ftol)
                    and (0.5 * ratio <= 1)
                    and (self.status == 2)
                ):
                    self.status = 3
                if self.status != 0:
                    break

                # Tests for termination and stringent tolerances
                if self.niter >= maxiter:
                    self.status = 5
                if (
                    (abs(actred) <= machep)
                    and (prered <= machep)
                    and (0.5 * ratio <= 1)
                ):
                    self.status = 6
                if delta <= machep * xnorm:
                    self.status = 7
                if gnorm <= machep:
                    self.status = 8
                if self.status != 0:
                    break

                # End of inner loop. Repeat if iteration unsuccessful
                if ratio >= 0.0001:
                    break

                # Check for over/underflow
                if ~np.all(
                    np.isfinite(wa1) & np.isfinite(wa2) & np.isfinite(x)
                ) or ~np.isfinite(ratio):
                    self.errmsg = (
                        "ERROR: parameter or function value(s) have become "
                        "infinite; check model function for over- and underflow"
                    )
                    self.status = -16
                    break

            if self.status != 0:
                break
        # End of outer loop.

        _logger.debug("in the termination phase")
        # Termination, either normal or user imposed.
        if len(self.params) == 0:
            return
        if nfree == 0:
            self.params = xall.copy()
        else:
            self.params[ifree] = x
        if (nprint > 0) and (self.status > 0):
            _logger.debug(f"calling {fcn}")
            [status, fvec] = self.call(fcn, self.params, functkw)
            _logger.debug("in the termination phase")
            self.fnorm = self.enorm(fvec)

        if (self.fnorm is not None) and (fnorm1 is not None):
            self.fnorm = np.max([self.fnorm, fnorm1])
            self.fnorm **= 2.0

        self.covar = None
        self.perror = None
        # (very carefully) set the covariance matrix COVAR
        if (
            (self.status > 0)
            and (nocovar is False)
            and (n is not None)
            and (fjac is not None)
            and (ipvt is not None)
        ):
            sz = fjac.shape
            if (n > 0) and (sz[0] >= n) and (sz[1] >= n) and (len(ipvt) >= n):

                _logger.debug("computing the covariance matrix")
                cv = self.calc_covar(fjac[0:n, 0:n], ipvt[0:n])
                cv.shape = [n, n]
                nn = len(xall)

                # Fill in actual covariance matrix, accounting for fixed
                # parameters.
                self.covar = np.zeros([nn, nn])
                for i in range(n):
                    self.covar[ifree, ifree[i]] = cv[:, i]

                # Compute errors in parameters
                _logger.debug("computing parameter errors")
                self.perror = np.zeros(nn)
                d = np.diagonal(self.covar)
                wh = (np.nonzero(d >= 0))[0]
                if len(wh) > 0:
                    self.perror[wh] = np.sqrt(d[wh])

    @property
    def optimize_result(self):
        """Converts mpfit object to a scipy.optimize.OptimizeResult."""
        return OptimizeResult(
            x=self.params,
            covar=self.covar,
            perror=self.perror,
            nit=self.niter,
            nfev=self.nfev,
            success=(self.status > 0) and (self.status != 5),
            status=self.status,
            message=self.errmsg,
            dof=self.dof,
            fnorm=self.fnorm,
        )

    def __repr__(self):
        return str(self.optimize_result)

    def defiter(
        self,
        fcn,
        x,
        niter,
        fnorm=None,
        functkw=None,
        quiet=False,
        iterstop=None,
        parinfo=None,
        format=None,
        dof=1,
    ):
        """Default procedure to be called every iteration.

        It simply prints the parameter values.
        """
        if quiet:
            return

        if fnorm is None:
            [status, fvec] = self.call(fcn, x, functkw)
            fnorm = self.enorm(fvec) ** 2

        # Determine which parameters to print
        nprint = len(x)
        _logger.info(f"Iter {niter:d}, Residual={fnorm:10f}, DOF={dof:d}")
        for i in range(nprint):
            if (parinfo is not None) and ("parname" in parinfo[i]):
                p = f"   {parinfo[i]['parname']} = "
            else:
                p = f"   P{i} = "
            if (parinfo is not None) and ("mpprint" in parinfo[i]):
                iprint = parinfo[i]["mpprint"]
            else:
                iprint = 1
            if iprint:
                _logger.info(f"{p}{x[i]:10f}  ")

        return 0

    def parinfo(self, parinfo=None, key="a", default=None, n=0):
        """Procedure to parse the parameter values in PARINFO, which is a list of dicts."""
        if (n == 0) and (parinfo is not None):
            n = len(parinfo)
        if n == 0:
            values = default

            return values
        values = []
        for i in range(n):
            if (parinfo is not None) and (key in parinfo[i]):
                values.append(parinfo[i][key])
            else:
                values.append(default)

        # Convert to numeric arrays if possible
        test = default
        if isinstance(default, list):
            test = default[0]
        if isinstance(test, int):
            values = np.asarray(values, int)
        elif isinstance(test, float):
            values = np.asarray(values, float)
        return values

    def call(self, fcn, x, functkw, fjac=None):
        """Call user function or procedure, with _EXTRA or not, with derivatives or not."""
        if self.qanytied:
            x = self.tie(x, self.ptied)
        self.nfev += 1
        if fjac is None:
            [status, f] = fcn(x, fjac=fjac, **functkw)
            if self.damp > 0:
                # Apply the damping if requested. This replaces the residuals
                # with their hyperbolic tangent. Thus residuals larger than
                # DAMP are essentially clipped.
                f = np.tanh(f / self.damp)
            return [status, f]
        else:
            return fcn(x, fjac=fjac, **functkw)

    def enorm(self, vec):
        return self.blas_enorm(vec)

    def fdjac2(
        self,
        fcn,
        x,
        fvec,
        step=None,
        ulimited=None,
        ulimit=None,
        dside=None,
        epsfcn=None,
        autoderivative=True,
        functkw=None,
        xall=None,
        ifree=None,
        dstep=None,
    ):
        """Calculate the Jacobian either analytically or with finite differencing."""
        machep = self.machar.machep
        if epsfcn is None:
            epsfcn = machep
        if xall is None:
            xall = x
        if ifree is None:
            ifree = np.arange(len(xall))
        if step is None:
            step = x * 0.0
        nall = len(xall)

        eps = np.sqrt(np.max([epsfcn, machep]))
        m = len(fvec)
        n = len(x)

        # Compute analytical derivative if requested
        if not autoderivative:
            fjac = np.zeros(nall)
            fjac[ifree] = 1.0  # Specify which parameters need derivatives
            [status, fjac] = self.call(fcn, xall, functkw, fjac=fjac)

            if fjac.shape != (m, nall):
                _logger.warning("ERROR: Derivative matrix was not computed properly.")
                _logger.warning(fjac.shape)
                _logger.warning(m, nall)
                return None

            # This definition is consistent with CURVEFIT
            # Sign error found (thanks Jesus Fernandez <fernande@irm.chu-caen.fr>)
            #             fjac.shape = [m,nall]
            #             fjac = -fjac

            # Select only the free parameters
            if len(ifree) < nall:
                fjac = fjac[:, ifree]
                fjac.shape = [m, n]
            return fjac

        fjac = np.zeros([m, n])

        h = eps * abs(x)

        # if STEP is given, use that
        # STEP includes the fixed parameters
        if step is not None:
            stepi = step[ifree]
            wh = (np.nonzero(stepi > 0))[0]
            if len(wh) > 0:
                h[wh] = stepi[wh]

        # if relative step is given, use that
        # DSTEP includes the fixed parameters
        if len(dstep) > 0:
            dstepi = dstep[ifree]
            wh = (np.nonzero(dstepi > 0))[0]
            if len(wh) > 0:
                h[wh] = abs(dstepi[wh] * x[wh])

        # In case any of the step values are zero
        h[h == 0] = eps

        # Reverse the sign of the step if we are up against the parameter
        # limit, or if the user requested it.
        # DSIDE includes the fixed parameters (ULIMITED/ULIMIT have only
        # varying ones)
        mask = dside[ifree] == -1
        if len(ulimited) > 0 and len(ulimit) > 0:
            mask = mask | ((ulimited != 0) & (x > ulimit - h))
            wh = (np.nonzero(mask))[0]
            if len(wh) > 0:
                h[wh] = -h[wh]
        # Loop through parameters, computing the derivative for each
        for j in range(n):
            xp = xall.copy()
            xp[ifree[j]] = xp[ifree[j]] + h[j]
            [status, fp] = self.call(fcn, xp, functkw)
            if status < 0:
                return None

            if abs(dside[ifree[j]]) <= 1:
                # COMPUTE THE ONE-SIDED DERIVATIVE
                # Note optimization fjac(0:*,j)
                fjac[0:, j] = (fp - fvec) / h[j]

            else:
                # COMPUTE THE TWO-SIDED DERIVATIVE
                xp[ifree[j]] = xall[ifree[j]] - h[j]

                [status, fm] = self.call(fcn, xp, functkw)
                if status < 0:
                    return None

                # Note optimization fjac(0:*,j)
                fjac[0:, j] = (fp - fm) / (2 * h[j])

        return fjac

    def qrfac(self, a, pivot=False):
        """QR factorization of the matrix a.

        Note that it is usually never necessary to form the Q matrix
        explicitly, and MPFIT does not.

        This function uses householder transformations with column
        pivoting (optional) to compute a qr factorization of the
        m by n matrix a. that is, qrfac determines an orthogonal
        matrix q, a permutation matrix p, and an upper trapezoidal
        matrix r with diagonal elements of nonincreasing magnitude,
        such that a*p = q*r. the householder transformation for
        column k, k = 1,2,...,min(m,n), is of the form

                            t
            i - (1/u(k))*u*u

        where u has zeros in the first k-1 positions. the form of
        this transformation and the method of pivoting first
        appeared in the corresponding linpack subroutine.

        Parameters
        ----------
        a : np.ndarray, shape (m, n)
            The matrix for which the qr factorization is to be computed.
        pivot : bool, default False
            If True, then column pivoting is enforced.
            If False, then no column pivoting is done.

        Returns
        -------
        a : np.ndarray, shape (m, n)
            The strict upper trapezoidal part of a contains the strict
            upper trapezoidal part of r, and the lower trapezoidal
            part of a contains a factored form of q (the non-trivial
            elements of the u vectors described above).
        ipvt : np.ndarray, shape (n,)
            Defines the permutation matrix p such that a*p = q*r.
            column j of p is column ipvt(j) of the identity matrix.
            If pivot is false, ipvt is not referenced.
        rdiag : np.ndarray, shape (n,)
            Contains the diagonal elements of r.
        acnorm : np.ndarray, shape (n,)
            Contains the norms of the corresponding columns of the input matrix a.
            If this information is not needed, then acnorm can coincide with rdiag.

        Notes
        -----
        Upon return, A(*,*) is in standard parameter order, A(*,IPVT) is in
        permuted order. RDIAG is in permuted order. ACNORM is in standard
        parameter order. The matrix A is still m x n where m >= n.

        The "upper" triangular matrix R is actually stored in the strict
        lower left triangle of A under the standard notation of IDL. The
        reflectors that generate Q are in the upper trapezoid of A upon
        output.

        """
        machep = self.machar.machep
        sz = a.shape
        m = sz[0]
        n = sz[1]

        # Compute the initial column norms and initialize arrays
        acnorm = np.zeros(n)
        for j in range(n):
            acnorm[j] = self.enorm(a[:, j])
        rdiag = acnorm.copy()
        wa = rdiag.copy()
        ipvt = np.arange(n)

        # Reduce a to r with householder transformations
        minmn = np.min([m, n])
        for j in range(minmn):
            if pivot:
                # Bring the column of largest norm into the pivot position
                rmax = np.max(rdiag[j:])
                kmax = (np.nonzero(rdiag[j:] == rmax))[0]
                ct = len(kmax)
                kmax += j
                if ct > 0:
                    kmax = kmax[0]

                    # Exchange rows via the pivot only. Avoid actually exchanging
                    # the rows, in case there is lots of memory transfer. The
                    # exchange occurs later, within the body of MPFIT, after the
                    # extraneous columns of the matrix have been shed.
                    if kmax != j:
                        temp = ipvt[j]
                        ipvt[j] = ipvt[kmax]
                        ipvt[kmax] = temp
                        rdiag[kmax] = rdiag[j]
                        wa[kmax] = wa[j]

            # Compute the householder transformation to reduce the jth
            # column of A to a multiple of the jth unit vector
            lj = ipvt[j]
            ajj = a[j:, lj]
            ajnorm = self.enorm(ajj)
            if ajnorm == 0:
                break
            if a[j, lj] < 0:
                ajnorm = -ajnorm

            ajj = ajj / ajnorm
            ajj[0] += 1
            # *** Note optimization a(j:*,j)
            a[j:, lj] = ajj

            # Apply the transformation to the remaining columns
            # and update the norms

            # NOTE to SELF: tried to optimize this by removing the loop,
            # but it actually got slower. Reverted to "for" loop to keep
            # it simple.
            if j + 1 < n:
                for k in range(j + 1, n):
                    lk = ipvt[k]
                    ajk = a[j:, lk]
                    # *** Note optimization a(j:*,lk)
                    # (corrected 20 Jul 2000)
                    if a[j, lj] != 0:
                        a[j:, lk] = ajk - ajj * sum(ajk * ajj) / a[j, lj]
                        if pivot and (rdiag[k] != 0):
                            temp = a[j, lk] / rdiag[k]
                            rdiag[k] *= np.sqrt(np.max([(1.0 - temp ** 2), 0.0]))
                            temp = rdiag[k] / wa[k]
                            if (0.05 * temp * temp) <= machep:
                                rdiag[k] = self.enorm(a[j + 1 :, lk])
                                wa[k] = rdiag[k]
            rdiag[j] = -ajnorm
        return [a, ipvt, rdiag, acnorm]

    def qrsolv(self, r, ipvt, diag, qtb, sdiag):
        """Solve A*x=b, d*x=0 using QR factorization.

        Given an m by n matrix a, an n by n diagonal matrix d,
        and an m-vector b, the problem is to determine an x which
        solves the system

            a*x = b ,     d*x = 0 ,

        in the least squares sense.

        This function completes the solution of the problem
        if it is provided with the necessary information from the
        factorization, with column pivoting, of a. that is, if
        a*p = q*r, where p is a permutation matrix, q has orthogonal
        columns, and r is an upper triangular matrix with diagonal
        elements of nonincreasing magnitude, then qrsolv expects
        the full upper triangle of r, the permutation matrix p,
        and the first n components of (q transpose)*b. the system
        a*x = b, d*x = 0, is then equivalent to

                   t       t
            r*z = q *b ,  p *d*p*z = 0 ,

        where x = p*z. if this system does not have full rank,
        then a least squares solution is obtained. on output qrsolv
        also provides an upper triangular matrix s such that

             t   t               t
            p *(a *a + d*d)*p = s *s .

        s is computed within qrsolv and may be of separate interest.

        Parameters
        ----------
        r : np.ndarray, shape (n, n)
            The full upper triangle must contain the full
            upper triangle of the matrix r.
        ipvt : np.ndarray, shape (n,)
            Defines the permutation matrix p such that a*p = q*r.
            column j of p is column ipvt(j) of the identity matrix.
        diag : np.ndarray, shape (n,)
            The diagonal elements of the matrix d.
        qtb : np.ndarray, shape (n,)
            Must contain the first n elements of the vector (q transpose)*b.
        sdiag : np.ndarray, shape (n,)
            Array to store the diagonal elements of the upper
            triangular matrix s

        Returns
        -------
        r : np.ndarray, shape (n, n)
            From the input r, the full upper triangle is unaltered, and the
            strict lower triangle contains the strict upper triangle
            (transposed) of the upper triangular matrix s.
        x : np.ndarray, shape (n,)
            The least-squares solution of the system a*x = b, d*x = 0.
        sdiag : np.ndarray, shape (n,)
            The diagonal elements of the upper triangular matrix s

        """
        sz = r.shape
        m, n = sz

        # copy r and (q transpose)*b to preserve input and initialize s.
        # in particular, save the diagonal elements of r in x.

        for j in range(n):
            r[j:n, j] = r[j, j:n]
        x = np.diagonal(r).copy()
        wa = qtb.copy()

        # Eliminate the diagonal matrix d using a givens rotation
        for j in range(n):
            l_ = ipvt[j]
            if diag[l_] == 0:
                break
            sdiag[j:] = 0
            sdiag[j] = diag[l_]

            # The transformations to eliminate the row of d modify only a
            # single element of (q transpose)*b beyond the first n, which
            # is initially zero.

            qtbpj = 0.0
            for k in range(j, n):
                if sdiag[k] == 0:
                    break
                if abs(r[k, k]) < abs(sdiag[k]):
                    cotan = r[k, k] / sdiag[k]
                    sine = 0.5 / np.sqrt(0.25 + 0.25 * cotan * cotan)
                    cosine = sine * cotan
                else:
                    tang = sdiag[k] / r[k, k]
                    cosine = 0.5 / np.sqrt(0.25 + 0.25 * tang * tang)
                    sine = cosine * tang

                # Compute the modified diagonal element of r and the
                # modified element of ((q transpose)*b,0).
                r[k, k] = cosine * r[k, k] + sine * sdiag[k]
                temp = cosine * wa[k] + sine * qtbpj
                qtbpj = -sine * wa[k] + cosine * qtbpj
                wa[k] = temp

                # Accumulate the transformation in the row of s
                if n > k + 1:
                    temp = cosine * r[k + 1 : n, k] + sine * sdiag[k + 1 : n]
                    sdiag[k + 1 : n] = (
                        -sine * r[k + 1 : n, k] + cosine * sdiag[k + 1 : n]
                    )
                    r[k + 1 : n, k] = temp
            sdiag[j] = r[j, j]
            r[j, j] = x[j]

        # Solve the triangular system for z. If the system is singular
        # then obtain a least squares solution
        nsing = n
        wh = (np.nonzero(sdiag == 0))[0]
        if len(wh) > 0:
            nsing = wh[0]
            wa[nsing:] = 0

        if nsing >= 1:
            wa[nsing - 1] = wa[nsing - 1] / sdiag[nsing - 1]  # Degenerate case
            # *** Reverse loop ***
            for j in range(nsing - 2, -1, -1):
                sum0 = sum(r[j + 1 : nsing, j] * wa[j + 1 : nsing])
                wa[j] = (wa[j] - sum0) / sdiag[j]

        # Permute the components of z back to components of x
        x[ipvt] = wa
        return r, x, sdiag

    def lmpar(self, r, ipvt, diag, qtb, delta, x, sdiag, par=None):
        """Solve the least-squares system

        Given an m by n matrix a, an n by n nonsingular diagonal
        matrix d, an m-vector b, and a positive number delta,
        the problem is to determine a value for the parameter
        par such that if x solves the system:

            a*x = b ,     sqrt(par)*d*x = 0 ,

        in the least squares sense, and dxnorm is the euclidean
        norm of d*x, then either par is zero and

            (dxnorm-delta) .le. 0.1*delta ,

        or par is positive and

            abs(dxnorm-delta) .le. 0.1*delta .

        this function completes the solution of the problem
        if it is provided with the necessary information from the
        qr factorization, with column pivoting, of a. that is, if
        a*p = q*r, where p is a permutation matrix, q has orthogonal
        columns, and r is an upper triangular matrix with diagonal
        elements of nonincreasing magnitude, then lmpar expects
        the full upper triangle of r, the permutation matrix p,
        and the first n components of (q transpose)*b. on output
        lmpar also provides an upper triangular matrix s such that

             t   t                   t
            p *(a *a + par*d*d)*p = s *s .

        s is employed within lmpar and may be of separate interest.

        only a few iterations are generally needed for convergence
        of the algorithm. if, however, the limit of 10 iterations
        is reached, then the output par will contain the best
        value obtained so far.

        Parameters
        ----------
        n is a positive integer input variable set to the order of r.

        r is an n by n array. on input the full upper triangle
            must contain the full upper triangle of the matrix r.
            on output the full upper triangle is unaltered, and the
            strict lower triangle contains the strict upper triangle
            (transposed) of the upper triangular matrix s.

        ldr is a positive integer input variable not less than n
            which specifies the leading dimension of the array r.

        ipvt is an integer input array of length n which defines the
            permutation matrix p such that a*p = q*r. column j of p
            is column ipvt(j) of the identity matrix.

        diag is an input array of length n which must contain the
            diagonal elements of the matrix d.

        qtb is an input array of length n which must contain the first
            n elements of the vector (q transpose)*b.

        delta is a positive input variable which specifies an upper
            bound on the euclidean norm of d*x.

        par is a nonnegative variable. on input par contains an
            initial estimate of the levenberg-marquardt parameter.
            on output par contains the final estimate.

        x is an output array of length n which contains the least
            squares solution of the system a*x = b, sqrt(par)*d*x = 0,
            for the output par.

        sdiag is an output array of length n which contains the
            diagonal elements of the upper triangular matrix s.

        wa1 and wa2 are work arrays of length n.

        Returns
        -------
        TODO
        """
        dwarf = self.machar.minnum
        machep = self.machar.machep
        sz = r.shape
        m, n = sz

        # Compute and store in x the gauss-newton direction. If the
        # jacobian is rank-deficient, obtain a least-squares solution
        nsing = n
        wa1 = qtb.copy()
        rdiagabs = abs(np.diagonal(r))
        rthresh = np.max(rdiagabs) * machep
        wh = (np.nonzero(rdiagabs < rthresh))[0]
        if len(wh) > 0:
            nsing = wh[0]
            wa1[wh[0] :] = 0
        if nsing >= 1:
            # *** Reverse loop ***
            for j in range(nsing - 1, -1, -1):
                wa1[j] = wa1[j] / r[j, j]
                if j - 1 >= 0:
                    wa1[0:j] = wa1[0:j] - r[0:j, j] * wa1[j]

        # Note: ipvt here is a permutation array
        x[ipvt] = wa1

        # Initialize the iteration counter. Evaluate the function at the
        # origin, and test for acceptance of the gauss-newton direction
        wa2 = diag * x
        dxnorm = self.enorm(wa2)
        fp = dxnorm - delta
        if fp <= 0.1 * delta:
            return [r, 0.0, x, sdiag]

        # If the jacobian is not rank deficient, the newton step provides a
        # lower bound, parl, for the zero of the function. Otherwise set
        # this bound to zero.

        parl = 0.0
        if nsing >= n:
            wa1 = diag[ipvt] * wa2[ipvt] / dxnorm
            wa1[0] = wa1[0] / r[0, 0]  # Degenerate case
            for j in range(1, n):  # Note "1" here, not zero
                sum0 = sum(r[0:j, j] * wa1[0:j])
                wa1[j] = (wa1[j] - sum0) / r[j, j]

            temp = self.enorm(wa1)
            parl = ((fp / delta) / temp) / temp

        # Calculate an upper bound, paru, for the zero of the function
        for j in range(n):
            sum0 = sum(r[0 : j + 1, j] * qtb[0 : j + 1])
            wa1[j] = sum0 / diag[ipvt[j]]
        gnorm = self.enorm(wa1)
        paru = gnorm / delta
        if paru == 0:
            paru = dwarf / np.min([delta, 0.1])

        # If the input par lies outside of the interval (parl,paru), set
        # par to the closer endpoint

        par = np.max([par, parl])
        par = np.min([par, paru])
        if par == 0:
            par = gnorm / dxnorm

        # Beginning of an interation (max_iter here is 10,
        # was hard-coded in earlier versions?)
        for _niter in range(10):
            # Evaluate the function at the current value of par
            if par == 0:
                par = np.max([dwarf, paru * 0.001])
            temp = np.sqrt(par)
            wa1 = temp * diag
            [r, x, sdiag] = self.qrsolv(r, ipvt, wa1, qtb, sdiag)
            wa2 = diag * x
            dxnorm = self.enorm(wa2)
            temp = fp
            fp = dxnorm - delta

            if (abs(fp) <= 0.1 * delta) or (
                (parl == 0) and (fp <= temp) and (temp < 0)
            ):
                break

            # Compute the newton correction
            wa1 = diag[ipvt] * wa2[ipvt] / dxnorm

            for j in range(n - 1):
                wa1[j] = wa1[j] / sdiag[j]
                wa1[j + 1 : n] = wa1[j + 1 : n] - r[j + 1 : n, j] * wa1[j]
            wa1[n - 1] = wa1[n - 1] / sdiag[n - 1]  # Degenerate case

            temp = self.enorm(wa1)
            parc = ((fp / delta) / temp) / temp

            # Depending on the sign of the function, update parl or paru
            if fp > 0:
                parl = np.max([parl, par])
            if fp < 0:
                paru = np.min([paru, par])

            # Compute an improved estimate for par
            par = np.max([parl, par + parc])

        return [r, par, x, sdiag]

    def tie(self, p, ptied=None):
        """Tie one parameter to another."""
        if ptied is None:
            return
        for i in range(len(ptied)):
            if ptied[i] == "":
                continue
            cmd = f"p[{i}] = {ptied[i]}"
            exec(cmd)

        return p

    def calc_covar(self, rr, ipvt=None, tol=1e-14):
        """Calculate the covariance matrix.

        Given an m by n matrix a, the problem is to determine
        the covariance matrix corresponding to a, defined as

                     t
            inverse(a *a) .

        This function completes the solution of the problem
        if it is provided with the necessary information from the
        qr factorization, with column pivoting, of a. that is, if
        a*p = q*r, where p is a permutation matrix, q has orthogonal
        columns, and r is an upper triangular matrix with diagonal
        elements of nonincreasing magnitude, then covar expects
        the full upper triangle of r and the permutation matrix p.
        the covariance matrix is then computed as

                       t     t
            p*inverse(r *r)*p  .

        If a is nearly rank-deficient, it may be desirable to compute
        the covariance matrix corresponding to the linearly independent
        columns of a. to define the numerical rank of a, covar uses
        the tolerance tol. if l is the largest integer such that

            abs(r(l,l)) .gt. tol*abs(r(1,1)) ,

        then covar computes the covariance matrix corresponding to
        the first l columns of r. for k greater than l, column
        and row ipvt(k) of the covariance matrix are set to zero.

        Parameters
        ----------
        rr : np.ndarray, shape(n, n)
            Input matrix
        ipvt : None or np.ndarray, default None
            If not None, should be is an integer input array of length n
            which defines the permutation matrix p such that a*p = q*r.
            column j of p is column ipvt(j) of the identity matrix.
        tol : float, default 1e-14
            A nonnegative input variable used to define the
            numerical rank of a in the manner described above.

        Returns
        -------
        r : np.ndarray, shape(n, n)
            The square symmetric covariance matrix.
        """
        if np.ndim(rr) != 2:
            _logger.error("ERROR: r must be a two-dimensional matrix")
            return -1
        s = rr.shape
        n = s[0]
        if s[0] != s[1]:
            _logger.error("ERROR: r must be a square matrix")
            return -1

        if ipvt is None:
            ipvt = np.arange(n)
        r = rr.copy()
        r.shape = [n, n]

        # For the inverse of r in the full upper triangle of r
        l_ = -1
        tolr = tol * abs(r[0, 0])
        for k in range(n):
            if abs(r[k, k]) <= tolr:
                break
            r[k, k] = 1.0 / r[k, k]
            for j in range(k):
                temp = r[k, k] * r[j, k]
                r[j, k] = 0.0
                r[0 : j + 1, k] = r[0 : j + 1, k] - temp * r[0 : j + 1, j]
            l_ = k

        # Form the full upper triangle of the inverse of (r transpose)*r
        # in the full upper triangle of r
        if l_ >= 0:
            for k in range(l_ + 1):
                for j in range(k):
                    temp = r[j, k]
                    r[0 : j + 1, j] = r[0 : j + 1, j] + temp * r[0 : j + 1, k]
                temp = r[k, k]
                r[0 : k + 1, k] = temp * r[0 : k + 1, k]

        # For the full lower triangle of the covariance matrix
        # in the strict lower triangle or and in wa
        wa = np.repeat([r[0, 0]], n)
        for j in range(n):
            jj = ipvt[j]
            sing = j > l_
            for i in range(j + 1):
                if sing:
                    r[i, j] = 0.0
                ii = ipvt[i]
                if ii > jj:
                    r[ii, jj] = r[i, j]
                if ii < jj:
                    r[jj, ii] = r[i, j]
            wa[jj] = r[j, j]

        # Symmetrize the covariance matrix in r
        for j in range(n):
            r[0 : j + 1, j] = r[j, 0 : j + 1]
            r[j, j] = wa[j]

        return r
