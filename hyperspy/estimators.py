# -*- coding: utf-8 -*-
# Copyright 2007-2011 The Hyperspy developers
#
# This file is part of  Hyperspy.
#
#  Hyperspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  Hyperspy.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import math

import scipy.optimize
import numpy.linalg

_epsilon = (np.finfo(float).eps)**(1/4.)

def approx_fprime_k(xk,f,k,*args):
    f0 = f(*((xk,)+args))
    ei = np.zeros((len(xk),), float)
    ei[k] = _epsilon
    return (f(*((xk+ei,)+args)) - f0) / _epsilon

def approx_hessian(xk,f,*args):
    num_par = len(xk)
    hess = np.zeros((num_par, num_par))
    for k in xrange(num_par):
        for l in xrange(num_par):
            hess[k,l] = approx_fprime_k(xk, approx_fprime_k,k,f,l,*args)
    return hess  

class Estimators:
    """
    """
    def calculate_p_std(self, p0, method, *args):
        print "Estimating the standard deviation"
        f = self._poisson_likelihood_function if method == 'ml' \
        else self._errfunc2
        hess = approx_hessian(p0,f,*args)
        ihess = np.linalg.inv(hess)
        p_std = np.sqrt(1./np.diag(ihess))
        return p_std

    def _poisson_likelihood_function(self,param,y, weights = None):
        """Returns the likelihood function of the model for the given
        data and parameters
        """
        mf = self._model_function(param)
        return -(y*np.log(mf) - mf).sum()

    def _gradient_ml(self,param, y, weights = None):
        mf = self._model_function(param)
        return -(self._jacobian(param, y)*(y/mf - 1)).sum(1)


    def _errfunc(self,param, y, weights = None):
        errfunc = self._model_function(param) - y
        if weights is None:
            return errfunc
        else:
            return errfunc * weights
    def _errfunc2(self,param, y, weights = None):
        if weights is None:
            return ((self._errfunc(param, y))**2).sum()
        else:
            return ((weights * self._errfunc(param, y))**2).sum()

    def _gradient_ls(self,param, y, weights = None):
        gls =(2*self._errfunc(param, y, weights) * 
        self._jacobian(param, y)).sum(1)
        return gls
        
    def _errfunc4mpfit(self, p, fjac = None, x = None, y = None, weights = None):
        if fjac is None:
            errfunc = self._model_function(p) - y
            if weights is not None:
                errfunc *= weights
            jacobian = None
            status = 0
            return [status, errfunc]
        else:
            return [0, self._jacobian(p,y).T]

            
