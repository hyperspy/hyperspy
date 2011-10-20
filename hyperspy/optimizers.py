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
import scipy.odr as odr
from scipy.optimize import leastsq,fmin, fmin_cg, fmin_ncg, fmin_bfgs, \
fmin_cobyla, fmin_l_bfgs_b, fmin_tnc, fmin_powell

from hyperspy.defaults_parser import preferences
from hyperspy.estimators import Estimators
from hyperspy.misc.mpfit.mpfit import mpfit

def vst(x, kind = 'ascombe'):
    if kind == 'ascombe':
        return 2*np.sqrt(x+3/8.)

class Optimizers(Estimators):
    """
    """

    def fit(self, fitter = None, method = 'ls',
    	    grad = False, weights = None, ext_bounding = False, ascombe = True,
    	    update_plot = False, bounded = False, **kwargs):
        """
        Fits the model to the experimental data using the fitter e
        The covariance matrix calculated by the 'leastsq' fitter is not always
        reliable
        """
        if fitter is None:
            fitter = preferences.Model.default_fitter
            print('Fitter: %s' % fitter)
        switch_aap = (update_plot != self.auto_update_plot)
        if switch_aap is True:
            self.set_auto_update_plot(update_plot)
        self.p_std = None
        self._set_p0()
        if ext_bounding:
            self._enable_ext_bounding()
        if grad is False :
            approx_grad = True
            jacobian = None
            odr_jacobian = None
            grad_ml = None
            grad_ls = None
        else :
            approx_grad = False
            jacobian = self._jacobian
            odr_jacobian = self._jacobian4odr
            grad_ml = self._gradient_ml
            grad_ls = self._gradient_ls
        if method == 'ml':
            weights = None
        if weights is True:
            if self.spectrum.variance is None:
                self.spectrum.estimate_variance()
            weights = 1. / np.sqrt(self.spectrum.variance.__getitem__(
            self.axes_manager._getitem_tuple)[self.channel_switches])
        elif weights is not None:
            weights = weights.__getitem__(self.axes_manager._getitem_tuple)[
            self.channel_switches]
        args = (self.spectrum()[self.channel_switches], 
        weights)
        
        # Least squares "dedicated" fitters
        if fitter == "leastsq":
            output = \
            leastsq(self._errfunc, self.p0[:], Dfun = jacobian,
            col_deriv=1, args = args, full_output = True, **kwargs)
            
            self.p0 = output[0]
            var_matrix = output[1]
            # In Scipy 0.7 sometimes the variance matrix is None (maybe a 
            # bug?) so...
            if var_matrix is not None:
                self.p_std = np.sqrt(np.diag(var_matrix))
            self.fit_output = output
        
        elif fitter == "odr":
            modelo = odr.Model(fcn = self._function4odr, 
            fjacb = odr_jacobian)
            mydata = odr.RealData(self.axis.axis[self.channel_switches],
            self.spectrum()[self.channel_switches],
            sx = None,
            sy = (1/weights if weights is not None else None))
            myodr = odr.ODR(mydata, modelo, beta0=self.p0[:])
            myoutput = myodr.run()
            result = myoutput.beta
            self.p_std = myoutput.sd_beta
            self.p0 = result
            self.fit_output = myoutput
            
        elif fitter == 'mpfit':
            autoderivative = 1
            if grad is True:
                autoderivative = 0

            if bounded is True:
                self.set_mpfit_parameters_info()
            elif bounded is False:
                self.mpfit_parinfo = None
            m = mpfit(self._errfunc4mpfit, self.p0[:], 
            parinfo = self.mpfit_parinfo, functkw= {
                'y': self.spectrum()[self.channel_switches], 
                'weights' :weights}, autoderivative = autoderivative, quiet = 1)
            self.p0 = m.params
            self.p_std = m.perror
            self.fit_output = m
            
        else:          
        # General optimizers (incluiding constrained ones(tnc,l_bfgs_b)
        # Least squares or maximum likelihood
            if method == 'ml':
                tominimize = self._poisson_likelihood_function
                fprime = grad_ml
            elif method == 'ls':
                tominimize = self._errfunc2
                fprime = grad_ls
                        
            # OPTIMIZERS
            
            # Simple (don't use gradient)
            if fitter == "fmin" :
                self.p0 = fmin(tominimize, self.p0, args = args, **kwargs)
            elif fitter == "powell" :
                self.p0 = fmin_powell(tominimize, self.p0, args = args, 
                **kwargs)
            
            # Make use of the gradient
            elif fitter == "cg" :
                self.p0 = fmin_cg(tominimize, self.p0, fprime = fprime,
                args= args, **kwargs)
            elif fitter == "ncg" :
                self.p0 = fmin_ncg(tominimize, self.p0, fprime = fprime,
                args = args, **kwargs)
            elif fitter == "bfgs" :
                self.p0 = fmin_bfgs(tominimize, self.p0, fprime = fprime,
                args = args, **kwargs)
            
            # Constrainded optimizers
            
            # Use gradient
            elif fitter == "tnc":
                if bounded is True:
                    self.set_boundaries()
                elif bounded is False:
                    self.self.free_parameters_boundaries = None
                self.p0 = fmin_tnc(tominimize, self.p0, fprime = fprime,
                args = args, bounds = self.free_parameters_boundaries, 
                approx_grad = approx_grad, **kwargs)[0]
            elif fitter == "l_bfgs_b":
                if bounded is True:
                    self.set_boundaries()
                elif bounded is False:
                    self.self.free_parameters_boundaries = None
                self.p0 = fmin_l_bfgs_b(tominimize, self.p0, fprime = fprime, 
                args =  args,  bounds = self.free_parameters_boundaries, 
                approx_grad = approx_grad, **kwargs)[0]
            else:
                print \
                """
                The %s optimizer is not available.

                Available optimizers:
                Unconstrained:
                --------------
                Only least Squares: leastsq and odr
                General: fmin, powell, cg, ncg, bfgs

                Cosntrained:
                ------------
                tnc and l_bfgs_b
                """ % fitter
                
        
        if np.iterable(self.p0) == 0:
            self.p0 = (self.p0,)
        self._charge_p0(p_std = self.p_std)
        self.set()
        if ext_bounding is True:
            self._disable_ext_bounding()
        if switch_aap is True:
            self.set_auto_update_plot(not update_plot)
            if not update_plot and self.spectrum._plot is not None:
                self.update_plot()
