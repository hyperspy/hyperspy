# -*- coding: utf-8 -*-
# Copyright © 2007 Francisco Javier de la Peña
#
# This file is part of EELSLab.
#
# EELSLab is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# EELSLab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EELSLab; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  
# USA

import numpy as np
import scipy.odr as odr
from scipy.optimize import leastsq,fmin, fmin_cg, fmin_ncg, fmin_bfgs, \
fmin_cobyla, fmin_l_bfgs_b, fmin_tnc, fmin_powell
from defaults_parser import defaults
from estimators import Estimators

def vst(x, kind = 'ascombe'):
    if kind == 'ascombe':
        return 2*np.sqrt(x+3/8.)

class Optimizers(Estimators):
    '''
    '''

    def fit(self, fitter = defaults.fitter, method = 'ls', grad = False, 
    weights = None, ext_bounding = False, ascombe = True, update_plot = False, 
    **kwargs):
        '''
        Fits the model to the experimental data using the fitter e
        The covariance matrix calculated by the 'leastsq' fitter is not always
        reliable
        '''
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
            print "weighted"
            if self.hl.variance is None:
                self.hl.estimate_variance()
            weights = 1. / np.sqrt(
            self.hl.variance[self.channel_switches, self.coordinates.ix, self.coordinates.iy])
        elif weights is not None:
            print "weighted"
            weights = weights[self.channel_switches, self.coordinates.ix, self.coordinates.iy]
        args = (self.hl.data_cube[self.channel_switches, self.coordinates.ix, self.coordinates.iy], 
        weights)
        
        # Least squares "dedicated" fitters
        if fitter == "leastsq":
            self.least_squares_fit_output[self.coordinates.ix][self.coordinates.iy] = \
            leastsq(self._errfunc, self.p0[:], Dfun = jacobian,
            col_deriv=1, args = args, full_output = True, **kwargs)
            self.p0 = self.least_squares_fit_output[self.coordinates.ix][self.coordinates.iy][0]
            var_matrix = self.least_squares_fit_output[self.coordinates.ix][self.coordinates.iy][1]
            # In Scipy 0.7 sometimes the variance matrix is None (maybe a 
            # bug?) so...
            if var_matrix is not None:
                self.p_std = np.sqrt(np.diag(var_matrix))
        
        elif fitter == "odr":
            modelo = odr.Model(fcn = self._function4odr, 
            fjacb = odr_jacobian)
            mydata = odr.RealData(self.hl.energy_axis[self.channel_switches],
            self.hl.data_cube[self.channel_switches,self.coordinates.ix,self.coordinates.iy],
            sx = None,
            sy = (1/weights if weights is not None else None))
            myodr = odr.ODR(mydata, modelo, beta0=self.p0[:])
            myoutput = myodr.run()
            result = myoutput.beta
            self.p_std = myoutput.sd_beta
            self.p0 = result
            std = True
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
            elif fitter == "tnc" :
                self.p0 = fmin_tnc(tominimize, self.p0, fprime = fprime,
                args = args, bounds = self.free_parameters_boundaries, 
                approx_grad = approx_grad, **kwargs)[0]
            elif fitter == "l_bfgs_b" :
                self.p0 = fmin_l_bfgs_b(tominimize, self.p0, fprime = fprime, 
                args =  args,  bounds = self.free_parameters_boundaries, 
                approx_grad = approx_grad, **kwargs)[0]
            else:
                print \
                '''
                The %s optimizer is not available.

                Available optimizers:
                Unconstrained:
                --------------
                Only least Squares: leastsq and odr
                General: fmin, powell, cg, ncg, bfgs

                Cosntrained:
                ------------
                tnc and l_bfgs_b
                ''' % fitter
                
        
        if np.iterable(self.p0) == 0:
            self.p0 = (self.p0,)
#        if self.p_std is None:
#            self.p_std = self.calculate_p_std(self.p0, method, *args)
        self._charge_p0(p_std = self.p_std)
        self.set()
        self.model_cube[self.channel_switches,self.coordinates.ix,self.coordinates.iy] = self.__call__(
        not self.convolved, onlyactive = True)
        if ext_bounding:
            self._disable_ext_bounding()
        if switch_aap is True:
            self.set_auto_update_plot(not update_plot)
            if not update_plot and self.hl.hse is not None:
                for line in self.hl.hse.spectrum_plot.left_ax_lines:
                    line.update()
