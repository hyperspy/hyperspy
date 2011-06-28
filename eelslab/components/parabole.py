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

from eelslab.component import Component


class Parabole(Component):
    """
    """

    def __init__(self, a=0, b=1.,c = 1., origin = 0):
        Component.__init__(self, ['a', 'b', 'c', 'origin'])
        self.name = 'Parabole'
        self.a.value, self.b.value, self.c.value, self.origin.value = \
        a, b, c, origin
        self.isbackground = False
        self.convolved = True
        self.a.grad = self.grad_a
        self.b.grad = self.grad_b
        self.origin.grad = self.grad_origin


    def function( self, x ) :
        """
        Given an one dimensional array x containing the energies at which
        you want to evaluate the background model, returns the background
        model for the current parameters.
        """
        return self.a.value + self.b.value * (x - self.origin.value)**2
    def grad_a(self, x):
        """
        Given an one dimensional array x containing the energies at which
        you want to evaluate the gradient of the background model,
        returns the gradient of parameter A for the current value of the
        parameters.
        """
        return  np.ones(len(x))
    def grad_b(self,x):
        """
        Given an one dimensional array x containing the energies at which
        you want to evaluate the gradient of the background model,
        returns the gradient of parameter sigma for the current value of
        the parameters.
        """
        return (x - self.origin.value)**2

    def grad_origin(self,x):
        return -self.b.value * (x - self.origin.value)
        
