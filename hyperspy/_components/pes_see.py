# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
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

from hyperspy.component import Component
from .gaussian import Gaussian

sqrt2pi = np.sqrt(2 * np.pi)


class SEE(Component):

    """Secondary electron emission component for Photoemission Spectroscopy

    Attributes
    ----------
    A : float
    Phi : float
    B : float
    sigma : float
        Resolution parameter.

    """

    def __init__(self, A=1., Phi=1., B=0., sigma=0):
        Component.__init__(self, ('A', 'Phi', 'B', 'sigma'))
        self.A.value, self.Phi.value, self.B.value, self.sigma.value = \
            A, Phi, B, sigma

        self._position = self.Phi
        # Boundaries
        self.A.bmin = 0.
        self.A.bmax = None

        self.convolved = True

        # Gradients
        self.A.grad = self.grad_A
        self.Phi.grad = self.grad_Phi
        self.B.grad = self.grad_B
        self.sigma.grad = self.grad_sigma

        # Resolution functions
        self.gaussian = Gaussian()
        self.gaussian.centre.free, self.gaussian.A.free = False, False
        self.gaussian.sigma.free = True
        self.gaussian.A.value = 1.

        # Linearity
        self.A._is_linear = True

    def __repr__(self):
        return 'SEE'

    def function(self, x, multi=False):
        if multi:
            nav_shape = self.A.map['values'].shape
            A = self.A.map['values'][...,None]
            B = self.B.map['values'][...,None]
            Phi = self.Phi.map['values'][...,None]
            Sigma = self.sigma.map['values']

            if np.all(Sigma):
                # Create temporary model to enable para.map['values']
                # with correct shape
                temp_model = self.model.signal.create_model()
                temp_model.append(self.gaussian)

                convolved = np.zeros(nav_shape + x.shape)
                self.gaussian.A.map['values'] = self.gaussian.A.value
                self.gaussian.sigma.map['values'] = Sigma
                self.gaussian.centre.map['values'] = (x[-1] + x[0]) / 2
                G = self.gaussian.function(x, multi=True)
                for index in np.ndindex(nav_shape):
                    convolved[index] = np.convolve(
                        G[index], np.where(x > Phi[index], 
                        A[index]*(x-Phi[index])/(x-Phi[index]+B[index]) ** 4, 0),
                    'same')
                del temp_model
                return convolved
            else:
                return np.where(x > Phi, A*(x-Phi)/(x-Phi+B) ** 4, 0)
        else:
            A = self.A.value
            B = self.B.value
            Phi = self.Phi.value
            Sigma = self.sigma.value

            if Sigma:
                self.gaussian.sigma.value = Sigma
                self.gaussian.centre.value = (x[-1] + x[0]) / 2
                return np.convolve(
                    self.gaussian.function(x),
                    np.where(x > Phi, A*(x-Phi)/(x-Phi+B) ** 4, 0),
                    'same')
            else:
                return np.where(x > Phi, A*(x-Phi)/(x-Phi+B) ** 4, 0)

    def grad_A(self, x):
        """
        """
        if self.sigma.value:
            self.gaussian.sigma.value = self.sigma.value
            self.gaussian.centre.value = (x[-1] + x[0]) / 2
            return np.convolve(
                self.gaussian.function(x),
                np.where(
                    x > self.Phi.value,
                    (x - self.Phi.value) /
                    (x - self.Phi.value + self.B.value) ** 4, 0),
                'same')
        else:
            return np.where(x > self.Phi.value, (x - self.Phi.value) /
                            (x - self.Phi.value + self.B.value) ** 4, 0)

    def grad_sigma(self, x):
        """
        """
        self.gaussian.sigma.value = self.sigma.value
        self.gaussian.centre.value = (x[-1] + x[0]) / 2
        return np.convolve(
            self.gaussian.grad_sigma(x),
            np.where(
                x > self.Phi.value,
                self.A.value * (x - self.Phi.value) /
                (x - self.Phi.value + self.B.value) ** 4, 0),
            'same')

    def grad_Phi(self, x):
        """
        """
        if self.sigma.value:
            self.gaussian.sigma.value = self.sigma.value
            self.gaussian.centre.value = (x[-1] + x[0]) / 2
            return np.convolve(
                self.gaussian.function(x),
                np.where(
                    x > self.Phi.value,
                    (4 * (x - self.Phi.value) * self.A.value) /
                    (self.B.value + x - self.Phi.value) ** 5 -
                    self.A.value / (self.B.value + x - self.Phi.value) ** 4,
                    0),
                'same')
        else:
            return np.where(
                x > self.Phi.value,
                (4 * (x - self.Phi.value) * self.A.value) /
                (self.B.value + x - self.Phi.value) ** 5 -
                self.A.value / (self.B.value + x - self.Phi.value) ** 4, 0)

    def grad_B(self, x):
        if self.sigma.value:
            self.gaussian.sigma.value = self.sigma.value
            self.gaussian.centre.value = (x[-1] + x[0]) / 2
            return np.convolve(
                self.gaussian.function(x),
                np.where(
                    x > self.Phi.value,
                    -(4 * (x - self.Phi.value) * self.A.value) /
                    (self.B.value + x - self.Phi.value) ** 5, 0),
                'same')
        else:
            return np.where(
                x > self.Phi.value,
                -(4 * (x - self.Phi.value) * self.A.value) /
                (self.B.value + x - self.Phi.value) ** 5, 0)
