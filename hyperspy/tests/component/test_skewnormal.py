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

from hyperspy.components1d import SkewNormal
from hyperspy.signals import Signal1D


def test_function(A=1.,x0=1.,scale=1.,shape=5.,noise=0.02):
    """ Test function for the skew normal distribution component.
    
    Creates a simulated noisy skew normal distribution based on the input 
    parameters and fits a skew normal component to this data.
    
    Parameters:
    -----------
        A : float
            Height parameter of the peak.
        x0 : float
            Location of the peak position.
        scale : float
            Width (sigma) parameter.
        shape: float 
            Skewness (asymmetry) parameter.
        noise: float
            A * noise determines the magnitude of the added gaussian noise.
    
    """
    # create skew normal signal and add noise
    g = SkewNormal(A=A,x0=x0,scale=scale,shape=shape)
    x = np.arange(x0-scale*3,x0+scale*3,step=0.01*scale)
    s = Signal1D(g.function(x))
    s.axes_manager.signal_axes[0].axis = x
    s.add_gaussian_noise(std=noise*A)
    # fit skew normal component to signal
    g2 = SkewNormal()
    m = s.create_model()
    m.append(g2)
    g2.x0.bmin=x0-scale*3 # prevent parameters to run away
    g2.x0.bmax=x0+scale*3
    g2.x0.bounded=True
    m.fit(bounded=True)
    m.print_current_values() # print out parameter values
    m.plot() # plot fit
    return m
