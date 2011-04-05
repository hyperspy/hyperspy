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

def pca(data):
    """Perform PCA using SVD.
    data - MxN matrix of input data
    (M dimensions, N trials)
    signals - MxN matrix of projected data
    PC - each column is a PC
    V - Mx1 matrix of variances
    """
    print "Performing PCA with a SVD based algorithm"
    N, M = data.shape
    Y = data
    u, S, PC = np.linalg.svd(Y, full_matrices = False)
    v = PC.T
    V = S ** 2
    return v,V
