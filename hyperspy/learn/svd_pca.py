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


import scipy.linalg
try:
    from scikits.learn.utils.extmath import fast_svd
    sklearn = True
except:
    sklearn = False

from hyperspy import messages

def pca(data, fast = False, output_dimension = None):
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
    if fast is True and sklearn is True:
        if output_dimension is None:
            messages.warning_exit('When using fast_svd it is necessary to '
                                  'define the output_dimension')
        u, S, PC = fast_svd(Y, output_dimension, q = 3)
    else:
        u, S, PC = scipy.linalg.svd(Y, full_matrices = False)
    
    v = PC.T
    V = S ** 2
    return v,V
