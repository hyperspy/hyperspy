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

import numpy
import commands

def snica_b(X, A0, h0, T, M, Kn) :
    # [Y, A]
    #This is SNICA to perform blind decomposition of 
    #non-negative mixture matrix X into non-negative
    #least-dependent components Y by a constrained Monte Carlo search.
    #The estimate of the mixing matrix is given in A

    #Arguments:

    # X - input matrix of mixed signals (rows), or 
    # an estimate Y from the previous step
    #(if a multistage annealing is being performed);

    # A0 - either an identity matrix (at the first step),
    # or an estimate A from the previous step;

    # h0 - initial Monte Carlo step size (0<h0<1, of order 0.2);
    # T - Metropolis temperature parameter (of order of the mutual
    # information of pure sources)
    # M - the number of terminal Monte Carlo steps (stopping criterion)
    # Kn - the number of nearest neighbors used in the mutual information 
    # calculatioms

    #For typical values of the above parameters refer to the example
    # in test.m


    J,N=numpy.shape(X)


    Xdd=numpy.diff(X,1,1)

    #Savitzky-Golay smoothing differentiation, use instead 
    #of the above line in case of noisy signals 
    #(you may want to adjust PL and F to suit your signals)
    # PL=5; F=51;
    # [b,g]=sgolay(PL,F);
     
    # for j = 1:J
    #   for n = (F+1)/2:N-(F+1)/2
    #     Xdd(j,n-(F+1)/2+1)=2*g(:,3)'*X(j,n - (F+1)/2 + 1: n + (F+1)/2 - 1)';
    # end
    # end
     
    J, N1 = numpy.shape(Xdd)

#    X = numpy.transpose(X)
#    Xdd = numpy.transpose(Xdd)

    numpy.savetxt('mix', X.T)
    numpy.savetxt('mixd', Xdd.T)
    numpy.savetxt('a', A0) 

    # execute C Program (linux syntax)
    cmd = 'snica_b ' + str(J) + ' ' + str(N) + ' ' + str(N1) + ' ' + str(h0) + ' ' + str(T) + ' ' + str(M) + ' ' + str(Kn)
    failure, output = commands.getstatusoutput(cmd)
    Y = numpy.loadtxt('Y')
    A = numpy.loadtxt('A')
    return Y.T, A
