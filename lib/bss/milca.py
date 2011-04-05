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
import commands
from StringIO import StringIO

def milca(x,kneig=12,algo=2,harm=1,finetune=7):

    '''Mutual Information based Least Dependent Component Analysis
    References:
    H. Stogbauer, A. Kraskov, S. A. Astakhov, P.Grassberger, 
    Phys. Rev. E 70 (6)  066123, 2004
    A. Kraskov, H. Stogbauer, P. Grassberger,  
    Phys. Rev. E 69 (6) 066138, 2004 
    
    Parameters:
    -----------
    x : data MxN   M...channelnummer  N...sampling points  m<<n
    kneig: k nearest neighbor for MI algorithm
    algo: version of the MI estimator: 1=cubic  2=rectangular
    harm: fit the MI vs. angle curve with # harmonics
    finetune: Number 2^# of angles between 0 and pi/2 (default is 7)

    Output:
    -------
    most independent components under linear transformation and the 
    transformation matrix x
    
    '''

    N, M = x.shape
    Nb = 2 ** finetune
    
    print "MILCA parameters:"
    print "-----------------"
    print "k nearest neighbor for MI algorithm (kneig) = ", kneig
    print "Version of the MI estimator: 1=cubic  2=rectangular (algo) = ", algo
    print "Fit the MI vs. angle curve with # harmonics (harm) = ", harm
    print "Number 2^# of angles between 0 and pi/2 (finetune) = ",  finetune

    # save data for external Programm
    np.savetxt('zwspmilca.txt', x)

    # execute C Programm

    cmd = 'milca zwspmilca.txt ' + str(M) + ' ' + str(N) + ' ' + \
    str(kneig) + ' ' + str(algo - 1) + ' ' + str(Nb) + ' ' + str(harm)
    failure, output = commands.getstatusoutput(cmd)
#    print "MILCA output"
#    print output
    out = StringIO(output)
    out = np.loadtxt(out)
    return out
