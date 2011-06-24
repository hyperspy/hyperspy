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

import commands
from StringIO import StringIO

import numpy as np
import matplotlib.pyplot as plt
    
import utils

def ica_tests(x,emb_dim = 1,emb_tau = 1,kneig = 5,algo = 2,kmax = 15):
    """Reliability test for any ICA output (linear transformation version!)
    Output: Dependency Matrix, Variability Matrix, and MI vs rotation angle
    plot (to check if the components are indeed the most independent one)

    Parameters
    x: input data mxn   m: channelnummer  n: sampling points  m<<n
    kneig:  k nearest neigbor for MI algorithm
    algo :  1=cubic  2=rectangular
    kmax :  Number of angles between 0 and pi/2 
    emb_dim:  embedding dimension (default is 1, no embedding)
    emb_tau:  time-delay, only relevant when emb_dim>1 (default is 1)
    
    """
    
    N,Nd = x.shape
    np.savetxt('zwsptests.txt', x)

    # execute C Programm
    cmd = 'ICAtests zwsptests.txt ' + str(Nd) + ' ' + str(N) + ' ' + \
    str(kneig) + ' ' + str(emb_tau) + ' ' + str(emb_dim) + ' ' + str(algo-1) \
    + ' ' + str(kmax)
    failure, output = commands.getstatusoutput(cmd)
    output = output.replace('\t\t', '\t').replace(' ', '')
    olist = output.split('\n\n')
    mivsangle = utils.str2num(olist[0])
    minat = utils.str2num(olist[1])
    varminat = utils.str2num(olist[2])
    
    # plot output
    count = 0
#    mivsangle = mivsangle.T

    if Nd==2:   
        print 'MI value: %f \n' % minat[0,1]
        print 'Variability of the MI under rotation: %f' % varminat(0,1)
        plt.figure()
        plt.plot(mivsangle)
        plt.xlim([1, kmax])
        plt.xticks(np.round(np.linspace(1,kmax,4)), (np.round(np.linspace(1,kmax,4))-1)/kmax*90)
        plt.xlabel('Rotation angle')
        plt.ylabel('Mutual Information')
    else:
        plt.figure()
        plt.subplot(1,2,1)
        maxMI = minat.max()
        if maxMI < 0.3:
            maxMI = 0.3  
        plt.imshow(minat,vmin = 0, vmax = maxMI, interpolation = 'nearest', cmap = plt.cm.gray)
        plt.title('Dependency Matrix')
        plt.colorbar()
        plt.subplot(1,2,2)
        minVar = varminat.min()
        if minVar < -0.015:
            print 'Input seems to be not the most independent representation under linear transformation!'
        plt.imshow(varminat, interpolation = 'nearest', cmap = plt.cm.gray)
        plt.title('Variability Matrix')
        plt.colorbar()
        
        plt.figure()
        
        ymin = mivsangle.min()
        ymax = mivsangle.max()
        print Nd
        for i in range(1,Nd+1):
            for j in range(1,Nd+1):
                if i<j:
                    plt.subplot(Nd,Nd,j+(i-1)*Nd)
                    plt.plot(mivsangle[count,:],'.-')
                    plt.xlim([1, kmax])
                    plt.xticks(np.round(np.linspace(1,kmax,4)), (np.round(np.linspace(1,kmax,4))-1)/kmax*90)
                    if count == 0:
                        plt.xlabel('rotation angle')
                        plt.ylabel('Mutual Information')
                    count=count+1;





