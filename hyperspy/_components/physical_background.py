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
from math import sin
from hyperspy.component import Component


from hyperspy.misc.eds.utils import Wpercent 
from hyperspy.misc.eds.utils import Cabsorption
from hyperspy.misc.eds.utils import Windowabsorption
from hyperspy.misc.eds.utils import Mucoef
from hyperspy.misc.eds.utils import MeanZ



        
class Physical_background(Component):

    """
    Background component based on kramer's law and absorption coefficients
    Attributes
    ----------
    coefficients : float (length = 2) 
    	The only two free parameters of this component. If the function fix_background is used those two coefficients are fixed
    E0 : int
    	The beam energy of the acquisition
    Window : int 
    	Contain the signal of the detector efficiency (calculated thanks to the function Windowabsorption())
    quanti: dictionnary
    	Contain the referenced variable quantification (none, result of CL quantification or an array) 
	This dictionnary is call in the function Wpercent() to calculate an array of weight percent with the same dimension than the model and a length which correspond to the number of elements filled in the metadata
    """

    def __init__(self, E0, detector, quantification, emission_model, absorption_model,TOA,coating_thickness,phase_map,correct_for_backscatterring,standard):
        Component.__init__(self,['mt','a','b','E0','quanti','teta','coating_thickness'])

        self.mt.value=100
        self.a.value=0.001
        self.b.value=0
        
        self.E0.value=E0
        self.teta.value=TOA
        self.teta.value=np.deg2rad(self.teta.value)
        self.teta.value=(1/sin(self.teta.value))
        
        self.coating_thickness.value=coating_thickness
        
        self._whitelist['E0']=E0
        self._whitelist['teta']=self.teta.value
        self._whitelist['Backscattering_correction'] = correct_for_backscatterring
        self._whitelist['std'] = standard
        self._whitelist['quanti'] = quantification
        self._whitelist['detector'] = detector
        self._whitelist['emission_model'] = emission_model
        self._whitelist['absorption_model'] = absorption_model
        self._whitelist['carto'] = phase_map
        self.quanti.value=1

        
        self.mt.free=True
        self.a.free=True
        self.b.free=True
        self.E0.free=False
        self.teta.free=False
        self.coating_thickness.free=False
        self.quanti.free=False
        
        self.isbackground=True

        # Boundaries
        self.mt.bmin=0.
        self.mt.bmax=500000
        self.a.bmin=1e-9
        self.a.bmax=1e9
        self.b.bmin=0
        self.b.bmax=1e9
        
    def initialize(self): # this function is necessary to initialize the quant map

        E0=self.E0.value
        teta=self.teta.value
        coating_thickness=self.coating_thickness.value
        
        self.mt._create_array()
        self.a._create_array()
        self.b._create_array()
        
        self.quanti._number_of_elements=len(self.model._signal.metadata.Sample.elements)
        self.quanti._create_array()
        
        if len(self.model.axes_manager.shape)==1:
            self.quanti.value=Wpercent(self.model,self._whitelist['quanti'])
        elif len(self.model.axes_manager.shape)==2:
            self.quanti.map['values'][:] = Wpercent(self.model,self._whitelist['quanti'])
            self.quanti.map['is_set'][:] = True
            self.quanti.value=self.quanti.map['values'][0,:]
        else: 
            self.quanti.map['values'][:] = Wpercent(self.model,self._whitelist['quanti'])
            self.quanti.map['is_set'][:] = True
            self.quanti.value=self.quanti.map['values'][0,0,:]    

        self._whitelist['Window_absorption']=np.array(Windowabsorption(self.model,self._whitelist['detector']),dtype=np.float16)
        
        if self.coating_thickness.value>0:
            self._whitelist['Coating_absorption']=1
        else: 
            self._whitelist['Coating_absorption']=(np.exp(-Cabsorption(self.model)*1.3*self.coating_thickness.value*10**-7*teta))# absorption by the coating layer (1.3 is the density)

        if self._whitelist['quanti']=='Mean'or self._whitelist['std']is True:
            Mu=Mucoef(self.model,self.quanti.value)
            Z=MeanZ(self.model,self.quanti.value)            
            self._whitelist['Mu']=Mu
            self._whitelist['Z']=Z

        carto=self._whitelist['carto']
        if carto is not None:
            Mu=[]
            Z=[]
            for i in range (1,int(np.max(carto)+1)):
                Mu.append(Mucoef(self.model,np.mean(self.quanti.map['values'][carto==i],axis=0)))
                Z.append(MeanZ(self.model,np.mean(self.quanti.map['values'][carto==i],axis=0)))     
            self._whitelist['Mu']=np.array(Mu,dtype=np.float16)
            self._whitelist['Z']=np.array(Z,dtype=np.float16)

        return {'Quant map and absorption correction parameters have been created'}
        
    def function(self,x):
 
        mt=self.mt.value
        a=self.a.value
        b=self.b.value

        E0=self._whitelist['E0']# allow to keep the parameter fixed even with the free_background function
        cosec=self._whitelist['teta']# allow to keep the parameter fixed even with the free_background function

        carto=self._whitelist['carto']
        if carto is not None:
            index=self.model._signal.axes_manager.indices
            phaseN=carto[int(index[1]),int(index[0])]
            Mu=self._whitelist['Mu'][int(phaseN-1)]
            Z=self._whitelist['Z'][int(phaseN-1)]
            Mu=np.array(Mu,dtype=float)
            Mu=Mu[self.model.channel_switches]
        elif self._whitelist['quanti']=='Mean' or self._whitelist['std']is True:
            Mu=self._whitelist['Mu']
            Mu=np.array(Mu,dtype=float)
            Mu=Mu[self.model.channel_switches]
            Z=self._whitelist['Z']
        else:
            Mu=Mucoef(self.model,self.quanti.value)
            Mu=np.array(Mu,dtype=float)
            Z=MeanZ(self.model,self.quanti.value)
        
        Window=self._whitelist['Window_absorption']
        Window=Window[self.model.channel_switches]

        coating=np.array(self._whitelist['Coating_absorption'],dtype=float)
        coating=coating[self.model.channel_switches]

        if self._whitelist['emission_model'] == 'Kramer':
            np.seterr(divide = 'ignore',invalid='ignore')# deactivate the warnings for x value that are not calculated (<0.17)
            emission=(a*mt*((E0-x)/x))
            np.seterr(divide = 'warn',invalid='warn')
        
        if self._whitelist['emission_model'] == 'Small':
            M=0.00599*E0+1.05
            P=-0.0322*E0+5.80
            np.seterr(divide = 'ignore',invalid='ignore')
            emission=a*((np.exp(P)*((Z*((E0-x)/x))**M)))
            np.seterr(divide = 'warn',invalid='warn')

        if self._whitelist['emission_model'] == 'Lifshin_SEM':
            np.seterr(divide = 'ignore',invalid='ignore')
            emission=a*(((E0-x)/x))+(b*((E0-x)/x)**2)
            np.seterr(divide = 'warn',invalid='warn')

        if self._whitelist['emission_model'] == 'Lifshin':
            np.seterr(divide = 'ignore',invalid='ignore')
            emission=a*(mt*(((E0-x)/x))+(a/25*mt*4e-3*(Z**1.75*((E0-x)/x)**2)))
            np.seterr(divide = 'warn',invalid='warn')    

        if self._whitelist['emission_model'] == 'Castellano_SEM': 
            a1 = 68.52809192351341
            a2 = 254.15693461075367
            a3 = 29.789319335480027
            a4 = 1.7663705750525933
            a5 = 4.158196337627563
            a6 = 23.75886334576287
            a7 = 1.58392121218387
            np.seterr(divide = 'ignore',invalid='ignore')
            emission=a*((Z**(1/2)*((E0-x)/x))*(-a1-a2*x+a3*np.log(Z)+(a4*E0**a5)/Z)*(1+(-a6+a7*E0)*(Z/x)))
            np.seterr(divide = 'warn',invalid='warn')

        if self._whitelist['emission_model'] == 'Castellano_TEM':          
            a1 = -555.40679202773
            a2 = 0.00010152130164852309
            a3 = 134.4405336236044
            a4 = 3150.427300886565
            a5 = 0.007869434977170494
            a6 = 399.2369203698975
            a7 = -1.330745199140076
            np.seterr(divide = 'ignore',invalid='ignore')
            emission=(a*((Z**(1/2))*((E0-x)/x)))*(a1+a2*x+a3*np.log(Z)+(a4*E0**a5)/Z)*(1+(a6+a7*E0)*(Z/x))
            np.seterr(divide = 'warn',invalid='warn')

        np.seterr(divide = 'ignore',invalid='ignore')
        absorption=((1-np.exp(-2*Mu*(mt*1e-7)*cosec))/(2*Mu*(mt*1e-7)*cosec))#love and scott model.
        #absorption=((1-np.exp(-Mu*(mt*10**-7)*cosec))/Mu)#love and scott model. 
        METabsorption=np.exp(-Mu*(mt*10**-7)*cosec)#Cliff lorimer
        np.seterr(divide = 'ignore',invalid='ignore')
        
        if self._whitelist['Backscattering_correction'] is True :
            h=(1-np.exp(0.361*(x/E0)**2+0.288*(x/E0)-0.619))*10**-4
            j=(1-np.exp(0.153*(x/E0)**2+2.04*(x/E0)-2.17))*10**-2
            k=1.003+0.0407*(x/E0)
            Backscatter=h*Z**2-j*Z+k
        else:
            Backscatter=1

            
        if self._whitelist['absorption_model'] == 'quadrilateral':
            f=np.where((x>0.2) & (x<(E0)),(emission*METabsorption*Window*coating*Backscatter),0)# keep the warnings for values x>0.17
            if not np.all(np.isfinite(f)): #avoid "residuals are not finite in the initial point"
                self.mt.store_current_value_in_array()
                return 1
            else:
                return f

        
        if self._whitelist['absorption_model'] == 'CL':# cliff lorimer
            f=np.where((x>0.2) & (x<(E0)),(emission*METabsorption*Window*coating*Backscatter),0)
            if not np.all(np.isfinite(f)): #avoid "residuals are not finite in the initial point"
                self.mt.store_current_value_in_array()
                return 1
            else:
                return f