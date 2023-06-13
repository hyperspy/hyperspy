# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

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
    Background component. An analytical model of the Bremsstrahlung signal generated in EDS spectra.
    This component is based on kramer's emission law (or other empirical emission model), 
    the mass absorption coefficients and the absorption in the sample (which depend of the composition of the pixel) and of the detector efficiency curve.
    
    Attributes
    ----------
    a: float
        Kramer's constant of the emission. A free parameters of the physical_background component that is fitted to the data.
    b: float
        A second coefficient ot the emission function. A free parameters of the physical_background component that is fitted to the data.
    mt : float
    	Mass depth. A free parameters of the physical_background component that is fitted to the data. 
    quanti: dictionnary
    	Contain the referenced variable quantification (none, result of CL quantification or an array).
        The function Wpercent() is called to estblish an array of weight percent with the same dimension
        than the model and a length which correspond to the number of elements in the metadata.
        
    """

    def __init__(self, E0, detector, quantification, emission_model, absorption_model,TOA,coating_thickness,phase_map,correct_for_backscatterring,standard, **kwargs):
        Component.__init__(self,['a','b','mt','quanti'])
        

        self._dic={}
        self.mt.value=100
        self.a.value=0.001
        self.b.value=0.0
        
        teta=TOA
        teta=np.deg2rad(teta)
        teta=(1/sin(teta))
        
        self._dic['Coating']=coating_thickness
        self._dic['E0']=E0
        self._dic['teta']=teta
        self._dic['Backscattering_correction'] = correct_for_backscatterring
        self._dic['std'] = standard
        self._dic['quanti'] = quantification
        self._dic['detector'] = detector
        self._dic['emission_model'] = emission_model
        self._dic['absorption_model'] = absorption_model
        self._dic['phase_map'] = phase_map
        self.quanti.value=1

        
        self.mt.free=True
        self.a.free=True
        self.b.free=True
        self.quanti.free=False
        
        self.isbackground=True

        # Boundaries
        self.mt.bmin=0.
        self.mt.bmax=500000
        self.a.bmin=1e-9
        self.a.bmax=1e9
        self.b.bmin=0
        self.b.bmax=1e9
        
        
    def initialize(self):
        """
        Initialize the Physical_background component by creating a quant map and saving some information of the component in a dictionnary.
                    
        """
        
        self.mt._create_array()
        self.a._create_array()
        self.b._create_array()
        
        self.quanti._number_of_elements=len(self.model._signal.metadata.Sample.elements)
        self.quanti._create_array()
        
        if len(self.model.axes_manager.shape)==1:
            self.quanti.value=Wpercent(self.model,self._dic['quanti'])
        elif len(self.model.axes_manager.shape)==2:
            self.quanti.map['values'][:] = Wpercent(self.model,self._dic['quanti'])
            self.quanti.map['is_set'][:] = True
            self.quanti.value=self.quanti.map['values'][0,:]
        else: 
            self.quanti.map['values'][:] = Wpercent(self.model,self._dic['quanti'])
            self.quanti.map['is_set'][:] = True
            self.quanti.value=self.quanti.map['values'][0,0,:]    

        self._dic['Window_absorption']=np.array(Windowabsorption(self.model,self._dic['detector']),dtype=np.float16)
        
        self._dic['Coating_absorption']=(np.exp(-Cabsorption(self.model)*1.3*self._dic['Coating']*10**-7*self._dic['teta']))# absorption by the coating layer (1.3 is the density)
        
        if self._dic['quanti']=='Mean'or self._dic['std']is True:
            self._dic['Mu']=Mucoef(self.model,self.quanti.value)
            Z=MeanZ(self.model,self.quanti.value)            
            self._dic['Z']=Z

        phasem=self._dic['phase_map']
        if phasem is not None:
            Mu=[]
            Z=[]
            for i in range (1,int(np.max(phasem)+1)):
                Mu.append(Mucoef(self.model,np.mean(self.quanti.map['values'][phasem==i],axis=0)))
                Z.append(MeanZ(self.model,np.mean(self.quanti.map['values'][phasem==i],axis=0)))     
            self._dic['Mu']=np.array(Mu,dtype=np.float16)
            self._dic['Z']=np.array(Z,dtype=np.float16)

        return {'Quant map and absorption correction parameters have been created'}
    
    
    def reinitialize(self,dic): # this function is necessary to reinitialize the quant map for loading the data.
        """
        Re-initialize the Physical_background component by re-creating a quant map and loading some information of the component from a dictionnary that was saved with the model (._dic).
        
        Parameters
        ----------
            dic: dictionary
                A dictionary that contain information about the energy, the dector and other argument passed in m.add_physical_background.
            
        """
        
        self._dic=dic
        self.mt._create_array()
        self.a._create_array()
        self.b._create_array()
        
        self.quanti._number_of_elements=len(self.model._signal.metadata.Sample.elements)
        self.quanti._create_array()
        
        self._dic['Window_absorption']=self._dic['Window_absorption']=np.array(Windowabsorption(self.model,self._dic['detector']),dtype=np.float16)
        
        self._dic['Coating_absorption']=(np.exp(-Cabsorption(self.model)*1.3*self._dic['Coating']*10**-7*self.teta.value))# absorption by the coating layer (1.3 is the density)
        
        ### the Mu parameter could be saved as a basesignal maybe and reloaded in the model if it change at each pixel. Try to think about it for the next version.
        
        return {'Quant map and absorption correction parameters have been recreated'}
           
    def function(self,x):
 
        mt=self.mt.value
        a=self.a.value
        b=self.b.value
        

        E0=self._dic['E0']
        cosec=self._dic['teta']

        phasem=self._dic['phase_map']
        if phasem is not None:
            index=self.model._signal.axes_manager.indices
            phaseN=phasem[int(index[1]),int(index[0])]
            Mu=self._dic['Mu'][int(phaseN-1)]
            Z=self._dic['Z'][int(phaseN-1)]
            Mu=np.array(Mu,dtype=np.float16)
            Mu=Mu[self.model.channel_switches]
        elif self._dic['quanti']=='Mean' or self._dic['std']is True:
            Mu=self._dic['Mu']
            Mu=np.array(Mu,dtype=np.float16)
            Mu=Mu[self.model.channel_switches]
            Z=self._dic['Z']
        else:
            Mu=Mucoef(self.model,self.quanti.value)
            Mu=np.array(Mu,dtype=np.float16)
            Z=MeanZ(self.model,self.quanti.value)
        
        Window_absorption=self._dic['Window_absorption']
        Window_absorption=Window_absorption[self.model.channel_switches]

        if self._dic['Coating']>0:
            coating=np.array(self._dic['Coating_absorption'],dtype=np.float16)
        else:
            coating=1

        if self._dic['emission_model'] == 'Kramer':
            np.seterr(divide = 'ignore',invalid='ignore')# deactivate the warnings for x value that are not calculated (<0.17)
            emission=(a*mt*((E0-x)/x))
            np.seterr(divide = 'warn',invalid='warn')
        
        if self._dic['emission_model'] == 'Small':
            M=0.00599*E0+1.05
            P=-0.0322*E0+5.80
            np.seterr(divide = 'ignore',invalid='ignore')
            emission=a*((np.exp(P)*((Z*((E0-x)/x))**M)))
            np.seterr(divide = 'warn',invalid='warn')

        if self._dic['emission_model'] == 'Lifshin_SEM':
            np.seterr(divide = 'ignore',invalid='ignore')
            emission=a*(((E0-x)/x))+(b*((E0-x)/x)**2)
            np.seterr(divide = 'warn',invalid='warn')

        if self._dic['emission_model'] == 'Lifshin':
            np.seterr(divide = 'ignore',invalid='ignore')
            emission=a*(mt*(((E0-x)/x))+(a/25*mt*4e-3*(Z**1.75*((E0-x)/x)**2)))
            np.seterr(divide = 'warn',invalid='warn')    

        if self._dic['emission_model'] == 'Castellano_SEM': 
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

        if self._dic['emission_model'] == 'Castellano_TEM':          
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
            
        
        if self._dic['Backscattering_correction'] is True :
            h=(1-np.exp(0.361*(x/E0)**2+0.288*(x/E0)-0.619))*10**-4
            j=(1-np.exp(0.153*(x/E0)**2+2.04*(x/E0)-2.17))*10**-2
            k=1.003+0.0407*(x/E0)
            Backscatter=h*Z**2-j*Z+k
        else:
            Backscatter=1.0
            
                    
        np.seterr(divide = 'ignore',invalid='ignore')
        
        absorption=((1-np.exp(-Mu*(mt*10**-7)*cosec))/Mu)#love and scott model. 
        METabsorption=np.exp(-Mu*(mt*10**-7)*cosec)#Cliff lorimer
        
        np.seterr(divide = 'ignore',invalid='ignore')
                     
        if self._dic['absorption_model'] == 'quadrilateral':
            f=np.where((x>0.17) & (x<(E0)),(emission*absorption*Window_absorption*coating*Backscatter),0)# keep the warnings for values x>0.17
            if not np.all(np.isfinite(f)): #avoid "residuals are not finite in the initial point"
                self.mt.store_current_value_in_array()
                return 1
            else:
                return f

        
        if self._dic['absorption_model'] == 'CL':# cliff lorimer
            f=np.where((x>0.17) & (x<(E0)),(emission*METabsorption*Window_absorption*coating*Backscatter),0)
            if not np.all(np.isfinite(f)): #avoid "residuals are not finite in the initial point"
                self.mt.store_current_value_in_array()
                return 1
            else:
                return f

        