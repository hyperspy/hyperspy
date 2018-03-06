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
from hyperspy.misc.material import _mass_absorption_mixture as mass_absorption_mixture
from hyperspy.misc import elements as element_db
from hyperspy.external.mpfit import mpfit
from hyperspy.misc.material import atomic_to_weight
from hyperspy.misc.eds.detector_efficiency import detector_efficiency
from scipy.interpolate import interp1d

def Wpercent(model,E0,quantification):
    
    if quantification is None :                 #if quantification is None, this function calculate an  approximation of a quantification based on peaks ratio 
        model.signal.set_lines([])
        intensity=model.signal.get_lines_intensity(only_one=True)
        if len(np.shape(intensity[0]))>1 and np.shape(intensity[0])[1]>1 : #for 3D Data
            u=np.ones([np.shape(intensity[0])[0],np.shape(intensity[0])[1], len(model._signal.metadata.Sample.elements)] )
            for i in range (0,len(intensity)):
                if "Ka" in intensity [i].metadata.General.title :
                    u[:,:,i] =intensity[i].data
                elif "La" in intensity [i].metadata.General.title:
                    u[:,:,i] =intensity[i].data*2.5
                elif "La" in intensity [i].metadata.General.title:
                    u[:,:,i] =intensity[i].data*2.8
                weight=np.ones([np.shape(intensity[0])[0],np.shape(intensity[0])[1], len(model._signal.metadata.Sample.elements)] )
                for i in range (0,len(model._signal.metadata.Sample.elements)):
                    weight[:,:,i] =(u[:,:,i]/(np.sum(u,axis=-1))) *100          	    

        elif len(np.shape(intensity[0]))==1 and np.shape(intensity[0])[0]>1 : #for 2D Data
            u=np.ones([np.shape(intensity[0])[0], len(model._signal.metadata.Sample.elements)] )
            for i in range (0,len(intensity)):
                if "Ka" in intensity [i].metadata.General.title :
                    u[:,i] =intensity[i].data
                elif "La" in intensity [i].metadata.General.title:
                    u[:,i] =intensity[i].data*2.5
                elif "La" in intensity [i].metadata.General.title:
                    u[:,i] =intensity[i].data*2.8
            weight=np.ones([np.shape(intensity[0])[0], len(model._signal.metadata.Sample.elements)] )
            for i in range (0,len(model._signal.metadata.Sample.elements)):
                weight[:,i] =(u[:,i]/(np.sum(u,axis=-1))) *100

        elif len(np.shape(intensity[0]))>1 and np.shape(intensity[0])[1]==1 : #for 2D Data
            u=np.ones([np.shape(intensity[0])[0],np.shape(intensity[0])[1], len(model._signal.metadata.Sample.elements)] )
            for i in range (0,len(intensity)):
                if "Ka" in intensity [i].metadata.General.title :
                    u[:,:,i] =intensity[i].data
                elif "La" in intensity [i].metadata.General.title:
                    u[:,:,i] =intensity[i].data*2.5
                elif "La" in intensity [i].metadata.General.title:
                    u[:,:,i] =intensity[i].data*2.8
                weight=np.ones([np.shape(intensity[0])[0],np.shape(intensity[0])[1], len(model._signal.metadata.Sample.elements)] )
                for i in range (0,len(model._signal.metadata.Sample.elements)):
                    weight[:,:,i] =(u[:,:,i]/(np.sum(u,axis=-1))) *100     
        
        else:
            u=np.ones([len(model._signal.metadata.Sample.elements)] ) #for 1D 
            for i in range (0,len(intensity)):
                if "Ka" in intensity [i].metadata.General.title :
                    u[i] =intensity[i].data
                elif "La" in intensity [i].metadata.General.title:
                    u[i] =intensity[i].data*2.5
                elif "La" in intensity [i].metadata.General.title:
                    u[i] =intensity[i].data*2.8        
            weight=np.ones([len(model._signal.metadata.Sample.elements)] )
            t=u.sum() 
            for i in range (0,len(u)):
                weight[i] =u[i] /t*100
    elif type(quantification) is np.ndarray: # if quantification is already an array of weight percent, directly keep the array
        weight=quantification

    else:
        result=quantification # if quantification is the result of the hyperspy quantification function. This function convert the result in an array with the same navigation shape than the model and a length equal to the number of elements 
        if 'atomic percent' in result[0].metadata.General.title:
            result=atomic_to_weight(result)
        else:
            result=result
            
        if len(np.shape(result[0]))>1 and np.shape(result[0])[1]>1 : # for 3D
            weight=np.ones([np.shape(result[0])[0],np.shape(result[0])[1], len(model._signal.metadata.Sample.elements)] )
            for i in range (0,len(result)):
                weight[:,:,i] =result[i].data        
		    
        elif len(np.shape(result[0]))==1 and np.shape(result[0])[0]>1 : #for 2D
            weight=np.ones([np.shape(result[0])[0], len(model._signal.metadata.Sample.elements)] )
            for i in range (0,len(result)):
               weight[:,i] =result[i].data       
		    
        else: #for 1D
            weight=np.ones([len(model._signal.metadata.Sample.elements)] )
            for i in range (0,len(result)):            
                weight[i] =result[i].data
		    

    return weight



def Mucoef(model,quanti): # this function calculate the absorption coefficient for all energy. This, correspond to the Mu parameter in the function
    weight=quanti
    t=(np.linspace(model._signal.axes_manager[-1].offset,model._signal.axes_manager[-1].size*model._signal.axes_manager[-1].scale,model._signal.axes_manager[-1].size/5))
    
    Ac=mass_absorption_mixture(elements=model._signal.metadata.Sample.elements ,weight_percent=weight, energies=t)    
    b=(model._signal.axes_manager.signal_axes[-1].axis)-0.035
    Ac=np.interp(b,t,Ac)
    
    return Ac


def Windowabsorption(model,detector): # this function interpolate the detector efficiency based on dictionnary (create from personnal data) and the signal length. This correspond to the Window parameter  
 
    a=np.array(detector_efficiency[detector])
    b=(model._signal.axes_manager.signal_axes[-1].axis)-0.035
    x =a[:,0]
    y = a[:,1]
    Accc=np.interp(b, x, y)
    return Accc
        
class Physical_background(Component):

    """
    Background component based on kramer's law and absorption coefficients
    Attributes
    ----------
    a : float
    b : float
    E0 : int 
    """

    def __init__(self, E0, detector, quantification, coefficients=1, Window=0, quanti=0):
        Component.__init__(self,['coefficients','E0','Window','quanti'])

        self.coefficients._number_of_elements = 2
        self.coefficients.value = np.ones(2)
        
        self.E0.value=E0
        
        self._whitelist['quanti'] = quantification
        self._whitelist['detector'] = detector       
        self.quanti.value=quanti

        self.Window.value=Window

        self.E0.free=False
        self.coefficients.free=True
        self.Window.free=False
        self.quanti.free=False
        
        self.isbackground=True

        # Boundaries
        self.coefficients.bmin=0
        self.coefficients.bmax=None

    def initialize(self): # this function is necessary to initialyze the quant map

        E0=self.E0.value

        self.quanti._number_of_elements=len(self.model._signal.metadata.Sample.elements)
        self.quanti._create_array()

        if len(self.model.axes_manager.shape)==1:
            self.quanti.value=Wpercent(self.model,E0,self._whitelist['quanti'])
        else: 
            self.quanti.map['values'][:] = Wpercent(self.model,E0,self._whitelist['quanti'])
            self.quanti.map['is_set'][:] = True
        
        self.Window._number_of_elements=len(self.model._signal.axes_manager.signal_axes[-1].axis)
        self.Window._create_array()
        self.Window.value=Windowabsorption(self.model,self._whitelist['detector'])
        
        return {'Quant map has been created'}
        
    def function(self,x):
 
        b=self.coefficients.value[0]
        a=self.coefficients.value[1]
        
        E0=self.E0.value

        Mu=Mucoef(self.model,self.quanti.value)
        
        Mu=np.array(Mu,dtype=float)
        Mu=Mu[self.model.channel_switches]
        
        Window=np.array(self.Window.value,dtype=float)
        Window=Window[self.model.channel_switches]
	
        emission=(a*100*((E0-x)/x)) #kramer's law 
	absorption=((1-np.exp(-2*Mu*b*10**-5 ))/((2*Mu*b*10**-5))) #love and scott model
	
        return np.where((x>0.17) & (x<(E0)),(emission*absorption*Window),0) #implement choice for coating correction
