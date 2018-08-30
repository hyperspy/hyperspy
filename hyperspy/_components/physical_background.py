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
    """
    Return an array of weight percent for each elements

    Parameters
    ----------
    model: EDS model
    E0: int
            The Beam energy
    quantification: None or a list or an array
            if quantification is None, this function calculate an  approximation of a quantification based on peaks ratio thanks to the function s.get_lines_intensity(). 
            if quantification is the result of the hyperspy quantification function. This function only convert the result in an array with the same navigation shape than the model and a length equal to the number of elements 
            if quantification is already an array of weight percent, directly keep the array
    """
 	
    if quantification is None :                 
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
    elif type(quantification) is np.ndarray: 
        weight=quantification

    else:
        result=quantification
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
    """
    Calculate the mass absorption coefficient for all energy of the model axis for each pixel. Need the weigth percent array defined by the Wpercent function
    Return the Mu parameter as a signal (all energy) with same number of elements than the model
    This parameter is calculated at each iteration during the fit
    Parameters
    ----------
    model: EDS model
    quanti: Array
            Must contain an array of weight percent for each elements
            This array is automaticaly created through the Wpercent function
    """	
    weight=quanti
    
    if np.sum(quanti)==0:
        raise ValueError("The quantification cannot be nul, but an an array with all weight percents set to 0 have been provided" )
    else: 
        t=(np.linspace(model._signal.axes_manager[-1].offset,model._signal.axes_manager[-1].size*model._signal.axes_manager[-1].scale,model._signal.axes_manager[-1].size))
        t=t[model.channel_switches]
        u=t[0::5]
        Ac=mass_absorption_mixture(elements=model._signal.metadata.Sample.elements ,weight_percent=weight, energies=u)    
        b=t
        Ac=np.interp(b,u,Ac) # Interpolation allows to gain some time
    
    return Ac

def Cabsorption(model): # this function calculate the absorption coefficient for all energy. This, correspond to the MuC parameter in the function
    """
    Calculate the mass absorption coefficient due to the coating layer for all energy 
    Parameters
    ----------
    model: EDS model

    """	

    t=(np.linspace(model._signal.axes_manager[-1].offset,model._signal.axes_manager[-1].size*model._signal.axes_manager[-1].scale,model._signal.axes_manager[-1].size))
    
    Acc=mass_absorption_mixture(elements='C' ,weight_percent=[100], energies=t)    
    #b=(model._signal.axes_manager.signal_axes[-1].axis)
    #Acc=np.interp(b,t,Acc) # Interpolation allows to gain some time
    
    return Acc

def Windowabsorption(model,detector): 
    """
    Return the detector efficiency as a signal based on a dictionnary (create from personnal data) and the signal length. This correspond to the Window parameter of the physical background class  
    To obtain the same signal length compare to the model, data are interpolated

    Parameters
    ----------
    model: EDS model
    detector: str or array
            The Acquisition detector which correspond to the Dataset
            String can be 'Polymer_C' / 'Super_X' / '12µm_BE' / '25µm_BE' / '100µm_BE' / 'Polymer_C2' / 'Polymer_C3' 
            Data are contain in a dictionnary in hyperspy repository
            
            An array with values of detector efficiency can be used if personnal data are needed 
    """	
    if type(detector) is str:
        a=np.array(detector_efficiency[detector])
        b=(model._signal.axes_manager.signal_axes[-1].axis)
        x =a[:,0]
        y = a[:,1]
        Accc=np.interp(b, x, y)
    else :
        a=detector
        b=(model._signal.axes_manager.signal_axes[-1].axis)-0.04
        x =a[:,0]
        y = a[:,1]
        Accc=np.interp(b, x, y)
    return Accc
        
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

    def __init__(self, E0, detector, quantification, absorption_model,TOA ,coating_thickness, coefficients=1, Window=0, quanti=0, MuC=0):
        Component.__init__(self,['coefficients','E0','Window','MuC','quanti','teta','coating_thickness'])

        self.coefficients._number_of_elements = 2
        self.coefficients.value = np.ones(2)
        
        self.E0.value=E0
        self.teta.value=TOA
        self.coating_thickness.value=coating_thickness
        
        self._whitelist['quanti'] = quantification
        self._whitelist['detector'] = detector
        self._whitelist['absorption_model'] = absorption_model
        self.quanti.value=quanti

        self.Window.value=Window
        self.MuC.value=MuC

        self.E0.free=False
        self.coefficients.free=True
        self.Window.free=False
        self.MuC.free=False
        self.teta.free=False
        self.coating_thickness.free=False
        self.quanti.free=False
        
        self.isbackground=True

        # Boundaries
        self.coefficients.bmin=0
        self.coefficients.bmax=None

    def initialize(self): # this function is necessary to initialize the quant map

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

        self.MuC._number_of_elements=len(self.model._signal.axes_manager.signal_axes[-1].axis)
        self.MuC._create_array()
        self.MuC.value=Cabsorption(self.model)
        
        return {'Quant map has been created'}
        
    def function(self,x):
 
        b=self.coefficients.value[0]
        a=self.coefficients.value[1]
    
        
        E0=self.E0.value
        teta=np.radians(self.teta.value)
        Cthickness=self.coating_thickness.value
        
        Mu=Mucoef(self.model,self.quanti.value)
        Mu=np.array(Mu,dtype=float)
        #Mu=Mu[self.model.channel_switches]

        
        
        Window=np.array(self.Window.value,dtype=float)
        Window=Window[self.model.channel_switches]

        MuC=np.array(self.MuC.value,dtype=float)
        MuC=MuC[self.model.channel_switches]
        cosec=(1/np.sin(teta))
	
        emission=(a*((E0-x)/x)) #kramer's law
        absorption=((1-np.exp(-2*Mu*b*10**-5*cosec))/((2*Mu*b*10**-5*cosec))) #love and scott model. 
        METabsorption=(np.exp(-Mu*b*10**-5*cosec)) #CL model

        if self.coating_thickness.value>0:
            coating=(np.exp(-MuC*1.3*Cthickness*10**-7*cosec))# absorption by the coating layer (1.3 is the density)
        else:
            coating=1
	
        if self._whitelist['absorption_model'] is 'quadrilateral':
            return np.where((x>0.17) & (x<(E0)),(emission*absorption*Window*coating),0) 
        if self._whitelist['absorption_model'] is 'CL':
            return np.where((x>0.17) & (x<(E0)),(emission*METabsorption*Window*coating),0)
