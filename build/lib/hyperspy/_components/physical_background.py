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
        w=model.signal.get_lines_intensity(only_one=True)
        if len(np.shape(w[0]))>1 and np.shape(w[0])[1]>1 : #for 3D Data
            u=np.ones([np.shape(w[0])[0],np.shape(w[0])[1], len(model._signal.metadata.Sample.elements)] )
            for i in range (0,len(w)):
                if "Ka" in w [i].metadata.General.title :
                    u[:,:,i] =w[i].data
                elif "La" in w [i].metadata.General.title:
                    u[:,:,i] =w[i].data*2.5
                elif "La" in w [i].metadata.General.title:
                    u[:,:,i] =w[i].data*2.8
                g=np.ones([np.shape(w[0])[0],np.shape(w[0])[1], len(model._signal.metadata.Sample.elements)] )
                for i in range (0,len(model._signal.metadata.Sample.elements)):
                    g[:,:,i] =(u[:,:,i]/(np.sum(u,axis=-1))) *100          	    
        elif len(np.shape(w[0]))==1 and np.shape(w[0])[0]>1 : #for 2D Data
            u=np.ones([np.shape(w[0])[0], len(model._signal.metadata.Sample.elements)] )
            for i in range (0,len(w)):
                if "Ka" in w [i].metadata.General.title :
                    u[:,i] =w[i].data
                elif "La" in w [i].metadata.General.title:
                    u[:,i] =w[i].data*2.5
                elif "La" in w [i].metadata.General.title:
                    u[:,i] =w[i].data*2.8
            g=np.ones([np.shape(w[0])[0], len(model._signal.metadata.Sample.elements)] )
            for i in range (0,len(model._signal.metadata.Sample.elements)):
                g[:,i] =(u[:,i]/(np.sum(u,axis=-1))) *100          		
        else:
            u=np.ones([len(model._signal.metadata.Sample.elements)] ) #for 1D 
            for i in range (0,len(w)):
                if "Ka" in w [i].metadata.General.title :
                    u[i] =w[i].data
                elif "La" in w [i].metadata.General.title:
                    u[i] =w[i].data*2.5
                elif "La" in w [i].metadata.General.title:
                    u[i] =w[i].data*2.8        
            g=np.ones([len(model._signal.metadata.Sample.elements)] )
            t=u.sum() 
            for i in range (0,len(u)):
                g[i] =u[i] /t*100
    elif isinstance(quantification[0] , (float,int)): # if quantification is already an array of weight percent, directly keep the array
        g=quantification

    else:
        w=quantification # if quantification is the result of the hyperspy quantification function. This function convert the result in an array with the same navigation shape than the model and a length equal to the number of elements 
        if 'atomic percent' in w[0].metadata.General.title:
            w=atomic_to_weight(w)
        else:
            w=w
            
        if len(np.shape(w[0]))>1 and np.shape(w[0])[1]>1 : # for 3D
            u=np.ones([np.shape(w[0])[0],np.shape(w[0])[1], len(model._signal.metadata.Sample.elements)] )
            for i in range (0,len(w)):
                u[:,:,i] =w[i].data
	
            g=np.ones([np.shape(w[0])[0],np.shape(w[0])[1], len(model._signal.metadata.Sample.elements)] )
            for i in range (0,len(model._signal.metadata.Sample.elements)):
                g[:,:,i] =(u[:,:,i]/(np.sum(u,axis=2))) *100          
		    
        elif len(np.shape(w[0]))==1 and np.shape(w[0])[0]>1 : #for 2D
            u=np.ones([np.shape(w[0])[0], len(model._signal.metadata.Sample.elements)] )
            for i in range (0,len(w)):
               u[:,i] =w[i].data

            g=np.ones([np.shape(w[0])[0], len(model._signal.metadata.Sample.elements)] )
            for i in range (0,len(model._signal.metadata.Sample.elements)):
                g[:,i] =(u[:,i]/(np.sum(u,axis=1))) *100          
		    
        else: #for 1D
            u=np.ones([len(model._signal.metadata.Sample.elements)] )
            for i in range (0,len(w)):            
                u[i] =w[i].data
		    
            g=np.ones([len(model._signal.metadata.Sample.elements)] )
            t=u.sum() 
            for i in range (0,len(u)):
                g[i] =u[i] /t*100
    return g



def Mucoef(model,quanti): # this function calculate the absorption coefficient for all energy. This, correspond to the Mu parameter in the function
    w=quanti
    t=np.linspace(model._signal.axes_manager[-1].offset,model._signal.axes_manager[-1].size*model._signal.axes_manager[-1].scale,model._signal.axes_manager[-1].size)-0.05
    t=t
    
##    if len(np.shape(w))>2 and np.shape(w)[1]>1 : #for 3D
##        Ac=np.empty([np.shape(w)[0],np.shape(w)[1],len(t)],float)
##        for i in range (0,np.shape(w)[0]):
##            for k in range (0, np.shape(w)[1]):  
##                for j in range (0,len(t)): 
##                    Ac[i,k,j] = mass_absorption_mixture(elements=model._signal.metadata.Sample.elements,weight_percent=w[i,k], energies=t[j])
##
##    elif len(np.shape(w))==2 and np.shape(w)[1]>1 : #for 2D
##        Ac=np.empty([np.shape(w)[0],len(t)],float)
##        for i in range (0,np.shape(w)[0]):
##            for j in range (0,len(t)): 
##                Ac[i,j] = mass_absorption_mixture(elements=model._signal.metadata.Sample.elements,weight_percent=w[i,:], energies=t[j])

    Ac=mass_absorption_mixture(elements=model._signal.metadata.Sample.elements,weight_percent=w, energies=t)    #1D

    return Ac




def Windowabsorption(model,detector): # this function interpolate the detector efficiency based on dictionnary (create from personnal data) and the signal length. This correspond to the Window parameter  
 
    a=np.array(detector_efficiency[detector])
    b=np.linspace(model._signal.axes_manager[-1].offset,model._signal.axes_manager[-1].size*model._signal.axes_manager[-1].scale,model._signal.axes_manager[-1].size)-0.05
    x =a[:,0]
    y = a[:,1]
    Accc=np.interp(b, x, y)
    return Accc


##def Cabsorption(model): # This function is used to calculate absorption by the coating layer
##    t=np.linspace(model._signal.axes_manager[-1].offset,model._signal.axes_manager[-1].size*model._signal.axes_manager[-1].scale,model._signal.axes_manager[-1].size)-0.05
##    Acc=mass_absorption_mixture(elements=['C'],weight_percent=[100], energies=t)
##    *((1-np.exp(-2*C*e*10**-6 ))/((2*C*e*10**-6)))
##    return Acc


##def emissionlaw(model,E0): # This function is made for fit the parameter only on the end of the spectra where the absorption is mostly absent
##    E0=E0
##
##    axis = model._signal.axes_manager.signal_axes[-1]
##    if E0<20:
##        i1, i2 = axis.value_range_to_indices(E0/3,E0)
##    else :
##        i1, i2 = axis.value_range_to_indices(np.max(axis.axis)/3,np.max(axis.axis))
##    def myfunc(p, fjac=None, x=None, y=None, err=None):
##        return [0, eval('(y-(%s))/err' % func, globals(), locals())]
##    func='p[0]*((p[1] -x)/x)'
##    x=axis.axis[model.channel_switches][i1:i2]
##    y=model._signal.data[model.channel_switches][i1:i2]
##    err=np.ones(len(model._signal.data[model.channel_switches][i1:i2]))
##    start_params=[0,E0] 
##
##    fa = {'x': x, 
##          'y': y, 
##          'err': err}
##
##    parinfo =[{'value': 0., 'fixed': 0, 'limited': [0, 0], 'limits' : [0., 0.], 'tied' : ''}
##        for i in range(len(start_params))]
##    parinfo[0]['limited'][0] = 1
##    parinfo[0]['limits'][0]  = 0.
##    parinfo[1]['fixed'] = 1
##
##
##    res = mpfit.mpfit(myfunc, start_params, functkw=fa,parinfo=parinfo)
##    res.globals=dict()
##    yfit = eval(func, globals(), {'x': x, 'p': res.params})
##    return res.params[0]

        
class Physical_background(Component):

    """
    Background component based on kramer's law and absorption coefficients
    Attributes
    ----------
    a : float
    b : float
    E0 : int 
    """

    def __init__(self, E0, detector, quantification, coefficients=3, Mu=0 , Window=0, quanti=0):
        Component.__init__(self,['coefficients','E0', 'Mu','Window','quanti'])

        self.coefficients._number_of_elements = 2
        self.coefficients.value = np.ones(2)
        
        self.E0.value=E0
        
        self._whitelist['quanti'] = quantification
        self._whitelist['detector'] = detector       
        self.quanti.value=quanti

        self.Mu.value=Mu
        self.Window.value=Window

        self.E0.free=False
        self.coefficients.free=True
        self.Mu.free=False
        self.Window.free=False
        self.quanti.free=False
        
        self.isbackground=True
        #self.convolved = False

        # Boundaries
        self.coefficients.bmax = None
        self.coefficients.bmin = 0.

    def function(self,x):
 
        b=self.coefficients.value[0]
        a=self.coefficients.value[1]
        self.coefficients._create_array()
            
        E0=self.E0.value
        

        quanti=Wpercent(self.model,E0,self._whitelist['quanti'])

        self.quanti._number_of_elements=len(self.model._signal.metadata.Sample.elements)
        self.quanti._create_array()
               
        if len(self.model.axes_manager.shape)==1:
            self.quanti.value=quanti
        else: 
            self.quanti.map['values'][:] = quanti
            self.quanti.map['is_set'][:] = True

        Mu=Mucoef(self.model,self.quanti.value)

        self.Mu._number_of_elements=len(self.model._signal.axes_manager.signal_axes[-1].axis)
        self.Mu._create_array()
        self.Mu.value=Mu
                    
        Mu=np.array(Mu,dtype=float)
        Mu=Mu[self.model.channel_switches]
       

        Window=Windowabsorption(self.model,self._whitelist['detector'])

        self.Window._number_of_elements=len(self.model._signal.axes_manager.signal_axes[-1].axis)
        self.Window.value=Window
        self.Window._create_array()
        
        Window=np.array(Window,dtype=float)
        Window=Window[self.model.channel_switches]
     
        
            #t=emissionlaw(self.model,E0)
            #self.model.components.Bremsstrahlung.a.value=(t)
            #a=t
            #print('modifification of values : P0 = emission coefficient  / P1 = E0')
            #a=self.a.value 
            #a=self.a.value =False
            
        return np.where((x>0.05) & (x<(E0)),((a*100*((E0-x)/x))*((1-np.exp(-2*Mu*b*10**-5 ))/((2*Mu*b*10**-5)))*Window),0) #implement choice for coating correction
