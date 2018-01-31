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




def Wpercent(model,E0,quantification):
    if quantification is None :
        w=model.signal.get_lines_intensity(only_one=True)
        if len(np.shape(w[0]))>1 and np.shape(w[0])[1]>1 :
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
                    g[:,:,i] =(u[:,:,i]/(np.sum(u,axis=2))) *100          	    
        elif len(np.shape(w[0]))==1 and np.shape(w[0])[0]>1 :
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
                g[:,i] =(u[:,i]/(np.sum(u,axis=1))) *100          		
        else:
            u=np.ones([len(model._signal.metadata.Sample.elements)] )
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
    elif isinstance(quantification[0] , (float)):
        g=quantification
    else:
        w=quantification
        if 'atomic percent' in w[0].metadata.General.title:
            w=atomic_to_weight(w)
        else:
            w=w           
        if len(np.shape(w[0]))>1 and np.shape(w[0])[1]>1 :
            u=np.ones([np.shape(w[0])[0],np.shape(w[0])[1], len(model._signal.metadata.Sample.elements)] )
            for i in range (0,len(w)):
                u[:,:,i] =w[i].data
	
            g=np.ones([np.shape(w[0])[0],np.shape(w[0])[1], len(model._signal.metadata.Sample.elements)] )
            for i in range (0,len(model._signal.metadata.Sample.elements)):
                g[:,:,i] =(u[:,:,i]/(np.sum(u,axis=2))) *100          
		    
        elif len(np.shape(w[0]))==1 and np.shape(w[0])[0]>1 :
            u=np.ones([np.shape(w[0])[0], len(model._signal.metadata.Sample.elements)] )
            for i in range (0,len(w)):
               u[:,i] =w[i].data

            g=np.ones([np.shape(w[0])[0], len(model._signal.metadata.Sample.elements)] )
            for i in range (0,len(model._signal.metadata.Sample.elements)):
                g[:,i] =(u[:,i]/(np.sum(u,axis=1))) *100          
		    
        else:
            u=np.ones([len(model._signal.metadata.Sample.elements)] )
            for i in range (0,len(w)):            
                u[i] =w[i].data
		    
            g=np.ones([len(model._signal.metadata.Sample.elements)] )
            t=u.sum() 
            for i in range (0,len(u)):
                g[i] =u[i] /t*100
    return g



def Mucoef(model,E0,quanti): 
    w=quanti
    t=np.linspace(model._signal.axes_manager[-1].offset,model._signal.axes_manager[-1].size*model._signal.axes_manager[-1].scale,model._signal.axes_manager[-1].size)-0.05
    #t=t[model.channel_switches]
    
    if len(np.shape(w))>2 and np.shape(w)[1]>1 :
        Ac=np.empty([np.shape(w)[0],np.shape(w)[1],len(t)],float)
        for i in range (0,np.shape(w)[0]):
            for k in range (0, np.shape(w)[1]):  
                for j in range (0,len(t)): 
                    Ac[i,k,j] = mass_absorption_mixture(elements=model._signal.metadata.Sample.elements,weight_percent=w[i,k], energies=t[j])

    elif len(np.shape(w))==2 and np.shape(w)[1]>1 :
        Ac=np.empty([np.shape(w)[0],len(t)],float)
        for i in range (0,np.shape(w)[0]):
            for j in range (0,len(t)): 
                Ac[i,j] = mass_absorption_mixture(elements=model._signal.metadata.Sample.elements,weight_percent=w[i,:], energies=t[j])

    else: Ac=mass_absorption_mixture(elements=model._signal.metadata.Sample.elements,weight_percent=w, energies=t)    

    return Ac



##def Cabsorption(model): 
##    t=np.linspace(model._signal.axes_manager[-1].offset,model._signal.axes_manager[-1].size*model._signal.axes_manager[-1].scale,model._signal.axes_manager[-1].size)-0.05
##    Acc=mass_absorption_mixture(elements=['C'],weight_percent=[100], energies=t)
##    *((1-np.exp(-2*C*e*10**-6 ))/((2*C*e*10**-6)))
##    return Acc



def Windowabsorption(model,detector): 
    from hyperspy.misc.eds.detector_efficiency import detector_efficiency
    from scipy.interpolate import interp1d
    a=np.array(detector_efficiency[detector])
    b=np.linspace(model._signal.axes_manager[-1].offset,model._signal.axes_manager[-1].size*model._signal.axes_manager[-1].scale,model._signal.axes_manager[-1].size)-0.05
    x =a[:,0]
    y = a[:,1]
    Accc=np.interp(b, x, y)
    return Accc



def emissionlaw(model,E0):
    E0=E0

    axis = model._signal.axes_manager.signal_axes[-1]
    if E0<20:
        i1, i2 = axis.value_range_to_indices(E0/3,E0)
    else :
        i1, i2 = axis.value_range_to_indices(np.max(axis.axis)/3,np.max(axis.axis))
    def myfunc(p, fjac=None, x=None, y=None, err=None):
        return [0, eval('(y-(%s))/err' % func, globals(), locals())]
    func='p[0]*((p[1] -x)/x)'
    x=axis.axis[model.channel_switches][i1:i2]
    y=model._signal.data[model.channel_switches][i1:i2]
    err=np.ones(len(model._signal.data[model.channel_switches][i1:i2]))
    start_params=[0,E0] 

    fa = {'x': x, 
          'y': y, 
          'err': err}

    parinfo =[{'value': 0., 'fixed': 0, 'limited': [0, 0], 'limits' : [0., 0.], 'tied' : ''}
        for i in range(len(start_params))]
    parinfo[0]['limited'][0] = 1
    parinfo[0]['limits'][0]  = 0.
    parinfo[1]['fixed'] = 1


    res = mpfit.mpfit(myfunc, start_params, functkw=fa,parinfo=parinfo)
    res.globals=dict()
    yfit = eval(func, globals(), {'x': x, 'p': res.params})
    return res.params[0]

        
class Physical_background(Component):

    """
    Background component based on kramer's law and absorption coefficients
    Attributes
    ----------
    a : float
    b : float
    E0 : int 
    """

    def __init__(self,model, E0, detector, quantification, coefficients=3, Mu=0 , Window=0, a=0,quanti=0,d=0):
        Component.__init__(self,['coefficients','E0','detector','quantification', 'Mu','Window','a','d','quanti'])
        self.coefficients.value=coefficients
        self.a.value=a
        self.d.value=d
        self.E0.value=E0
        self.quanti._number_of_elements=len(model._signal.metadata.Sample.elements)
        self.quanti.value=Wpercent(model,E0,quantification)
        
                              
        self.Mu._number_of_elements=model.axis.axis.shape[0]
        self.Mu.value=Mucoef(model,E0,self.quanti.value)
        #self.C._number_of_elements=model.axis.axis.shape[-1]
        #self.C.value=Cabsorption(model)
        self.Window._number_of_elements=model.axis.axis.shape[0]
        self.Window.value=Windowabsorption(model,detector)


        self.E0.free=False
        self.quanti.free=False
        self.a.free=True
        self.d.free=False
        self.coefficients.free=True
        self.Mu.free=False
        self.Window.free=False
        #self.C.free=False
        
        self.isbackground=True

        # Boundaries
        self.coefficients.bmax = None
        self.coefficients.bmin = 0.

    def function(self,x):
 
        b=self.coefficients.value
        
        
        Mu=self.Mu.value
        self.Mu._create_array()
        Mu=np.array(Mu,dtype=float)
        Mu=Mu[self.model.channel_switches]

##        C=self.C.value
##        self.C._create_array()
##        C=np.array(C,dtype=float)
##        C=C[self.model.channel_switches]

        Window=self.Window.value
        self.Window._create_array()
        Window=np.array(Window,dtype=float)
        Window=Window[self.model.channel_switches]

        E0=self.E0.value
               
        a=self.a.value     
##        if a==0:
##            t=emissionlaw(self.model,E0)
##            self.model.components.Bremsstrahlung.a.value=(t)
##            a=t
##            print('modifification of values : P0 = emission coefficient  / P1 = E0')
##            
##        else: a=self.a.value

        #d=np.max(self.model.axis.axis)/np.max(x)
         
        return np.where((x>0.1) & (x<(E0+0.05)),((a*100*((E0-x)/x))*((1-np.exp(-2*Mu*b*10**-5 ))/((2*Mu*b*10**-5)))*Window),0) #implement choice for coating correction
