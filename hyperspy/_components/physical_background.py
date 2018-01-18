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


def Wpercent(model): 
    w=np.ones(len(model._signal.metadata.Sample.elements))
    for i in range (0,len(w)):
        if "Ka" in model._signal.metadata.Sample.xray_lines[i]:
            w[i]=(model.get_lines_intensity([model._signal.metadata.Sample.xray_lines[i]]))[0].data
            #raise an error if no xray lines are indicated/ not the same as element indicated in metadata
        elif "La" in model._signal.metadata.Sample.xray_lines[i]:
            w[i]=(model.get_lines_intensity([model._signal.metadata.Sample.xray_lines[i]]))[0].data*2.5
        elif "Ma" in model._signal.metadata.Sample.xray_lines[i]:
            w[i]=(model.get_lines_intensity([model._signal.metadata.Sample.xray_lines[i]]))[0].data*2.8
    t=sum(w) 
    for i in range (0,len(w)):
        w[i]=w[i]/t*100
    return w

def MeanZ (model):
    z=np.ones(1)
    w=Wpercent(model)
    for i in range (0,len(model._signal.metadata.Sample.elements)):
        z+=(element_db.elements[model._signal.metadata.Sample.elements[i]]['General_properties']['Z'])*(w[i]/100)
    return z

def Mucoef(model): 
    t=np.linspace(model._signal.axes_manager[0].offset,model._signal.axes_manager[0].size*model._signal.axes_manager[0].scale,model._signal.axes_manager[0].size)-0.05
    Ac=mass_absorption_mixture(elements=model._signal.metadata.Sample.elements,weight_percent=Wpercent(model), energies=t)
    return Ac

def Cabsorption(model): 
    t=np.linspace(model._signal.axes_manager[0].offset,model._signal.axes_manager[0].size*model._signal.axes_manager[0].scale,model._signal.axes_manager[0].size)-0.05
    Acc=mass_absorption_mixture(elements=['C'],weight_percent=[100], energies=t)
    return Acc

def Windowabsorption(model): 
    t=np.linspace(model._signal.axes_manager[0].offset,model._signal.axes_manager[0].size*model._signal.axes_manager[0].scale,model._signal.axes_manager[0].size)-0.05
    Accc=mass_absorption_mixture(elements=['C','O'],weight_percent=[60,40], energies=t)
    return Accc

def emissionlaw(model,E0,Z):
    Z=Z
    E0=E0

    axis = model._signal.axes_manager.signal_axes[0]
    i1, i2 = axis.value_range_to_indices(max(axis.axis)/2,max(axis.axis))
    def myfunc(p, fjac=None, x=None, y=None, err=None):
        return [0, eval('(y-(%s))/err' % func, globals(), locals())]
    func='p[0]*p[1] *((p[2] -x)/x)'
    x=axis.axis[model.channel_switches][i1:i2]
    y=model._signal.data[model.channel_switches][i1:i2]
    err=np.ones(len(model._signal.data[model.channel_switches][i1:i2]))
    start_params=[20,Z,E0] 

    fa = {'x': x, 
          'y': y, 
          'err': err}

    parinfo =[{'value': 0., 'fixed': 0, 'limited': [0, 0], 'limits' : [0., 0.], 'tied' : ''}
        for i in range(len(start_params))]
    parinfo[1]['fixed'] = 1
    parinfo[2]['fixed'] = 1


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

    def __init__(self,model, E0, coefficients=3, Z=15,Mu=0,e=1.30,f=2.25,a=0):
        Component.__init__(self,['coefficients','E0','Z','Mu','C','Window','e','f','a'])
        self.coefficients.value=coefficients
        self.e.value=e
        self.f.value=f
        self.a.value=a
        self.E0.value=E0

        self.Z.value=MeanZ(model)
        
        self.Mu._number_of_elements=model.axis.axis.shape[0]
        self.Mu.value=Mucoef(model)
        self.C._number_of_elements=model.axis.axis.shape[0]
        self.C.value=Cabsorption(model)
        self.Window._number_of_elements=model.axis.axis.shape[0]
        self.Window.value=Windowabsorption(model)

        self.E0.free=False
        self.Z.free=False
        self.a.free=False
        self.coefficients.free=True
        self.e.free=False
        self.f.free=False
        self.Mu.free=False
        self.Window.free=False
        self.C.free=False
        
        self.isbackground=True

        # Boundaries
        self.coefficients.bmin = 100
        self.coefficients.bmax = 0

    def function(self,x):
 
        b=self.coefficients.value
        e=self.e.value
        Z=self.Z.value
        f=self.f.value

        Mu=self.Mu.value
        self.Mu._create_array()
        Mu=np.array(Mu,dtype=float)
        Mu=Mu[self.model.channel_switches]

        C=self.C.value
        self.C._create_array()
        C=np.array(C,dtype=float)
        C=C[self.model.channel_switches]

        Window=self.Window.value
        self.Window._create_array()
        Window=np.array(Window,dtype=float)
        Window=Window[self.model.channel_switches]

        E0=self.E0.value
        
        a=self.a.value
        if a==0:
            t=emissionlaw(self.model,E0,Z)
            a=self.model.components.Bremsstrahlung.a.value=t
            print('modifification of values : P0 = emission coefficient / P1 = Mean Z  / P2 = E0')
            
        else: a=self.a.value
        
        return np.where((x>0.05) & (x<(E0+0.05)),((a*Z*100*((E0-x)/x))*((1-np.exp(-2*Mu*b*10**-5 ))/((2*Mu*b*10**-5)))*((1-np.exp(-2*C*e*10**-6 ))/((2*C*e*10**-6)))*((1-np.exp(-2*Window*f*10**-5 ))/((2*Window*f*10**-5)))),0) #implement choice for coating correction and detector window correction
   
