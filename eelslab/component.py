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
#from widgets import cursors



class Parameter(object):
    """
    class_documentation
    """

    def __init__(self, value=0., free=True, bmin=None, 
    bmax=None, twin = None):
        
        self.connection_active = False
        self.connected_functions = []
        self.ext_bounded = False
        self._number_of_elements = 1
        self._bounds = (None, None)
        self.bmin = bmin
        self.bmax = bmax
        self.__twin = None
        self.twin = twin
        self.twin_function = lambda x: x
        self._twins = []
        self.ext_force_positive = False
        self.value = value
        self.free = free
        self.map = None
        self.std_map = None
        self.grad = None
        self.already_set_map = None
        self.name = ''
        self.units = ''
        self.std = None
        

    # Define the bounding and coupling propertires
    
    def connect(self, f):
        if f not in self.connected_functions:
            self.connected_functions.append(f)
    def disconnect(self, f):
        if f in self.connected_functions:
            self.connected_functions.remove(f)
            
    def _coerce(self):
        if self.twin is None:
            return self.__value
        else:
            return self.twin_function(self.twin.value)
    def _decoerce(self, arg):
        if self.connection_active is True:
            for f in self.connected_functions:
                try:
                    f()
                except:
                    self.disconnect(f)
        if self.ext_bounded is False :
                self.__value = arg
        else:
            if self.ext_force_positive is True :
                self.__value = abs(arg)
            else :
                if self._number_of_elements == 1:
                    if arg <= self.bmin:
                        self.__value=self.bmin
                    elif arg >= self.bmax:
                        self.__value=self.bmax
                    else:
                        self.__value=arg
                else :
                    self.__value=arg
    value = property(_coerce, _decoerce)

    # Fix the parameter when coupled
    def _getfree(self):
        if self.twin is None:
            return self.__free
        else:
            return False
    def _setfree(self,arg):
        self.__free = arg
    free = property(_getfree,_setfree)

    def _set_twin(self,arg):
        if arg is None :
            if self.__twin is not None :
                if self in self.__twin._twins:
                    self.__twin._twins.remove(self)
        else :
            if self not in arg._twins :
                arg._twins.append(self)
        self.__twin = arg

    def _get_twin(self):
        return self.__twin
    twin = property(_get_twin, _set_twin)

    def _get_bmin(self):
        if isinstance(self._bounds, tuple):
            return self._bounds[0]
        elif isinstance(self._bounds, list):
            return self._bounds[0][0]
    def _set_bmin(self,arg):
        if self._number_of_elements == 1 :
            self._bounds = (arg,self.bmax)
        elif self._number_of_elements > 1 :
            self._bounds = [(arg, self.bmax)] * self._number_of_elements
    bmin = property(_get_bmin,_set_bmin)

    def _get_bmax(self):
        if isinstance(self._bounds, tuple):
            return self._bounds[1]
        elif isinstance(self._bounds, list):
            return self._bounds[0][1]
    def _set_bmax(self,arg):
        if self._number_of_elements == 1 :
            self._bounds = (self.bmin, arg)
        elif self._number_of_elements > 1 :
            self._bounds = [(self.bmin, arg)] * self._number_of_elements
    bmax = property(_get_bmax,_set_bmax)

    def store_current_value_in_array(self,indexes):
        self.map['values'][indexes] = self.value
        self.map['is_set'][indexes] = True
        if self.std is not None:
            self.map['std'][indexes] = self.std
    def assign_current_value_to_all(self, mask = None):
        '''Stores in the map the current value for all the rest of the pixels
        
        Parameters
        ----------
        mask: numpy array
        '''
        if mask is None:
            mask = np.zeros(self.map.shape, dtype = 'bool')
        self.map['values'][mask == False] = self.value
        self.map['is_set'][mask == False] = True
                    

class Component:
    def __init__(self, parameter_name_list):
        self.parameters = []
        self.init_parameters(parameter_name_list)
        self.refresh_free_parameters()
        self.active = True
        self.isbackground = False
        self.convolved = True
        self.parameters = tuple(self.parameters)

    def init_parameters(self, parameter_name_list):
        for par in parameter_name_list:
            exec('self.%s = Parameter()' % par)
            exec('self.%s.name = \'%s\'' % (par, par))
            exec('self.parameters.append(self.%s)' % par)
            exec('try:\n    self.%s.grad = self.grad_%s\nexcept:\n    '
            'self.%s.grad = None' % (par, par, par))
    
    def __repr__(self):
        return self.name

    def refresh_free_parameters(self):
        self.free_parameters = set()
        for parameter in self.parameters:
            if parameter.free:
                self.free_parameters.add(parameter)
        self.update_number_free_parameters()

    def update_number_free_parameters(self):
        i=0
        for parameter in self.free_parameters:
            i += parameter._number_of_elements
        self.nfree_param=i

    def update_number_parameters(self):
        i=0
        for parameter in self.parameters:
            i += parameter._number_of_elements
        self.nparam=i

    def charge( self, p, p_std = None, onlyfree = False):
        if onlyfree is True:
            parameters = self.free_parameters
        else:
            parameters = self.parameters
        i=0
        for parameter in parameters:
            lenght = parameter._number_of_elements
            parameter.value = (p[i] if lenght == 1 else 
            p[i:i+lenght].tolist())
            if p_std is not None:
                parameter.map['std'] = (p_std[i] if lenght == 1 else 
                p_std[i:i+lenght].tolist())
            
            i+=lenght           
                
    def create_arrays(self, shape):
        for parameter in self.parameters:
            # It will overwrite the arrays if the dimension changes.
            # This is only useful if the number of knots of the fine structure
            # component has changed
            if parameter.map is None  or parameter.map.shape != shape:
                parameter.map = np.zeros(shape, 
                dtype = [
                ('values','float', parameter._number_of_elements), 
                ('std', 'float', parameter._number_of_elements), 
                ('is_set', 'bool', 1)])
                parameter.map['std'][:] = np.nan
    
    def store_current_parameters_in_map(self, indexes):
        for parameter in self.parameters:
            parameter.store_current_value_in_array(indexes)
        
    def charge_value_from_map(self, indexes, only_fixed = False):
        if only_fixed is True:
            parameters = set(self.parameters) - set(self.free_parameters)
        else:
            parameters = self.parameters

        for parameter in parameters:
            if parameter.map['is_set'][indexes]:
                parameter.value = parameter.map['values'][indexes]
                parameter.std = parameter.map['std'][indexes]
                if parameter._number_of_elements > 1:
                    parameter.value = parameter.value.tolist()
                    parameter.value = parameter.std.tolist()

#    def plot_maps(self):
#        for parameter in self.parameters:
#            if (parameter.map is not None) and (parameter.twin is None):
#                dim = len(parameter.map.squeeze().shape)
#                title = '%s - %s' % (self.name, parameter.name)
#                if dim == 2:
#                    fig = plt.figure()
#                    ax = fig.add_subplot(111)
#                    this_map = ax.matshow(parameter.map.squeeze().T)
#                    ax.set_title(title)
#                    fig.canvas.set_window_title(title)
#                    fig.colorbar(this_map)
#                    cursors.add_axes(ax)
#                    fig.canvas.draw()
#                elif dim == 1:
#                    fig = plt.figure()
#                    ax = fig.add_subplot(111)
#                    this_graph = ax.plot(parameter.map.squeeze())
#                    ax.set_title(title)
#                    fig.canvas.set_window_title(title)
#                    ax.set_title(title)
#                    ax.set_ylabel('%s (%s)' % (parameter.name, parameter.units))
#                    ax.set_xlabel('Pixel')
#                    fig.canvas.draw()
#                elif dim == 3:
#                    pass
    def summary(self):
        for parameter in self.parameters:
            dim = len(parameter.map.squeeze().shape)
            if (parameter.map is not None) and (parameter.twin is None):
                if dim <= 1:
                    print '%s = %s ± %s %s' % (parameter.name, parameter.value, 
                    parameter.std, parameter.units)

    def __call__(self, p, x, onlyfree = True) :
        self.charge(p , onlyfree = onlyfree)
        return self.function(x)
