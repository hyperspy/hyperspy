# -*- coding: utf-8 -*-
# Copyright 2007-2011 The Hyperspy developers
#
# This file is part of  Hyperspy.
#
#  Hyperspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  Hyperspy.  If not, see <http://www.gnu.org/licenses/>.

import os

import numpy as np

from hyperspy.defaults_parser import preferences
from hyperspy.misc.utils import (incremental_filename,
                                  append2pathname,
                                  slugify)
from hyperspy.exceptions import NavigationDimensionError

class Parameter(object):
    """Model parameter
    
    Attributes
    ----------
    value : float or array
        The value of the parameter for the current location. The value
        for other locations is stored in map.
    bmin, bmax: float
        Lower and upper bounds of the parameter value.
    twin : {None, Parameter}
        If it is not None, the value of the current parameter is 
        a function of the given Parameter. The function is by default
        the identity function, but it can be defined by twin_function
    twin_function : function
        Function that takes Parameter.value as its only argument
        and returns a float or array that is set to be the current 
        Parameter.value
    ext_force_positive : bool
        If True, the parameter value is set to be the absolute value 
        of the input value i.e. if we set Parameter.value = -3, the 
        value stored is 3 instead. This is useful to bound a value 
        to be positive in an optimization without actually using an 
        optimizer that supports bounding.
    ext_bounded : bool
        Similar to ext_force_positive, but in this case the bounds are
        defined by bmin and bmax. It is a better idea to use
        an optimizer that supports bounding though.
        
    Methods
    -------
    as_signal(field = 'values')
        Get a parameter map as a signal object
    plot()
        Plots the value of the Parameter at all locations.
    export(folder=None, name=None, format=None, save_std=False)
        Saves the value of the parameter map to the specified format
    
    
    """

    def __init__(self, value=0., free=True, bmin=None, bmax=None,
                 twin=None):
        
        self.component = None
        self.connection_active = False
        self.connected_functions = list()
        self.ext_bounded = False
        self._number_of_elements = 1
        self._bounds = (None, None)
        self.bmin = None
        self.bmax = None
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
        self._axes_manager = None

    # Define the bounding and coupling propertires
    
    def __repr__(self):
        text = ''
        text += 'Parameter %s' % self.name
        if self.component is not None:
            text += ' of %s' % self.component._get_short_description()
        text = '<' + text + '>'
        return text
    
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

        if self.ext_bounded is False:
                self.__value = arg
        else:
            if self.ext_force_positive is True :
                self.__value = abs(arg)
            else :
                if self._number_of_elements == 1:
                    if self.bmin is not None and arg <= self.bmin:
                        self.__value=self.bmin
                    elif self.bmax is not None and arg >= self.bmax:
                        self.__value=self.bmax
                    else:
                        self.__value=arg
                else :
                    self.__value=ar
        if self.connection_active is True:
            for f in self.connected_functions:
                try:
                    f()
                except:
                    self.disconnect(f)
    value = property(_coerce, _decoerce)

    # Fix the parameter when coupled
    def _getfree(self):
        if self.twin is None:
            return self.__free
        else:
            return False
    def _setfree(self,arg):
        self.__free = arg
        if self.component is not None:
            self.component._update_free_parameters()
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
        
    def create_array(self, shape):
        if len(shape) == 1 and shape[0] == 0:
            shape = [1,]
        dtype_ = np.dtype([
            ('values','float', self._number_of_elements), 
            ('std', 'float', self._number_of_elements), 
            ('is_set', 'bool', 1)])
        if (self.map is None  or self.map.shape != shape or 
                    self.map.dtype != dtype_):
            self.map = np.zeros(shape, dtype_)       
            self.map['std'][:] = np.nan
            # TODO: in the future this class should have access to 
            # axes manager and should be able to charge its own
            # values. Until then, the next line is necessary to avoid
            # erros when self.std is defined and the shape is different
            # from the newly defined arrays
            self.std = None
            
    def as_signal(self, field='values'):
        """Get a parameter map as a signal object.
        
        Please note that this method only works when the navigation dimension
        is greater than 0.
        
        Parameters
        ----------
        field : {'values', 'std', 'is_set'}
        
        Raises
        ------
        
        NavigationDimensionError : if the navigation dimension is 0
        
        """
        from hyperspy.signal import Signal
        if self._axes_manager.navigation_dimension == 0:
            raise NavigationDimensionError(0, '>0')
            
        s = Signal({'data' : self.map[field],
                    'axes' : self._axes_manager._get_navigation_axes_dicts()})
        s.mapped_parameters.title = self.name
        for axis in s.axes_manager.axes:
            axis.navigate = False
        if self._number_of_elements > 1:
            s.axes_manager.append_axis(size=self._number_of_elements,
                                       name=self.name,
                                       index_in_array=len(s.axes_manager.axes),
                                       navigate=True)
        return s
        
    def plot(self):
        self.as_signal().plot()
        
    def export(self, folder=None, name=None, format=None, save_std=False):
        '''Save the data to a file.
        
        All the arguments are optional.
        
        Parameters
        ----------
        folder : str or None
            The path to the folder where the file will be saved. If `None` the
            current folder is used by default.
        name : str or None
            The name of the file. If `None` the Components name followed by the
            Parameter `name` attributes will be used by default. If a file with 
            the same name exists the name will be modified by appending a number
            to the file path.
        save_std : bool
            If True, also the standard deviation will be saved
        
        '''
        if format is None:
            format = preferences.General.default_export_format
        if name is None:
            name = self.component.name + '_' + self.name
        filename = incremental_filename(slugify(name) + '.' + format)
        if folder is not None:
            filename = os.path.join(folder, filename)
        self.as_signal().save(filename)
        if save_std is True:
            self.as_signal(field = 'std').save(append2pathname(
                filename,'_std'))
                    
class Component(object):
    def __init__(self, parameter_name_list):
        self.connected_functions = list()
        self.parameters = []
        self.init_parameters(parameter_name_list)
        self._update_free_parameters()
        self.active = True
        self.isbackground = False
        self.convolved = True
        self.parameters = tuple(self.parameters)
        self.name = ''
        self._id_name = self.__class__.__name__
        self._id_version = '1.0'
        
    def connect(self, f):
        if f not in self.connected_functions:
            self.connected_functions.append(f)
    def disconnect(self, f):
        if f in self.connected_functions:
            self.connected_functions.remove(f)
            
    def _get_active(self):
        return self.__active
    def _set_active(self, arg):
        self.__active = arg
        for f in self.connected_functions:
            try:
                f()
            except:
                self.disconnect(f)
    active = property(_get_active, _set_active)

    def init_parameters(self, parameter_name_list):
        for name in parameter_name_list:
            parameter = Parameter()
            self.parameters.append(parameter)
            parameter.name = name
            setattr(self, name, parameter)
            if hasattr(self, 'grad_' + name):
                parameter.grad = getattr(self, 'grad_' + name)
            parameter.component = self
            
    def _get_long_description(self):
        if self.name:
            text = '%s (%s component)' % (self.name, self._id_name)
        else:
            text = '%s component' % self._id_name
        return text
        
    def _get_short_description(self):
        text = ''
        if self.name:
            text += self.name
        else:
            text += self._id_name
        text += ' component'
        return text
    
    def __repr__(self):
        text = '<%s>' % self._get_long_description()
        return text

    def _update_free_parameters(self):
        self.free_parameters = set()
        for parameter in self.parameters:
            if parameter.free:
                self.free_parameters.add(parameter)
        # update_number_free_parameters(self):
        i=0
        for parameter in self.free_parameters:
            i += parameter._number_of_elements
        self._nfree_param=i

    def update_number_parameters(self):
        i=0
        for parameter in self.parameters:
            i += parameter._number_of_elements
        self.nparam=i

    def charge(self, p, p_std = None, onlyfree = False):
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
                parameter.std = (p_std[i] if lenght == 1 else 
                p_std[i:i+lenght].tolist())
            
            i+=lenght           
                
    def create_arrays(self, shape):
        for parameter in self.parameters:
            parameter.create_array(shape)
    
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
                    parameter.std = parameter.std.tolist()

    def plot(self, only_free = True):
        """Plot the value of the parameters of the model
        
        Parameters
        ----------
        only_free : bool
            If True, only the value of the parameters that are free will be
            plotted
              
        """
        if only_free:
            parameters = self.free_parameters
        else:
            parameters = self.parameters
            
        parameters = [k for k in parameters if k.twin is None]
        for parameter in parameters:
            parameter.plot()
            
    def export(self, folder=None, format=None, save_std=False, only_free=True):
        """Plot the value of the parameters of the model
        
        Parameters
        ----------
        folder : str or None
            The path to the folder where the file will be saved. If `None` the
            current folder is used by default.
        format : str
            The format to which the data will be exported. It must be the
            extension of any format supported by Hyperspy. If None, the default
            format for exporting as defined in the `Preferences` will be used.
        save_std : bool
            If True, also the standard deviation will be saved.
        only_free : bool
            If True, only the value of the parameters that are free will be
            exported.
            
        Notes
        -----
        The name of the files will be determined by each the Component and
        each Parameter name attributes. Therefore, it is possible to customise
        the file names modify the name attributes.
              
        """
        if only_free:
            parameters = self.free_parameters
        else:
            parameters = self.parameters
            
        parameters = [k for k in parameters if k.twin is None]
        for parameter in parameters:
            parameter.export(folder=folder, format=format, save_std=save_std,)
            
    def summary(self):
        for parameter in self.parameters:
            dim = len(parameter.map.squeeze().shape) if parameter.map \
                        is not None else 0
            if parameter.twin is None:
                if dim <= 1:
                    print '%s = %s ± %s %s' % (parameter.name, parameter.value, 
                    parameter.std, parameter.units)

    def __call__(self, p, x, onlyfree = True) :
        self.charge(p , onlyfree = onlyfree)
        return self.function(x)
        
    def set_axes(self, axes_manager):
        for parameter in self.parameters:
            parameter._axes_manager = axes_manager
