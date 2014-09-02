# -*- coding: utf-8 -*-
# Copyright 2007-2011 The HyperSpy developers
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

import os

import numpy as np
import warnings

from hyperspy.defaults_parser import preferences
from hyperspy.misc.utils import slugify
from hyperspy.misc.io.tools import (incremental_filename,
                                    append2pathname,)
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
        Function that, if selt.twin is not None, takes self.twin.value
        as its only argument and returns a float or array that is
        returned when getting Parameter.value
    twin_inverse_function : function
        The inverse of twin_function. If it is None then it is not
        possible to set the value of the parameter twin by setting
        the value of the current parameter.
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
    connect, disconnect(function)
        Call the functions connected when the value attribute changes.

    """
    __number_of_elements = 1
    __value = 0
    _bounds = (None, None)
    __twin = None
    _axes_manager = None
    __ext_bounded = False
    __ext_force_positive = False

    def __init__(self):
        self._twins = set()
        self.connected_functions = list()
        self.twin_function = lambda x: x
        self.twin_inverse_function = lambda x: x
        self.value = 0
        self.std = None
        self.component = None
        self.free = True
        self.grad = None
        self.name = ''
        self.units = ''
        self.map = None
        self.model = None

    def __repr__(self):
        text = ''
        text += 'Parameter %s' % self.name
        if self.component is not None:
            text += ' of %s' % self.component._get_short_description()
        text = '<' + text + '>'
        return text.encode('utf8')

    def __len__(self):
        return self._number_of_elements

    def connect(self, f):
        if f not in self.connected_functions:
            self.connected_functions.append(f)
            if self.twin:
                self.twin.connect(f)

    def disconnect(self, f):
        if f in self.connected_functions:
            self.connected_functions.remove(f)
            if self.twin:
                self.twin.disconnect(f)

    def _getvalue(self):
        if self.twin is None:
            return self.__value
        else:
            return self.twin_function(self.twin.value)

    def _setvalue(self, arg):
        try:
            # Use try/except instead of hasattr("__len__") because a numpy
            # memmap has a __len__ wrapper even for numbers that raises a
            # TypeError when calling. See issue #349.
            if len(arg) != self._number_of_elements:
                raise ValueError(
                    "The lenght of the parameter must be ",
                    self._number_of_elements)
            else:
                if not isinstance(arg, tuple):
                    arg = tuple(arg)
        except TypeError:
            if self._number_of_elements != 1:
                raise ValueError(
                    "The lenght of the parameter must be ",
                    self._number_of_elements)
        old_value = self.__value

        if self.twin is not None:
            if self.twin_inverse_function is not None:
                self.twin.value = self.twin_inverse_function(arg)
            return

        if self.ext_bounded is False:
            self.__value = arg
        else:
            if self.ext_force_positive is True:
                arg = np.abs(arg)
            if self._number_of_elements == 1:
                if self.bmin is not None and arg <= self.bmin:
                    self.__value = self.bmin
                elif self.bmax is not None and arg >= self.bmax:
                    self.__value = self.bmax
                else:
                    self.__value = arg
            else:
                bmin = (self.bmin if self.bmin is not None
                        else -np.inf)
                bmax = (self.bmax if self.bmin is not None
                        else np.inf)
                self.__value = np.clip(arg, bmin, bmax)

        if (self._number_of_elements != 1 and
                not isinstance(self.__value, tuple)):
            self.__value = tuple(self.__value)
        if old_value != self.__value:
            for f in self.connected_functions:
                try:
                    f()
                except:
                    self.disconnect(f)
    value = property(_getvalue, _setvalue)

    # Fix the parameter when coupled
    def _getfree(self):
        if self.twin is None:
            return self.__free
        else:
            return False

    def _setfree(self, arg):
        self.__free = arg
        if self.component is not None:
            self.component._update_free_parameters()
    free = property(_getfree, _setfree)

    def _set_twin(self, arg):
        if arg is None:
            if self.twin is not None:
                # Store the value of the twin in order to set the
                # value of the parameter when it is uncoupled
                twin_value = self.value
                if self in self.twin._twins:
                    self.twin._twins.remove(self)
                    for f in self.connected_functions:
                        self.twin.disconnect(f)

                self.__twin = arg
                self.value = twin_value
        else:
            if self not in arg._twins:
                arg._twins.add(self)
                for f in self.connected_functions:
                    arg.connect(f)
            self.__twin = arg

        if self.component is not None:
            self.component._update_free_parameters()

    def _get_twin(self):
        return self.__twin
    twin = property(_get_twin, _set_twin)

    def _get_bmin(self):
        if self._number_of_elements == 1:
            return self._bounds[0]
        else:
            return self._bounds[0][0]

    def _set_bmin(self, arg):
        if self._number_of_elements == 1:
            self._bounds = (arg, self.bmax)
        else:
            self._bounds = ((arg, self.bmax),) * self._number_of_elements
        # Update the value to take into account the new bounds
        self.value = self.value
    bmin = property(_get_bmin, _set_bmin)

    def _get_bmax(self):
        if self._number_of_elements == 1:
            return self._bounds[1]
        else:
            return self._bounds[0][1]

    def _set_bmax(self, arg):
        if self._number_of_elements == 1:
            self._bounds = (self.bmin, arg)
        else:
            self._bounds = ((self.bmin, arg),) * self._number_of_elements
        # Update the value to take into account the new bounds
        self.value = self.value
    bmax = property(_get_bmax, _set_bmax)

    @property
    def _number_of_elements(self):
        return self.__number_of_elements

    @_number_of_elements.setter
    def _number_of_elements(self, arg):
        # Do nothing if the number of arguments stays the same
        if self.__number_of_elements == arg:
            return
        if arg <= 1:
            raise ValueError("Please provide an integer number equal "
                             "or greater to 1")
        self._bounds = ((self.bmin, self.bmax),) * arg
        self.__number_of_elements = arg

        if arg == 1:
            self._Parameter__value = 0
        else:
            self._Parameter__value = (0,) * arg
        if self.component is not None:
            self.component.update_number_parameters()

    @property
    def ext_bounded(self):
        return self.__ext_bounded

    @ext_bounded.setter
    def ext_bounded(self, arg):
        if arg is not self.__ext_bounded:
            self.__ext_bounded = arg
            # Update the value to take into account the new bounds
            self.value = self.value

    @property
    def ext_force_positive(self):
        return self.__ext_force_positive

    @ext_force_positive.setter
    def ext_force_positive(self, arg):
        if arg is not self.__ext_force_positive:
            self.__ext_force_positive = arg
            # Update the value to take into account the new bounds
            self.value = self.value

    def store_current_value_in_array(self):
        """Store the value and std attributes.

        See also
        --------
        fetch, assign_current_value_to_all

        """
        indices = self._axes_manager.indices[::-1]
        # If it is a single spectrum indices is ()
        if not indices:
            indices = (0,)
        self.map['values'][indices] = self.value
        self.map['is_set'][indices] = True
        if self.std is not None:
            self.map['std'][indices] = self.std

    def fetch(self):
        """Fetch the stored value and std attributes.


        See Also
        --------
        store_current_value_in_array, assign_current_value_to_all

        """
        indices = self._axes_manager.indices[::-1]
        # If it is a single spectrum indices is ()
        if not indices:
            indices = (0,)
        if self.map['is_set'][indices]:
            self.value = self.map['values'][indices]
            self.std = self.map['std'][indices]

    def assign_current_value_to_all(self, mask=None):
        '''Assign the current value attribute to all the  indices

        Parameters
        ----------
        mask: {None, boolean numpy array}
            Set only the indices that are not masked i.e. where
            mask is False.

        See Also
        --------
        store_current_value_in_array, fetch

        '''
        if mask is None:
            mask = np.zeros(self.map.shape, dtype='bool')
        self.map['values'][mask == False] = self.value
        self.map['is_set'][mask == False] = True

    def _create_array(self):
        """Create the map array to store the information in
        multidimensional datasets.

        """
        shape = self._axes_manager._navigation_shape_in_array
        if len(shape) == 1 and shape[0] == 0:
            shape = [1, ]
        dtype_ = np.dtype([
            ('values', 'float', self._number_of_elements),
            ('std', 'float', self._number_of_elements),
            ('is_set', 'bool', 1)])
        if (self.map is None or self.map.shape != shape or
                self.map.dtype != dtype_):
            self.map = np.zeros(shape, dtype_)
            self.map['std'].fill(np.nan)
            # TODO: in the future this class should have access to
            # axes manager and should be able to fetch its own
            # values. Until then, the next line is necessary to avoid
            # erros when self.std is defined and the shape is different
            # from the newly defined arrays
            self.std = None

    def as_signal(self, field='values'):
        """Get a parameter map as a signal object.

        Please note that this method only works when the navigation
        dimension is greater than 0.

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

        s = Signal(data=self.map[field],
                   axes=self._axes_manager._get_navigation_axes_dicts())
        s.metadata.General.title = ("%s parameter" % self.name
                                    if self.component is None
                                    else "%s parameter of %s component" %
                                    (self.name, self.component.name))
        for axis in s.axes_manager._axes:
            axis.navigate = False
        if self._number_of_elements > 1:
            s.axes_manager.append_axis(
                size=self._number_of_elements,
                name=self.name,
                navigate=True)
        return s

    def plot(self):
        self.as_signal().plot()

    def export(self, folder=None, name=None, format=None,
               save_std=False):
        '''Save the data to a file.

        All the arguments are optional.

        Parameters
        ----------
        folder : str or None
            The path to the folder where the file will be saved.
             If `None` the current folder is used by default.
        name : str or None
            The name of the file. If `None` the Components name followed
             by the Parameter `name` attributes will be used by default.
              If a file with the same name exists the name will be
              modified by appending a number to the file path.
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
            self.as_signal(field='std').save(append2pathname(
                filename, '_std'))


class Component(object):
    __axes_manager = None

    def __init__(self, parameter_name_list):
        self.connected_functions = list()
        self.parameters = []
        self.init_parameters(parameter_name_list)
        self._update_free_parameters()
        self.active = True
        self._active_array = None
        self.isbackground = False
        self.convolved = True
        self.parameters = tuple(self.parameters)
        self._name = ''
        self._id_name = self.__class__.__name__
        self._id_version = '1.0'
        self._position = None
        self.model = None

    _active_is_multidimensional = False
    _active = True

    @property
    def active_is_multidimensional(self):
        return self._active_is_multidimensional

    @active_is_multidimensional.setter
    def active_is_multidimensional(self, value):
        if not isinstance(value, bool):
            raise ValueError('Only boolean values are permitted')

        if value == self.active_is_multidimensional:
            warnings.warn(
                '`active_is_multidimensional` already %s for %s' %
                (str(value), self.name), RuntimeWarning)
            return

        if value:  # Turn on
            if self._axes_manager.navigation_size < 2:
                warnings.warn(
                    '`navigation_size` < 2, skipping',
                    RuntimeWarning)
                return
            # Store value at current position
            self._create_active_array()
            self._store_active_value_in_array(self._active)
            self._active_is_multidimensional = True
        else:  # Turn off
            # Get the value at the current position before switching it off
            self._active = self.active
            self._active_array = None
            self._active_is_multidimensional = False

    @property
    def name(self):
        return(self._name)

    @name.setter
    def name(self, value):
        if self.model:
            for component in self.model:
                if value == component.name:
                    if not (component is self):
                        raise ValueError(
                            "Another component already has "
                            "the name " + str(value))
                else:
                    self._name = value
        else:
            self._name = value

    @property
    def _axes_manager(self):
        return self.__axes_manager

    @_axes_manager.setter
    def _axes_manager(self, value):
        for parameter in self.parameters:
            parameter._axes_manager = value
        self.__axes_manager = value

    def connect(self, f):
        if f not in self.connected_functions:
            self.connected_functions.append(f)

    def disconnect(self, f):
        if f in self.connected_functions:
            self.connected_functions.remove(f)

    @property
    def active(self):
        if self.active_is_multidimensional is True:
            # The following should set
            self.active = self._active_array[self._axes_manager.indices[::-1]]
        return self._active

    def _store_active_value_in_array(self, value):
        self._active_array[self._axes_manager.indices[::-1]] = value

    @active.setter
    def active(self, arg):
        if self._active == arg:
            return
        self._active = arg
        if self.active_is_multidimensional is True:
            self._store_active_value_in_array(arg)

        for f in self.connected_functions:
            try:
                f()
            except:
                self.disconnect(f)

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
        i = 0
        for parameter in self.free_parameters:
            i += parameter._number_of_elements
        self._nfree_param = i

    def update_number_parameters(self):
        i = 0
        for parameter in self.parameters:
            i += parameter._number_of_elements
        self.nparam = i
        self._update_free_parameters()

    def fetch_values_from_array(self, p, p_std=None, onlyfree=False):
        if onlyfree is True:
            parameters = self.free_parameters
        else:
            parameters = self.parameters
        i = 0
        for parameter in parameters:
            lenght = parameter._number_of_elements
            parameter.value = (p[i] if lenght == 1 else p[i:i + lenght])
            if p_std is not None:
                parameter.std = (p_std[i] if lenght == 1 else
                                 tuple(p_std[i:i + lenght]))

            i += lenght

    def _create_active_array(self):
        shape = self._axes_manager._navigation_shape_in_array
        if len(shape) == 1 and shape[0] == 0:
            shape = [1, ]
        if (not isinstance(self._active_array, np.ndarray)
                or self._active_array.shape != shape):
            self._active_array = np.ones(shape, dtype=bool)

    def _create_arrays(self):
        if self.active_is_multidimensional:
            self._create_active_array()
        for parameter in self.parameters:
            parameter._create_array()

    def store_current_parameters_in_map(self):
        for parameter in self.parameters:
            parameter.store_current_value_in_array()

    def fetch_stored_values(self, only_fixed=False):
        if self.active_is_multidimensional:
            # Store the stored value in self._active and trigger the connected
            # functions.
            self.active = self.active
        if only_fixed is True:
            parameters = (set(self.parameters) -
                          set(self.free_parameters))
        else:
            parameters = self.parameters
        parameters = [parameter for parameter in parameters
                      if (parameter.twin is None or
                          not isinstance(parameter.twin, Parameter))]
        for parameter in parameters:
            parameter.fetch()

    def plot(self, only_free=True):
        """Plot the value of the parameters of the model

        Parameters
        ----------
        only_free : bool
            If True, only the value of the parameters that are free will
             be plotted

        """
        if only_free:
            parameters = self.free_parameters
        else:
            parameters = self.parameters

        parameters = [k for k in parameters if k.twin is None]
        for parameter in parameters:
            parameter.plot()

    def export(self, folder=None, format=None, save_std=False,
               only_free=True):
        """Plot the value of the parameters of the model

        Parameters
        ----------
        folder : str or None
            The path to the folder where the file will be saved. If
            `None` the
            current folder is used by default.
        format : str
            The format to which the data will be exported. It must be
            the
            extension of any format supported by HyperSpy. If None, the
            default
            format for exporting as defined in the `Preferences` will be
             used.
        save_std : bool
            If True, also the standard deviation will be saved.
        only_free : bool
            If True, only the value of the parameters that are free will
             be
            exported.

        Notes
        -----
        The name of the files will be determined by each the Component
        and
        each Parameter name attributes. Therefore, it is possible to
        customise
        the file names modify the name attributes.

        """
        if only_free:
            parameters = self.free_parameters
        else:
            parameters = self.parameters

        parameters = [k for k in parameters if k.twin is None]
        for parameter in parameters:
            parameter.export(folder=folder, format=format,
                             save_std=save_std,)

    def summary(self):
        for parameter in self.parameters:
            dim = len(parameter.map.squeeze().shape) if parameter.map \
                is not None else 0
            if parameter.twin is None:
                if dim <= 1:
                    print '%s = %s ± %s %s' % (parameter.name,
                                               parameter.value,
                                               parameter.std,
                                               parameter.units)

    def __tempcall__(self, p, x, onlyfree=True):
        self.fetch_values_from_array(p, onlyfree=onlyfree)
        return self.function(x)

    def __call__(self):
        """Returns the corresponding model for the current coordinates

        Returns
        -------
        numpy array
        """

        axis = self.model.axis.axis[self.model.channel_switches]
        component_array = self.function(axis)
        return component_array

    def _component2plot(self, axes_manager, out_of_range2nans=True):
        old_axes_manager = None
        if axes_manager is not self.model.axes_manager:
            old_axes_manager = self.model.axes_manager
            self.model.axes_manager = axes_manager
            self.charge()
        s = self.__call__()
        if not self.active:
            s.fill(np.nan)
        if self.model.spectrum.metadata.Signal.binned is True:
            s *= self.model.spectrum.axes_manager.signal_axes[0].scale
        if old_axes_manager is not None:
            self.model.axes_manager = old_axes_manager
            self.charge()
        if out_of_range2nans is True:
            ns = np.empty((self.model.axis.axis.shape))
            ns.fill(np.nan)
            ns[self.model.channel_switches] = s
            s = ns
        return s

    def set_parameters_free(self, parameter_name_list=None):
        """
        Sets parameters in a component to free.

        Parameters
        ----------
        parameter_name_list : None or list of strings, optional
            If None, will set all the parameters to free.
            If list of strings, will set all the parameters with the same name
            as the strings in parameter_name_list to free.

        Examples
        --------
        >>> v1 = components.Voigt()
        >>> v1.set_parameters_free()
        >>> v1.set_parameters_free(parameter_name_list=['area','centre'])

        See also
        --------
        set_parameters_not_free
        hyperspy.model.Model.set_parameters_free
        hyperspy.model.Model.set_parameters_not_free
        """

        parameter_list = []
        if not parameter_name_list:
            parameter_list = self.parameters
        else:
            for _parameter in self.parameters:
                if _parameter.name in parameter_name_list:
                    parameter_list.append(_parameter)

        for _parameter in parameter_list:
            _parameter.free = True

    def set_parameters_not_free(self, parameter_name_list=None):
        """
        Sets parameters in a component to not free.

        Parameters
        ----------
        parameter_name_list : None or list of strings, optional
            If None, will set all the parameters to not free.
            If list of strings, will set all the parameters with the same name
            as the strings in parameter_name_list to not free.

        Examples
        --------
        >>> v1 = components.Voigt()
        >>> v1.set_parameters_not_free()
        >>> v1.set_parameters_not_free(parameter_name_list=['area','centre'])

        See also
        --------
        set_parameters_free
        hyperspy.model.Model.set_parameters_free
        hyperspy.model.Model.set_parameters_not_free
        """

        parameter_list = []
        if not parameter_name_list:
            parameter_list = self.parameters
        else:
            for _parameter in self.parameters:
                if _parameter.name in parameter_name_list:
                    parameter_list.append(_parameter)

        for _parameter in parameter_list:
            _parameter.free = False
