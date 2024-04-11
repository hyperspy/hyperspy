# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
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

import copy

import numpy as np

from hyperspy.model import BaseModel, ModelComponents

_SIGNAL_RANGE_VALUES = """x1, x2 : None or float
            Start and end of the range in the first axis (horizontal)
            in units.
        y1, y2 : None or float
            Start and end of the range in the second axis (vertical)
            in units.
        """


_SIGNAL_RANGE_PIXELS = """i1, i2 : None or float
            Start and end of the range in the first axis (horizontal)
            in pixels.
        j1, j2 : None or float
            Start and end of the range in the second axis (vertical)
            in pixels.
        """


class Model2D(BaseModel):
    """Model and data fitting for two dimensional signals.

    A model is constructed as a linear combination of
    :mod:`~hyperspy.api.model.components2D` that are added to the model using
    :meth:`~hyperspy.model.BaseModel.append` or
    :meth:`~hyperspy.model.BaseModel.extend`. There are predifined components
    available in the :mod:`~hyperspy.api.model.components2D` module
    and custom components can made using the :class:`~.api.model.components1D.Expression`.
    If needed, new components can be created easily using the code of existing
    components as a template.

    Once defined, the model can be fitted to the data using
    :meth:`~hyperspy.model.BaseModel.fit` or :meth:`~hyperspy.model.BaseModel.multifit`.
    Once the optimizer reaches the convergence criteria or the maximum number
    of iterations the new value of the component parameters are stored in the
    components.

    It is possible to access the components in the model by their name or by
    the index in the model. An example is given at the end of this docstring.

    Methods
    -------
    add_signal_range
    remove_signal_range
    reset_signal_range
    set_signal_range

    Notes
    -----
    Methods are not yet defined for plotting 2D models or using gradient based
    optimisation methods.

    See Also
    --------
    hyperspy.model.BaseModel, hyperspy.models.model1d.Model1D

    """

    _signal_dimension = 2

    def __init__(self, signal2D, dictionary=None):
        super().__init__()
        self.signal = signal2D
        self.axes_manager = self.signal.axes_manager
        self._plot = None
        self._position_widgets = {}
        self._adjust_position_all = None
        self._plot_components = False
        self._suspend_update = False
        self._model_line = None
        self.xaxis, self.yaxis = np.meshgrid(
            self.axes_manager.signal_axes[0].axis, self.axes_manager.signal_axes[1].axis
        )
        self.axes_manager.events.indices_changed.connect(self._on_navigating, [])
        self._channel_switches = np.ones(
            self.axes_manager._signal_shape_in_array, dtype=bool
        )
        self._chisq = signal2D._get_navigation_signal()
        self.chisq.change_dtype("float")
        self.chisq.data.fill(np.nan)
        self.chisq.metadata.General.title = (
            self.signal.metadata.General.title + " chi-squared"
        )
        self._dof = self.chisq._deepcopy_with_new_data(
            np.zeros_like(self.chisq.data, dtype="int")
        )
        self.dof.metadata.General.title = (
            self.signal.metadata.General.title + " degrees of freedom"
        )
        self.free_parameters_boundaries = None
        self._components = ModelComponents(self)
        if dictionary is not None:
            self._load_dictionary(dictionary)
        self._whitelist = {
            "_channel_switches": None,
            "free_parameters_boundaries": None,
            "chisq.data": None,
            "dof.data": None,
        }
        self._slicing_whitelist = {
            "_channel_switches": "isig",
            "chisq.data": "inav",
            "dof.data": "inav",
        }

    def _get_current_data(self, onlyactive=False, component_list=None, binned=None):
        """Returns the corresponding 2D model for the current coordinates

        Parameters
        ----------
        onlyactive : bool
            If True, only the active components will be used to build the
            model.
        component_list : list or None
            If None, the sum of all the components is returned. If list, only
            the provided components are returned
        binned : None
            Not Implemented for Model2D

        Returns
        -------
        numpy array
        """
        if component_list is None:
            component_list = self
        if not isinstance(component_list, (list, tuple)):
            raise ValueError("'Component_list' parameter needs to be a list or None.")

        if onlyactive:
            component_list = [
                component for component in component_list if component.active
            ]

        sum_ = np.zeros_like(self.xaxis)
        if onlyactive is True:
            for component in component_list:  # Cut the parameters list
                if component.active:
                    np.add(sum_, component.function(self.xaxis, self.yaxis), sum_)
        else:
            for component in component_list:  # Cut the parameters list
                np.add(sum_, component.function(self.xaxis, self.yaxis), sum_)

        return sum_[self._channel_switches]

    def _errfunc(self, param, y, weights=None):
        if weights is None:
            weights = 1.0
        errfunc = self._model_function(param).ravel() - y
        return errfunc * weights

    def _set_signal_range_in_pixels(
        self,
        i1=None,
        i2=None,
        j1=None,
        j2=None,
    ):
        """
        Use only the selected range defined in pixels in the
        fitting routine.

        Parameters
        ----------
        %s
        """
        self._backup_channel_switches = copy.copy(self._channel_switches)

        self._channel_switches[:, :] = False
        if i2 is not None:
            i2 += 1
        if j2 is not None:
            j2 += 1
        self._channel_switches[slice(i1, i2), slice(j1, j2)] = True
        self.update_plot(render_figure=True)

    _set_signal_range_in_pixels.__doc__ %= _SIGNAL_RANGE_PIXELS

    def set_signal_range(self, x1=None, x2=None, y1=None, y2=None):
        """
        Use only the selected range defined in its own units in the
        fitting routine.

        Parameters
        ----------
        %s

        See Also
        --------
        add_signal_range, remove_signal_range, reset_signal_range,
        hyperspy.model.BaseModel.set_signal_range_from_mask
        """
        xaxis = self.axes_manager.signal_axes[0]
        yaxis = self.axes_manager.signal_axes[1]
        i_indices = xaxis.value_range_to_indices(x1, x2)
        j_indices = yaxis.value_range_to_indices(y1, y2)
        self._set_signal_range_in_pixels(*(i_indices + j_indices))

    set_signal_range.__doc__ %= _SIGNAL_RANGE_VALUES

    def _remove_signal_range_in_pixels(self, i1=None, i2=None, j1=None, j2=None):
        """
        Removes the data in the given range (pixels) from the data
        range that will be used by the fitting rountine

        Parameters
        ----------
        %s
        """
        if i2 is not None:
            i2 += 1
        if j2 is not None:
            j2 += 1
        self._channel_switches[slice(i1, i2), slice(j1, j2)] = False
        self.update_plot()

    _remove_signal_range_in_pixels.__doc__ %= _SIGNAL_RANGE_PIXELS

    def remove_signal_range(self, x1=None, x2=None, y1=None, y2=None):
        """
        Removes the data in the given range (calibrated values) from
        the data range that will be used by the fitting rountine

        Parameters
        ----------
        %s

        See Also
        --------
        set_signal_range, add_signal_range, reset_signal_range,
        hyperspy.model.BaseModel.set_signal_range_from_mask
        """
        xaxis = self.axes_manager.signal_axes[0]
        yaxis = self.axes_manager.signal_axes[1]
        i_indices = xaxis.value_range_to_indices(x1, x2)
        j_indices = yaxis.value_range_to_indices(y1, y2)
        self._remove_signal_range_in_pixels(*i_indices, *j_indices)

    remove_signal_range.__doc__ %= _SIGNAL_RANGE_VALUES

    def reset_signal_range(self):
        """
        Resets the data range.

        See Also
        --------
        set_signal_range, add_signal_range, remove_signal_range,
        hyperspy.model.BaseModel.set_signal_range_from_mask
        """
        self._set_signal_range_in_pixels()

    def _add_signal_range_in_pixels(self, i1=None, i2=None, j1=None, j2=None):
        """
        Adds the data in the given range from the data range (pixels)
        that will be used by the fitting rountine

        Parameters
        ----------
        %s
        """
        if i2 is not None:
            i2 += 1
        if j2 is not None:
            j2 += 1
        self._channel_switches[slice(i1, i2), slice(j1, j2)] = True
        self.update_plot()

    _add_signal_range_in_pixels.__doc__ %= _SIGNAL_RANGE_PIXELS

    def add_signal_range(self, x1=None, x2=None, y1=None, y2=None):
        """
        Adds the data in the given range from the data range
        (calibrated values) that will be used by the fitting rountine.

        Parameters
        ----------
        %s

        See Also
        --------
        set_signal_range, reset_signal_range, remove_signal_range,
        hyperspy.model.BaseModel.set_signal_range_from_mask
        """
        xaxis = self.axes_manager.signal_axes[0]
        yaxis = self.axes_manager.signal_axes[1]
        i_indices = xaxis.value_range_to_indices(x1, x2)
        j_indices = yaxis.value_range_to_indices(y1, y2)
        self._add_signal_range_in_pixels(*(i_indices + j_indices))

    add_signal_range.__doc__ %= _SIGNAL_RANGE_VALUES

    def _check_analytical_jacobian(self):
        """Check all components have analytical gradients.

        If they do, return True and an empty string.
        If they do not, return False and an error message.
        """
        return False, "Analytical gradients not implemented for Model2D"

    def _jacobian(self, param, y, weights=None):
        raise NotImplementedError

    def _function4odr(self, param, x):
        raise NotImplementedError

    def _jacobian4odr(self, param, x):
        raise NotImplementedError

    def _poisson_likelihood_function(self, param, y, weights=None):
        raise NotImplementedError

    def _gradient_ml(self, param, y, weights=None):
        raise NotImplementedError

    def _gradient_ls(self, param, y, weights=None):
        raise NotImplementedError

    def _huber_loss_function(self, param, y, weights=None, huber_delta=None):
        raise NotImplementedError

    def _gradient_huber(self, param, y, weights=None, huber_delta=None):
        raise NotImplementedError

    def _model2plot(self, axes_manager, out_of_range2nans=True):
        old_axes_manager = None
        if axes_manager is not self.axes_manager:
            old_axes_manager = self.axes_manager
            self.axes_manager = axes_manager
            self.fetch_stored_values()
        s = self._get_current_data(onlyactive=True)
        if old_axes_manager is not None:
            self.axes_manager = old_axes_manager
            self.fetch_stored_values()
        if out_of_range2nans is True:
            ns = np.empty(self.xaxis.shape)
            ns.fill(np.nan)
            ns[np.where(self._channel_switches)] = s.ravel()
            s = ns
        return s

    def plot(self, plot_components=False):
        raise NotImplementedError

    @staticmethod
    def _connect_component_line(component):
        raise NotImplementedError

    @staticmethod
    def _disconnect_component_line(component):
        raise NotImplementedError

    def _plot_component(self, component):
        raise NotImplementedError

    def enable_adjust_position(self, components=None, fix_them=True, show_label=True):
        raise NotImplementedError

    def disable_adjust_position(self):
        raise NotImplementedError
