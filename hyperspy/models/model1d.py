# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
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
import scipy
import traits.api as t

import hyperspy.drawing
from hyperspy import signal_tools
from hyperspy.decorators import interactive_range_selector
from hyperspy.events import EventSuppressor
from hyperspy.exceptions import SignalDimensionError
from hyperspy.misc import utils
from hyperspy.model import BaseModel, ModelComponents
from hyperspy.ui_registry import DISPLAY_DT, TOOLKIT_DT, add_gui_method


@add_gui_method(toolkey="hyperspy.Model1D.fit_component")
class ComponentFit(signal_tools.SpanSelectorInSignal1D):
    only_current = t.Bool(True)
    iterpath = t.Enum(
        "flyback",
        "serpentine",
        default="serpentine",
        desc="Define the iterating pattern over the navigation space.",
    )

    def __init__(
        self,
        model,
        component,
        signal_range=None,
        estimate_parameters=True,
        fit_independent=False,
        only_current=True,
        iterpath="serpentine",
        **kwargs,
    ):
        if model.signal.axes_manager.signal_dimension != 1:
            raise SignalDimensionError(model.signal.axes_manager.signal_dimension, 1)

        self.signal = model.signal
        self.axis = self.signal.axes_manager.signal_axes[0]
        self.span_selector = None
        self.model = model
        self.component = component
        self.signal_range = signal_range
        self.estimate_parameters = estimate_parameters
        self.fit_independent = fit_independent
        self.fit_kwargs = kwargs
        self.only_current = only_current
        self.iterpath = iterpath
        if signal_range == "interactive":
            if (
                not hasattr(self.model, "_plot")
                or self.model._plot is None
                or not self.model._plot.is_active
            ):
                self.model.plot()
            self.span_selector_switch(on=True)

    def _fit_fired(self):
        if self.signal_range != "interactive" and self.signal_range is not None:
            self.model.set_signal_range(*self.signal_range)
        elif self.signal_range == "interactive":
            self.model.set_signal_range(self.ss_left_value, self.ss_right_value)

        # Backup "free state" of the parameters and fix all but those
        # of the chosen component
        if self.fit_independent:
            active_state = []
            for component_ in self.model:
                active_state.append(component_.active)
                if component_ is not self.component:
                    component_.active = False
                else:
                    component_.active = True
        else:
            free_state = []
            for component_ in self.model:
                for parameter in component_.parameters:
                    free_state.append(parameter.free)
                    if component_ is not self.component:
                        parameter.free = False

        # Setting reasonable initial value for parameters through
        # the components estimate_parameters function (if it has one)
        only_current = self.only_current
        if self.estimate_parameters:
            if hasattr(self.component, "estimate_parameters"):
                if self.signal_range == "interactive":
                    self.component.estimate_parameters(
                        self.signal,
                        self.ss_left_value,
                        self.ss_right_value,
                        only_current=only_current,
                    )
                elif self.signal_range is not None:
                    self.component.estimate_parameters(
                        self.signal,
                        self.signal_range[0],
                        self.signal_range[1],
                        only_current=only_current,
                    )

        if only_current:
            self.model.fit(**self.fit_kwargs)
        else:
            self.model.multifit(iterpath=self.iterpath, **self.fit_kwargs)

        # Restore the signal range
        if self.signal_range is not None:
            self.model._channel_switches = self.model._backup_channel_switches.copy()

        self.model.update_plot()

        if self.fit_independent:
            for component_ in self.model:
                component_.active = active_state.pop(0)
        else:
            # Restore the "free state" of the components
            for component_ in self.model:
                for parameter in component_.parameters:
                    parameter.free = free_state.pop(0)

    def apply(self):
        self._fit_fired()


class Model1D(BaseModel):
    """Model and data fitting for one dimensional signals.

    A model is constructed as a linear combination of
    :mod:`~hyperspy.api.model.components1D` that are added to the model using
    :meth:`~hyperspy.model.BaseModel.append` or :meth:`~hyperspy.model.BaseModel.extend`.
    There are many predifined components available in the
    :mod:`~hyperspy.api.model.components1D` module. If needed, new
    components can be created easily using the
    :class:`~.api.model.components1D.Expression` component or by
    using the code of existing components as a template.

    Once defined, the model can be fitted to the data using
    :meth:`~hyperspy.model.BaseModel.fit` or
    :meth:`~hyperspy.model.BaseModel.multifit`. Once the optimizer reaches
    the convergence criteria or the maximum number of iterations the new value
    of the component parameters are stored in the components.

    It is possible to access the components in the model by their name or by
    the index in the model. An example is given at the end of this docstring.

    Methods
    -------
    fit_component
    enable_adjust_position
    disable_adjust_position
    plot
    set_signal_range
    remove_signal_range
    reset_signal_range
    add_signal_range

    Examples
    --------
    In the following example we create a histogram from a normal distribution
    and fit it with a gaussian component. It demonstrates how to create
    a model from a :class:`~.api.signals.Signal1D` instance, add
    components to it, adjust the value of the parameters of the components,
    fit the model to the data and access the components in the model.

    >>> s = hs.signals.Signal1D(
    ...    np.random.normal(scale=2, size=10000)).get_histogram()
    >>> g = hs.model.components1D.Gaussian()
    >>> m = s.create_model()
    >>> m.append(g)
    >>> m.print_current_values()
    Model1D:  histogram
    CurrentComponentValues: Gaussian
    Active: True
    Parameter Name |    Free |      Value |        Std |        Min |        Max | Linear
    ============== | ======= | ========== | ========== | ========== | ========== | ======
                 A |    True |        1.0 |       None |        0.0 |       None |   True
            centre |    True |        0.0 |       None |       None |       None |  False
             sigma |    True |        1.0 |       None |        0.0 |       None |  False
    >>> g.centre.value = 3
    >>> m.print_current_values()
    Model1D:  histogram
    CurrentComponentValues: Gaussian
    Active: True
    Parameter Name |    Free |      Value |        Std |        Min |        Max | Linear
    ============== | ======= | ========== | ========== | ========== | ========== | ======
                 A |    True |        1.0 |       None |        0.0 |       None |   True
            centre |    True |        3.0 |       None |       None |       None |  False
             sigma |    True |        1.0 |       None |        0.0 |       None |  False
    >>> g.sigma.value
    1.0
    >>> m.fit() # doctest: +SKIP
    >>> g.sigma.value # doctest: +SKIP
    1.9779042300856682
    >>> m[0].sigma.value # doctest: +SKIP
    1.9779042300856682
    >>> m["Gaussian"].centre.value # doctest: +SKIP
    -0.072121936813224569

    See Also
    --------
    hyperspy.model.BaseModel, hyperspy.models.model2d.Model2D

    """

    _signal_dimension = 1

    def __init__(self, signal1D, dictionary=None):
        super().__init__()
        self._signal = signal1D
        self.axes_manager = self.signal.axes_manager
        self._plot = None
        self._position_widgets = {}
        self._adjust_position_all = None
        self._plot_components = False
        self._suspend_update = False
        self._model_line = None
        self._residual_line = None
        self.axis = self.axes_manager.signal_axes[0]
        self.axes_manager.events.indices_changed.connect(self._on_navigating, [])
        self._channel_switches = np.array([True] * len(self.axis.axis))
        self._chisq = signal1D._get_navigation_signal()
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

    def append(self, thing):
        """
        Add component to Model.

        Parameters
        ----------
        thing : :class:`~.component.Component`
            The component to add to the model.
        """
        cm = self.suspend_update if self._plot_active else utils.dummy_context_manager
        with cm(update_on_resume=False):
            super().append(thing)
        if self._plot_components:
            self._plot_component(thing)
        if self._adjust_position_all:
            self._make_position_adjuster(
                thing, self._adjust_position_all[0], self._adjust_position_all[1]
            )
        if self._plot_active:
            self.signal._plot.signal_plot.update()

    append.__doc__ = BaseModel.append.__doc__

    def remove(self, things):
        things = self._get_component(things)
        if not np.iterable(things):
            things = [things]
        for thing in things:
            parameter = thing._position
            if parameter in self._position_widgets:
                for pw in reversed(self._position_widgets[parameter]):
                    pw.close()
            if hasattr(thing, "_component_line"):
                line = thing._component_line
                line.close()
        super().remove(things)
        self._disconnect_parameters2update_plot(things)

    remove.__doc__ = BaseModel.remove.__doc__

    def _get_current_data(
        self,
        onlyactive=False,
        component_list=None,
        binned=None,
        ignore_channel_switches=False,
    ):
        """
        Returns the corresponding model for the current coordinates

        Parameters
        ----------
        onlyactive : bool
            If True, only the active components will be used to build the
            model.
        component_list : list or None
            If None, the model is constructed with all active components. Otherwise,
            the model is constructed with the components in component_list.
        binned : bool or None
            Specify whether the binned attribute of the signal axes needs to be
            taken into account.
        ignore_channel_switches: bool
            If true, the entire signal axis are returned
            without checking _channel_switches.

        Returns
        -------
        numpy array
        """

        if component_list is None:
            component_list = self
        if not isinstance(component_list, (list, tuple)):
            raise ValueError("'Component_list' parameter need to be a list or None")

        if onlyactive:
            component_list = [
                component for component in component_list if component.active
            ]

        slice_ = slice(None) if ignore_channel_switches else self._channel_switches

        try:
            model_convolved = self.convolved
            convolution_supported = True
        except NotImplementedError:
            convolution_supported = False

        if convolution_supported and model_convolved:
            sum_convolved = np.zeros_like(self._convolution_axis, dtype=float)
            sum_ = np.zeros_like(self.axis.axis, dtype=float)
            for component in component_list:
                if component.convolved:
                    sum_convolved += component.function(self._convolution_axis)
                else:
                    sum_ += component.function(self.axis.axis)
            model_data = sum_ + np.convolve(
                self._signal_to_convolve._get_current_data(self.axes_manager),
                sum_convolved,
                mode="valid",
            )
            model_data = model_data[slice_]
        else:
            axis = self.axis.axis[slice_]
            model_data = np.zeros(len(axis))
            for component in component_list:
                model_data += component.function(axis)

        if binned is None:
            # use self.axis instead of self.signal.axes_manager[-1]
            # to avoid small overhead (~10 us) which isn't negligeable when
            # __call__ is called repeatably, typically when fitting!
            binned = self.axis.is_binned

        if binned:
            if self.axis.is_uniform:
                model_data *= self.axis.scale
            else:
                model_data *= np.gradient(self.axis.axis)
        return model_data

    def _errfunc(self, param, y, weights=None):
        if weights is None:
            weights = 1.0
        errfunc = self._model_function(param) - y
        return errfunc * weights

    def _set_signal_range_in_pixels(self, i1=None, i2=None):
        """
        Use only the selected spectral range in the fitting routine.

        Parameters
        ----------
        i1 : Int
        i2 : Int

        Notes
        -----
        To use the full energy range call the function without arguments.
        """

        self._backup_channel_switches = copy.copy(self._channel_switches)
        self._channel_switches[:] = False
        if i2 is not None:
            i2 += 1
        self._channel_switches[i1:i2] = True
        self.update_plot(render_figure=True)

    def _parse_signal_range_values(self, x1=None, x2=None):
        """
        Parse signal range values to be used by the `set_signal_range`,
        `add_signal_range` and `remove_signal_range` and return sorted indices.
        """
        try:
            x1, x2 = x1
        except TypeError:
            # It was not a ROI, we carry on
            pass
        return self.axis.value_range_to_indices(x1, x2)

    @interactive_range_selector
    def set_signal_range(self, x1=None, x2=None):
        """
        Use only the selected spectral range defined in its own units in the
        fitting routine.

        Parameters
        ----------
        x1, x2 : None or float

        See Also
        --------
        add_signal_range, remove_signal_range, reset_signal_range,
        hyperspy.model.BaseModel.set_signal_range_from_mask
        """
        indices = self._parse_signal_range_values(x1, x2)
        self._set_signal_range_in_pixels(*indices)

    def _remove_signal_range_in_pixels(self, i1=None, i2=None):
        """
        Removes the data in the given range from the data range that
        will be used by the fitting rountine.

        Parameters
        ----------
        i1, i2 : None or integer
        """
        if i2 is not None:
            i2 += 1
        self._channel_switches[i1:i2] = False
        self.update_plot()

    @interactive_range_selector
    def remove_signal_range(self, x1=None, x2=None):
        """
        Removes the data in the given range from the data range that
        will be used by the fitting rountine.

        Parameters
        ----------
        x1, x2 : None or float

        See Also
        --------
        set_signal_range, add_signal_range, reset_signal_range,
        hyperspy.model.BaseModel.set_signal_range_from_mask
        """
        indices = self._parse_signal_range_values(x1, x2)
        self._remove_signal_range_in_pixels(*indices)

    def reset_signal_range(self):
        """
        Resets the data range

        See Also
        --------
        set_signal_range, add_signal_range, remove_signal_range
        """
        self._set_signal_range_in_pixels()

    def _add_signal_range_in_pixels(self, i1=None, i2=None):
        """
        Adds the data in the given range from the data range that
        will be used by the fitting rountine.

        Parameters
        ----------
        i1, i2 : None or integer
        """
        if i2 is not None:
            i2 += 1
        self._channel_switches[i1:i2] = True
        self.update_plot()

    @interactive_range_selector
    def add_signal_range(self, x1=None, x2=None):
        """
        Adds the data in the given range from the data range that
        will be used by the fitting rountine.

        Parameters
        ----------
        x1, x2 : None or float

        See Also
        --------
        set_signal_range, reset_signal_range, remove_signal_range
        """
        indices = self._parse_signal_range_values(x1, x2)
        self._add_signal_range_in_pixels(*indices)

    def _check_analytical_jacobian(self):
        """Check all components have analytical gradients.

        If they do, return True and an empty string.
        If they do not, return False and an error message.
        """
        missing_gradients = []
        for component in self:
            if component.active:
                for parameter in component.free_parameters:
                    if not callable(parameter.grad):
                        missing_gradients.append(parameter)

                    if parameter._twins:
                        for par in parameter._twins:
                            if not callable(par.grad):
                                missing_gradients.append(par)

        if len(missing_gradients) > 0:
            pars = ", ".join(str(x) for x in missing_gradients)
            return False, f"Analytical gradient not available for {pars}"
        else:
            return True, ""

    def _jacobian(self, param, y, weights=None):
        if weights is None:
            weights = 1.0

        counter = 0
        grad = np.zeros(len(self.axis.axis))
        for component in self:  # Cut the parameters list
            if component.active:
                component.fetch_values_from_array(
                    param[counter : counter + component._nfree_param], onlyfree=True
                )
                for parameter in component.free_parameters:
                    if self._convolved and component.convolved:
                        par_grad = np.convolve(
                            parameter.grad(self._convolution_axis),
                            self._signal_to_convolve._get_current_data(
                                self.axes_manager
                            ),
                            mode="valid",
                        )
                    else:
                        par_grad = parameter.grad(self.axis.axis)

                    if parameter._twins:
                        for par in parameter._twins:
                            if self._convolved and component.convolved:
                                par_grad_twin = np.convolve(
                                    par.grad(self._convolution_axis),
                                    self._signal_to_convolve._get_current_data(
                                        self.axes_manager
                                    ),
                                    mode="valid",
                                )
                            else:
                                par_grad_twin = par.grad(self.axis.axis)
                            np.add(par_grad, par_grad_twin, par_grad)

                    grad = np.vstack((grad, par_grad))

                counter += component._nfree_param

        to_return = grad[1:, self._channel_switches] * weights

        if self.axis.is_binned:
            if self.axis.is_uniform:
                to_return *= self.axis.scale
            else:
                to_return *= np.gradient(self.axis.axis)

        return to_return

    def _function4odr(self, x, param):
        return self._model_function(param)

    def _jacobian4odr(self, x, param):
        return self._jacobian(param, x)

    def _poisson_likelihood_function(self, param, y, weights=None):
        """Returns the likelihood function of the model for the given
        data and parameters
        """
        mf = self._model_function(param)
        with np.errstate(invalid="ignore"):
            return -(y * np.log(mf) - mf).sum()

    def _gradient_ml(self, param, y, weights=None):
        mf = self._model_function(param)
        return -(self._jacobian(param, y) * (y / mf - 1)).sum(1)

    def _hessian_ml(self, param, y, weights=None):
        """Calculate the Hessian of the negative log-likelihood for Poisson ML fitting.

        This computes the Fisher Information Matrix, which is used to estimate
        parameter uncertainties via the Cramér-Rao bound.

        Parameters
        ----------
        param : array-like
            Model parameters
        y : array-like
            Observed data
        weights : array-like, optional
            Weights (not used for Poisson fitting)

        Returns
        -------
        hessian : ndarray
            Hessian matrix (Fisher Information Matrix)
        """
        mf = self._model_function(param)
        jac = self._jacobian(param, y)

        # For Poisson likelihood, the Hessian has two terms:
        # H_ij = sum_k [ (y_k / mu_k^2) * J_ki * J_kj + (y_k / mu_k - 1) * H_kij ]
        # where mu_k = model_function(x_k), J_ki = d(mu_k)/d(p_i), H_kij = d^2(mu_k)/d(p_i)d(p_j)

        # First term: (y / mu^2) * J^T @ J
        with np.errstate(divide="ignore", invalid="ignore"):
            weights_hess = y / (mf * mf)
            # Handle division by zero - set to 0 where mf is 0
            weights_hess = np.nan_to_num(weights_hess, nan=0.0, posinf=0.0, neginf=0.0)

        # Weighted Jacobian outer product: sum_k (y_k / mu_k^2) * J_ki * J_kj
        hessian_first = np.einsum("k,ki,kj->ij", weights_hess, jac.T, jac.T)

        # Second term: (y / mu - 1) * d^2(mu)/d(p_i)d(p_j)
        # For most components, the second derivatives are small compared to first term
        # and can be neglected (Gauss-Newton approximation)
        # This is commonly done in practice for Poisson ML fitting

        return hessian_first

    def _hessian_ls(self, param, y, weights=None):
        """Calculate the Hessian matrix for least squares loss function.

        This computes the approximate Hessian using the Gauss-Newton method,
        which is used to estimate parameter uncertainties.

        Parameters
        ----------
        param : array-like
            Model parameters
        y : array-like
            Observed data
        weights : array-like, optional
            Weights for weighted least squares

        Returns
        -------
        hessian : ndarray
            Hessian matrix (Gauss-Newton approximation)
        """
        jac = self._jacobian(param, y)

        # For least squares, the Hessian has two terms:
        # H_ij = sum_k [ J_ki * J_kj + (y_k - f_k) * H_kij ]
        # where J_ki = d(f_k)/d(p_i), H_kij = d^2(f_k)/d(p_i)d(p_j)

        # Gauss-Newton approximation: neglect second term, H_ij ≈ J^T @ W @ J
        # where W is the weight matrix (identity for unweighted case)

        if weights is None:
            # Unweighted case: H = J^T @ J
            hessian = np.dot(jac, jac.T)
        else:
            # Weighted case: H = J^T @ W @ J
            # Convert weights to appropriate shape if needed
            if np.isscalar(weights):
                weights = np.full(len(y), weights)
            elif weights.ndim == 0:
                weights = np.full(len(y), float(weights))

            # Apply weights: sum_k (w_k * J_ki * J_kj)
            hessian = np.einsum("k,ki,kj->ij", weights, jac.T, jac.T)

        return hessian

    def _gradient_ls(self, param, y, weights=None):
        gls = (2 * self._errfunc(param, y, weights) * self._jacobian(param, y)).sum(1)
        return gls

    def _huber_loss_function(self, param, y, weights=None, huber_delta=None):
        if weights is None:
            weights = 1.0
        if huber_delta is None:
            huber_delta = 1.0
        return scipy.special.huber(huber_delta, weights * self._errfunc(param, y)).sum()

    def _gradient_huber(self, param, y, weights=None, huber_delta=None):
        if huber_delta is None:
            huber_delta = 1.0
        return (
            self._jacobian(param, y)
            * np.clip(self._errfunc(param, y, weights), -huber_delta, huber_delta)
        ).sum(axis=1)

    def _hessian_huber(self, param, y, weights=None, huber_delta=None):
        """Calculate the Hessian matrix for Huber loss function.

        This computes the approximate Hessian using the Gauss-Newton method,
        which is used to estimate parameter uncertainties.

        Parameters
        ----------
        param : array-like
            Model parameters
        y : array-like
            Observed data
        weights : array-like, optional
            Weights for weighted fitting
        huber_delta : float, optional
            Delta parameter for Huber loss function (default: 1.0)

        Returns
        -------
        hessian : ndarray
            Hessian matrix (Gauss-Newton approximation)
        """
        if huber_delta is None:
            huber_delta = 1.0

        jac = self._jacobian(param, y)
        residuals = self._errfunc(param, y, weights)

        # For Huber loss, the second derivative w.r.t. residuals is:
        # d²L/dr² = 1 if |r| ≤ δ, 0 if |r| > δ
        # This creates a weight matrix for the Gauss-Newton approximation
        huber_weights = (np.abs(residuals) <= huber_delta).astype(float)

        # Apply additional weights if provided
        if weights is not None:
            if np.isscalar(weights):
                huber_weights = huber_weights * weights
            elif weights.ndim == 0:
                huber_weights = huber_weights * float(weights)
            else:
                huber_weights = huber_weights * weights

        # Gauss-Newton approximation: H = J^T @ W @ J
        # where W is the weight matrix (huber_weights in this case)
        hessian = np.einsum("k,ki,kj->ij", huber_weights, jac.T, jac.T)

        return hessian

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
            ns = np.empty(self.axis.axis.shape)
            ns.fill(np.nan)
            ns[np.where(self._channel_switches)] = s
            s = ns
        return s

    def _residual_for_plot(self, **kwargs):
        """From an model1D object, the original signal is subtracted
        by the model signal then returns the residual
        """

        return self.signal._get_current_data() - self._get_current_data(
            ignore_channel_switches=True
        )

    def plot(self, plot_components=False, plot_residual=False, **kwargs):
        """Plot the current spectrum to the screen and a map with a
        cursor to explore the SI.

        Parameters
        ----------
        plot_components : bool
            If True, add a line per component to the signal figure.
        plot_residual : bool
            If True, add a residual line (Signal - Model) to the signal figure.
        **kwargs : dict
            All extra keyword arguements are passed to
            :meth:`~.api.signals.Signal1D.plot`
        """

        # If new coordinates are assigned
        self.signal.plot(**kwargs)
        _plot = self.signal._plot
        l1 = _plot.signal_plot.ax_lines[0]
        color = l1.line.get_color()
        l1.set_line_properties(color=color, type="scatter")

        l2 = hyperspy.drawing.signal1d.Signal1DLine()
        l2.data_function = self._model2plot
        l2.set_line_properties(color="blue", type="line")
        # Add the line to the figure
        _plot.signal_plot.add_line(l2)
        l2.plot()
        _plot.signal_plot.events.closed.connect(self._close_plot, [])

        self._model_line = l2
        self._plot = self.signal._plot

        # Optional to plot the residual of (Signal - Model)
        if plot_residual:
            l3 = hyperspy.drawing.signal1d.Signal1DLine()
            # _residual_for_plot outputs the residual (Signal - Model)
            l3.data_function = self._residual_for_plot
            l3.set_line_properties(color="green", type="line")
            # Add the line to the figure
            _plot.signal_plot.add_line(l3)
            l3.plot()
            # Quick access to _residual_line if needed
            self._residual_line = l3

        self._connect_parameters2update_plot(self)

        if plot_components is True:
            self.enable_plot_components()
        else:
            # If we were plotted before, make sure we reset state here
            self.disable_plot_components()
        # If we were plotted before, make sure we reset state here
        self.disable_adjust_position()

    @staticmethod
    def _connect_component_line(component):
        if hasattr(component, "_component_line"):
            f = component._component_line._auto_update_line
            component.events.active_changed.connect(f, [])
            for parameter in component.parameters:
                parameter.events.value_changed.connect(f, [])

    @staticmethod
    def _disconnect_component_line(component):
        if hasattr(component, "_component_line"):
            f = component._component_line._auto_update_line
            component.events.active_changed.disconnect(f)
            for parameter in component.parameters:
                parameter.events.value_changed.disconnect(f)

    @staticmethod
    def _update_component_line(component):
        if hasattr(component, "_component_line"):
            component._component_line.update(render_figure=False, update_ylimits=False)

    def _plot_component(self, component):
        line = hyperspy.drawing.signal1d.Signal1DLine()
        line.data_function = component._component2plot
        # Add the line to the figure
        self._plot.signal_plot.add_line(line)
        line.plot()
        component._component_line = line
        self._connect_component_line(component)

    def _disable_plot_component(self, component):
        self._disconnect_component_line(component)
        if hasattr(component, "_component_line"):
            component._component_line.close()
            del component._component_line
        self._plot_components = False

    def _close_plot(self):
        self.disable_adjust_position()
        super()._close_plot()

    def enable_plot_components(self):
        if self._plot is None or self._plot_components:  # pragma: no cover
            return
        self._plot_components = True
        for component in [component for component in self if component.active]:
            self._plot_component(component)

    enable_plot_components.__doc__ = BaseModel.enable_plot_components.__doc__

    def disable_plot_components(self):
        self._plot_components = False
        if self._plot is None:  # pragma: no cover
            return
        for component in self:
            self._disable_plot_component(component)

    disable_plot_components.__doc__ = BaseModel.disable_plot_components.__doc__

    def enable_adjust_position(self, components=None, fix_them=True, show_label=True):
        """Allow changing the *x* position of component by dragging
        a vertical line that is plotted in the signal model figure

        Parameters
        ----------
        components : None, list of :class:`~.component.Component`
            If None, the position of all the active components of the
            model that has a well defined *x* position with a value
            in the axis range will get a position adjustment line.
            Otherwise the feature is added only to the given components.
            The components can be specified by name, index or themselves.
        fix_them : bool, default True
            If True the position parameter of the components will be
            temporarily fixed until adjust position is disable.
            This can
            be useful to iteratively adjust the component positions and
            fit the model.
        show_label : bool, default True
            If True, a label showing the component name is added to the
            plot next to the vertical line.

        See Also
        --------
        disable_adjust_position

        """
        if self._plot is None or not self._plot.is_active:
            self.plot()
        if self._position_widgets:
            self.disable_adjust_position()
        if components:
            components = [self._get_component(x) for x in components]
        else:
            self._adjust_position_all = (fix_them, show_label)

        components = components if components else self
        if not components:
            # The model does not have components so we do nothing
            return
        components = [component for component in components if component.active]
        for component in components:
            self._make_position_adjuster(component, fix_them, show_label)

    def _make_position_adjuster(self, component, fix_it, show_label):
        if component._position is None or component._position.twin:
            return
        axis = self.axes_manager.signal_axes[0]
        # Create the vertical line and labels
        widgets = [hyperspy.drawing.widgets.VerticalLineWidget(self.axes_manager)]
        if show_label:
            label = hyperspy.drawing.widgets.LabelWidget(self.axes_manager)
            label.string = component._get_short_description().replace(" component", "")
            widgets.append(label)

        self._position_widgets[component._position] = widgets
        for w in widgets:
            # Setup widget
            w.axes = (axis,)
            w.snap_position = False
            w.position = (component._position.value,)
            w.set_mpl_ax(self._plot.signal_plot.ax)
            # Create widget -> parameter connection
            w.events.moved.connect(self._on_widget_moved, {"obj": "widget"})
            # Create parameter -> widget connection
            component._position.events.value_changed.connect(
                w._set_position, dict(value="position")
            )
            # Map relation for close event
            w.events.closed.connect(self._on_position_widget_close, {"obj": "widget"})

    def _reverse_lookup_position_widget(self, widget):
        for parameter, widgets in self._position_widgets.items():
            if widget in widgets:
                return parameter
        raise KeyError()

    def _on_widget_moved(self, widget):
        parameter = self._reverse_lookup_position_widget(widget)
        es = EventSuppressor()
        for w in self._position_widgets[parameter]:
            es.add((w.events.moved, w._set_position))
        with es.suppress():
            parameter.value = widget.position[0]

    def _on_position_widget_close(self, widget):
        widget.events.closed.disconnect(self._on_position_widget_close)
        parameter = self._reverse_lookup_position_widget(widget)
        self._position_widgets[parameter].remove(widget)
        if len(self._position_widgets[parameter]) == 0:
            self._position_widgets.pop(parameter)
        parameter.events.value_changed.disconnect(widget._set_position)
        widget.events.moved.disconnect(self._on_widget_moved)

    def disable_adjust_position(self):
        """Disable the interactive adjust position feature

        See Also
        --------
        enable_adjust_position

        """
        self._adjust_position_all = False
        for pws in list(self._position_widgets.values()):
            # Iteration works on a copied collection, so changes during
            # iteration should be ok
            for pw in reversed(pws):  # pws is reference, so work in reverse
                pw.close()

    def fit_component(
        self,
        component,
        signal_range="interactive",
        estimate_parameters=True,
        fit_independent=False,
        only_current=True,
        display=True,
        toolkit=None,
        **kwargs,
    ):
        component = self._get_component(component)
        cf = ComponentFit(
            self,
            component,
            signal_range,
            estimate_parameters,
            fit_independent,
            only_current,
            **kwargs,
        )
        if signal_range == "interactive":
            return cf.gui(display=display, toolkit=toolkit)
        else:
            cf.apply()

    fit_component.__doc__ = """
        Fit the given component in the given signal range.

        This method is useful to obtain starting parameters for the
        components. Any keyword arguments are passed to the fit method.

        Parameters
        ----------
        component : :class:`~hyperspy.component.Component`
            The component must be in the model, otherwise an exception
            is raised. The component can be specified by name, index or itself.
        signal_range : str, tuple of None
            If ``'interactive'`` the signal range is selected using the span
            selector on the spectrum plot. The signal range can also
            be manually specified by passing a tuple of floats (left, right).
            If None the current signal range is used. Note that ROIs can be used
            in place of a tuple.
        estimate_parameters : bool, default True
            If True will check if the component has an
            estimate_parameters function, and use it to estimate the
            parameters in the component.
        fit_independent : bool, default False
            If True, all other components are disabled. If False, all other
            component paramemeters are fixed.
        %s
        %s
        **kwargs : dict
            All extra keyword arguments are passed to the
            py:meth:`~hyperspy.model.BaseModel.fit` or
            py:meth:`~hyperspy.model.BaseModel.multifit`
            method, depending if ``only_current`` is True or False.

        Examples
        --------
        Signal range set interactivly

        >>> s = hs.signals.Signal1D([0, 1, 2, 4, 8, 4, 2, 1, 0])
        >>> m = s.create_model()
        >>> g1 = hs.model.components1D.Gaussian()
        >>> m.append(g1)
        >>> m.fit_component(g1) # doctest: +SKIP

        Signal range set through direct input

        >>> m.fit_component(g1, signal_range=(1, 7))

        """ % (DISPLAY_DT, TOOLKIT_DT)
