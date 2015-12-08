# -*- coding: utf-8 -*-
# Copyright 2007-2015 The HyperSpy developers
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

import copy
import os
import tempfile
import warnings
import numbers
import numpy as np
import scipy.odr as odr
from scipy.optimize import (leastsq,
                            fmin,
                            fmin_cg,
                            fmin_ncg,
                            fmin_bfgs,
                            fmin_l_bfgs_b,
                            fmin_tnc,
                            fmin_powell)
from traits.trait_errors import TraitError

from hyperspy.model import BaseModel, ModelComponents
from hyperspy import messages
import hyperspy.drawing.spectrum
from hyperspy.drawing.utils import on_figure_window_close
from hyperspy.external import progressbar
from hyperspy._signals.eels import Spectrum
from hyperspy._signals.image import Image
from hyperspy.defaults_parser import preferences
from hyperspy.axes import generate_axis
from hyperspy.exceptions import WrongObjectError
from hyperspy.decorators import interactive_range_selector
from hyperspy.external.mpfit.mpfit import mpfit
from hyperspy.axes import AxesManager
from hyperspy.drawing.widgets import (DraggableVerticalLine,
                                      DraggableLabel)
from hyperspy.gui.tools import ComponentFit
from hyperspy.component import Component
from hyperspy import components
from hyperspy.signal import Signal
from hyperspy.misc.export_dictionary import (export_to_dictionary,
                                             load_from_dictionary,
                                             parse_flag_string,
                                             reconstruct_object)
from hyperspy.misc.utils import slugify, shorten_name
from hyperspy.misc.slicing import copy_slice_from_whitelist

class Model2D(BaseModel):

    """
    The class for models of two-dimensional signals i.e. images.

    Methods are defined for creating and fitting 2D models but plotting features
    are not yet provided.
    """

    def __init__(self, image, dictionary=None):
        self.image = image
        self.signal = self.image
        self.axes_manager = self.signal.axes_manager
        self.xaxis, self.yaxis = np.meshgrid(
            self.axes_manager.signal_axes[0].axis,
            self.axes_manager.signal_axes[1].axis)
        self.axes_manager.connect(self.fetch_stored_values)
        self.free_parameters_boundaries = None
        self.channel_switches = None
        # self._position_widgets = []
        self._plot = None
        self.chisq = image._get_navigation_signal()
        self.chisq.change_dtype("float")
        self.chisq.data.fill(np.nan)
        self.chisq.metadata.General.title = self.signal.metadata.General.title + \
            ' chi-squared'
        self.dof = self.chisq._deepcopy_with_new_data(
            np.zeros_like(
                self.chisq.data,
                dtype='int'))
        self.dof.metadata.General.title = self.signal.metadata.General.title + \
            ' degrees of freedom'
        # self._suspend_update = False
        self._adjust_position_all = None
        self._plot_components = False
        self.components = ModelComponents(self)
        if dictionary is not None:
            self._load_dictionary(dictionary)
        self.inav = ModelSpecialSlicers(self, True)
        self.isig = ModelSpecialSlicers(self, False)

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, value):
        if isinstance(value, Image):
            self._image = value
        else:
            raise WrongObjectError(str(type(value)), 'Image')

    # TODO: write 2D secific plotting tools
    # def _connect_parameters2update_plot(self):
    #    pass

    # Plotting code to rewrite
    # def _disconnect_parameters2update_plot(self):
    #    pass

    # To rewrite
    # def as_signal(self):
    #    pass

    # Plotting code to rewrite
    # def update_plot(self):
    #    pass

    def __call__(self, onlyactive=False):
        """Returns the corresponding 2D model for the current coordinates

        Parameters
        ----------
        only_active : bool
            If true, only the active components will be used to build the model.

        Returns
        -------
        numpy array
        """

        sum_ = np.zeros_like(self.xaxis)
        if onlyactive is True:
            for component in self:  # Cut the parameters list
                if component.active:
                    np.add(sum_, component.function(self.xaxis, self.yaxis),
                           sum_)
        else:
            for component in self:  # Cut the parameters list
                np.add(sum_, component.function(self.xaxis, self.yaxis),
                       sum_)
        return sum_

    def _errfunc(self, param, y, weights=None):
        if weights is None:
            weights = 1.
        errfunc = self._model_function(param) - y
        return (errfunc * weights).ravel()

    # TODO: The methods below are implemented only for Model1D and should be
    # added eventually also for Model2D. Probably there are smarter ways to do
    # it than redefining every method, but it is structured this way now to make
    # clear what is and isn't available
    def _connect_parameters2update_plot(self):
        raise NotImplementedError

    def _disconnect_parameters2update_plot(self):
        raise NotImplementedError

    def as_signal(self, component_list=None, out_of_range_to_nan=True,
                  show_progressbar=None):
        raise NotImplementedError

    @property
    def _plot_active(self):
        raise NotImplementedError

    def update_plot(self, *args, **kwargs):
        raise NotImplementedError

    def suspend_update(self):
        raise NotImplementedError

    def resume_update(self, update=True):
        raise NotImplementedError

    def _update_model_line(self):
        raise NotImplementedError

    def _set_signal_range_in_pixels(self, i1=None, i2=None):
        raise NotImplementedError

    @interactive_range_selector
    def set_signal_range(self, x1=None, x2=None):
        raise NotImplementedError

    def _remove_signal_range_in_pixels(self, i1=None, i2=None):
        raise NotImplementedError

    @interactive_range_selector
    def remove_signal_range(self, x1=None, x2=None):
        raise NotImplementedError

    def reset_signal_range(self):
        raise NotImplementedError

    def _add_signal_range_in_pixels(self, i1=None, i2=None):
        raise NotImplementedError

    @interactive_range_selector
    def add_signal_range(self, x1=None, x2=None):
        raise NotImplementedError

    def reset_the_signal_range(self):
        raise NotImplementedError

    def _jacobian(self, param, y, weights=None):
        raise NotImplementedError

    def _jacobian4odr(self, param, x):
        raise NotImplementedError

    def _gradient_ml(self, param, y, weights=None):
        raise NotImplementedError

    def _gradient_ls(self, param, y, weights=None):
        raise NotImplementedError

    def plot(self, plot_components=False):
        raise NotImplementedError

    @staticmethod
    def _connect_component_line(component):
        raise NotImplementedError

    @staticmethod
    def _disconnect_component_line(component):
        raise NotImplementedError

    def _connect_component_lines(self):
        raise NotImplementedError

    def _disconnect_component_lines(self):
        raise NotImplementedError

    def _plot_component(self, component):
        raise NotImplementedError

    @staticmethod
    def _update_component_line(component):
        raise NotImplementedError

    def _disable_plot_component(self, component):
        raise NotImplementedError

    def _close_plot(self):
        raise NotImplementedError

    def enable_plot_components(self):
        raise NotImplementedError

    def disable_plot_components(self):
        raise NotImplementedError

    def enable_adjust_position(
            self, components=None, fix_them=True, show_label=True):
        raise NotImplementedError

    def disable_adjust_position(self):
        raise NotImplementedError


