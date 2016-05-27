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

import copy
import os.path
import warnings
import inspect
from contextlib import contextmanager

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

from hyperspy.axes import AxesManager
from hyperspy import io
from hyperspy.drawing import mpl_hie, mpl_hse, mpl_he
from hyperspy.learn.mva import MVA, LearningResults
import hyperspy.misc.utils
from hyperspy.misc.utils import DictionaryTreeBrowser
from hyperspy.drawing import signal as sigdraw
from hyperspy.decorators import auto_replot
from hyperspy.defaults_parser import preferences
from hyperspy.misc.io.tools import ensure_directory
from hyperspy.external.progressbar import progressbar
from hyperspy.exceptions import SignalDimensionError, DataDimensionError
from hyperspy.misc import array_tools
from hyperspy.misc import rgb_tools
from hyperspy.misc.utils import underline
from hyperspy.external.astroML.histtools import histogram
from hyperspy.drawing.utils import animate_legend
from hyperspy.misc.hspy_warnings import VisibleDeprecationWarning
from hyperspy.misc.signal_tools import are_signals_aligned


class MVATools(object):
    # TODO: All of the plotting methods here should move to drawing

    def _plot_factors_or_pchars(self, factors, comp_ids=None,
                                calibrate=True, avg_char=False,
                                same_window=None, comp_label='PC',
                                img_data=None,
                                plot_shifts=True, plot_char=4,
                                cmap=plt.cm.gray, quiver_color='white',
                                vector_scale=1,
                                per_row=3, ax=None):
        """Plot components from PCA or ICA, or peak characteristics

        Parameters
        ----------

        comp_ids : None, int, or list of ints
            if None, returns maps of all components.
            if int, returns maps of components with ids from 0 to given
            int.
            if list of ints, returns maps of components with ids in
            given list.
        calibrate : bool
            if True, plots are calibrated according to the data in the
            axes
            manager.
        same_window : bool
            if True, plots each factor to the same window.  They are
            not scaled.
        comp_label : string, the label that is either the plot title
        (if plotting in
            separate windows) or the label in the legend (if plotting
            in the
            same window)
        cmap : a matplotlib colormap
            The colormap used for factor images or
            any peak characteristic scatter map
            overlay.

        Parameters only valid for peak characteristics (or pk char factors):
        --------------------------------------------------------------------

        img_data - 2D numpy array,
            The array to overlay peak characteristics onto.  If None,
            defaults to the average image of your stack.

        plot_shifts - bool, default is True
            If true, plots a quiver (arrow) plot showing the shifts for
            each
            peak present in the component being plotted.

        plot_char - None or int
            If int, the id of the characteristic to plot as the colored
            scatter plot.
            Possible components are:
               4: peak height
               5: peak orientation
               6: peak eccentricity

       quiver_color : any color recognized by matplotlib
           Determines the color of vectors drawn for
           plotting peak shifts.

       vector_scale : integer or None
           Scales the quiver plot arrows.  The vector
           is defined as one data unit along the X axis.
           If shifts are small, set vector_scale so
           that when they are multiplied by vector_scale,
           they are on the scale of the image plot.
           If None, uses matplotlib's autoscaling.

        """
        if same_window is None:
            same_window = preferences.MachineLearning.same_window
        if comp_ids is None:
            comp_ids = range(factors.shape[1])

        elif not hasattr(comp_ids, '__iter__'):
            comp_ids = range(comp_ids)

        n = len(comp_ids)
        if same_window:
            rows = int(np.ceil(n / float(per_row)))

        fig_list = []

        if n < per_row:
            per_row = n

        if same_window and self.axes_manager.signal_dimension == 2:
            f = plt.figure(figsize=(4 * per_row, 3 * rows))
        else:
            f = plt.figure()
        for i in range(len(comp_ids)):
            if self.axes_manager.signal_dimension == 1:
                if same_window:
                    ax = plt.gca()
                else:
                    if i > 0:
                        f = plt.figure()
                    ax = f.add_subplot(111)
                ax = sigdraw._plot_1D_component(
                    factors=factors,
                    idx=comp_ids[i],
                    axes_manager=self.axes_manager,
                    ax=ax,
                    calibrate=calibrate,
                    comp_label=comp_label,
                    same_window=same_window)
                if same_window:
                    plt.legend(ncol=factors.shape[1] // 2, loc='best')
            elif self.axes_manager.signal_dimension == 2:
                if same_window:
                    ax = f.add_subplot(rows, per_row, i + 1)
                else:
                    if i > 0:
                        f = plt.figure()
                    ax = f.add_subplot(111)

                sigdraw._plot_2D_component(factors=factors,
                                           idx=comp_ids[i],
                                           axes_manager=self.axes_manager,
                                           calibrate=calibrate, ax=ax,
                                           cmap=cmap, comp_label=comp_label)
            if not same_window:
                fig_list.append(f)
        try:
            plt.tight_layout()
        except:
            pass
        if not same_window:
            return fig_list
        else:
            return f

    def _plot_loadings(self, loadings, comp_ids=None, calibrate=True,
                       same_window=None, comp_label=None,
                       with_factors=False, factors=None,
                       cmap=plt.cm.gray, no_nans=False, per_row=3):
        if same_window is None:
            same_window = preferences.MachineLearning.same_window
        if comp_ids is None:
            comp_ids = range(loadings.shape[0])

        elif not hasattr(comp_ids, '__iter__'):
            comp_ids = range(comp_ids)

        n = len(comp_ids)
        if same_window:
            rows = int(np.ceil(n / float(per_row)))

        fig_list = []

        if n < per_row:
            per_row = n

        if same_window and self.axes_manager.signal_dimension == 2:
            f = plt.figure(figsize=(4 * per_row, 3 * rows))
        else:
            f = plt.figure()

        for i in range(n):
            if self.axes_manager.navigation_dimension == 1:
                if same_window:
                    ax = plt.gca()
                else:
                    if i > 0:
                        f = plt.figure()
                    ax = f.add_subplot(111)
            elif self.axes_manager.navigation_dimension == 2:
                if same_window:
                    ax = f.add_subplot(rows, per_row, i + 1)
                else:
                    if i > 0:
                        f = plt.figure()
                    ax = f.add_subplot(111)
            sigdraw._plot_loading(
                loadings, idx=comp_ids[i], axes_manager=self.axes_manager,
                no_nans=no_nans, calibrate=calibrate, cmap=cmap,
                comp_label=comp_label, ax=ax, same_window=same_window)
            if not same_window:
                fig_list.append(f)
        try:
            plt.tight_layout()
        except:
            pass
        if not same_window:
            if with_factors:
                return fig_list, self._plot_factors_or_pchars(
                    factors, comp_ids=comp_ids, calibrate=calibrate,
                    same_window=same_window, comp_label=comp_label,
                    per_row=per_row)
            else:
                return fig_list
        else:
            if self.axes_manager.navigation_dimension == 1:
                plt.legend(ncol=loadings.shape[0] // 2, loc='best')
                animate_legend()
            if with_factors:
                return f, self._plot_factors_or_pchars(factors,
                                                       comp_ids=comp_ids,
                                                       calibrate=calibrate,
                                                       same_window=same_window,
                                                       comp_label=comp_label,
                                                       per_row=per_row)
            else:
                return f

    def _export_factors(self,
                        factors,
                        folder=None,
                        comp_ids=None,
                        multiple_files=None,
                        save_figures=False,
                        save_figures_format='png',
                        factor_prefix=None,
                        factor_format=None,
                        comp_label=None,
                        cmap=plt.cm.gray,
                        plot_shifts=True,
                        plot_char=4,
                        img_data=None,
                        same_window=False,
                        calibrate=True,
                        quiver_color='white',
                        vector_scale=1,
                        no_nans=True, per_row=3):

        from hyperspy._signals.image import Image
        from hyperspy._signals.spectrum import Spectrum

        if multiple_files is None:
            multiple_files = preferences.MachineLearning.multiple_files

        if factor_format is None:
            factor_format = preferences.MachineLearning.\
                export_factors_default_file_format

        # Select the desired factors
        if comp_ids is None:
            comp_ids = range(factors.shape[1])
        elif not hasattr(comp_ids, '__iter__'):
            comp_ids = range(comp_ids)
        mask = np.zeros(factors.shape[1], dtype=np.bool)
        for idx in comp_ids:
            mask[idx] = 1
        factors = factors[:, mask]

        if save_figures is True:
            plt.ioff()
            fac_plots = self._plot_factors_or_pchars(factors,
                                                     comp_ids=comp_ids,
                                                     same_window=same_window,
                                                     comp_label=comp_label,
                                                     img_data=img_data,
                                                     plot_shifts=plot_shifts,
                                                     plot_char=plot_char,
                                                     cmap=cmap,
                                                     per_row=per_row,
                                                     quiver_color=quiver_color,
                                                     vector_scale=vector_scale)
            for idx in range(len(comp_ids)):
                filename = '%s_%02i.%s' % (factor_prefix, comp_ids[idx],
                                           save_figures_format)
                if folder is not None:
                    filename = os.path.join(folder, filename)
                ensure_directory(filename)
                _args = {'dpi': 600,
                         'format': save_figures_format}
                fac_plots[idx].savefig(filename, **_args)
            plt.ion()

        elif multiple_files is False:
            if self.axes_manager.signal_dimension == 2:
                # factor images
                axes_dicts = []
                axes = self.axes_manager.signal_axes[::-1]
                shape = (axes[1].size, axes[0].size)
                factor_data = np.rollaxis(
                    factors.reshape((shape[0], shape[1], -1)), 2)
                axes_dicts.append(axes[0].get_axis_dictionary())
                axes_dicts.append(axes[1].get_axis_dictionary())
                axes_dicts.append({'name': 'factor_index',
                                   'scale': 1.,
                                   'offset': 0.,
                                   'size': int(factors.shape[1]),
                                   'units': 'factor',
                                   'index_in_array': 0, })
                s = Image(factor_data,
                          axes=axes_dicts,
                          metadata={
                              'General': {'title': '%s from %s' % (
                                  factor_prefix,
                                  self.metadata.General.title),
                              }})
            elif self.axes_manager.signal_dimension == 1:
                axes = [self.axes_manager.signal_axes[0].get_axis_dictionary(),
                        {'name': 'factor_index',
                         'scale': 1.,
                         'offset': 0.,
                         'size': int(factors.shape[1]),
                         'units': 'factor',
                         'index_in_array': 0,
                         }]
                axes[0]['index_in_array'] = 1
                s = Spectrum(
                    factors.T, axes=axes, metadata={
                        "General": {
                            'title': '%s from %s' %
                            (factor_prefix, self.metadata.General.title), }})
            filename = '%ss.%s' % (factor_prefix, factor_format)
            if folder is not None:
                filename = os.path.join(folder, filename)
            s.save(filename)
        else:  # Separate files
            if self.axes_manager.signal_dimension == 1:

                axis_dict = self.axes_manager.signal_axes[0].\
                    get_axis_dictionary()
                axis_dict['index_in_array'] = 0
                for dim, index in zip(comp_ids, range(len(comp_ids))):
                    s = Spectrum(factors[:, index],
                                 axes=[axis_dict, ],
                                 metadata={
                                     "General": {'title': '%s from %s' % (
                                         factor_prefix,
                                         self.metadata.General.title),
                                     }})
                    filename = '%s-%i.%s' % (factor_prefix,
                                             dim,
                                             factor_format)
                    if folder is not None:
                        filename = os.path.join(folder, filename)
                    s.save(filename)

            if self.axes_manager.signal_dimension == 2:
                axes = self.axes_manager.signal_axes
                axes_dicts = [axes[0].get_axis_dictionary(),
                              axes[1].get_axis_dictionary()]
                axes_dicts[0]['index_in_array'] = 0
                axes_dicts[1]['index_in_array'] = 1

                factor_data = factors.reshape(
                    self.axes_manager._signal_shape_in_array + [-1, ])

                for dim, index in zip(comp_ids, range(len(comp_ids))):
                    im = Image(factor_data[..., index],
                               axes=axes_dicts,
                               metadata={
                                   "General": {'title': '%s from %s' % (
                                       factor_prefix,
                                       self.metadata.General.title),
                                   }})
                    filename = '%s-%i.%s' % (factor_prefix,
                                             dim,
                                             factor_format)
                    if folder is not None:
                        filename = os.path.join(folder, filename)
                    im.save(filename)

    def _export_loadings(self,
                         loadings,
                         folder=None,
                         comp_ids=None,
                         multiple_files=None,
                         loading_prefix=None,
                         loading_format=None,
                         save_figures_format='png',
                         comp_label=None,
                         cmap=plt.cm.gray,
                         save_figures=False,
                         same_window=False,
                         calibrate=True,
                         no_nans=True,
                         per_row=3):

        from hyperspy._signals.image import Image
        from hyperspy._signals.spectrum import Spectrum

        if multiple_files is None:
            multiple_files = preferences.MachineLearning.multiple_files

        if loading_format is None:
            loading_format = preferences.MachineLearning.\
                export_loadings_default_file_format

        if comp_ids is None:
            comp_ids = range(loadings.shape[0])
        elif not hasattr(comp_ids, '__iter__'):
            comp_ids = range(comp_ids)
        mask = np.zeros(loadings.shape[0], dtype=np.bool)
        for idx in comp_ids:
            mask[idx] = 1
        loadings = loadings[mask]

        if save_figures is True:
            plt.ioff()
            sc_plots = self._plot_loadings(loadings, comp_ids=comp_ids,
                                           calibrate=calibrate,
                                           same_window=same_window,
                                           comp_label=comp_label,
                                           cmap=cmap, no_nans=no_nans,
                                           per_row=per_row)
            for idx in range(len(comp_ids)):
                filename = '%s_%02i.%s' % (loading_prefix, comp_ids[idx],
                                           save_figures_format)
                if folder is not None:
                    filename = os.path.join(folder, filename)
                ensure_directory(filename)
                _args = {'dpi': 600,
                         'format': save_figures_format}
                sc_plots[idx].savefig(filename, **_args)
            plt.ion()
        elif multiple_files is False:
            if self.axes_manager.navigation_dimension == 2:
                axes_dicts = []
                axes = self.axes_manager.navigation_axes[::-1]
                shape = (axes[1].size, axes[0].size)
                loading_data = loadings.reshape((-1, shape[0], shape[1]))
                axes_dicts.append(axes[0].get_axis_dictionary())
                axes_dicts[0]['index_in_array'] = 1
                axes_dicts.append(axes[1].get_axis_dictionary())
                axes_dicts[1]['index_in_array'] = 2
                axes_dicts.append({'name': 'loading_index',
                                   'scale': 1.,
                                   'offset': 0.,
                                   'size': int(loadings.shape[0]),
                                   'units': 'factor',
                                   'index_in_array': 0, })
                s = Image(loading_data,
                          axes=axes_dicts,
                          metadata={
                              "General": {'title': '%s from %s' % (
                                  loading_prefix,
                                  self.metadata.General.title),
                              }})
            elif self.axes_manager.navigation_dimension == 1:
                cal_axis = self.axes_manager.navigation_axes[0].\
                    get_axis_dictionary()
                cal_axis['index_in_array'] = 1
                axes = [{'name': 'loading_index',
                         'scale': 1.,
                         'offset': 0.,
                         'size': int(loadings.shape[0]),
                         'units': 'comp_id',
                         'index_in_array': 0, },
                        cal_axis]
                s = Image(loadings,
                          axes=axes,
                          metadata={
                              "General": {'title': '%s from %s' % (
                                  loading_prefix,
                                  self.metadata.General.title),
                              }})
            filename = '%ss.%s' % (loading_prefix, loading_format)
            if folder is not None:
                filename = os.path.join(folder, filename)
            s.save(filename)
        else:  # Separate files
            if self.axes_manager.navigation_dimension == 1:
                axis_dict = self.axes_manager.navigation_axes[0].\
                    get_axis_dictionary()
                axis_dict['index_in_array'] = 0
                for dim, index in zip(comp_ids, range(len(comp_ids))):
                    s = Spectrum(loadings[index],
                                 axes=[axis_dict, ])
                    filename = '%s-%i.%s' % (loading_prefix,
                                             dim,
                                             loading_format)
                    if folder is not None:
                        filename = os.path.join(folder, filename)
                    s.save(filename)
            elif self.axes_manager.navigation_dimension == 2:
                axes_dicts = []
                axes = self.axes_manager.navigation_axes[::-1]
                shape = (axes[0].size, axes[1].size)
                loading_data = loadings.reshape((-1, shape[0], shape[1]))
                axes_dicts.append(axes[0].get_axis_dictionary())
                axes_dicts[0]['index_in_array'] = 0
                axes_dicts.append(axes[1].get_axis_dictionary())
                axes_dicts[1]['index_in_array'] = 1
                for dim, index in zip(comp_ids, range(len(comp_ids))):
                    s = Image(loading_data[index, ...],
                              axes=axes_dicts,
                              metadata={
                                  "General": {'title': '%s from %s' % (
                                      loading_prefix,
                                      self.metadata.General.title),
                                  }})
                    filename = '%s-%i.%s' % (loading_prefix,
                                             dim,
                                             loading_format)
                    if folder is not None:
                        filename = os.path.join(folder, filename)
                    s.save(filename)

    def plot_decomposition_factors(self,
                                   comp_ids=None,
                                   calibrate=True,
                                   same_window=None,
                                   comp_label='Decomposition factor',
                                   per_row=3):
        """Plot factors from a decomposition.

        Parameters
        ----------

        comp_ids : None, int, or list of ints
            if None, returns maps of all components.
            if int, returns maps of components with ids from 0 to given
            int.
            if list of ints, returns maps of components with ids in
            given list.

        calibrate : bool
            if True, calibrates plots where calibration is available
            from
            the axes_manager.  If False, plots are in pixels/channels.

        same_window : bool
            if True, plots each factor to the same window.  They are
            not scaled.

        comp_label : string, the label that is either the plot title
        (if plotting in
            separate windows) or the label in the legend (if plotting
            in the
            same window)

        cmap : The colormap used for the factor image, or for peak
            characteristics, the colormap used for the scatter plot of
            some peak characteristic.

        per_row : int, the number of plots in each row, when the
        same_window
            parameter is True.

        See Also
        --------
        plot_decomposition_loadings, plot_decomposition_results.

        """
        if self.axes_manager.signal_dimension > 2:
            raise NotImplementedError("This method cannot plot factors of "
                                      "signals of dimension higher than 2."
                                      "You can use "
                                      "`plot_decomposition_results` instead.")
        if same_window is None:
            same_window = preferences.MachineLearning.same_window
        factors = self.learning_results.factors
        if comp_ids is None:
            comp_ids = self.learning_results.output_dimension

        return self._plot_factors_or_pchars(factors,
                                            comp_ids=comp_ids,
                                            calibrate=calibrate,
                                            same_window=same_window,
                                            comp_label=comp_label,
                                            per_row=per_row)

    def plot_bss_factors(self, comp_ids=None, calibrate=True,
                         same_window=None, comp_label='BSS factor',
                         per_row=3):
        """Plot factors from blind source separation results.

        Parameters
        ----------

        comp_ids : None, int, or list of ints
            if None, returns maps of all components.
            if int, returns maps of components with ids from 0 to
            given int.
            if list of ints, returns maps of components with ids in
            given list.

        calibrate : bool
            if True, calibrates plots where calibration is available
            from
            the axes_manager.  If False, plots are in pixels/channels.

        same_window : bool
            if True, plots each factor to the same window.  They are
            not scaled.

        comp_label : string, the label that is either the plot title
        (if plotting in
            separate windows) or the label in the legend (if plotting
            in the
            same window)

        cmap : The colormap used for the factor image, or for peak
            characteristics, the colormap used for the scatter plot of
            some peak characteristic.

        per_row : int, the number of plots in each row, when the
        same_window
            parameter is True.

        See Also
        --------
        plot_bss_loadings, plot_bss_results.

        """
        if self.axes_manager.signal_dimension > 2:
            raise NotImplementedError("This method cannot plot factors of "
                                      "signals of dimension higher than 2."
                                      "You can use "
                                      "`plot_decomposition_results` instead.")

        if same_window is None:
            same_window = preferences.MachineLearning.same_window
        factors = self.learning_results.bss_factors
        return self._plot_factors_or_pchars(factors,
                                            comp_ids=comp_ids,
                                            calibrate=calibrate,
                                            same_window=same_window,
                                            comp_label=comp_label,
                                            per_row=per_row)

    def plot_decomposition_loadings(self,
                                    comp_ids=None,
                                    calibrate=True,
                                    same_window=None,
                                    comp_label='Decomposition loading',
                                    with_factors=False,
                                    cmap=plt.cm.gray,
                                    no_nans=False,
                                    per_row=3):
        """Plot loadings from PCA.

        Parameters
        ----------

        comp_ids : None, int, or list of ints
            if None, returns maps of all components.
            if int, returns maps of components with ids from 0 to
            given int.
            if list of ints, returns maps of components with ids in
            given list.

        calibrate : bool
            if True, calibrates plots where calibration is available
            from
            the axes_manager.  If False, plots are in pixels/channels.

        same_window : bool
            if True, plots each factor to the same window.  They are
            not scaled.

        comp_label : string,
            The label that is either the plot title (if plotting in
            separate windows) or the label in the legend (if plotting
            in the same window). In this case, each loading line can be
            toggled on and off by clicking on the legended line.

        with_factors : bool
            If True, also returns figure(s) with the factors for the
            given comp_ids.

        cmap : matplotlib colormap
            The colormap used for the factor image, or for peak
            characteristics, the colormap used for the scatter plot of
            some peak characteristic.

        no_nans : bool
            If True, removes NaN's from the loading plots.

        per_row : int
            the number of plots in each row, when the same_window
            parameter is True.

        See Also
        --------
        plot_decomposition_factors, plot_decomposition_results.

        """
        if self.axes_manager.navigation_dimension > 2:
            raise NotImplementedError("This method cannot plot loadings of "
                                      "dimension higher than 2."
                                      "You can use "
                                      "`plot_decomposition_results` instead.")
        if same_window is None:
            same_window = preferences.MachineLearning.same_window
        loadings = self.learning_results.loadings.T
        if with_factors:
            factors = self.learning_results.factors
        else:
            factors = None

        if comp_ids is None:
            comp_ids = self.learning_results.output_dimension
        return self._plot_loadings(
            loadings,
            comp_ids=comp_ids,
            with_factors=with_factors,
            factors=factors,
            same_window=same_window,
            comp_label=comp_label,
            cmap=cmap,
            no_nans=no_nans,
            per_row=per_row)

    def plot_bss_loadings(self, comp_ids=None, calibrate=True,
                          same_window=None, comp_label='BSS loading',
                          with_factors=False, cmap=plt.cm.gray,
                          no_nans=False, per_row=3):
        """Plot loadings from ICA

        Parameters
        ----------

        comp_ids : None, int, or list of ints
            if None, returns maps of all components.
            if int, returns maps of components with ids from 0 to
            given int.
            if list of ints, returns maps of components with ids in
            given list.

        calibrate : bool
            if True, calibrates plots where calibration is available
            from
            the axes_manager.  If False, plots are in pixels/channels.

        same_window : bool
            if True, plots each factor to the same window.  They are
            not scaled.

        comp_label : string,
            The label that is either the plot title (if plotting in
            separate windows) or the label in the legend (if plotting
            in the same window). In this case, each loading line can be
            toggled on and off by clicking on the legended line.

        with_factors : bool
            If True, also returns figure(s) with the factors for the
            given comp_ids.

        cmap : matplotlib colormap
            The colormap used for the factor image, or for peak
            characteristics, the colormap used for the scatter plot of
            some peak characteristic.

        no_nans : bool
            If True, removes NaN's from the loading plots.

        per_row : int
            the number of plots in each row, when the same_window
            parameter is True.

        See Also
        --------
        plot_bss_factors, plot_bss_results.

        """
        if self.axes_manager.navigation_dimension > 2:
            raise NotImplementedError("This method cannot plot loadings of "
                                      "dimension higher than 2."
                                      "You can use "
                                      "`plot_bss_results` instead.")
        if same_window is None:
            same_window = preferences.MachineLearning.same_window
        loadings = self.learning_results.bss_loadings.T
        if with_factors:
            factors = self.learning_results.bss_factors
        else:
            factors = None
        return self._plot_loadings(
            loadings,
            comp_ids=comp_ids,
            with_factors=with_factors,
            factors=factors,
            same_window=same_window,
            comp_label=comp_label,
            cmap=cmap,
            no_nans=no_nans,
            per_row=per_row)

    def export_decomposition_results(self, comp_ids=None,
                                     folder=None,
                                     calibrate=True,
                                     factor_prefix='factor',
                                     factor_format=None,
                                     loading_prefix='loading',
                                     loading_format=None,
                                     comp_label=None,
                                     cmap=plt.cm.gray,
                                     same_window=False,
                                     multiple_files=None,
                                     no_nans=True,
                                     per_row=3,
                                     save_figures=False,
                                     save_figures_format='png'):
        """Export results from a decomposition to any of the supported
        formats.

        Parameters
        ----------
        comp_ids : None, int, or list of ints
            if None, returns all components/loadings.
            if int, returns components/loadings with ids from 0 to
            given int.
            if list of ints, returns components/loadings with ids in
            given list.
        folder : str or None
            The path to the folder where the file will be saved.
            If `None` the
            current folder is used by default.
        factor_prefix : string
            The prefix that any exported filenames for
            factors/components
            begin with
        factor_format : string
            The extension of the format that you wish to save to.
        loading_prefix : string
            The prefix that any exported filenames for
            factors/components
            begin with
        loading_format : string
            The extension of the format that you wish to save to.
            Determines
            the kind of output.
                - For image formats (tif, png, jpg, etc.), plots are
                created
                  using the plotting flags as below, and saved at
                  600 dpi.
                  One plot per loading is saved.
                - For multidimensional formats (rpl, hdf5), arrays are
                saved
                  in single files.  All loadings are contained in the
                  one
                  file.
                - For spectral formats (msa), each loading is saved to a
                  separate file.
        multiple_files : Bool
            If True, on exporting a file per factor and per loading will
             be
            created. Otherwise only two files will be created, one for
            the
            factors and another for the loadings. The default value can
            be
            chosen in the preferences.
        save_figures : Bool
            If True the same figures that are obtained when using the
            plot
            methods will be saved with 600 dpi resolution

        Plotting options (for save_figures = True ONLY)
        ----------------------------------------------

        calibrate : bool
            if True, calibrates plots where calibration is available
            from
            the axes_manager.  If False, plots are in pixels/channels.
        same_window : bool
            if True, plots each factor to the same window.
        comp_label : string, the label that is either the plot title
            (if plotting in separate windows) or the label in the legend
            (if plotting in the same window)
        cmap : The colormap used for the factor image, or for peak
            characteristics, the colormap used for the scatter plot of
            some peak characteristic.
        per_row : int, the number of plots in each row, when the
        same_window
            parameter is True.
        save_figures_format : str
            The image format extension.

        See Also
        --------
        get_decomposition_factors,
        get_decomposition_loadings.

        """

        factors = self.learning_results.factors
        loadings = self.learning_results.loadings.T
        self._export_factors(
            factors,
            folder=folder,
            comp_ids=comp_ids,
            calibrate=calibrate,
            multiple_files=multiple_files,
            factor_prefix=factor_prefix,
            factor_format=factor_format,
            comp_label=comp_label,
            save_figures=save_figures,
            cmap=cmap,
            no_nans=no_nans,
            same_window=same_window,
            per_row=per_row,
            save_figures_format=save_figures_format)
        self._export_loadings(
            loadings,
            comp_ids=comp_ids, folder=folder,
            calibrate=calibrate,
            multiple_files=multiple_files,
            loading_prefix=loading_prefix,
            loading_format=loading_format,
            comp_label=comp_label,
            cmap=cmap,
            save_figures=save_figures,
            same_window=same_window,
            no_nans=no_nans,
            per_row=per_row)

    def export_bss_results(self,
                           comp_ids=None,
                           folder=None,
                           calibrate=True,
                           multiple_files=None,
                           save_figures=False,
                           factor_prefix='bss_factor',
                           factor_format=None,
                           loading_prefix='bss_loading',
                           loading_format=None,
                           comp_label=None, cmap=plt.cm.gray,
                           same_window=False,
                           no_nans=True,
                           per_row=3,
                           save_figures_format='png'):
        """Export results from ICA to any of the supported formats.

        Parameters
        ----------
        comp_ids : None, int, or list of ints
            if None, returns all components/loadings.
            if int, returns components/loadings with ids from 0 to given
             int.
            if list of ints, returns components/loadings with ids in
            iven list.
        folder : str or None
            The path to the folder where the file will be saved. If
            `None` the
            current folder is used by default.
        factor_prefix : string
            The prefix that any exported filenames for
            factors/components
            begin with
        factor_format : string
            The extension of the format that you wish to save to.
            Determines
            the kind of output.
                - For image formats (tif, png, jpg, etc.), plots are
                created
                  using the plotting flags as below, and saved at
                  600 dpi.
                  One plot per factor is saved.
                - For multidimensional formats (rpl, hdf5), arrays are
                saved
                  in single files.  All factors are contained in the one
                  file.
                - For spectral formats (msa), each factor is saved to a
                  separate file.

        loading_prefix : string
            The prefix that any exported filenames for
            factors/components
            begin with
        loading_format : string
            The extension of the format that you wish to save to.
        multiple_files : Bool
            If True, on exporting a file per factor and per loading
            will be
            created. Otherwise only two files will be created, one
            for the
            factors and another for the loadings. The default value
            can be
            chosen in the preferences.
        save_figures : Bool
            If True the same figures that are obtained when using the
            plot
            methods will be saved with 600 dpi resolution

        Plotting options (for save_figures = True ONLY)
        ----------------------------------------------
        calibrate : bool
            if True, calibrates plots where calibration is available
            from
            the axes_manager.  If False, plots are in pixels/channels.
        same_window : bool
            if True, plots each factor to the same window.
        comp_label : string
            the label that is either the plot title (if plotting in
            separate windows) or the label in the legend (if plotting
            in the
            same window)
        cmap : The colormap used for the factor image, or for peak
            characteristics, the colormap used for the scatter plot of
            some peak characteristic.
        per_row : int, the number of plots in each row, when the
        same_window
            parameter is True.
        save_figures_format : str
            The image format extension.

        See Also
        --------
        get_bss_factors,
        get_bss_loadings.

        """

        factors = self.learning_results.bss_factors
        loadings = self.learning_results.bss_loadings.T
        self._export_factors(factors,
                             folder=folder,
                             comp_ids=comp_ids,
                             calibrate=calibrate,
                             multiple_files=multiple_files,
                             factor_prefix=factor_prefix,
                             factor_format=factor_format,
                             comp_label=comp_label,
                             save_figures=save_figures,
                             cmap=cmap,
                             no_nans=no_nans,
                             same_window=same_window,
                             per_row=per_row,
                             save_figures_format=save_figures_format)

        self._export_loadings(loadings,
                              comp_ids=comp_ids,
                              folder=folder,
                              calibrate=calibrate,
                              multiple_files=multiple_files,
                              loading_prefix=loading_prefix,
                              loading_format=loading_format,
                              comp_label=comp_label,
                              cmap=cmap,
                              save_figures=save_figures,
                              same_window=same_window,
                              no_nans=no_nans,
                              per_row=per_row,
                              save_figures_format=save_figures_format)

    def _get_loadings(self, loadings):
        from hyperspy.api import signals
        data = loadings.T.reshape(
            (-1,) + self.axes_manager.navigation_shape[::-1])
        signal = signals.Signal(
            data,
            axes=(
                [{"size": data.shape[0], "navigate": True}] +
                self.axes_manager._get_navigation_axes_dicts()))
        signal.set_signal_origin(self.metadata.Signal.signal_origin)
        for axis in signal.axes_manager._axes[1:]:
            axis.navigate = False
        return signal

    def _get_factors(self, factors):
        signal = self.__class__(
            factors.T.reshape((-1,) + self.axes_manager.signal_shape[::-1]),
            axes=[{"size": factors.shape[-1], "navigate": True}] +
            self.axes_manager._get_signal_axes_dicts())
        signal.set_signal_origin(self.metadata.Signal.signal_origin)
        signal.set_signal_type(self.metadata.Signal.signal_type)
        for axis in signal.axes_manager._axes[1:]:
            axis.navigate = False
        return signal

    def get_decomposition_loadings(self):
        """Return the decomposition loadings as a Signal.

        See Also
        -------
        get_decomposition_factors, export_decomposition_results.

        """
        signal = self._get_loadings(self.learning_results.loadings)
        signal.axes_manager._axes[0].name = "Decomposition component index"
        signal.metadata.General.title = "Decomposition loadings of " + \
            self.metadata.General.title
        return signal

    def get_decomposition_factors(self):
        """Return the decomposition factors as a Signal.

        See Also
        -------
        get_decomposition_loadings, export_decomposition_results.

        """
        signal = self._get_factors(self.learning_results.factors)
        signal.axes_manager._axes[0].name = "Decomposition component index"
        signal.metadata.General.title = ("Decomposition factors of " +
                                         self.metadata.General.title)
        return signal

    def get_bss_loadings(self):
        """Return the blind source separtion loadings as a Signal.

        See Also
        -------
        get_bss_factors, export_bss_results.

        """
        signal = self._get_loadings(
            self.learning_results.bss_loadings)
        signal.axes_manager[0].name = "BSS component index"
        signal.metadata.General.title = ("BSS loadings of " +
                                         self.metadata.General.title)
        return signal

    def get_bss_factors(self):
        """Return the blind source separtion factors as a Signal.

        See Also
        -------
        get_bss_loadings, export_bss_results.

        """
        signal = self._get_factors(self.learning_results.bss_factors)
        signal.axes_manager[0].name = "BSS component index"
        signal.metadata.General.title = ("BSS factors of " +
                                         self.metadata.General.title)
        return signal

    def plot_bss_results(self,
                         factors_navigator="auto",
                         loadings_navigator="auto",
                         factors_dim=2,
                         loadings_dim=2,):
        """Plot the blind source separation factors and loadings.

        Unlike `plot_bss_factors` and `plot_bss_loadings`, this method displays
        one component at a time. Therefore it provides a more compact
        visualization than then other two methods.  The loadings and factors
        are displayed in different windows and each has its own
        navigator/sliders to navigate them if they are multidimensional. The
        component index axis is syncronize between the two.

        Parameters
        ----------
        factor_navigator, loadings_navigator : {"auto", None, "spectrum",
        Signal}
            See `plot` documentation for details.
        factors_dim, loadings_dim: int
            Currently HyperSpy cannot plot signals of dimension higher than
            two. Therefore, to visualize the BSS results when the
            factors or the loadings have signal dimension greater than 2
            we can view the data as spectra(images) by setting this parameter
            to 1(2). (Default 2)

        See Also
        --------
        plot_bss_factors, plot_bss_loadings, plot_decomposition_results.

        """
        factors = self.get_bss_factors()
        loadings = self.get_bss_loadings()
        factors.axes_manager._axes[0] = loadings.axes_manager._axes[0]
        if loadings.axes_manager.signal_dimension > 2:
            loadings.axes_manager.set_signal_dimension(loadings_dim)
        if factors.axes_manager.signal_dimension > 2:
            factors.axes_manager.set_signal_dimension(factors_dim)
        loadings.plot(navigator=loadings_navigator)
        factors.plot(navigator=factors_navigator)

    def plot_decomposition_results(self,
                                   factors_navigator="auto",
                                   loadings_navigator="auto",
                                   factors_dim=2,
                                   loadings_dim=2):
        """Plot the decompostion factors and loadings.

        Unlike `plot_factors` and `plot_loadings`, this method displays
        one component at a time. Therefore it provides a more compact
        visualization than then other two methods.  The loadings and factors
        are displayed in different windows and each has its own
        navigator/sliders to navigate them if they are multidimensional. The
        component index axis is syncronize between the two.

        Parameters
        ----------
        factor_navigator, loadings_navigator : {"auto", None, "spectrum",
        Signal}
            See `plot` documentation for details.
        factors_dim, loadings_dim : int
            Currently HyperSpy cannot plot signals of dimension higher than
            two. Therefore, to visualize the BSS results when the
            factors or the loadings have signal dimension greater than 2
            we can view the data as spectra(images) by setting this parameter
            to 1(2). (Default 2)

        See Also
        --------
        plot_factors, plot_loadings, plot_bss_results.

        """
        factors = self.get_decomposition_factors()
        loadings = self.get_decomposition_loadings()
        factors.axes_manager._axes[0] = loadings.axes_manager._axes[0]
        if loadings.axes_manager.signal_dimension > 2:
            loadings.axes_manager.set_signal_dimension(loadings_dim)
        if factors.axes_manager.signal_dimension > 2:
            factors.axes_manager.set_signal_dimension(factors_dim)
        loadings.plot(navigator=loadings_navigator)
        factors.plot(navigator=factors_navigator)


class BaseSignal(MVA,
                 MVATools,):

    _record_by = ""
    _signal_type = ""
    _signal_origin = ""

    def __init__(self, data, **kwds):
        """Create a Signal from a numpy array.

        Parameters
        ----------
        data : numpy array
           The signal data. It can be an array of any dimensions.
        axes : dictionary (optional)
            Dictionary to define the axes (see the
            documentation of the AxesManager class for more details).
        attributes : dictionary (optional)
            A dictionary whose items are stored as attributes.
        metadata : dictionary (optional)
            A dictionary containing a set of parameters
            that will to stores in the `metadata` attribute.
            Some parameters might be mandatory in some cases.
        original_metadata : dictionary (optional)
            A dictionary containing a set of parameters
            that will to stores in the `original_metadata` attribute. It
            typically contains all the parameters that has been
            imported from the original data file.

        """

        self._create_metadata()
        self.learning_results = LearningResults()
        kwds['data'] = data
        self._load_dictionary(kwds)
        self._plot = None
        self.auto_replot = True
        self.inav = SpecialSlicers(self, True)
        self.isig = SpecialSlicers(self, False)

    def _create_metadata(self):
        self.metadata = DictionaryTreeBrowser()
        mp = self.metadata
        mp.add_node("_HyperSpy")
        mp.add_node("General")
        mp.add_node("Signal")
        mp._HyperSpy.add_node("Folding")
        folding = mp._HyperSpy.Folding
        folding.unfolded = False
        folding.signal_unfolded = False
        folding.original_shape = None
        folding.original_axes_manager = None
        mp.Signal.binned = False
        self.original_metadata = DictionaryTreeBrowser()
        self.tmp_parameters = DictionaryTreeBrowser()

    def __repr__(self):
        if self.metadata._HyperSpy.Folding.unfolded:
            unfolded = "unfolded "
        else:
            unfolded = ""
        string = '<'
        string += self.__class__.__name__
        string += ", title: %s" % self.metadata.General.title
        string += ", %sdimensions: %s" % (
            unfolded,
            self.axes_manager._get_dimension_str())

        string += '>'

        return string

    def __getitem__(self, slices, isNavigation=None):
        try:
            len(slices)
        except TypeError:
            slices = (slices,)
        _orig_slices = slices
        if isNavigation is None:
            warnings.warn(
                "Indexing the `Signal` class is deprecated and will be removed "
                "in Hyperspy 1.0. Please use `.isig` and/or `.inav` instead.",
                VisibleDeprecationWarning)

        has_nav = True if isNavigation is None else isNavigation
        has_signal = True if isNavigation is None else not isNavigation

        # Create a deepcopy of self that contains a view of self.data
        _signal = self._deepcopy_with_new_data(self.data)

        nav_idx = [el.index_in_array for el in
                   _signal.axes_manager.navigation_axes]
        signal_idx = [el.index_in_array for el in
                      _signal.axes_manager.signal_axes]

        if not has_signal:
            idx = nav_idx
        elif not has_nav:
            idx = signal_idx
        else:
            idx = nav_idx + signal_idx

        # Add support for Ellipsis
        if Ellipsis in _orig_slices:
            _orig_slices = list(_orig_slices)
            # Expand the first Ellipsis
            ellipsis_index = _orig_slices.index(Ellipsis)
            _orig_slices.remove(Ellipsis)
            _orig_slices = (
                _orig_slices[:ellipsis_index] +
                [slice(None), ] * max(0, len(idx) - len(_orig_slices)) +
                _orig_slices[ellipsis_index:])
            # Replace all the following Ellipses by :
            while Ellipsis in _orig_slices:
                _orig_slices[_orig_slices.index(Ellipsis)] = slice(None)
            _orig_slices = tuple(_orig_slices)

        if len(_orig_slices) > len(idx):
            raise IndexError("too many indices")

        slices = np.array([slice(None,)] *
                          len(_signal.axes_manager._axes))

        slices[idx] = _orig_slices + (slice(None),) * max(
            0, len(idx) - len(_orig_slices))

        array_slices = []
        to_remove = []
        for slice_, axis in zip(slices, _signal.axes_manager._axes):
            if (isinstance(slice_, slice) or
                    (len(_signal.data.shape) - len(to_remove)) < 2):
                array_slices.append(axis._slice_me(slice_))
            else:
                if isinstance(slice_, float):
                    slice_ = axis.value2index(slice_)
                array_slices.append(slice_)
                to_remove.append(axis.index_in_axes_manager)
        for _idx in reversed(sorted(to_remove)):
            _signal._remove_axis(_idx)
        array_slices = tuple(array_slices)
        _signal.data = _signal.data[array_slices]
        if self.metadata.has_item('Signal.Noise_properties.variance'):
            variance = self.metadata.Signal.Noise_properties.variance
            if isinstance(variance, BaseSignal):
                _signal.metadata.Signal.Noise_properties.variance = \
                    variance.__getitem__(_orig_slices, isNavigation)
        _signal.get_dimensions_from_data()

        return _signal

    def __setitem__(self, i, j):
        """x.__setitem__(i, y) <==> x[i]=y

        """
        if isinstance(j, BaseSignal):
            j = j.data
        self.__getitem__(i).data[:] = j

    def _binary_operator_ruler(self, other, op_name):
        exception_message = (
            "Invalid dimensions for this operation")
        if isinstance(other, BaseSignal):
            # Both objects are signals
            oam = other.axes_manager
            sam = self.axes_manager
            if sam.navigation_shape == oam.navigation_shape and \
                    sam.signal_shape == oam.signal_shape:
                # They have the same signal shape.
                # The signal axes are aligned but there is
                # no guarantee that data axes area aligned so we make sure that
                # they are aligned for the operation.
                sdata = self._data_aligned_with_axes
                odata = other._data_aligned_with_axes
                if op_name in INPLACE_OPERATORS:
                    self.data = getattr(sdata, op_name)(odata)
                    self.axes_manager._sort_axes()
                    return self
                else:
                    ns = self._deepcopy_with_new_data(
                        getattr(sdata, op_name)(odata))
                    ns.axes_manager._sort_axes()
                    return ns
            else:
                # Different navigation and/or signal shapes
                if not are_signals_aligned(self, other):
                    raise ValueError(exception_message)
                else:
                    # They are broadcastable but have different number of axes
                    new_nav_axes = []
                    for saxis, oaxis in zip(
                            sam.navigation_axes, oam.navigation_axes):
                        new_nav_axes.append(saxis if saxis.size > 1 or
                                            oaxis.size == 1 else
                                            oaxis)
                    bigger_am = None
                    if sam.navigation_dimension != oam.navigation_dimension:
                        bigger_am = (sam
                                     if sam.navigation_dimension >
                                     oam.navigation_dimension
                                     else oam)
                        new_nav_axes.extend(
                            bigger_am.navigation_axes[len(new_nav_axes):])
                    # Because they are broadcastable and navigation axes come
                    # first in the data array, we don't need to pad the data
                    # array.
                    new_sig_axes = []
                    for saxis, oaxis in zip(
                            sam.signal_axes, oam.signal_axes):
                        new_sig_axes.append(saxis if saxis.size > 1 or
                                            oaxis.size == 1 else
                                            oaxis)
                    if sam.signal_dimension != oam.signal_dimension:
                        bigger_am = (
                            sam if sam.signal_dimension > oam.signal_dimension
                            else oam)
                        new_sig_axes.extend(
                            bigger_am.signal_axes[len(new_sig_axes):])
                    sdim_diff = abs(sam.signal_dimension -
                                    oam.signal_dimension)
                    sdata = self._data_aligned_with_axes
                    odata = other._data_aligned_with_axes
                    if len(new_nav_axes) and sdim_diff:
                        if bigger_am is sam:
                            # Pad odata
                            while sdim_diff:
                                odata = np.expand_dims(
                                    odata, oam.navigation_dimension)
                                sdim_diff -= 1
                        else:
                            # Pad sdata
                            while sdim_diff:
                                sdata = np.expand_dims(
                                    sdata, sam.navigation_dimension)
                                sdim_diff -= 1
                    if op_name in INPLACE_OPERATORS:
                        # This should raise a ValueError if the operation
                        # changes the shape of the object on the left.
                        self.data = getattr(sdata, op_name)(odata)
                        self.axes_manager._sort_axes()
                        return self
                    else:
                        ns = self._deepcopy_with_new_data(
                            getattr(sdata, op_name)(odata))
                        new_axes = new_nav_axes[::-1] + new_sig_axes[::-1]
                        ns.axes_manager._axes = [axis.copy()
                                                 for axis in new_axes]
                        if bigger_am is oam:
                            ns.metadata.Signal.record_by = \
                                other.metadata.Signal.record_by
                            ns._assign_subclass()
                        return ns

        else:
            # Second object is not a Signal
            if op_name in INPLACE_OPERATORS:
                getattr(self.data, op_name)(other)
                return self
            else:
                return self._deepcopy_with_new_data(
                    getattr(self.data, op_name)(other))

    def _unary_operator_ruler(self, op_name):
        return self._deepcopy_with_new_data(getattr(self.data, op_name)())

    def _check_signal_dimension_equals_one(self):
        if self.axes_manager.signal_dimension != 1:
            raise SignalDimensionError(self.axes_manager.signal_dimension, 1)

    def _check_signal_dimension_equals_two(self):
        if self.axes_manager.signal_dimension != 2:
            raise SignalDimensionError(self.axes_manager.signal_dimension, 2)

    def _check_navigation_mask(self, mask):
        if mask is not None:
            if not isinstance(mask, BaseSignal):
                raise ValueError("mask must be a Signal instance.")
            elif mask.axes_manager.signal_dimension not in (0, 1):
                raise ValueError("mask must be a Signal with signal_dimension "
                                 "equal to 1")
            elif (mask.axes_manager.navigation_dimension !=
                  self.axes_manager.navigation_dimension):
                raise ValueError("mask must be a Signal with the same "
                                 "navigation_dimension as the current signal.")

    def _deepcopy_with_new_data(self, data=None):
        """Returns a deepcopy of itself replacing the data.

        This method has the advantage over deepcopy that it does not
        copy the data what can save precious memory

        Parameters
        ---------
        data : {None | np.array}

        Returns
        -------
        ns : Signal

        """
        try:
            old_data = self.data
            self.data = None
            old_plot = self._plot
            self._plot = None
            ns = self.deepcopy()
            ns.data = np.atleast_1d(data)
            return ns
        finally:
            self.data = old_data
            self._plot = old_plot

    def _print_summary(self):
        string = "\n\tTitle: "
        string += self.metadata.General.title
        if self.metadata.has_item("Signal.signal_type"):
            string += "\n\tSignal type: "
            string += self.metadata.Signal.signal_type
        string += "\n\tData dimensions: "
        string += str(self.axes_manager.shape)
        if self.metadata.has_item('Signal.record_by'):
            string += "\n\tData representation: "
            string += self.metadata.Signal.record_by
            string += "\n\tData type: "
            string += str(self.data.dtype)
        print(string)

    def _load_dictionary(self, file_data_dict):
        """Load data from dictionary.

        Parameters
        ----------
        file_data_dict : dictionary
            A dictionary containing at least a 'data' keyword with an array of
            arbitrary dimensions. Additionally the dictionary can contain the
            following items:
            data : numpy array
               The signal data. It can be an array of any dimensions.
            axes : dictionary (optional)
                Dictionary to define the axes (see the
                documentation of the AxesManager class for more details).
            attributes : dictionary (optional)
                A dictionary whose items are stored as attributes.
            metadata : dictionary (optional)
                A dictionary containing a set of parameters
                that will to stores in the `metadata` attribute.
                Some parameters might be mandatory in some cases.
            original_metadata : dictionary (optional)
                A dictionary containing a set of parameters
                that will to stores in the `original_metadata` attribute. It
                typically contains all the parameters that has been
                imported from the original data file.

        """

        self.data = np.atleast_1d(np.asanyarray(file_data_dict['data']))
        if 'axes' not in file_data_dict:
            file_data_dict['axes'] = self._get_undefined_axes_list()
        self.axes_manager = AxesManager(
            file_data_dict['axes'])
        if 'metadata' not in file_data_dict:
            file_data_dict['metadata'] = {}
        if 'original_metadata' not in file_data_dict:
            file_data_dict['original_metadata'] = {}
        if 'attributes' in file_data_dict:
            for key, value in file_data_dict['attributes'].items():
                if hasattr(self, key):
                    if isinstance(value, dict):
                        for k, v in value.items():
                            eval('self.%s.__setattr__(k,v)' % key)
                    else:
                        self.__setattr__(key, value)
        self.original_metadata.add_dictionary(
            file_data_dict['original_metadata'])
        self.metadata.add_dictionary(
            file_data_dict['metadata'])
        if "title" not in self.metadata.General:
            self.metadata.General.title = ''
        if (self._record_by or
                "Signal.record_by" not in self.metadata):
            self.metadata.Signal.record_by = self._record_by
        if (self._signal_origin or
                "Signal.signal_origin" not in self.metadata):
            self.metadata.Signal.signal_origin = self._signal_origin
        if (self._signal_type or
                not self.metadata.has_item("Signal.signal_type")):
            self.metadata.Signal.signal_type = self._signal_type

    def squeeze(self):
        """Remove single-dimensional entries from the shape of an array
        and the axes.

        """
        # We deepcopy everything but data
        self = self._deepcopy_with_new_data(self.data)
        for axis in self.axes_manager._axes:
            if axis.size == 1:
                self._remove_axis(axis.index_in_axes_manager)
        self.data = self.data.squeeze()
        return self

    def _to_dictionary(self, add_learning_results=True):
        """Returns a dictionary that can be used to recreate the signal.

        All items but `data` are copies.

        Parameters
        ----------
        add_learning_results : bool

        Returns
        -------
        dic : dictionary

        """
        dic = {'data': self.data,
               'axes': self.axes_manager._get_axes_dicts(),
               'metadata': self.metadata.deepcopy().as_dictionary(),
               'original_metadata': self.original_metadata.deepcopy().as_dictionary(),
               'tmp_parameters': self.tmp_parameters.deepcopy().as_dictionary()}
        if add_learning_results and hasattr(self, 'learning_results'):
            dic['learning_results'] = copy.deepcopy(
                self.learning_results.__dict__)
        return dic

    def _get_undefined_axes_list(self):
        axes = []
        for i in range(len(self.data.shape)):
            axes.append({'size': int(self.data.shape[i]), })
        return axes

    def __call__(self, axes_manager=None):
        if axes_manager is None:
            axes_manager = self.axes_manager
        return np.atleast_1d(
            self.data.__getitem__(axes_manager._getitem_tuple))

    def plot(self, navigator="auto", axes_manager=None, **kwargs):
        """Plot the signal at the current coordinates.

        For multidimensional datasets an optional figure,
        the "navigator", with a cursor to navigate that data is
        raised. In any case it is possible to navigate the data using
        the sliders. Currently only signals with signal_dimension equal to
        0, 1 and 2 can be plotted.

        Parameters
        ----------
        navigator : {"auto", None, "slider", "spectrum", Signal}
            If "auto", if navigation_dimension > 0, a navigator is
            provided to explore the data.
            If navigation_dimension is 1 and the signal is an image
            the navigator is a spectrum obtained by integrating
            over the signal axes (the image).
            If navigation_dimension is 1 and the signal is a spectrum
            the navigator is an image obtained by stacking horizontally
            all the spectra in the dataset.
            If navigation_dimension is > 1, the navigator is an image
            obtained by integrating the data over the signal axes.
            Additionaly, if navigation_dimension > 2 a window
            with one slider per axis is raised to navigate the data.
            For example,
            if the dataset consists of 3 navigation axes X, Y, Z and one
            signal axis, E, the default navigator will be an image
            obtained by integrating the data over E at the current Z
            index and a window with sliders for the X, Y and Z axes
            will be raised. Notice that changing the Z-axis index
            changes the navigator in this case.
            If "slider" and the navigation dimension > 0 a window
            with one slider per axis is raised to navigate the data.
            If "spectrum" and navigation_dimension > 0 the navigator
            is always a spectrum obtained by integrating the data
            over all other axes.
            If None, no navigator will be provided.
            Alternatively a Signal instance can be provided. The signal
            dimension must be 1 (for a spectrum navigator) or 2 (for a
            image navigator) and navigation_shape must be 0 (for a static
            navigator) or navigation_shape + signal_shape must be equal
            to the navigator_shape of the current object (for a dynamic
            navigator).
            If the signal dtype is RGB or RGBA this parameters has no
            effect and is always "slider".

        axes_manager : {None, axes_manager}
            If None `axes_manager` is used.

        **kwargs : optional
            Any extra keyword arguments are passed to the signal plot.

        """

        if self._plot is not None:
            try:
                self._plot.close()
            except:
                # If it was already closed it will raise an exception,
                # but we want to carry on...
                pass

        if axes_manager is None:
            axes_manager = self.axes_manager
        if self.is_rgbx is True:
            if axes_manager.navigation_size < 2:
                navigator = None
            else:
                navigator = "slider"
        if axes_manager.signal_dimension == 0:
            self._plot = mpl_he.MPL_HyperExplorer()
        elif axes_manager.signal_dimension == 1:
            # Hyperspectrum
            self._plot = mpl_hse.MPL_HyperSignal1D_Explorer()
        elif axes_manager.signal_dimension == 2:
            self._plot = mpl_hie.MPL_HyperImage_Explorer()
        else:
            raise ValueError('Plotting is not supported for this view')

        self._plot.axes_manager = axes_manager
        self._plot.signal_data_function = self.__call__
        if self.metadata.General.title:
            self._plot.signal_title = self.metadata.General.title
        elif self.tmp_parameters.has_item('filename'):
            self._plot.signal_title = self.tmp_parameters.filename

        def get_static_explorer_wrapper(*args, **kwargs):
            return navigator()

        def get_1D_sum_explorer_wrapper(*args, **kwargs):
            navigator = self
            # Sum over all but the first navigation axis.
            while len(navigator.axes_manager.shape) > 1:
                navigator = navigator.sum(-1)
            return np.nan_to_num(navigator.data).squeeze()

        def get_dynamic_explorer_wrapper(*args, **kwargs):
            navigator.axes_manager.indices = self.axes_manager.indices[
                navigator.axes_manager.signal_dimension:]
            navigator.axes_manager._update_attributes()
            return navigator()

        if not isinstance(navigator, BaseSignal) and navigator == "auto":
            if (self.axes_manager.navigation_dimension == 1 and
                    self.axes_manager.signal_dimension == 1):
                navigator = "data"
            elif self.axes_manager.navigation_dimension > 0:
                if self.axes_manager.signal_dimension == 0:
                    navigator = self.deepcopy()
                else:
                    navigator = self
                    while navigator.axes_manager.signal_dimension > 0:
                        navigator = navigator.sum(-1)
                if navigator.axes_manager.navigation_dimension == 1:
                    navigator = navigator.as_spectrum(0)
                else:
                    navigator = navigator.as_image((0, 1))
            else:
                navigator = None
        # Navigator properties
        if axes_manager.navigation_axes:
            if navigator is "slider":
                self._plot.navigator_data_function = "slider"
            elif navigator is None:
                self._plot.navigator_data_function = None
            elif isinstance(navigator, BaseSignal):
                # Dynamic navigator
                if (axes_manager.navigation_shape ==
                        navigator.axes_manager.signal_shape +
                        navigator.axes_manager.navigation_shape):
                    self._plot.navigator_data_function = \
                        get_dynamic_explorer_wrapper

                elif (axes_manager.navigation_shape ==
                        navigator.axes_manager.signal_shape or
                        axes_manager.navigation_shape[:2] ==
                        navigator.axes_manager.signal_shape or
                        (axes_manager.navigation_shape[0],) ==
                        navigator.axes_manager.signal_shape):
                    self._plot.navigator_data_function = \
                        get_static_explorer_wrapper
                else:
                    raise ValueError(
                        "The navigator dimensions are not compatible with "
                        "those of self.")
            elif navigator == "data":
                self._plot.navigator_data_function = \
                    lambda axes_manager=None: self.data
            elif navigator == "spectrum":
                self._plot.navigator_data_function = \
                    get_1D_sum_explorer_wrapper
            else:
                raise ValueError(
                    "navigator must be one of \"spectrum\",\"auto\","
                    " \"slider\", None, a Signal instance")

        self._plot.plot(**kwargs)

    def save(self, filename=None, overwrite=None, extension=None, **kwds):
        """Saves the signal in the specified format.

        The function gets the format from the extension.:
            - hdf5 for HDF5
            - rpl for Ripple (useful to export to Digital Micrograph)
            - msa for EMSA/MSA single spectrum saving.
            - Many image formats such as png, tiff, jpeg...

        If no extension is provided the default file format as defined
        in the `preferences` is used.
        Please note that not all the formats supports saving datasets of
        arbitrary dimensions, e.g. msa only supports 1D data.

        Each format accepts a different set of parameters. For details
        see the specific format documentation.

        Parameters
        ----------
        filename : str or None
            If None (default) and tmp_parameters.filename and
            `tmp_paramters.folder` are defined, the
            filename and path will be taken from there. A valid
            extension can be provided e.g. "my_file.rpl", see `extension`.
        overwrite : None, bool
            If None, if the file exists it will query the user. If
            True(False) it (does not) overwrites the file if it exists.
        extension : {None, 'hdf5', 'rpl', 'msa',common image extensions e.g.
                    'tiff', 'png'}
            The extension of the file that defines the file format.
            If None, the extension is taken from the first not None in the
            following list:
            i) the filename
            ii)  `tmp_parameters.extension`
            iii) `preferences.General.default_file_format` in this order.

        """
        if filename is None:
            if (self.tmp_parameters.has_item('filename') and
                    self.tmp_parameters.has_item('folder')):
                filename = os.path.join(self.tmp_parameters.folder,
                                        self.tmp_parameters.filename)
                extension = (self.tmp_parameters.extension
                             if not extension
                             else extension)
            elif self.metadata.has_item('General.original_filename'):
                filename = self.metadata.General.original_filename
            else:
                raise ValueError('File name not defined')
        if extension is not None:
            basename, ext = os.path.splitext(filename)
            filename = basename + '.' + extension
        io.save(filename, self, overwrite=overwrite, **kwds)

    def _replot(self):
        if self._plot is not None:
            if self._plot.is_active() is True:
                self.plot()

    @auto_replot
    def get_dimensions_from_data(self):
        """Get the dimension parameters from the data_cube. Useful when
        the data_cube was externally modified, or when the SI was not
        loaded from a file

        """
        dc = self.data
        for axis in self.axes_manager._axes:
            axis.size = int(dc.shape[axis.index_in_array])

    def crop(self, axis, start=None, end=None):
        """Crops the data in a given axis. The range is given in pixels

        Parameters
        ----------
        axis : {int | string}
            Specify the data axis in which to perform the cropping
            operation. The axis can be specified using the index of the
            axis in `axes_manager` or the axis name.
        start, end : {int | float | None}
            The beginning and end of the cropping interval. If int
            the value is taken as the axis index. If float the index
            is calculated using the axis calibration. If start/end is
            None crop from/to the low/high end of the axis.

        """
        axis = self.axes_manager[axis]
        i1, i2 = axis._get_index(start), axis._get_index(end)
        if i1 is not None:
            new_offset = axis.axis[i1]
        # We take a copy to guarantee the continuity of the data
        self.data = self.data[
            (slice(None),) * axis.index_in_array + (slice(i1, i2),
                                                    Ellipsis)]

        if i1 is not None:
            axis.offset = new_offset
        self.get_dimensions_from_data()
        self.squeeze()

    def swap_axes(self, axis1, axis2):
        """Swaps the axes.

        Parameters
        ----------
        axis1, axis2 : {int | str}
            Specify the data axes in which to perform the operation.
            The axis can be specified using the index of the
            axis in `axes_manager` or the axis name.

        Returns
        -------
        s : a copy of the object with the axes swapped.

        """
        axis1 = self.axes_manager[axis1].index_in_array
        axis2 = self.axes_manager[axis2].index_in_array
        s = self._deepcopy_with_new_data(self.data.swapaxes(axis1, axis2))
        c1 = s.axes_manager._axes[axis1]
        c2 = s.axes_manager._axes[axis2]
        c1.slice, c2.slice = c2.slice, c1.slice
        c1.navigate, c2.navigate = c2.navigate, c1.navigate
        s.axes_manager._axes[axis1] = c2
        s.axes_manager._axes[axis2] = c1
        s.axes_manager._update_attributes()
        am = s.axes_manager
        am.connect(am._update_attributes)
        s._make_sure_data_is_contiguous()
        return s

    def rollaxis(self, axis, to_axis):
        """Roll the specified axis backwards, until it lies in a given position.

        Parameters
        ----------
        axis : {int, str}
            The axis to roll backwards.  The positions of the other axes do not
            change relative to one another.
        to_axis : {int, str}
            The axis is rolled until it lies before this other axis.

        Returns
        -------
        s : Signal or subclass
            Output signal.

        See Also
        --------
        roll : swap_axes

        Examples
        --------
        >>> s = hs.signals.Signal1D(np.ones((5,4,3,6)))
        >>> s
        <Spectrum, title: , dimensions: (3, 4, 5, 6)>
        >>> s.rollaxis(3, 1)
        <Spectrum, title: , dimensions: (3, 4, 5, 6)>
        >>> s.rollaxis(2,0)
        <Spectrum, title: , dimensions: (5, 3, 4, 6)>

        """
        axis = self.axes_manager[axis].index_in_array
        to_index = self.axes_manager[to_axis].index_in_array
        if axis == to_index:
            return self.deepcopy()
        new_axes_indices = hyperspy.misc.utils.rollelem(
            [axis_.index_in_array for axis_ in self.axes_manager._axes],
            index=axis,
            to_index=to_index)

        s = self._deepcopy_with_new_data(self.data.transpose(new_axes_indices))
        s.axes_manager._axes = hyperspy.misc.utils.rollelem(
            s.axes_manager._axes,
            index=axis,
            to_index=to_index)
        s.axes_manager._update_attributes()
        s._make_sure_data_is_contiguous()
        return s

    @property
    def _data_aligned_with_axes(self):
        """Returns a view of `data` with is axes aligned with the Signal axes.

        """
        if self.axes_manager.axes_are_aligned_with_data:
            return self.data
        else:
            am = self.axes_manager
            nav_iia_r = am.navigation_indices_in_array[::-1]
            sig_iia_r = am.signal_indices_in_array[::-1]
            # nav_sort = np.argsort(nav_iia_r)
            # sig_sort = np.argsort(sig_iia_r) + len(nav_sort)
            data = self.data.transpose(nav_iia_r + sig_iia_r)
            return data

    def rebin(self, new_shape):
        """Returns the object with the data rebinned.

        Parameters
        ----------
        new_shape: tuple of ints
            The new shape elements must be divisors of the original shape
            elements.

        Returns
        -------
        s : Signal subclass

        Raises
        ------
        ValueError
            When there is a mismatch between the number of elements in the
            signal shape and `new_shape` or `new_shape` elements are not
            divisors of the original signal shape.


        Examples
        --------
        >>> import hyperspy.api as hs
        >>> s = hs.signals.Signal1D(np.zeros((10, 100)))
        >>> s
        <Spectrum, title: , dimensions: (10|100)>
        >>> s.rebin((5, 100))
        <Spectrum, title: , dimensions: (5|100)>
        I
        """
        if len(new_shape) != len(self.data.shape):
            raise ValueError("Wrong shape size")
        new_shape_in_array = []
        for axis in self.axes_manager._axes:
            new_shape_in_array.append(
                new_shape[axis.index_in_axes_manager])
        factors = (np.array(self.data.shape) /
                   np.array(new_shape_in_array))
        s = self._deepcopy_with_new_data(
            array_tools.rebin(self.data, new_shape_in_array))
        for axis in s.axes_manager._axes:
            axis.scale *= factors[axis.index_in_array]
        s.get_dimensions_from_data()
        if s.metadata.has_item('Signal.Noise_properties.variance'):
            if isinstance(s.metadata.Signal.Noise_properties.variance,
                          BaseSignal):
                var = s.metadata.Signal.Noise_properties.variance
                s.metadata.Signal.Noise_properties.variance = var.rebin(
                    new_shape)
        return s

    def split(self,
              axis='auto',
              number_of_parts='auto',
              step_sizes='auto'):
        """Splits the data into several signals.

        The split can be defined by giving the number_of_parts, a homogeneous
        step size or a list of customized step sizes. By default ('auto'),
        the function is the reverse of utils.stack().

        Parameters
        ----------
        axis : {'auto' | int | string}
            Specify the data axis in which to perform the splitting
            operation.  The axis can be specified using the index of the
            axis in `axes_manager` or the axis name.
            - If 'auto' and if the object has been created with utils.stack,
            split will return the former list of signals
            (options stored in 'metadata._HyperSpy.Stacking_history'
             else the last navigation axis will be used.
        number_of_parts : {'auto' | int}
            Number of parts in which the SI will be splitted. The
            splitting is homegenous. When the axis size is not divisible
            by the number_of_parts the reminder data is lost without
            warning. If number_of_parts and step_sizes is 'auto',
            number_of_parts equals the length of the axis,
            step_sizes equals one  and the axis is supress from each
            sub_spectra.
        step_sizes : {'auto' | list of ints | int}
            Size of the splitted parts. If 'auto', the step_sizes equals one.
            If int, the splitting is homogenous.

        Examples
        --------
        >>> s = hs.signals.Signal1D(random.random([4,3,2]))
        >>> s
            <Spectrum, title: , dimensions: (3, 4|2)>
        >>> s.split()
            [<Spectrum, title: , dimensions: (3 |2)>,
            <Spectrum, title: , dimensions: (3 |2)>,
            <Spectrum, title: , dimensions: (3 |2)>,
            <Spectrum, title: , dimensions: (3 |2)>]
        >>> s.split(step_sizes=2)
            [<Spectrum, title: , dimensions: (3, 2|2)>,
            <Spectrum, title: , dimensions: (3, 2|2)>]
        >>> s.split(step_sizes=[1,2])
            [<Spectrum, title: , dimensions: (3, 1|2)>,
            <Spectrum, title: , dimensions: (3, 2|2)>]

        Returns
        -------
        list of the splitted signals
        """

        shape = self.data.shape
        signal_dict = self._to_dictionary(add_learning_results=False)

        if axis == 'auto':
            mode = 'auto'
            if hasattr(self.metadata._HyperSpy, 'Stacking_history'):
                stack_history = self.metadata._HyperSpy.Stacking_history
                axis_in_manager = stack_history.axis
                step_sizes = stack_history.step_sizes
            else:
                axis_in_manager = \
                    self.axes_manager[-1 + 1j].index_in_axes_manager
        else:
            mode = 'manual'
            axis_in_manager = self.axes_manager[axis].index_in_axes_manager

        axis = self.axes_manager[axis_in_manager].index_in_array
        len_axis = self.axes_manager[axis_in_manager].size

        if number_of_parts is 'auto' and step_sizes is 'auto':
            step_sizes = 1
            number_of_parts = len_axis
        elif number_of_parts is not 'auto' and step_sizes is not 'auto':
            raise ValueError(
                "You can define step_sizes or number_of_parts "
                "but not both.")
        elif step_sizes is 'auto':
            if number_of_parts > shape[axis]:
                raise ValueError(
                    "The number of parts is greater than "
                    "the axis size.")
            else:
                step_sizes = ([shape[axis] // number_of_parts, ] *
                              number_of_parts)

        if isinstance(step_sizes, int):
            step_sizes = [step_sizes] * int(len_axis / step_sizes)

        splitted = []
        cut_index = np.array([0] + step_sizes).cumsum()

        axes_dict = signal_dict['axes']
        for i in range(len(cut_index) - 1):
            axes_dict[axis]['offset'] = \
                self.axes_manager._axes[axis].index2value(cut_index[i])
            axes_dict[axis]['size'] = cut_index[i + 1] - cut_index[i]
            data = self.data[
                (slice(None), ) * axis +
                (slice(cut_index[i], cut_index[i + 1]), Ellipsis)]
            signal_dict['data'] = data
            splitted += self.__class__(**signal_dict),

        if number_of_parts == len_axis \
                or step_sizes == [1] * len_axis:
            for i, signal1D in enumerate(splitted):
                signal1D.data = signal1D.data[
                    signal1D.axes_manager._get_data_slice([(axis, 0)])]
                signal1D._remove_axis(axis_in_manager)

        if mode == 'auto' and hasattr(
                self.original_metadata, 'stack_elements'):
            for i, spectrum in enumerate(splitted):
                se = self.original_metadata.stack_elements['element' + str(i)]
                spectrum.metadata = copy.deepcopy(
                    se['metadata'])
                spectrum.original_metadata = copy.deepcopy(
                    se['original_metadata'])
                spectrum.metadata.General.title = se.metadata.General.title

        return splitted

    # TODO: remove in Hyperspy 1.0
    @staticmethod
    def unfold_if_multidim():
        """Unfold the datacube if it is >2D

        Deprecated method, please use unfold.

        """
        warnings.warn(
            "`unfold_if_multidim` is deprecated and will be removed in "
            "Hyperspy 1.0. Please use `unfold` instead.",
            VisibleDeprecationWarning)
        if len(self.axes_manager._axes) > 2:
            print("Automatically unfolding the data")
            self.unfold()
            return True
        else:
            return False

    @auto_replot
    def _unfold(self, steady_axes, unfolded_axis):
        """Modify the shape of the data by specifying the axes whose
        dimension do not change and the axis over which the remaining axes will
        be unfolded

        Parameters
        ----------
        steady_axes : list
            The indices of the axes which dimensions do not change
        unfolded_axis : int
            The index of the axis over which all the rest of the axes (except
            the steady axes) will be unfolded

        See also
        --------
        fold
        """

        # It doesn't make sense unfolding when dim < 2
        if self.data.squeeze().ndim < 2:
            return

        # We need to store the original shape and coordinates to be used
        # by
        # the fold function only if it has not been already stored by a
        # previous unfold
        folding = self.metadata._HyperSpy.Folding
        if folding.unfolded is False:
            folding.original_shape = self.data.shape
            folding.original_axes_manager = self.axes_manager
            folding.unfolded = True

        new_shape = [1] * len(self.data.shape)
        for index in steady_axes:
            new_shape[index] = self.data.shape[index]
        new_shape[unfolded_axis] = -1
        self.data = self.data.reshape(new_shape)
        self.axes_manager = self.axes_manager.deepcopy()
        uname = ''
        uunits = ''
        to_remove = []
        for axis, dim in zip(self.axes_manager._axes, new_shape):
            if dim == 1:
                uname += ',' + str(axis)
                uunits = ',' + str(axis.units)
                to_remove.append(axis)
        ua = self.axes_manager._axes[unfolded_axis]
        ua.name = str(ua) + uname
        ua.units = str(ua.units) + uunits
        ua.size = self.data.shape[unfolded_axis]
        for axis in to_remove:
            self.axes_manager.remove(axis.index_in_axes_manager)
        self.data = self.data.squeeze()
        if self.metadata.has_item('Signal.Noise_properties.variance'):
            variance = self.metadata.Signal.Noise_properties.variance
            if isinstance(variance, BaseSignal):
                variance._unfold(steady_axes, unfolded_axis)

    def unfold(self, unfold_navigation=True, unfold_signal=True):
        """Modifies the shape of the data by unfolding the signal and
        navigation dimensions separately

        Returns
        -------
        needed_unfolding : bool


        """
        unfolded = False
        if unfold_navigation:
            if self.unfold_navigation_space():
                unfolded = True
        if unfold_signal:
            if self.unfold_signal_space():
                unfolded = True
        return unfolded

    @contextmanager
    def unfolded(self, unfold_navigation=True, unfold_signal=True):
        """Use this function together with a `with` statement to have the
        signal be unfolded for the scope of the `with` block, before
        automatically refolding when passing out of scope.

        See also
        --------
        unfold, fold

        Examples
        --------
        >>> import numpy as np
        >>> s = Signal(np.random.random((64,64,1024)))
        >>> with s.unfolded():
                # Do whatever needs doing while unfolded here
                pass
        """
        unfolded = self.unfold(unfold_navigation, unfold_signal)
        try:
            yield unfolded
        finally:
            if unfolded is not False:
                self.fold()

    def unfold_navigation_space(self):
        """Modify the shape of the data to obtain a navigation space of
        dimension 1

        Returns
        -------
        needed_unfolding : bool

        """

        if self.axes_manager.navigation_dimension < 2:
            needed_unfolding = False
        else:
            needed_unfolding = True
            steady_axes = [
                axis.index_in_array for axis in
                self.axes_manager.signal_axes]
            unfolded_axis = (
                self.axes_manager.navigation_axes[0].index_in_array)
            self._unfold(steady_axes, unfolded_axis)
        return needed_unfolding

    def unfold_signal_space(self):
        """Modify the shape of the data to obtain a signal space of
        dimension 1

        Returns
        -------
        needed_unfolding : bool

        """
        if self.axes_manager.signal_dimension < 2:
            needed_unfolding = False
        else:
            needed_unfolding = True
            steady_axes = [
                axis.index_in_array for axis in
                self.axes_manager.navigation_axes]
            unfolded_axis = self.axes_manager.signal_axes[0].index_in_array
            self._unfold(steady_axes, unfolded_axis)
            self.metadata._HyperSpy.Folding.signal_unfolded = True
        return needed_unfolding

    @auto_replot
    def fold(self):
        """If the signal was previously unfolded, folds it back"""
        folding = self.metadata._HyperSpy.Folding
        # Note that == must be used instead of is True because
        # if the value was loaded from a file its type can be np.bool_
        if folding.unfolded is True:
            self.data = self.data.reshape(folding.original_shape)
            self.axes_manager = folding.original_axes_manager
            folding.original_shape = None
            folding.original_axes_manager = None
            folding.unfolded = False
            folding.signal_unfolded = False
            if self.metadata.has_item('Signal.Noise_properties.variance'):
                variance = self.metadata.Signal.Noise_properties.variance
                if isinstance(variance, BaseSignal):
                    variance.fold()

    def _make_sure_data_is_contiguous(self):
        if self.data.flags['C_CONTIGUOUS'] is False:
            self.data = np.ascontiguousarray(self.data)

    def _iterate_signal(self):
        """Iterates over the signal data.

        It is faster than using the signal iterator.

        """
        if self.axes_manager.navigation_size < 2:
            yield self()
            return
        self._make_sure_data_is_contiguous()
        axes = [axis.index_in_array for
                axis in self.axes_manager.signal_axes]
        unfolded_axis = (
            self.axes_manager.navigation_axes[0].index_in_array)
        new_shape = [1] * len(self.data.shape)
        for axis in axes:
            new_shape[axis] = self.data.shape[axis]
        new_shape[unfolded_axis] = -1
        # Warning! if the data is not contigous it will make a copy!!
        data = self.data.reshape(new_shape)
        for i in range(data.shape[unfolded_axis]):
            getitem = [0] * len(data.shape)
            for axis in axes:
                getitem[axis] = slice(None)
            getitem[unfolded_axis] = i
            yield(data[getitem])

    def _remove_axis(self, axis):
        am = self.axes_manager
        axis = am[axis]
        if am.navigation_dimension + am.signal_dimension > 1:
            am.remove(axis.index_in_axes_manager)
            if axis.navigate is False:  # The removed axis is a signal axis
                if am.signal_dimension == 2:
                    self._record_by = "image"
                elif am.signal_dimension == 1:
                    self._record_by = "spectrum"
                elif am.signal_dimension == 0:
                    self._record_by = ""
                else:
                    return
                self.metadata.Signal.record_by = self._record_by
                self._assign_subclass()
        else:
            # Create a "Scalar" axis because the axis is the last one left and
            # HyperSpy does not # support 0 dimensions
            am.remove(axis.index_in_axes_manager)
            am._append_axis(
                size=1,
                scale=1,
                offset=0,
                name="Scalar",
                navigate=False,)

    def _apply_function_on_data_and_remove_axis(self, function, axis):
        s = self._deepcopy_with_new_data(
            function(self.data,
                     axis=self.axes_manager[axis].index_in_array))
        s._remove_axis(axis)
        return s

    def sum(self, axis):
        """Sum the data over the given axis.

        Parameters
        ----------
        axis : {int, string}
           The axis can be specified using the index of the axis in
           `axes_manager` or the axis name.

        Returns
        -------
        s : Signal

        See also
        --------
        sum_in_mask, mean

        Examples
        --------
        >>> import numpy as np
        >>> s = Signal(np.random.random((64,64,1024)))
        >>> s.data.shape
        (64,64,1024)
        >>> s.sum(-1).data.shape
        (64,64)
        # If we just want to plot the result of the operation
        s.sum(-1, True).plot()

        """
        return self._apply_function_on_data_and_remove_axis(np.sum, axis)

    def max(self, axis, return_signal=False):
        """Returns a signal with the maximum of the signal along an axis.

        Parameters
        ----------
        axis : {int | string}
           The axis can be specified using the index of the axis in
           `axes_manager` or the axis name.

        Returns
        -------
        s : Signal

        See also
        --------
        sum, mean, min

        Examples
        --------
        >>> import numpy as np
        >>> s = Signal(np.random.random((64,64,1024)))
        >>> s.data.shape
        (64,64,1024)
        >>> s.max(-1).data.shape
        (64,64)

        """
        return self._apply_function_on_data_and_remove_axis(np.max, axis)

    def min(self, axis):
        """Returns a signal with the minimum of the signal along an axis.

        Parameters
        ----------
        axis : {int | string}
           The axis can be specified using the index of the axis in
           `axes_manager` or the axis name.

        Returns
        -------
        s : Signal

        See also
        --------
        sum, mean, max, std, var

        Examples
        --------
        >>> import numpy as np
        >>> s = Signal(np.random.random((64,64,1024)))
        >>> s.data.shape
        (64,64,1024)
        >>> s.min(-1).data.shape
        (64,64)

        """

        return self._apply_function_on_data_and_remove_axis(np.min, axis)

    def mean(self, axis):
        """Returns a signal with the average of the signal along an axis.

        Parameters
        ----------
        axis : {int | string}
           The axis can be specified using the index of the axis in
           `axes_manager` or the axis name.

        Returns
        -------
        s : Signal

        See also
        --------
        sum_in_mask, mean

        Examples
        --------
        >>> import numpy as np
        >>> s = Signal(np.random.random((64,64,1024)))
        >>> s.data.shape
        (64,64,1024)
        >>> s.mean(-1).data.shape
        (64,64)

        """
        return self._apply_function_on_data_and_remove_axis(np.mean,
                                                            axis)

    def std(self, axis):
        """Returns a signal with the standard deviation of the signal along
        an axis.

        Parameters
        ----------
        axis : {int | string}
           The axis can be specified using the index of the axis in
           `axes_manager` or the axis name.

        Returns
        -------
        s : Signal

        See also
        --------
        sum_in_mask, mean

        Examples
        --------
        >>> import numpy as np
        >>> s = Signal(np.random.random((64,64,1024)))
        >>> s.data.shape
        (64,64,1024)
        >>> s.std(-1).data.shape
        (64,64)

        """
        return self._apply_function_on_data_and_remove_axis(np.std, axis)

    def var(self, axis):
        """Returns a signal with the variances of the signal along an axis.

        Parameters
        ----------
        axis : {int | string}
           The axis can be specified using the index of the axis in
           `axes_manager` or the axis name.

        Returns
        -------
        s : Signal

        See also
        --------
        sum_in_mask, mean

        Examples
        --------
        >>> import numpy as np
        >>> s = Signal(np.random.random((64,64,1024)))
        >>> s.data.shape
        (64,64,1024)
        >>> s.var(-1).data.shape
        (64,64)

        """
        return self._apply_function_on_data_and_remove_axis(np.var, axis)

    def diff(self, axis, order=1):
        """Returns a signal with the n-th order discrete difference along
        given axis.
        Parameters
        ----------
        axis : {int | string}
           The axis can be specified using the index of the axis in
           `axes_manager` or the axis name.
        order: the order of the derivative
        See also
        --------
        mean, sum
        Examples
        --------
        >>> import numpy as np
        >>> s = Signal(np.random.random((64,64,1024)))
        >>> s.data.shape
        (64,64,1024)
        >>> s.diff(-1).data.shape
        (64,64,1023)
        """

        s = self._deepcopy_with_new_data(
            np.diff(self.data,
                    n=order,
                    axis=self.axes_manager[axis].index_in_array))
        axis = s.axes_manager[axis]
        axis.offset += (order * axis.scale / 2)
        s.get_dimensions_from_data()
        return s

    def derivative(self, axis, order=1):
        """Numerical derivative along the given axis.

        Currently only the first order finite difference method is implemented.

        Parameters
        ----------
        axis : {int | string}
           The axis can be specified using the index of the axis in
           `axes_manager` or the axis name.
        order: int
            The order of the derivative. (Note that this is the order of the
            derivative i.e. `order=2` does not use second order finite
            differences method.)

        Returns
        -------
        der : Signal
            Note that the size of the data on the given `axis` decreases by the
            given `order` i.e. if `axis` is "x" and `order` is 2 the x dimension
            is N, der's x dimension is N - 2.

        See also
        --------
        diff

        """

        der = self.diff(order=order, axis=axis)
        axis = self.axes_manager[axis]
        der.data /= axis.scale ** order
        return der

    def integrate_simpson(self, axis):
        """Returns a signal with the result of calculating the integral
        of the signal along an axis using Simpson's rule.

        Parameters
        ----------
        axis : {int | string}
           The axis can be specified using the index of the axis in
           `axes_manager` or the axis name.

        Returns
        -------
        s : Signal

        See also
        --------
        sum_in_mask, mean

        Examples
        --------
        >>> import numpy as np
        >>> s = Signal(np.random.random((64,64,1024)))
        >>> s.data.shape
        (64,64,1024)
        >>> s.var(-1).data.shape
        (64,64)

        """
        axis = self.axes_manager[axis]
        s = self._deepcopy_with_new_data(
            sp.integrate.simps(y=self.data,
                               x=axis.axis,
                               axis=axis.index_in_array))
        s._remove_axis(axis.index_in_axes_manager)
        return s

    def integrate1D(self, axis):
        """Integrate the signal over the given axis.

        The integration is performed using Simpson's rule if
        `metadata.Signal.binned` is False and summation over the given axis if
        True.

        Parameters
        ----------
        axis : {int | string}
           The axis can be specified using the index of the axis in
           `axes_manager` or the axis name.

        Returns
        -------
        s : Signal

        See also
        --------
        sum_in_mask, mean

        Examples
        --------
        >>> import numpy as np
        >>> s = Signal(np.random.random((64,64,1024)))
        >>> s.data.shape
        (64,64,1024)
        >>> s.var(-1).data.shape
        (64,64)

        """
        if self.metadata.Signal.binned is False:
            return self.integrate_simpson(axis)
        else:
            return self.sum(axis)

    def indexmax(self, axis):
        """Returns a signal with the index of the maximum along an axis.

        Parameters
        ----------
        axis : {int | string}
           The axis can be specified using the index of the axis in
           `axes_manager` or the axis name.

        Returns
        -------
        s : Signal
            The data dtype is always int.

        See also
        --------
        sum, mean, min

        Usage
        -----
        >>> import numpy as np
        >>> s = Signal(np.random.random((64,64,1024)))
        >>> s.data.shape
        (64,64,1024)
        >>> s.indexmax(-1).data.shape
        (64,64)

        """
        return self._apply_function_on_data_and_remove_axis(np.argmax, axis)

    def valuemax(self, axis):
        """Returns a signal with the value of the maximum along an axis.

        Parameters
        ----------
        axis : {int | string}
           The axis can be specified using the index of the axis in
           `axes_manager` or the axis name.

        Returns
        -------
        s : Signal
            The data dtype is always int.

        See also
        --------
        sum, mean, min

        Usage
        -----
        >>> import numpy as np
        >>> s = Signal(np.random.random((64,64,1024)))
        >>> s.data.shape
        (64,64,1024)
        >>> s.valuemax(-1).data.shape
        (64,64)

        """
        s = self.indexmax(axis)
        s.data = self.axes_manager[axis].index2value(s.data)
        return s

    def get_histogram(self, bins='freedman', range_bins=None, **kwargs):
        """Return a histogram of the signal data.

        More sophisticated algorithms for determining bins can be used.
        Aside from the `bins` argument allowing a string specified how bins
        are computed, the parameters are the same as numpy.histogram().

        Parameters
        ----------
        bins : int or list or str, optional
            If bins is a string, then it must be one of:
            'knuth' : use Knuth's rule to determine bins
            'scotts' : use Scott's rule to determine bins
            'freedman' : use the Freedman-diaconis rule to determine bins
            'blocks' : use bayesian blocks for dynamic bin widths
        range_bins : tuple or None, optional
            the minimum and maximum range for the histogram. If not specified,
            it will be (x.min(), x.max())
        **kwargs
            other keyword arguments (weight and density) are described in
            np.histogram().

        Returns
        -------
        hist_spec : An 1D spectrum instance containing the histogram.

        See Also
        --------
        print_summary_statistics
        astroML.density_estimation.histogram, numpy.histogram : these are the
            functions that hyperspy uses to compute the histogram.

        Notes
        -----
        The number of bins estimators are taken from AstroML. Read
        their documentation for more info.

        Examples
        --------
        >>> s = hs.signals.Signal1D(np.random.normal(size=(10, 100)))
        Plot the data histogram
        >>> s.get_histogram().plot()
        Plot the histogram of the signal at the current coordinates
        >>> s.get_current_signal().get_histogram().plot()

        """
        from hyperspy import signals
        data = self.data[~np.isnan(self.data)].flatten()
        hist, bin_edges = histogram(data,
                                    bins=bins,
                                    range=range_bins,
                                    **kwargs)
        hist_spec = signals.Signal1D(hist)
        if bins == 'blocks':
            hist_spec.axes_manager.signal_axes[0].axis = bin_edges[:-1]
            warnings.warn(
                "The options `bins = 'blocks'` is not fully supported in this "
                "versions of hyperspy. It should be used for plotting purpose"
                "only.")
        else:
            hist_spec.axes_manager[0].scale = bin_edges[1] - bin_edges[0]
            hist_spec.axes_manager[0].offset = bin_edges[0]

        hist_spec.axes_manager[0].name = 'value'
        hist_spec.metadata.General.title = (self.metadata.General.title +
                                            " histogram")
        hist_spec.metadata.Signal.binned = True
        return hist_spec

    def map(self, function,
            show_progressbar=None, **kwargs):
        """Apply a function to the signal data at all the coordinates.

        The function must operate on numpy arrays and the output *must have the
        same dimensions as the input*. The function is applied to the data at
        each coordinate and the result is stored in the current signal i.e.
        this method operates *in-place*.  Any extra keyword argument is passed
        to the function. The keywords can take different values at different
        coordinates. If the function takes an `axis` or `axes` argument, the
        function is assumed to be vectorial and the signal axes are assigned to
        `axis` or `axes`.  Otherwise, the signal is iterated over the
        navigation axes and a progress bar is displayed to monitor the
        progress.

        Parameters
        ----------

        function : function
            A function that can be applied to the signal.
        show_progressbar : None or bool
            If True, display a progress bar. If None the default is set in
            `preferences`.
        keyword arguments : any valid keyword argument
            All extra keyword arguments are passed to the

        Notes
        -----
        This method is similar to Python's :func:`map` that can also be utilize
        with a :class:`Signal` instance for similar purposes. However, this
        method has the advantage of being faster because it iterates the numpy
        array instead of the :class:`Signal`.

        Examples
        --------
        Apply a gaussian filter to all the images in the dataset. The sigma
        parameter is constant.

        >>> import scipy.ndimage
        >>> im = hs.signals.Signal2D(np.random.random((10, 64, 64)))
        >>> im.map(scipy.ndimage.gaussian_filter, sigma=2.5)

        Apply a gaussian filter to all the images in the dataset. The sigmal
        parameter is variable.

        >>> im = hs.signals.Signal2D(np.random.random((10, 64, 64)))
        >>> sigmas = hs.signals.Signal(np.linspace(2,5,10))
        >>> sigmas.axes_manager.set_signal_dimension(0)
        >>> im.map(scipy.ndimage.gaussian_filter, sigma=sigmas)

        """
        if show_progressbar is None:
            show_progressbar = preferences.General.show_progressbar
        # Sepate ndkwargs
        ndkwargs = ()
        for key, value in kwargs.items():
            if isinstance(value, BaseSignal):
                ndkwargs += ((key, value),)

        # Check if the signal axes have inhomogenous scales and/or units and
        # display in warning if yes.
        scale = set()
        units = set()
        for i in range(len(self.axes_manager.signal_axes)):
            scale.add(self.axes_manager.signal_axes[i].scale)
            units.add(self.axes_manager.signal_axes[i].units)
        if len(units) != 1 or len(scale) != 1:
            warnings.warn(
                "The function you applied does not take into "
                "account the difference of units and of scales in-between"
                " axes.")
        # If the function has an axis argument and the signal dimension is 1,
        # we suppose that it can operate on the full array and we don't
        # interate over the coordinates.
        try:
            fargs = inspect.getargspec(function).args
        except TypeError:
            # This is probably a Cython function that is not supported by
            # inspect.
            fargs = []

        if not ndkwargs and (self.axes_manager.signal_dimension == 1 and
                             "axis" in fargs):
            kwargs['axis'] = \
                self.axes_manager.signal_axes[-1].index_in_array

            self.data = function(self.data, **kwargs)
        # If the function has an axes argument
        # we suppose that it can operate on the full array and we don't
        # interate over the coordinates.
        elif not ndkwargs and "axes" in fargs:
            kwargs['axes'] = tuple([axis.index_in_array for axis in
                                    self.axes_manager.signal_axes])
            self.data = function(self.data, **kwargs)
        else:
            # Iteration over coordinates.
            pbar = progressbar(
                maxval=self.axes_manager.navigation_size,
                disabled=not show_progressbar)
            iterators = [signal[1]._iterate_signal() for signal in ndkwargs]
            iterators = tuple([self._iterate_signal()] + iterators)
            for data in zip(*iterators):
                for (key, value), datum in zip(ndkwargs, data[1:]):
                    kwargs[key] = datum[0]
                data[0][:] = function(data[0], **kwargs)
                next(pbar)
            pbar.finish()

    def copy(self):
        try:
            backup_plot = self._plot
            self._plot = None
            return copy.copy(self)
        finally:
            self._plot = backup_plot

    def __deepcopy__(self, memo):
        dc = type(self)(**self._to_dictionary())
        if dc.data is not None:
            dc.data = dc.data.copy()
        # The Signal subclasses might change the view on init
        # The following code just copies the original view
        for oaxis, caxis in zip(self.axes_manager._axes,
                                dc.axes_manager._axes):
            caxis.navigate = oaxis.navigate
        return dc

    def deepcopy(self):
        return copy.deepcopy(self)

    def change_dtype(self, dtype):
        """Change the data type.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast. In
            addition to all standard numpy dtypes HyperSpy
            supports four extra dtypes for RGB images:
            "rgb8", "rgba8", "rgb16" and "rgba16". Changing from
            and to any rgbx dtype is more constrained than most
            other dtype conversions. To change to a rgbx dtype
            the signal `record_by` must be "spectrum",
            `signal_dimension` must be 3(4) for rgb(rgba) dtypes
            and the dtype must be uint8(uint16) for rgbx8(rgbx16).
            After conversion `record_by` becomes `image` and the
            spectra dimension is removed. The dtype of images of
            dtype rgbx8(rgbx16) can only be changed to uint8(uint16)
            and the `record_by` becomes "spectrum".


        Examples
        --------
        >>> s = hs.signals.Signal1D([1,2,3,4,5])
        >>> s.data
        array([1, 2, 3, 4, 5])
        >>> s.change_dtype('float')
        >>> s.data
        array([ 1.,  2.,  3.,  4.,  5.])

        """
        if not isinstance(dtype, np.dtype):
            if dtype in rgb_tools.rgb_dtypes:
                if self.metadata.Signal.record_by != "spectrum":
                    raise AttributeError(
                        "Only spectrum signals can be converted "
                        "to RGB images.")
                if "8" in dtype and self.data.dtype.name != "uint8":
                    raise AttributeError(
                        "Only signals with dtype uint8 can be converted to "
                        "rgb8 images")
                elif "16" in dtype and self.data.dtype.name != "uint16":
                    raise AttributeError(
                        "Only signals with dtype uint16 can be converted to "
                        "rgb16 images")
                dtype = rgb_tools.rgb_dtypes[dtype]
                self.data = rgb_tools.regular_array2rgbx(self.data)
                self.axes_manager.remove(-1)
                self.metadata.Signal.record_by = "image"
                self._assign_subclass()
                return
            else:
                dtype = np.dtype(dtype)
        if rgb_tools.is_rgbx(self.data) is True:
            ddtype = self.data.dtype.fields["B"][0]

            if ddtype != dtype:
                raise ValueError(
                    "It is only possibile to change to %s." %
                    ddtype)
            self.data = rgb_tools.rgbx2regular_array(self.data)
            self.get_dimensions_from_data()
            self.metadata.Signal.record_by = "spectrum"
            self.axes_manager[-1 + 2j].name = "RGB index"
            self._assign_subclass()
            return
        else:
            self.data = self.data.astype(dtype)

    def estimate_poissonian_noise_variance(self,
                                           expected_value=None,
                                           gain_factor=None,
                                           gain_offset=None,
                                           correlation_factor=None):
        """Estimate the poissonian noise variance of the signal.

        The variance is stored in the
        ``metadata.Signal.Noise_properties.variance`` attribute.

        A poissonian noise  variance is equal to the expected value. With the
        default arguments, this method simply sets the variance attribute to
        the given `expected_value`. However, more generally (although then
        noise is not strictly poissonian), the variance may be proportional to
        the expected value. Moreover, when the noise is a mixture of white
        (gaussian) and poissonian noise, the variance is described by the
        following linear model:

            .. math::

                \mathrm{Var}[X] = (a * \mathrm{E}[X] + b) * c

        Where `a` is the `gain_factor`, `b` is the `gain_offset` (the gaussian
        noise variance) and `c` the `correlation_factor`. The correlation
        factor accounts for correlation of adjacent signal elements that can
        be modeled as a convolution with a gaussian point spread function.


        Parameters
        ----------
        expected_value : None or Signal instance.
            If None, the signal data is taken as the expected value. Note that
            this may be inaccurate where `data` is small.
        gain_factor, gain_offset, correlation_factor: None or float.
            All three must be positive. If None, take the values from
            ``metadata.Signal.Noise_properties.Variance_linear_model`` if
            defined. Otherwise suppose poissonian noise i.e. ``gain_factor=1``,
            ``gain_offset=0``, ``correlation_factor=1``. If not None, the
            values are stored in
            ``metadata.Signal.Noise_properties.Variance_linear_model``.

        """
        if expected_value is None:
            dc = self.data.copy()
        else:
            dc = expected_value.data.copy()
        if self.metadata.has_item(
                "Signal.Noise_properties.Variance_linear_model"):
            vlm = self.metadata.Signal.Noise_properties.Variance_linear_model
        else:
            self.metadata.add_node(
                "Signal.Noise_properties.Variance_linear_model")
            vlm = self.metadata.Signal.Noise_properties.Variance_linear_model

        if gain_factor is None:
            if not vlm.has_item("gain_factor"):
                vlm.gain_factor = 1
            gain_factor = vlm.gain_factor

        if gain_offset is None:
            if not vlm.has_item("gain_offset"):
                vlm.gain_offset = 0
            gain_offset = vlm.gain_offset

        if correlation_factor is None:
            if not vlm.has_item("correlation_factor"):
                vlm.correlation_factor = 1
            correlation_factor = vlm.correlation_factor

        if gain_offset < 0:
            raise ValueError("`gain_offset` must be positive.")
        if gain_factor < 0:
            raise ValueError("`gain_factor` must be positive.")
        if correlation_factor < 0:
            raise ValueError("`correlation_factor` must be positive.")

        variance = (dc * gain_factor + gain_offset) * correlation_factor
        # The lower bound of the variance is the gaussian noise.
        variance = np.clip(variance, gain_offset * correlation_factor, np.inf)
        variance = type(self)(variance)
        variance.axes_manager = self.axes_manager
        variance.metadata.General.title = ("Variance of " +
                                           self.metadata.General.title)
        self.metadata.set_item(
            "Signal.Noise_properties.variance", variance)

    def get_current_signal(self, auto_title=True, auto_filename=True):
        """Returns the data at the current coordinates as a Signal subclass.

        The signal subclass is the same as that of the current object. All the
        axes navigation attribute are set to False.

        Parameters
        ----------
        auto_title : bool
            If True an space followed by the current indices in parenthesis
            are appended to the title.
        auto_filename : bool
            If True and `tmp_parameters.filename` is defined
            (what is always the case when the Signal has been read from a
            file), the filename is modified by appending an underscore and a
            parenthesis containing the current indices.

        Returns
        -------
        cs : Signal subclass instance.

        Examples
        --------
        >>> im = hs.signals.Signal2D(np.zeros((2,3, 32,32)))
        >>> im
        <Image, title: , dimensions: (3, 2, 32, 32)>
        >>> im.axes_manager.indices = 2,1
        >>> im.get_current_signal()
        <Image, title:  (2, 1), dimensions: (32, 32)>

        """
        cs = self.__class__(
            self(),
            axes=self.axes_manager._get_signal_axes_dicts(),
            metadata=self.metadata.as_dictionary(),)

        if auto_filename is True and self.tmp_parameters.has_item('filename'):
            cs.tmp_parameters.filename = (self.tmp_parameters.filename +
                                          '_' +
                                          str(self.axes_manager.indices))
            cs.tmp_parameters.extension = self.tmp_parameters.extension
            cs.tmp_parameters.folder = self.tmp_parameters.folder
        if auto_title is True:
            cs.metadata.General.title = (cs.metadata.General.title +
                                         ' ' + str(self.axes_manager.indices))
        cs.axes_manager._set_axis_attribute_values("navigate", False)
        return cs

    def _get_navigation_signal(self, data=None, dtype=None):
        """Return a signal with the same axes as the navigation space.

        Parameters
        ----------
        data : {None, numpy array}, optional
            If None the `Signal` data is an array of the same dtype as the
            current one filled with zeros. If a numpy array, the array must
            have the correct dimensions.

        dtype : data-type, optional
            The desired data-type for the data array when `data` is None,
            e.g., `numpy.int8`.  Default is the data type of the current signal
            data.


        """
        if data is not None:
            ref_shape = (self.axes_manager._navigation_shape_in_array
                         if self.axes_manager.navigation_dimension != 0
                         else (1,))
            if data.shape != ref_shape:
                raise ValueError(
                    ("data.shape %s is not equal to the current navigation "
                     "shape in array which is %s") %
                    (str(data.shape), str(ref_shape)))
        else:
            if dtype is None:
                dtype = self.data.dtype
            if self.axes_manager.navigation_dimension == 0:
                data = np.array([0, ], dtype=dtype)
            else:
                data = np.zeros(self.axes_manager._navigation_shape_in_array,
                                dtype=dtype)
        if self.axes_manager.navigation_dimension == 0:
            s = BaseSignal(data)
        elif self.axes_manager.navigation_dimension == 1:
            from hyperspy._signals.signal1d import Signal1D
            s = Signal1D(data,
                         axes=self.axes_manager._get_navigation_axes_dicts())
        elif self.axes_manager.navigation_dimension == 2:
            from hyperspy._signals.signal2d import Signal2D
            s = Signal2D(data,
                         axes=self.axes_manager._get_navigation_axes_dicts())
        else:
            s = BaseSignal(np.zeros(self.axes_manager._navigation_shape_in_array,
                                    dtype=self.data.dtype),
                           axes=self.axes_manager._get_navigation_axes_dicts())
            s.axes_manager.set_signal_dimension(
                self.axes_manager.navigation_dimension)
        return s

    def _get_signal_signal(self, data=None, dtype=None):
        """Return a signal with the same axes as the signal space.

        Parameters
        ----------
        data : {None, numpy array}, optional
            If None the `Signal` data is an array of the same dtype as the
            current one filled with zeros. If a numpy array, the array must
            have the correct dimensions.
        dtype : data-type, optional
            The desired data-type for the data array when `data` is None,
            e.g., `numpy.int8`.  Default is the data type of the current signal
            data.

        """

        if data is not None:
            ref_shape = (self.axes_manager._signal_shape_in_array
                         if self.axes_manager.signal_dimension != 0
                         else (1,))
            if data.shape != ref_shape:
                raise ValueError(
                    "data.shape %s is not equal to the current signal shape in"
                    " array which is %s" % (str(data.shape), str(ref_shape)))
        else:
            if dtype is None:
                dtype = self.data.dtype
            if self.axes_manager.signal_dimension == 0:
                data = np.array([0, ], dtype=dtype)
            else:
                data = np.zeros(self.axes_manager._signal_shape_in_array,
                                dtype=dtype)

        if self.axes_manager.signal_dimension == 0:
            s = BaseSignal(data)
            s.set_signal_type(self.metadata.Signal.signal_type)
        else:
            s = self.__class__(data,
                               axes=self.axes_manager._get_signal_axes_dicts())
        return s

    def __iter__(self):
        # Reset AxesManager iteration index
        self.axes_manager.__iter__()
        return self

    def __next__(self):
        next(self.axes_manager)
        return self.get_current_signal()

    def __len__(self):
        nitem = int(self.axes_manager.navigation_size)
        nitem = nitem if nitem > 0 else 1
        return nitem

    def as_signal1D(self, spectral_axis):
        """Return the Signal as a one-dimensional Signal1D.

        The chosen spectral axis is moved to the last index in the
        array and the data is made contiguous for effecient
        iteration over spectra.


        Parameters
        ----------
        spectral_axis : {int, complex, str}
            Select the spectral axis to-be using its index or name.

        Examples
        --------
        >>> img = hs.signals.Signal2D(np.ones((3,4,5,6)))
        >>> img
        <Signal2D, title: , dimensions: (4, 3, 6, 5)>
        >>> img.to_signal1D(-1+1j)
        <Signal1D, title: , dimensions: (6, 5, 4, 3)>
        >>> img.to_signal1D(0)
        <Signal1D, title: , dimensions: (6, 5, 3, 4)>

        """
        # Roll the spectral axis to-be to the latex index in the array
        sp = self.rollaxis(spectral_axis, -1 + 3j)
        sp.metadata.Signal.record_by = "spectrum"
        import hyperspy.io
        hyperspy.io.BAN_DEPRECATED = True
        try:
            sp._assign_subclass()
        except:
            raise
        finally:
            hyperspy.io.BAN_DEPRECATED = False
        return sp

    def as_spectrum(self, spectral_axis):
        """Return the Signal as a spectrum.

        The chosen spectral axis is moved to the last index in the
        array and the data is made contiguous for effecient
        iteration over spectra.


        Parameters
        ----------
        spectral_axis : {int, complex, str}
            Select the spectral axis to-be using its index or name.

        Examples
        --------
        >>> img = hs.signals.Signal2D(np.ones((3,4,5,6)))
        >>> img
        <Image, title: , dimensions: (4, 3, 6, 5)>
        >>> img.to_spectrum(-1+1j)
        <Spectrum, title: , dimensions: (6, 5, 4, 3)>
        >>> img.to_spectrum(0)
        <Spectrum, title: , dimensions: (6, 5, 3, 4)>

        """
        # Roll the spectral axis to-be to the latex index in the array
        warnings.warn("The as_spectrum method will be deprecated from version"
                      " 1.0.0 and replaced with as_signal1D",
                      VisibleDeprecationWarning)
        sp = self.rollaxis(spectral_axis, -1 + 3j)
        sp.metadata.Signal.record_by = "spectrum"
        sp._assign_subclass()
        return sp

    def as_signal2D(self, image_axes):
        """Convert signal to image.

        The chosen image axes are moved to the last indices in the
        array and the data is made contiguous for effecient
        iteration over images.

        Parameters
        ----------
        image_axes : tuple of {int, complex, str}
            Select the image axes. Note that the order of the axes matters
            and it is given in the "natural" i.e. X, Y, Z... order.

        Examples
        --------
        >>> s = hs.signals.Signal1D(np.ones((2,3,4,5)))
        >>> s
        <Spectrum, title: , dimensions: (4, 3, 2, 5)>
        >>> s.as_image((0,1))
        <Image, title: , dimensions: (5, 2, 4, 3)>

        >>> s.to_image((1,2))
        <Image, title: , dimensions: (4, 5, 3, 2)>

        Raises
        ------
        DataDimensionError : when data.ndim < 2

        """
        if self.data.ndim < 2:
            raise DataDimensionError(
                "A Signal dimension must be >= 2 to be converted to an Image")
        axes = (self.axes_manager[image_axes[0]],
                self.axes_manager[image_axes[1]])
        iaxes = [axis.index_in_array for axis in axes]
        im = self.rollaxis(iaxes[0] + 3j, -1 + 3j).rollaxis(
            iaxes[1] - np.argmax(iaxes) + 3j, -2 + 3j)
        im.metadata.Signal.record_by = "image"
        import hyperspy.io
        hyperspy.io.BAN_DEPRECATED = True
        try:
            im._assign_subclass()
        except:
            raise
        finally:
            hyperspy.io.BAN_DEPRECATED = False
        return im

    def as_image(self, image_axes):
        """Convert signal to image.

        The chosen image axes are moved to the last indices in the
        array and the data is made contiguous for effecient
        iteration over images.

        Parameters
        ----------
        image_axes : tuple of {int, complex, str}
            Select the image axes. Note that the order of the axes matters
            and it is given in the "natural" i.e. X, Y, Z... order.

        Examples
        --------
        >>> s = hs.signals.Signal1D(np.ones((2,3,4,5)))
        >>> s
        <Spectrum, title: , dimensions: (4, 3, 2, 5)>
        >>> s.as_image((0,1))
        <Image, title: , dimensions: (5, 2, 4, 3)>

        >>> s.to_image((1,2))
        <Image, title: , dimensions: (4, 5, 3, 2)>

        Raises
        ------
        DataDimensionError : when data.ndim < 2

        """
        warnings.warn("The as_image method will be deprecated from version"
                      " 1.0.0 and replaced with as_signal2D",
                      VisibleDeprecationWarning)
        if self.data.ndim < 2:
            raise DataDimensionError(
                "A Signal dimension must be >= 2 to be converted to an Image")
        axes = (self.axes_manager[image_axes[0]],
                self.axes_manager[image_axes[1]])
        iaxes = [axis.index_in_array for axis in axes]
        im = self.rollaxis(iaxes[0] + 3j, -1 + 3j).rollaxis(
            iaxes[1] - np.argmax(iaxes) + 3j, -2 + 3j)
        im.metadata.Signal.record_by = "image"
        im._assign_subclass()
        return im

    def _assign_subclass(self):
        mp = self.metadata
        self.__class__ = hyperspy.io.assign_signal_subclass(
            record_by=mp.Signal.record_by
            if "Signal.record_by" in mp
            else self._record_by,
            signal_type=mp.Signal.signal_type
            if "Signal.signal_type" in mp
            else self._signal_type,
            signal_origin=mp.Signal.signal_origin
            if "Signal.signal_origin" in mp
            else self._signal_origin)
        self.__init__(**self._to_dictionary())

    def set_signal_type(self, signal_type):
        """Set the signal type and change the current class
        accordingly if pertinent.

        The signal_type attribute specifies the kind of data that the signal
        containts e.g. "EELS" for electron energy-loss spectroscopy,
        "PES" for photoemission spectroscopy. There are some methods that are
        only available for certain kind of signals, so setting this
        parameter can enable/disable features.

        Parameters
        ----------
        signal_type : {"EELS", "EDS_TEM", "EDS_SEM", "DielectricFunction"}
            Currently there are special features for "EELS" (electron
            energy-loss spectroscopy), "EDS_TEM" (energy dispersive X-rays of
            thin samples, normally obtained in a transmission electron
            microscope), "EDS_SEM" (energy dispersive X-rays of thick samples,
            normally obtained in a scanning electron microscope) and
            "DielectricFuction". Setting the signal_type to the correct acronym
            is highly advisable when analyzing any signal for which HyperSpy
            provides extra features. Even if HyperSpy does not provide extra
            features for the signal that you are analyzing, it is good practice
            to set signal_type to a value that best describes the data signal
            type.

        """
        self.metadata.Signal.signal_type = signal_type
        self._assign_subclass()

    def set_signal_origin(self, origin):
        """Set the origin of the signal and change the current class
        accordingly if pertinent.

        The signal_origin attribute specifies if the data was obtained
        through experiment or simulation. There are some methods that are
        only available for experimental or simulated data, so setting this
        parameter can enable/disable features.


        Parameters
        ----------
        origin : {'experiment', 'simulation', None, ""}
            None an the empty string mean that the signal origin is uknown.

        Raises
        ------
        ValueError if origin is not 'experiment' or 'simulation'

        """
        if origin not in ['experiment', 'simulation', "", None]:
            raise ValueError("`origin` must be one of: experiment, simulation")
        if origin is None:
            origin = ""
        self.metadata.Signal.signal_origin = origin
        self._assign_subclass()

    def print_summary_statistics(self, formatter="%.3f"):
        """Prints the five-number summary statistics of the data, the mean and
        the standard deviation.

        Prints the mean, standandard deviation (std), maximum (max), minimum
        (min), first quartile (Q1), median and third quartile. nans are
        removed from the calculations.

        Parameters
        ----------
        formatter : bool
           Number formatter.

        See Also
        --------
        get_histogram

        """
        data = self.data
        # To make it work with nans
        data = data[~np.isnan(data)]
        print(underline("Summary statistics"))
        print("mean:\t" + formatter % data.mean())
        print("std:\t" + formatter % data.std())
        print()
        print("min:\t" + formatter % data.min())
        print("Q1:\t" + formatter % np.percentile(data,
                                                  25))
        print("median:\t" + formatter % np.median(data))
        print("Q3:\t" + formatter % np.percentile(data,
                                                  75))
        print("max:\t" + formatter % data.max())

    @property
    def is_rgba(self):
        return rgb_tools.is_rgba(self.data)

    @property
    def is_rgb(self):
        return rgb_tools.is_rgb(self.data)

    @property
    def is_rgbx(self):
        return rgb_tools.is_rgbx(self.data)

    def add_marker(self, marker, plot_on_signal=True, plot_marker=True):
        """
        Add a marker to the signal or navigator plot.

        Plot the signal, if not yet plotted

        Parameters
        ----------
        marker: `hyperspy.drawing._markers`
            the marker to add. see `plot.markers`
        plot_on_signal: bool
            If True, add the marker to the signal
            If False, add the marker to the navigator
        plot_marker: bool
            if True, plot the marker

        Examples
        -------
        >>> import scipy.misc
        >>> im = hs.signals.Signal2D(scipy.misc.lena())
        >>> m = hs.plot.markers.rectangle(x1=150, y1=100, x2=400,
        >>>                                  y2=400, color='red')
        >>> im.add_marker(m)

        """
        if self._plot is None:
            self.plot()
        if plot_on_signal:
            self._plot.signal_plot.add_marker(marker)
        else:
            self._plot.navigator_plot.add_marker(marker)
        if plot_marker:
            marker.plot()

    def create_model(self):
        """Create a model for the current signal

        Returns
        -------
        A Model class

        """
        from hyperspy.model import Model
        return Model(self)


ARITHMETIC_OPERATORS = (
    "__add__",
    "__sub__",
    "__mul__",
    "__floordiv__",
    "__mod__",
    "__divmod__",
    "__pow__",
    "__lshift__",
    "__rshift__",
    "__and__",
    "__xor__",
    "__or__",
    "__mod__",
    "__truediv__",
)
INPLACE_OPERATORS = (
    "__iadd__",
    "__isub__",
    "__imul__",
    "__itruediv__",
    "__ifloordiv__",
    "__imod__",
    "__ipow__",
    "__ilshift__",
    "__irshift__",
    "__iand__",
    "__ixor__",
    "__ior__",
)
COMPARISON_OPERATORS = (
    "__lt__",
    "__le__",
    "__eq__",
    "__ne__",
    "__ge__",
    "__gt__",
)
UNARY_OPERATORS = (
    "__neg__",
    "__pos__",
    "__abs__",
    "__invert__",
)
for name in ARITHMETIC_OPERATORS + INPLACE_OPERATORS + COMPARISON_OPERATORS:
    exec(
        ("def %s(self, other):\n" % name) +
        ("   return self._binary_operator_ruler(other, \'%s\')\n" %
         name))
    exec("%s.__doc__ = np.ndarray.%s.__doc__" % (name, name))
    exec("setattr(BaseSignal, \'%s\', %s)" % (name, name))
    # The following commented line enables the operators with swapped
    # operands. They should be defined only for commutative operators
    # but for simplicity we don't support this at all atm.

    # exec("setattr(BaseSignal, \'%s\', %s)" % (name[:2] + "r" + name[2:],
    # name))

# Implement unary arithmetic operations
for name in UNARY_OPERATORS:
    exec(
        ("def %s(self):" % name) +
        ("   return self._unary_operator_ruler(\'%s\')" % name))
    exec("%s.__doc__ = int.%s.__doc__" % (name, name))
    exec("setattr(BaseSignal, \'%s\', %s)" % (name, name))


class SpecialSlicers:

    def __init__(self, signal, isNavigation):
        self.isNavigation = isNavigation
        self.signal = signal

    def __getitem__(self, slices):
        return self.signal.__getitem__(slices, self.isNavigation)

    def __setitem__(self, i, j):
        """x.__setitem__(i, y) <==> x[i]=y
        """
        if isinstance(j, BaseSignal):
            j = j.data
        self.signal.__getitem__(i, self.isNavigation).data[:] = j

    def __len__(self):
        return self.signal.axes_manager.signal_shape[0]
