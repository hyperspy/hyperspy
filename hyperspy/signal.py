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
import inspect
import logging
import numbers
import warnings
from collections.abc import MutableMapping
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from itertools import product
from pathlib import Path

import dask.array as da
import numpy as np
import traits.api as t
from dask.diagnostics import ProgressBar
from matplotlib import pyplot as plt
from pint import UndefinedUnitError
from rsciio.utils import rgb_tools
from rsciio.utils.tools import ensure_directory
from scipy import integrate
from scipy import signal as sp_signal
from scipy.interpolate import make_interp_spline
from tlz import concat

from hyperspy.api import _ureg
from hyperspy.axes import AxesManager, create_axis
from hyperspy.docstrings.plot import (
    BASE_PLOT_DOCSTRING,
    BASE_PLOT_DOCSTRING_PARAMETERS,
    PLOT1D_DOCSTRING,
    PLOT2D_KWARGS_DOCSTRING,
)
from hyperspy.docstrings.signal import (
    CLUSTER_SIGNALS_ARG,
    HISTOGRAM_BIN_ARGS,
    HISTOGRAM_MAX_BIN_ARGS,
    HISTOGRAM_RANGE_ARGS,
    LAZY_OUTPUT_ARG,
    MANY_AXIS_PARAMETER,
    NAN_FUNC,
    NUM_WORKERS_ARG,
    ONE_AXIS_PARAMETER,
    OPTIMIZE_ARG,
    OUT_ARG,
    RECHUNK_ARG,
    SHOW_PROGRESSBAR_ARG,
)
from hyperspy.docstrings.utils import REBIN_ARGS
from hyperspy.drawing import mpl_he, mpl_hie, mpl_hse
from hyperspy.drawing import signal as sigdraw
from hyperspy.drawing.markers import markers_dict_to_markers
from hyperspy.drawing.utils import animate_legend
from hyperspy.events import Event, Events
from hyperspy.exceptions import (
    DataDimensionError,
    LazyCupyConversion,
    SignalDimensionError,
    VisibleDeprecationWarning,
)
from hyperspy.interactive import interactive
from hyperspy.io import assign_signal_subclass
from hyperspy.io import save as io_save
from hyperspy.learn.mva import MVA, LearningResults
from hyperspy.misc.array_tools import rebin as array_rebin
from hyperspy.misc.hist_tools import _set_histogram_metadata, histogram
from hyperspy.misc.math_tools import check_random_state, hann_window_nth_order, outer_nd
from hyperspy.misc.signal_tools import are_signals_aligned, broadcast_signals
from hyperspy.misc.slicing import FancySlicing, SpecialSlicers
from hyperspy.misc.utils import (
    DictionaryTreeBrowser,
    _get_block_pattern,
    add_scalar_axis,
    dummy_context_manager,
    guess_output_signal_size,
    is_cupy_array,
    isiterable,
    iterable_not_string,
    process_function_blockwise,
    rollelem,
    slugify,
    to_numpy,
    underline,
)

_logger = logging.getLogger(__name__)

try:
    import cupy as cp

    CUPY_INSTALLED = True  # pragma: no cover
except ImportError:
    CUPY_INSTALLED = False


def _dic_get_hs_obj_paths(dic, axes_managers, signals, containers, axes):
    for key in dic:
        if key.startswith("_sig_"):
            signals.append((key, dic))
        elif key.startswith("_hspy_AxesManager_"):
            axes_managers.append((key, dic))
        elif key.startswith("_hspy_Axis_"):
            axes.append((key, dic))
        elif isinstance(dic[key], (list, tuple)):
            signals = []
            # Support storing signals in containers
            for i, item in enumerate(dic[key]):
                if isinstance(item, dict) and "_sig_" in item:
                    signals.append(i)
            if signals:
                containers.append(((dic, key), signals))
        elif isinstance(dic[key], dict):
            _dic_get_hs_obj_paths(
                dic[key],
                axes_managers=axes_managers,
                signals=signals,
                containers=containers,
                axes=axes,
            )


def _obj_in_dict2hspy(dic, lazy):
    """
    Recursively walk nested dicts substituting dicts with their hyperspy
    object where relevant

    Parameters
    ----------
    d: dictionary
        The nested dictionary
    lazy: bool

    """
    from hyperspy.io import dict2signal

    axes_managers, signals, containers, axes = [], [], [], []
    _dic_get_hs_obj_paths(
        dic,
        axes_managers=axes_managers,
        signals=signals,
        containers=containers,
        axes=axes,
    )
    for key, dic in axes_managers:
        dic[key[len("_hspy_AxesManager_") :]] = AxesManager(dic[key])
        del dic[key]
    for key, dic in axes:
        dic[key[len("_hspy_Axis_") :]] = create_axis(**dic[key])
    for key, dic in signals:
        dic[key[len("_sig_") :]] = dict2signal(dic[key], lazy=lazy)
        del dic[key]
    for dickey, signals_idx in containers:
        dic, key = dickey
        container = dic[key]
        to_tuple = False
        if type(container) is tuple:
            container = list(container)
            to_tuple = True
        for idx in signals_idx:
            container[idx] = dict2signal(container[idx]["_sig_"], lazy=lazy)
        if to_tuple:
            dic[key] = tuple(container)


class ModelManager(object):
    """Container for models"""

    class ModelStub(object):
        def __init__(self, mm, name):
            self._name = name
            self._mm = mm
            self.restore = lambda: mm.restore(self._name)
            self.remove = lambda: mm.remove(self._name)
            self.pop = lambda: mm.pop(self._name)
            self.restore.__doc__ = "Returns the stored model"
            self.remove.__doc__ = "Removes the stored model"
            self.pop.__doc__ = "Returns the stored model and removes it from storage"

        def __repr__(self):
            return repr(self._mm._models[self._name])

    def __init__(self, signal, dictionary=None):
        self._signal = signal
        self._models = DictionaryTreeBrowser()
        self._add_dictionary(dictionary)

    def _add_dictionary(self, dictionary=None):
        if dictionary is not None:
            for k, v in dictionary.items():
                if k.startswith("_") or k in ["restore", "remove"]:
                    raise KeyError("Can't add dictionary with key '%s'" % k)
                k = slugify(k, True)
                self._models.set_item(k, v)
                setattr(self, k, self.ModelStub(self, k))

    def _set_nice_description(self, node, names):
        ans = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dimensions": self._signal.axes_manager._get_dimension_str(),
        }
        node.add_dictionary(ans)
        for n in names:
            node.add_node("components." + n)

    def _save(self, name, dictionary):
        _abc = "abcdefghijklmnopqrstuvwxyz"

        def get_letter(models):
            howmany = len(models)
            if not howmany:
                return "a"
            order = int(np.log(howmany) / np.log(26)) + 1
            letters = [
                _abc,
            ] * order
            for comb in product(*letters):
                guess = "".join(comb)
                if guess not in models.keys():
                    return guess

        if name is None:
            name = get_letter(self._models)
        else:
            name = self._check_name(name)

        if name in self._models:
            self.remove(name)

        self._models.add_node(name)
        node = self._models.get_item(name)
        names = [c["name"] for c in dictionary["components"]]
        self._set_nice_description(node, names)

        node.set_item("_dict", dictionary)
        setattr(self, name, self.ModelStub(self, name))

    def store(self, model, name=None):
        """If the given model was created from this signal, stores it

        Parameters
        ----------
        model : :class:`~hyperspy.model.BaseModel` (or subclass)
            The model to store in the signal
        name : str or None
            The name for the model to be stored with

        See Also
        --------
        remove
        restore
        pop

        """
        if model.signal is self._signal:
            self._save(name, model.as_dictionary())
        else:
            raise ValueError(
                "The model is created from a different signal, "
                "you should store it there"
            )

    def _check_name(self, name, existing=False):
        if not isinstance(name, str):
            raise KeyError("Name has to be a string")
        if name.startswith("_"):
            raise KeyError('Name cannot start with "_" symbol')
        if "." in name:
            raise KeyError('Name cannot contain dots (".")')
        name = slugify(name, True)
        if existing:
            if name not in self._models:
                raise KeyError("Model named '%s' is not currently stored" % name)
        return name

    def remove(self, name):
        """Removes the given model

        Parameters
        ----------
        name : str
            The name of the model to remove

        See Also
        --------
        restore
        store
        pop

        """
        name = self._check_name(name, True)
        delattr(self, name)
        self._models.__delattr__(name)

    def pop(self, name):
        """Returns the restored model and removes it from storage

        Parameters
        ----------
        name : str
            The name of the model to restore and remove

        See Also
        --------
        restore
        store
        remove

        """
        name = self._check_name(name, True)
        model = self.restore(name)
        self.remove(name)
        return model

    def restore(self, name):
        """Returns the restored model

        Parameters
        ----------
        name : str
            The name of the model to restore

        See Also
        --------
        remove
        store
        pop

        """
        name = self._check_name(name, True)
        d = self._models.get_item(name + "._dict").as_dictionary()
        for c in d["components"]:
            # Restoring of polynomials saved with Hyperspy <v2.1.1 and >= 1.5
            # the order initialisation argument was missing from the whitelist
            # We patch the dictionary to be able to load these files
            if (
                c["_id_name"] == "Polynomial"
                # make sure that this is not an old polynomial
                and c["parameters"][0]["_id_name"] != "coefficients"
                and "order" not in c.keys()
            ):
                c["_whitelist"]["order"] = "init"
                c["order"] = len(c["parameters"]) - 1
        return self._signal.create_model(dictionary=copy.deepcopy(d))

    def __repr__(self):
        return repr(self._models)

    def __len__(self):
        return len(self._models)

    def __getitem__(self, name):
        name = self._check_name(name, True)
        return getattr(self, name)


class MVATools(object):
    # TODO: All of the plotting methods here should move to drawing

    def _plot_factors_or_pchars(
        self,
        factors,
        comp_ids=None,
        calibrate=True,
        avg_char=False,
        same_window=True,
        comp_label="PC",
        img_data=None,
        plot_shifts=True,
        plot_char=4,
        cmap=plt.cm.gray,
        quiver_color="white",
        vector_scale=1,
        per_row=3,
        ax=None,
    ):
        """Plot components from PCA or ICA, or peak characteristics.

        Parameters
        ----------
        comp_ids : None, int, or list of ints
            If None, returns maps of all components.
            If int, returns maps of components with ids from 0 to given
            int.
            if list of ints, returns maps of components with ids in
            given list.
        calibrate : bool
            If True, plots are calibrated according to the data in the
            axes manager.
        same_window : bool
            If True, plots each factor to the same window. They are not scaled.
            Default True.
        comp_label : str
            Title of the plot
        cmap : a matplotlib colormap
            The colormap used for factor images or any peak characteristic
            scatter map overlay. Default is the matplotlib gray colormap
            (``plt.cm.gray``).

        Other Parameters
        ----------------
        img_data : 2D numpy array,
            The array to overlay peak characteristics onto. If None,
            defaults to the average image of your stack.
        plot_shifts : bool, default is True
            If true, plots a quiver (arrow) plot showing the shifts for
            each
            peak present in the component being plotted.
        plot_char : None or int
            If int, the id of the characteristic to plot as the colored
            scatter plot.
            Possible components are:

            * 4: peak height
            * 5: peak orientation
            * 6: peak eccentricity
        quiver_color : any color recognized by matplotlib
            Determines the color of vectors drawn for
            plotting peak shifts.
        vector_scale : integer or None
            Scales the quiver plot arrows. The vector is defined as one data
            unit along the X axis. If shifts are small, set vector_scale so
            that when they are multiplied by vector_scale, they are on the
            scale of the image plot. If None, uses matplotlib's autoscaling.

        Returns
        -------
        matplotlib figure or list of figure if same_window=False

        """
        if same_window is None:
            same_window = True
        if comp_ids is None:
            comp_ids = range(factors.shape[1])

        elif not hasattr(comp_ids, "__iter__"):
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
                        plt.title("%s" % comp_label)
                    ax = f.add_subplot(111)
                ax = sigdraw._plot_1D_component(
                    factors=factors,
                    idx=comp_ids[i],
                    axes_manager=self.axes_manager,
                    ax=ax,
                    calibrate=calibrate,
                    comp_label=comp_label,
                    same_window=same_window,
                )
                if same_window:
                    plt.legend(ncol=factors.shape[1] // 2, loc="best")
            elif self.axes_manager.signal_dimension == 2:
                if same_window:
                    ax = f.add_subplot(rows, per_row, i + 1)
                else:
                    if i > 0:
                        f = plt.figure()
                        plt.title("%s" % comp_label)
                    ax = f.add_subplot(111)

                sigdraw._plot_2D_component(
                    factors=factors,
                    idx=comp_ids[i],
                    axes_manager=self.axes_manager,
                    calibrate=calibrate,
                    ax=ax,
                    cmap=cmap,
                    comp_label=comp_label,
                )
            if not same_window:
                fig_list.append(f)
        if same_window:  # Main title for same window
            title = "%s" % comp_label
            if self.axes_manager.signal_dimension == 1:
                plt.title(title)
            else:
                plt.suptitle(title)
        try:
            plt.tight_layout()
        except BaseException:
            pass
        if not same_window:
            return fig_list
        else:
            return f

    def _plot_loadings(
        self,
        loadings,
        comp_ids,
        calibrate=True,
        same_window=True,
        comp_label=None,
        with_factors=False,
        factors=None,
        cmap=plt.cm.gray,
        no_nans=False,
        per_row=3,
        axes_decor="all",
    ):
        if same_window is None:
            same_window = True
        if comp_ids is None:
            comp_ids = range(loadings.shape[0])

        elif not hasattr(comp_ids, "__iter__"):
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
                        plt.title("%s" % comp_label)
                    ax = f.add_subplot(111)
            elif self.axes_manager.navigation_dimension == 2:
                if same_window:
                    ax = f.add_subplot(rows, per_row, i + 1)
                else:
                    if i > 0:
                        f = plt.figure()
                        plt.title("%s" % comp_label)
                    ax = f.add_subplot(111)
            sigdraw._plot_loading(
                loadings,
                idx=comp_ids[i],
                axes_manager=self.axes_manager,
                no_nans=no_nans,
                calibrate=calibrate,
                cmap=cmap,
                comp_label=comp_label,
                ax=ax,
                same_window=same_window,
                axes_decor=axes_decor,
            )
            if not same_window:
                fig_list.append(f)
        if same_window:  # Main title for same window
            title = "%s" % comp_label
            if self.axes_manager.navigation_dimension == 1:
                plt.title(title)
            else:
                plt.suptitle(title)
        try:
            plt.tight_layout()
        except BaseException:
            pass
        if not same_window:
            if with_factors:
                return fig_list, self._plot_factors_or_pchars(
                    factors,
                    comp_ids=comp_ids,
                    calibrate=calibrate,
                    same_window=same_window,
                    comp_label=comp_label,
                    per_row=per_row,
                )
            else:
                return fig_list
        else:
            if self.axes_manager.navigation_dimension == 1:
                plt.legend(ncol=loadings.shape[0] // 2, loc="best")
                animate_legend(f)
            if with_factors:
                return f, self._plot_factors_or_pchars(
                    factors,
                    comp_ids=comp_ids,
                    calibrate=calibrate,
                    same_window=same_window,
                    comp_label=comp_label,
                    per_row=per_row,
                )
            else:
                return f

    def _export_factors(
        self,
        factors,
        folder=None,
        comp_ids=None,
        multiple_files=True,
        save_figures=False,
        save_figures_format="png",
        factor_prefix=None,
        factor_format=None,
        comp_label=None,
        cmap=plt.cm.gray,
        plot_shifts=True,
        plot_char=4,
        img_data=None,
        same_window=False,
        calibrate=True,
        quiver_color="white",
        vector_scale=1,
        no_nans=True,
        per_row=3,
    ):
        from hyperspy._signals.signal1d import Signal1D
        from hyperspy._signals.signal2d import Signal2D

        if multiple_files is None:
            multiple_files = True

        if factor_format is None:
            factor_format = "hspy"

        # Select the desired factors
        if comp_ids is None:
            comp_ids = range(factors.shape[1])
        elif not hasattr(comp_ids, "__iter__"):
            comp_ids = range(comp_ids)
        mask = np.zeros(factors.shape[1], dtype=np.bool)
        for idx in comp_ids:
            mask[idx] = 1
        factors = factors[:, mask]

        if save_figures is True:
            plt.ioff()
            fac_plots = self._plot_factors_or_pchars(
                factors,
                comp_ids=comp_ids,
                same_window=same_window,
                comp_label=comp_label,
                img_data=img_data,
                plot_shifts=plot_shifts,
                plot_char=plot_char,
                cmap=cmap,
                per_row=per_row,
                quiver_color=quiver_color,
                vector_scale=vector_scale,
            )
            for idx in range(len(comp_ids)):
                filename = "%s_%02i.%s" % (
                    factor_prefix,
                    comp_ids[idx],
                    save_figures_format,
                )
                if folder is not None:
                    filename = Path(folder, filename)
                ensure_directory(filename)
                _args = {"dpi": 600, "format": save_figures_format}
                fac_plots[idx].savefig(filename, **_args)
            plt.ion()

        elif multiple_files is False:
            if self.axes_manager.signal_dimension == 2:
                # factor images
                axes_dicts = []
                axes = self.axes_manager.signal_axes[::-1]
                shape = (axes[1].size, axes[0].size)
                factor_data = np.rollaxis(factors.reshape((shape[0], shape[1], -1)), 2)
                axes_dicts.append(axes[0].get_axis_dictionary())
                axes_dicts.append(axes[1].get_axis_dictionary())
                axes_dicts.append(
                    {
                        "name": "factor_index",
                        "scale": 1.0,
                        "offset": 0.0,
                        "size": int(factors.shape[1]),
                        "units": "factor",
                        "index_in_array": 0,
                    }
                )
                s = Signal2D(
                    factor_data,
                    axes=axes_dicts,
                    metadata={
                        "General": {
                            "title": "%s from %s"
                            % (factor_prefix, self.metadata.General.title),
                        }
                    },
                )
            elif self.axes_manager.signal_dimension == 1:
                axes = [
                    self.axes_manager.signal_axes[0].get_axis_dictionary(),
                    {
                        "name": "factor_index",
                        "scale": 1.0,
                        "offset": 0.0,
                        "size": int(factors.shape[1]),
                        "units": "factor",
                        "index_in_array": 0,
                    },
                ]
                axes[0]["index_in_array"] = 1
                s = Signal1D(
                    factors.T,
                    axes=axes,
                    metadata={
                        "General": {
                            "title": "%s from %s"
                            % (factor_prefix, self.metadata.General.title),
                        }
                    },
                )
            filename = "%ss.%s" % (factor_prefix, factor_format)
            if folder is not None:
                filename = Path(folder, filename)
            s.save(filename)
        else:  # Separate files
            if self.axes_manager.signal_dimension == 1:
                axis_dict = self.axes_manager.signal_axes[0].get_axis_dictionary()
                axis_dict["index_in_array"] = 0
                for dim, index in zip(comp_ids, range(len(comp_ids))):
                    s = Signal1D(
                        factors[:, index],
                        axes=[
                            axis_dict,
                        ],
                        metadata={
                            "General": {
                                "title": "%s from %s"
                                % (factor_prefix, self.metadata.General.title),
                            }
                        },
                    )
                    filename = "%s-%i.%s" % (factor_prefix, dim, factor_format)
                    if folder is not None:
                        filename = Path(folder, filename)
                    s.save(filename)

            if self.axes_manager.signal_dimension == 2:
                axes = self.axes_manager.signal_axes
                axes_dicts = [
                    axes[0].get_axis_dictionary(),
                    axes[1].get_axis_dictionary(),
                ]
                axes_dicts[0]["index_in_array"] = 0
                axes_dicts[1]["index_in_array"] = 1

                factor_data = factors.reshape(
                    self.axes_manager._signal_shape_in_array
                    + [
                        -1,
                    ]
                )

                for dim, index in zip(comp_ids, range(len(comp_ids))):
                    im = Signal2D(
                        factor_data[..., index],
                        axes=axes_dicts,
                        metadata={
                            "General": {
                                "title": "%s from %s"
                                % (factor_prefix, self.metadata.General.title),
                            }
                        },
                    )
                    filename = "%s-%i.%s" % (factor_prefix, dim, factor_format)
                    if folder is not None:
                        filename = Path(folder, filename)
                    im.save(filename)

    def _export_loadings(
        self,
        loadings,
        folder=None,
        comp_ids=None,
        multiple_files=True,
        loading_prefix=None,
        loading_format="hspy",
        save_figures_format="png",
        comp_label=None,
        cmap=plt.cm.gray,
        save_figures=False,
        same_window=False,
        calibrate=True,
        no_nans=True,
        per_row=3,
    ):
        from hyperspy._signals.signal1d import Signal1D
        from hyperspy._signals.signal2d import Signal2D

        if multiple_files is None:
            multiple_files = True

        if loading_format is None:
            loading_format = "hspy"

        if comp_ids is None:
            comp_ids = range(loadings.shape[0])
        elif not hasattr(comp_ids, "__iter__"):
            comp_ids = range(comp_ids)
        mask = np.zeros(loadings.shape[0], dtype=np.bool)
        for idx in comp_ids:
            mask[idx] = 1
        loadings = loadings[mask]

        if save_figures is True:
            plt.ioff()
            sc_plots = self._plot_loadings(
                loadings,
                comp_ids=comp_ids,
                calibrate=calibrate,
                same_window=same_window,
                comp_label=comp_label,
                cmap=cmap,
                no_nans=no_nans,
                per_row=per_row,
            )
            for idx in range(len(comp_ids)):
                filename = "%s_%02i.%s" % (
                    loading_prefix,
                    comp_ids[idx],
                    save_figures_format,
                )
                if folder is not None:
                    filename = Path(folder, filename)
                ensure_directory(filename)
                _args = {"dpi": 600, "format": save_figures_format}
                sc_plots[idx].savefig(filename, **_args)
            plt.ion()
        elif multiple_files is False:
            if self.axes_manager.navigation_dimension == 2:
                axes_dicts = []
                axes = self.axes_manager.navigation_axes[::-1]
                shape = (axes[1].size, axes[0].size)
                loading_data = loadings.reshape((-1, shape[0], shape[1]))
                axes_dicts.append(axes[0].get_axis_dictionary())
                axes_dicts[0]["index_in_array"] = 1
                axes_dicts.append(axes[1].get_axis_dictionary())
                axes_dicts[1]["index_in_array"] = 2
                axes_dicts.append(
                    {
                        "name": "loading_index",
                        "scale": 1.0,
                        "offset": 0.0,
                        "size": int(loadings.shape[0]),
                        "units": "factor",
                        "index_in_array": 0,
                    }
                )
                s = Signal2D(
                    loading_data,
                    axes=axes_dicts,
                    metadata={
                        "General": {
                            "title": "%s from %s"
                            % (loading_prefix, self.metadata.General.title),
                        }
                    },
                )
            elif self.axes_manager.navigation_dimension == 1:
                cal_axis = self.axes_manager.navigation_axes[0].get_axis_dictionary()
                cal_axis["index_in_array"] = 1
                axes = [
                    {
                        "name": "loading_index",
                        "scale": 1.0,
                        "offset": 0.0,
                        "size": int(loadings.shape[0]),
                        "units": "comp_id",
                        "index_in_array": 0,
                    },
                    cal_axis,
                ]
                s = Signal2D(
                    loadings,
                    axes=axes,
                    metadata={
                        "General": {
                            "title": "%s from %s"
                            % (loading_prefix, self.metadata.General.title),
                        }
                    },
                )
            filename = "%ss.%s" % (loading_prefix, loading_format)
            if folder is not None:
                filename = Path(folder, filename)
            s.save(filename)
        else:  # Separate files
            if self.axes_manager.navigation_dimension == 1:
                axis_dict = self.axes_manager.navigation_axes[0].get_axis_dictionary()
                axis_dict["index_in_array"] = 0
                for dim, index in zip(comp_ids, range(len(comp_ids))):
                    s = Signal1D(
                        loadings[index],
                        axes=[
                            axis_dict,
                        ],
                    )
                    filename = "%s-%i.%s" % (loading_prefix, dim, loading_format)
                    if folder is not None:
                        filename = Path(folder, filename)
                    s.save(filename)
            elif self.axes_manager.navigation_dimension == 2:
                axes_dicts = []
                axes = self.axes_manager.navigation_axes[::-1]
                shape = (axes[0].size, axes[1].size)
                loading_data = loadings.reshape((-1, shape[0], shape[1]))
                axes_dicts.append(axes[0].get_axis_dictionary())
                axes_dicts[0]["index_in_array"] = 0
                axes_dicts.append(axes[1].get_axis_dictionary())
                axes_dicts[1]["index_in_array"] = 1
                for dim, index in zip(comp_ids, range(len(comp_ids))):
                    s = Signal2D(
                        loading_data[index, ...],
                        axes=axes_dicts,
                        metadata={
                            "General": {
                                "title": "%s from %s"
                                % (loading_prefix, self.metadata.General.title),
                            }
                        },
                    )
                    filename = "%s-%i.%s" % (loading_prefix, dim, loading_format)
                    if folder is not None:
                        filename = Path(folder, filename)
                    s.save(filename)

    def plot_decomposition_factors(
        self,
        comp_ids=None,
        calibrate=True,
        same_window=True,
        title=None,
        cmap=plt.cm.gray,
        per_row=3,
        **kwargs,
    ):
        """Plot factors from a decomposition. In case of 1D signal axis, each
        factors line can be toggled on and off by clicking on their
        corresponding line in the legend.

        Parameters
        ----------
        comp_ids : None, int or list of int
            If `comp_ids` is ``None``, maps of all components will be
            returned if the `output_dimension` was defined when executing
            :meth:`~hyperspy.api.signals.BaseSignal.decomposition`. Otherwise it
            raises a :exc:`ValueError`.
            If `comp_ids` is an int, maps of components with ids from 0 to
            the given value will be returned. If `comp_ids` is a list of
            ints, maps of components with ids contained in the list will be
            returned.
        calibrate : bool
            If ``True``, calibrates plots where calibration is available
            from the axes_manager.  If ``False``, plots are in pixels/channels.
        same_window : bool
            If ``True``, plots each factor to the same window.  They are
            not scaled. Default is ``True``.
        title : str
            Title of the matplotlib plot or label of the line in the legend
            when the dimension of factors is 1 and ``same_window`` is ``True``.
        cmap : :class:`~matplotlib.colors.Colormap`
            The colormap used for the factor images, or for peak
            characteristics. Default is the matplotlib gray colormap
            (``plt.cm.gray``).
        per_row : int
            The number of plots in each row, when the `same_window`
            parameter is ``True``.

        See Also
        --------
        plot_decomposition_loadings, plot_decomposition_results

        """
        if self.axes_manager.signal_dimension > 2:
            raise NotImplementedError(
                "This method cannot plot factors of "
                "signals of dimension higher than 2."
                "You can use "
                "`plot_decomposition_results` instead."
            )
        if self.learning_results.factors is None:
            raise RuntimeError(
                "No learning results found. A 'decomposition' "
                "needs to be performed first."
            )
        if same_window is None:
            same_window = True
        if self.learning_results.factors is None:
            raise RuntimeError("Run a decomposition first.")
        factors = self.learning_results.factors
        if comp_ids is None:
            if self.learning_results.output_dimension:
                comp_ids = self.learning_results.output_dimension
            else:
                raise ValueError(
                    "Please provide the number of components to plot via the "
                    "`comp_ids` argument."
                )
        if title is None:
            title = self._get_plot_title(
                "Decomposition factors of", same_window=same_window
            )

        return self._plot_factors_or_pchars(
            factors,
            comp_ids=comp_ids,
            calibrate=calibrate,
            same_window=same_window,
            comp_label=title,
            cmap=cmap,
            per_row=per_row,
        )

    def plot_bss_factors(
        self,
        comp_ids=None,
        calibrate=True,
        same_window=True,
        title=None,
        cmap=plt.cm.gray,
        per_row=3,
        **kwargs,
    ):
        """Plot factors from blind source separation results. In case of 1D
        signal axis, each factors line can be toggled on and off by clicking
        on their corresponding line in the legend.

        Parameters
        ----------

        comp_ids : None, int, or list of int
            If `comp_ids` is ``None``, maps of all components will be
            returned. If it is an int, maps of components with ids from 0 to
            the given value will be returned. If `comp_ids` is a list of
            ints, maps of components with ids contained in the list will be
            returned.
        calibrate : bool
            If ``True``, calibrates plots where calibration is available
            from the axes_manager.  If ``False``, plots are in pixels/channels.
        same_window : bool
            if ``True``, plots each factor to the same window.  They are
            not scaled. Default is ``True``.
        title : str
            Title of the matplotlib plot or label of the line in the legend
            when the dimension of factors is 1 and ``same_window`` is ``True``.
        cmap : :class:`~matplotlib.colors.Colormap`
            The colormap used for the factor images, or for peak
            characteristics. Default is the matplotlib gray colormap
            (``plt.cm.gray``).
        per_row : int
            The number of plots in each row, when the `same_window`
            parameter is ``True``.

        See Also
        --------
        plot_bss_loadings, plot_bss_results

        """
        if self.axes_manager.signal_dimension > 2:
            raise NotImplementedError(
                "This method cannot plot factors of "
                "signals of dimension higher than 2."
                "You can use "
                "`plot_decomposition_results` instead."
            )
        if self.learning_results.bss_factors is None:
            raise RuntimeError(
                "No learning results found. A "
                "'blind_source_separation' needs to be "
                "performed first."
            )

        if same_window is None:
            same_window = True
        factors = self.learning_results.bss_factors
        if title is None:
            title = self._get_plot_title("BSS factors of", same_window=same_window)

        return self._plot_factors_or_pchars(
            factors,
            comp_ids=comp_ids,
            calibrate=calibrate,
            same_window=same_window,
            comp_label=title,
            per_row=per_row,
        )

    def plot_decomposition_loadings(
        self,
        comp_ids=None,
        calibrate=True,
        same_window=True,
        title=None,
        with_factors=False,
        cmap=plt.cm.gray,
        no_nans=False,
        per_row=3,
        axes_decor="all",
        **kwargs,
    ):
        """Plot loadings from a decomposition. In case of 1D navigation axis,
        each loading line can be toggled on and off by clicking on the legended
        line.

        Parameters
        ----------
        comp_ids : None, int, or list of int
            If `comp_ids` is ``None``, maps of all components will be
            returned if the `output_dimension` was defined when executing
            :meth:`~hyperspy.api.signals.BaseSignal.decomposition`.
            Otherwise it raises a :exc:`ValueError`.
            If `comp_ids` is an int, maps of components with ids from 0 to
            the given value will be returned. If `comp_ids` is a list of
            ints, maps of components with ids contained in the list will be
            returned.
        calibrate : bool
            if ``True``, calibrates plots where calibration is available
            from the axes_manager. If ``False``, plots are in pixels/channels.
        same_window : bool
            if ``True``, plots each factor to the same window. They are
            not scaled. Default is ``True``.
        title : str
            Title of the matplotlib plot or label of the line in the legend
            when the dimension of loadings is 1 and ``same_window`` is ``True``.
        with_factors : bool
            If ``True``, also returns figure(s) with the factors for the
            given comp_ids.
        cmap : :class:`~matplotlib.colors.Colormap`
            The colormap used for the loadings images, or for peak
            characteristics. Default is the matplotlib gray colormap
            (``plt.cm.gray``).
        no_nans : bool
            If ``True``, removes ``NaN``'s from the loading plots.
        per_row : int
            The number of plots in each row, when the `same_window`
            parameter is ``True``.
        axes_decor : str or None, optional
            One of: ``'all'``, ``'ticks'``, ``'off'``, or ``None``
            Controls how the axes are displayed on each image; default is
            ``'all'``
            If ``'all'``, both ticks and axis labels will be shown.
            If ``'ticks'``, no axis labels will be shown, but ticks/labels will.
            If ``'off'``, all decorations and frame will be disabled.
            If ``None``, no axis decorations will be shown, but ticks/frame
            will.

        See Also
        --------
        plot_decomposition_factors, plot_decomposition_results

        """
        if self.axes_manager.navigation_dimension > 2:
            raise NotImplementedError(
                "This method cannot plot loadings of "
                "dimension higher than 2."
                "You can use "
                "`plot_decomposition_results` instead."
            )
        if self.learning_results.loadings is None:
            raise RuntimeError(
                "No learning results found. A 'decomposition' "
                "needs to be performed first."
            )
        if same_window is None:
            same_window = True
        if self.learning_results.loadings is None:
            raise RuntimeError("Run a decomposition first.")
        loadings = self.learning_results.loadings.T
        if with_factors:
            factors = self.learning_results.factors
        else:
            factors = None

        if comp_ids is None:
            if self.learning_results.output_dimension:
                comp_ids = self.learning_results.output_dimension
            else:
                raise ValueError(
                    "Please provide the number of components to plot via the "
                    "`comp_ids` argument"
                )

        if title is None:
            title = self._get_plot_title(
                "Decomposition loadings of", same_window=same_window
            )

        return self._plot_loadings(
            loadings,
            comp_ids=comp_ids,
            with_factors=with_factors,
            factors=factors,
            same_window=same_window,
            comp_label=title,
            cmap=cmap,
            no_nans=no_nans,
            per_row=per_row,
            axes_decor=axes_decor,
        )

    def plot_bss_loadings(
        self,
        comp_ids=None,
        calibrate=True,
        same_window=True,
        title=None,
        with_factors=False,
        cmap=plt.cm.gray,
        no_nans=False,
        per_row=3,
        axes_decor="all",
        **kwargs,
    ):
        """Plot loadings from blind source separation results. In case of 1D
        navigation axis, each loading line can be toggled on and off by
        clicking on their corresponding line in the legend.

        Parameters
        ----------
        comp_ids : None, int or list of int
            If ``comp_ids=None``, maps of all components will be
            returned. If it is an int, maps of components with ids from 0 to
            the given value will be returned. If `comp_ids` is a list of
            ints, maps of components with ids contained in the list will be
            returned.
        calibrate : bool
            if ``True``, calibrates plots where calibration is available
            from the axes_manager.  If ``False``, plots are in pixels/channels.
        same_window : bool
            If ``True``, plots each factor to the same window. They are
            not scaled. Default is ``True``.
        title : str
            Title of the matplotlib plot or label of the line in the legend
            when the dimension of loadings is 1 and ``same_window`` is ``True``.
        with_factors : bool
            If `True`, also returns figure(s) with the factors for the
            given `comp_ids`.
        cmap : :class:`~matplotlib.colors.Colormap`
            The colormap used for the loading image, or for peak
            characteristics,. Default is the matplotlib gray colormap
            (``plt.cm.gray``).
        no_nans : bool
            If ``True``, removes ``NaN``'s from the loading plots.
        per_row : int
            The number of plots in each row, when the `same_window`
            parameter is ``True``.
        axes_decor : str or None, optional
            One of: ``'all'``, ``'ticks'``, ``'off'``, or ``None``
            Controls how the axes are displayed on each image;
            default is ``'all'``
            If ``'all'``, both ticks and axis labels will be shown
            If ``'ticks'``, no axis labels will be shown, but ticks/labels will
            If ``'off'``, all decorations and frame will be disabled
            If ``None``, no axis decorations will be shown, but ticks/frame will

        See Also
        --------
        plot_bss_factors, plot_bss_results

        """
        if self.axes_manager.navigation_dimension > 2:
            raise NotImplementedError(
                "This method cannot plot loadings of "
                "dimension higher than 2."
                "You can use "
                "`plot_bss_results` instead."
            )
        if self.learning_results.bss_loadings is None:
            raise RuntimeError(
                "No learning results found. A "
                "'blind_source_separation' needs to be "
                "performed first."
            )
        if same_window is None:
            same_window = True
        if title is None:
            title = self._get_plot_title("BSS loadings of", same_window=same_window)
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
            comp_label=title,
            cmap=cmap,
            no_nans=no_nans,
            per_row=per_row,
            axes_decor=axes_decor,
        )

    def _get_plot_title(self, base_title="Loadings", same_window=True):
        title_md = self.metadata.General.title
        title = "%s %s" % (base_title, title_md)
        if title_md == "":  # remove the 'of' if 'title' is a empty string
            title = title.replace(" of ", "")
        if not same_window:
            title = title.replace("loadings", "loading")
        return title

    def export_decomposition_results(
        self,
        comp_ids=None,
        folder=None,
        calibrate=True,
        factor_prefix="factor",
        factor_format="hspy",
        loading_prefix="loading",
        loading_format="hspy",
        comp_label=None,
        cmap=plt.cm.gray,
        same_window=False,
        multiple_files=True,
        no_nans=True,
        per_row=3,
        save_figures=False,
        save_figures_format="png",
    ):
        """Export results from a decomposition to any of the supported formats.

        Parameters
        ----------
        comp_ids : None, int or list of int
            If None, returns all components/loadings.
            If an int, returns components/loadings with ids from 0 to the
            given value.
            If a list of ints, returns components/loadings with ids provided in
            the given list.
        folder : str or None
            The path to the folder where the file will be saved.
            If ``None``, the current folder is used by default.
        factor_prefix : str
            The prefix that any exported filenames for factors/components
            begin with
        factor_format : str
            The extension of the format that you wish to save the factors to.
            Default is ``'hspy'``. See `loading_format` for more details.
        loading_prefix : str
            The prefix that any exported filenames for factors/components
            begin with
        loading_format : str
            The extension of the format that you wish to save to. default
            is ``'hspy'``. The format determines the kind of output:

            * For image formats (``'tif'``, ``'png'``, ``'jpg'``, etc.),
              plots are created using the plotting flags as below, and saved
              at 600 dpi. One plot is saved per loading.
            * For multidimensional formats (``'rpl'``, ``'hspy'``), arrays
              are saved in single files.  All loadings are contained in the
              one file.
            * For spectral formats (``'msa'``), each loading is saved to a
              separate file.

        multiple_files : bool
            If ``True``, one file will be created for each factor and loading.
            Otherwise, only two files will be created, one for
            the factors and another for the loadings. The default value can
            be chosen in the preferences.
        save_figures : bool
            If ``True`` the same figures that are obtained when using the plot
            methods will be saved with 600 dpi resolution

        Notes
        -----
        The following parameters are only used when ``save_figures = True``

        Other Parameters
        ----------------
        calibrate : :class:`bool`
            If ``True``, calibrates plots where calibration is available
            from the axes_manager. If ``False``, plots are in pixels/channels.
        same_window : :class:`bool`
            If ``True``, plots each factor to the same window.
        comp_label : :class:`str`
            the label that is either the plot title (if plotting in separate
            windows) or the label in the legend (if plotting in the same window)
        cmap : :class:`~matplotlib.colors.Colormap`
            The colormap used for images, such as factors, loadings, or for peak
            characteristics. Default is the matplotlib gray colormap
            (``plt.cm.gray``).
        per_row : :class:`int`
            The number of plots in each row, when the `same_window`
            parameter is ``True``.
        save_figures_format : :class:`str`
            The image format extension.

        See Also
        --------
        get_decomposition_factors, get_decomposition_loadings
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
            save_figures_format=save_figures_format,
        )
        self._export_loadings(
            loadings,
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
        )

    def export_cluster_results(
        self,
        cluster_ids=None,
        folder=None,
        calibrate=True,
        center_prefix="cluster_center",
        center_format="hspy",
        membership_prefix="cluster_label",
        membership_format="hspy",
        comp_label=None,
        cmap=plt.cm.gray,
        same_window=False,
        multiple_files=True,
        no_nans=True,
        per_row=3,
        save_figures=False,
        save_figures_format="png",
    ):
        """Export results from a cluster analysis to any of the supported
        formats.

        Parameters
        ----------
        cluster_ids : None, int or list of int
            if None, returns all clusters/centers.
            if int, returns clusters/centers with ids from 0 to
            given int.
            if list of ints, returnsclusters/centers with ids in
            given list.
        folder : str or None
            The path to the folder where the file will be saved.
            If `None` the
            current folder is used by default.
        center_prefix : str
            The prefix that any exported filenames for
            cluster centers
            begin with
        center_format : str
            The extension of the format that you wish to save to. Default is
            "hspy". See `loading format` for more details.
        label_prefix : str
            The prefix that any exported filenames for
            cluster labels
            begin with
        label_format : str
            The extension of the format that you wish to save to. default
            is "hspy". The format determines the kind of output.

            * For image formats (``'tif'``, ``'png'``, ``'jpg'``, etc.),
                plots are created using the plotting flags as below, and saved
                at 600 dpi. One plot is saved per loading.
            * For multidimensional formats (``'rpl'``, ``'hspy'``), arrays
                are saved in single files.  All loadings are contained in the
                one file.
            * For spectral formats (``'msa'``), each loading is saved to a
                separate file.

        multiple_files : bool, default False
            If True, on exporting a file per center will
            be created. Otherwise only two files will be created, one for
            the centers and another for the membership. The default value can
            be chosen in the preferences.
        save_figures : bool, default False
            If True the same figures that are obtained when using the plot
            methods will be saved with 600 dpi resolution

        Other Parameters
        ----------------
        These parameters are plotting options and only used when
        ``save_figures=True``.

        calibrate : bool
            if True, calibrates plots where calibration is available
            from
            the axes_manager.  If False, plots are in pixels/channels.
        same_window : bool
            if True, plots each factor to the same window.
        comp_label : str
            The label that is either the plot title
            (if plotting in separate windows) or the label in the legend
            (if plotting in the same window)
        cmap : matplotlib.colors.Colormap
            The colormap used for the factor image, or for peak
            characteristics, the colormap used for the scatter plot of
            some peak characteristic.
        per_row : int
            the number of plots in each row, when the ``same_window=True``.
        save_figures_format : str
            The image format extension.

        See Also
        --------
        get_cluster_signals,
        get_cluster_labels

        """

        factors = self.learning_results.cluster_centers.T
        loadings = self.learning_results.cluster_labels
        self._export_factors(
            factors,
            folder=folder,
            comp_ids=cluster_ids,
            calibrate=calibrate,
            multiple_files=multiple_files,
            factor_prefix=center_prefix,
            factor_format=center_format,
            comp_label=comp_label,
            save_figures=save_figures,
            cmap=cmap,
            no_nans=no_nans,
            same_window=same_window,
            per_row=per_row,
            save_figures_format=save_figures_format,
        )
        self._export_loadings(
            loadings,
            comp_ids=cluster_ids,
            folder=folder,
            calibrate=calibrate,
            multiple_files=multiple_files,
            loading_prefix=membership_prefix,
            loading_format=membership_format,
            comp_label=comp_label,
            cmap=cmap,
            save_figures=save_figures,
            same_window=same_window,
            no_nans=no_nans,
            per_row=per_row,
        )

    def export_bss_results(
        self,
        comp_ids=None,
        folder=None,
        calibrate=True,
        multiple_files=True,
        save_figures=False,
        factor_prefix="bss_factor",
        factor_format="hspy",
        loading_prefix="bss_loading",
        loading_format="hspy",
        comp_label=None,
        cmap=plt.cm.gray,
        same_window=False,
        no_nans=True,
        per_row=3,
        save_figures_format="png",
    ):
        """Export results from ICA to any of the supported formats.

        Parameters
        ----------
        comp_ids : None, int or list of int
            If None, returns all components/loadings.
            If an int, returns components/loadings with ids from 0 to the
            given value.
            If a list of ints, returns components/loadings with ids provided in
            the given list.
        folder : str or None
            The path to the folder where the file will be saved.
            If ``None`` the current folder is used by default.
        factor_prefix : str
            The prefix that any exported filenames for factors/components
            begin with
        factor_format : str
            The extension of the format that you wish to save the factors to.
            Default is ``'hspy'``. See `loading_format` for more details.
        loading_prefix : str
            The prefix that any exported filenames for factors/components
            begin with
        loading_format : str
            The extension of the format that you wish to save to. default
            is ``'hspy'``. The format determines the kind of output:

            * For image formats (``'tif'``, ``'png'``, ``'jpg'``, etc.),
              plots are created using the plotting flags as below, and saved
              at 600 dpi. One plot is saved per loading.
            * For multidimensional formats (``'rpl'``, ``'hspy'``), arrays
              are saved in single files.  All loadings are contained in the
              one file.
            * For spectral formats (``'msa'``), each loading is saved to a
              separate file.

        multiple_files : bool
            If ``True``, one file will be created for each factor and loading.
            Otherwise, only two files will be created, one for
            the factors and another for the loadings. The default value can
            be chosen in the preferences.
        save_figures : bool
            If ``True``, the same figures that are obtained when using the plot
            methods will be saved with 600 dpi resolution

        Notes
        -----
        The following parameters are only used when ``save_figures = True``

        Other Parameters
        ----------------
        calibrate : :class:`bool`
            If ``True``, calibrates plots where calibration is available
            from the axes_manager. If ``False``, plots are in pixels/channels.
        same_window : :class:`bool`
            If ``True``, plots each factor to the same window.
        comp_label : :class:`str`
            the label that is either the plot title (if plotting in separate
            windows) or the label in the legend (if plotting in the same window)
        cmap : :class:`~matplotlib.colors.Colormap`
            The colormap used for images, such as factors, loadings, or
            for peak characteristics. Default is the matplotlib gray colormap
            (``plt.cm.gray``).
        per_row : :class:`int`
            The number of plots in each row, when the `same_window`
            parameter is ``True``.
        save_figures_format : :class:`str`
            The image format extension.

        See Also
        --------
        get_bss_factors, get_bss_loadings
        """

        factors = self.learning_results.bss_factors
        loadings = self.learning_results.bss_loadings.T
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
            save_figures_format=save_figures_format,
        )

        self._export_loadings(
            loadings,
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
            save_figures_format=save_figures_format,
        )

    def _get_loadings(self, loadings):
        if loadings is None:
            raise RuntimeError("No learning results found.")
        from hyperspy.api import signals

        data = loadings.T.reshape((-1,) + self.axes_manager.navigation_shape[::-1])
        if data.shape[0] > 1:
            signal = signals.BaseSignal(
                data,
                axes=(
                    [{"size": data.shape[0], "navigate": True}]
                    + self.axes_manager._get_navigation_axes_dicts()
                ),
            )
            for axis in signal.axes_manager._axes[1:]:
                axis.navigate = False
        else:
            signal = self._get_navigation_signal(data.squeeze())
        return signal

    def _get_factors(self, factors):
        if factors is None:
            raise RuntimeError("No learning results found.")
        signal = self.__class__(
            factors.T.reshape((-1,) + self.axes_manager.signal_shape[::-1]),
            axes=[{"size": factors.shape[-1], "navigate": True}]
            + self.axes_manager._get_signal_axes_dicts(),
        )
        signal.set_signal_type(self.metadata.Signal.signal_type)
        for axis in signal.axes_manager._axes[1:]:
            axis.navigate = False
        return signal

    def get_decomposition_loadings(self):
        """Return the decomposition loadings.

        Returns
        -------
        signal : :class:`~hyperspy.signal.BaseSignal` (or subclass)

        See Also
        --------
        get_decomposition_factors, export_decomposition_results

        """
        if self.learning_results.loadings is None:
            raise RuntimeError("Run a decomposition first.")
        signal = self._get_loadings(self.learning_results.loadings)
        signal.axes_manager._axes[0].name = "Decomposition component index"
        signal.metadata.General.title = (
            "Decomposition loadings of " + self.metadata.General.title
        )
        return signal

    def get_decomposition_factors(self):
        """Return the decomposition factors.

        Returns
        -------
        signal : :class:`~hyperspy.signal.BaseSignal` (or subclass)

        See Also
        --------
        get_decomposition_loadings, export_decomposition_results

        """
        if self.learning_results.factors is None:
            raise RuntimeError("Run a decomposition first.")
        signal = self._get_factors(self.learning_results.factors)
        signal.axes_manager._axes[0].name = "Decomposition component index"
        signal.metadata.General.title = (
            "Decomposition factors of " + self.metadata.General.title
        )
        return signal

    def get_bss_loadings(self):
        """Return the blind source separation loadings.

        Returns
        -------
        :class:`~hyperspy.signal.BaseSignal` (or subclass)

        See Also
        --------
        get_bss_factors, export_bss_results

        """
        signal = self._get_loadings(self.learning_results.bss_loadings)
        signal.axes_manager[0].name = "BSS component index"
        signal.metadata.General.title = "BSS loadings of " + self.metadata.General.title
        return signal

    def get_bss_factors(self):
        """Return the blind source separation factors.

        Returns
        -------
        :class:`~hyperspy.signal.BaseSignal` (or subclass)

        See Also
        --------
        get_bss_loadings, export_bss_results

        """
        signal = self._get_factors(self.learning_results.bss_factors)
        signal.axes_manager[0].name = "BSS component index"
        signal.metadata.General.title = "BSS factors of " + self.metadata.General.title
        return signal

    def plot_bss_results(
        self,
        factors_navigator="smart_auto",
        loadings_navigator="smart_auto",
        factors_dim=2,
        loadings_dim=2,
    ):
        """Plot the blind source separation factors and loadings.

        Unlike :meth:`~hyperspy.api.signals.BaseSignal.plot_bss_factors` and
        :meth:`~hyperspy.api.signals.BaseSignal.plot_bss_loadings`,
        this method displays one component at a time. Therefore it provides a
        more compact visualization than then other two methods.
        The loadings and factors are displayed in different windows and each
        has its own navigator/sliders to navigate them if they are
        multidimensional. The component index axis is synchronized between
        the two.

        Parameters
        ----------
        factors_navigator : str, None, or :class:`~hyperspy.api.signals.BaseSignal` (or subclass)
            One of: ``'smart_auto'``, ``'auto'``, ``None``, ``'spectrum'`` or a
            :class:`~hyperspy.api.signals.BaseSignal` object.
            ``'smart_auto'`` (default) displays sliders if the navigation
            dimension is less than 3. For a description of the other options
            see the :meth:`~hyperspy.api.signals.BaseSignal.plot` documentation
            for details.
        loadings_navigator : str, None, or :class:`~hyperspy.api.signals.BaseSignal` (or subclass)
            See the `factors_navigator` parameter
        factors_dim : int
            Currently HyperSpy cannot plot a signal when the signal dimension is
            higher than two. Therefore, to visualize the BSS results when the
            factors or the loadings have signal dimension greater than 2,
            the data can be viewed as spectra (or images) by setting this
            parameter to 1 (or 2). (The default is 2)
        loadings_dim : int
            See the ``factors_dim`` parameter

        See Also
        --------
        plot_bss_factors, plot_bss_loadings, plot_decomposition_results

        """
        factors = self.get_bss_factors()
        loadings = self.get_bss_loadings()
        _plot_x_results(
            factors=factors,
            loadings=loadings,
            factors_navigator=factors_navigator,
            loadings_navigator=loadings_navigator,
            factors_dim=factors_dim,
            loadings_dim=loadings_dim,
        )

    def plot_decomposition_results(
        self,
        factors_navigator="smart_auto",
        loadings_navigator="smart_auto",
        factors_dim=2,
        loadings_dim=2,
    ):
        """Plot the decomposition factors and loadings.

        Unlike :meth:`~hyperspy.api.signals.BaseSignal.plot_decomposition_factors`
        and :meth:`~hyperspy.api.signals.BaseSignal.plot_decomposition_loadings`,
        this method displays one component at a time. Therefore it provides a
        more compact visualization than then other two methods. The loadings
        and factors are displayed in different windows and each has its own
        navigator/sliders to navigate them if they are multidimensional. The
        component index axis is synchronized between the two.

        Parameters
        ----------
        factors_navigator : str, None, or :class:`~hyperspy.api.signals.BaseSignal` (or subclass)
            One of: ``'smart_auto'``, ``'auto'``, ``None``, ``'spectrum'`` or a
            :class:`~hyperspy.api.signals.BaseSignal` object.
            ``'smart_auto'`` (default) displays sliders if the navigation
            dimension is less than 3. For a description of the other options
            see the :meth:`~hyperspy.api.signals.BaseSignal.plot` documentation
            for details.
        loadings_navigator : str, None, or :class:`~hyperspy.api.signals.BaseSignal` (or subclass)
            See the `factors_navigator` parameter
        factors_dim, loadings_dim : int
            Currently HyperSpy cannot plot a signal when the signal dimension is
            higher than two. Therefore, to visualize the BSS results when the
            factors or the loadings have signal dimension greater than 2,
            the data can be viewed as spectra (or images) by setting this
            parameter to 1 (or 2). (The default is 2)

        See Also
        --------
        plot_decomposition_factors, plot_decomposition_loadings, plot_bss_results

        """

        factors = self.get_decomposition_factors()
        loadings = self.get_decomposition_loadings()
        _plot_x_results(
            factors=factors,
            loadings=loadings,
            factors_navigator=factors_navigator,
            loadings_navigator=loadings_navigator,
            factors_dim=factors_dim,
            loadings_dim=loadings_dim,
        )

    def get_cluster_labels(self, merged=False):
        """Return cluster labels as a Signal.

        Parameters
        ----------
        merged : bool, default False
            If False the cluster label signal has a navigation axes of length
            number_of_clusters and the signal along the the navigation
            direction is binary - 0 the point is not in the cluster, 1 it is
            included. If True, the cluster labels are merged (no navigation
            axes). The value of the signal at any point will be between -1 and
            the number of clusters. -1 represents the points that
            were masked for cluster analysis if any.

        See Also
        --------
        get_cluster_signals

        Returns
        -------
        :class:`~hyperspy.api.signals.BaseSignal`
            The cluster labels
        """
        if self.learning_results.cluster_labels is None:
            raise RuntimeError("Cluster analysis needs to be performed first.")
        if merged:
            data = (
                np.arange(1, self.learning_results.number_of_clusters + 1)[
                    :, np.newaxis
                ]
                * self.learning_results.cluster_labels
            ).sum(0) - 1
            label_signal = self._get_loadings(data)
        else:
            label_signal = self._get_loadings(self.learning_results.cluster_labels.T)
            label_signal.axes_manager._axes[0].name = "Cluster index"
        label_signal.metadata.General.title = (
            "Cluster labels of " + self.metadata.General.title
        )
        return label_signal

    def _get_cluster_signals_factors(self, signal):
        if self.learning_results.cluster_centroid_signals is None:
            raise RuntimeError("Cluster analysis needs to be performed first.")
        if signal == "mean":
            members = self.learning_results.cluster_labels.sum(1, keepdims=True)
            cs = self.learning_results.cluster_sum_signals / members
        elif signal == "sum":
            cs = self.learning_results.cluster_sum_signals
        elif signal == "centroid":
            cs = self.learning_results.cluster_centroid_signals
        return cs

    def get_cluster_signals(self, signal="mean"):
        """Return the cluster centers as a Signal.

        Parameters
        ----------
        %s

        See Also
        --------
        get_cluster_labels

        """
        cs = self._get_cluster_signals_factors(signal=signal)
        signal = self._get_factors(cs.T)
        signal.axes_manager._axes[0].name = "Cluster index"
        signal.metadata.General.title = (
            f"Cluster {signal} signals of {self.metadata.General.title}"
        )
        return signal

    get_cluster_signals.__doc__ %= CLUSTER_SIGNALS_ARG

    def get_cluster_distances(self):
        """Euclidian distances to the centroid of each cluster

        See Also
        --------
        get_cluster_signals

        Returns
        -------
        signal
            Hyperspy signal of cluster distances

        """
        if self.learning_results.cluster_distances is None:
            raise RuntimeError("Cluster analysis needs to be performed first.")
        distance_signal = self._get_loadings(self.learning_results.cluster_distances.T)
        distance_signal.axes_manager._axes[0].name = "Cluster index"
        distance_signal.metadata.General.title = (
            "Cluster distances of " + self.metadata.General.title
        )
        return distance_signal

    def plot_cluster_signals(
        self,
        signal="mean",
        cluster_ids=None,
        calibrate=True,
        same_window=True,
        title=None,
        per_row=3,
    ):
        """Plot centers from a cluster analysis.

        Parameters
        ----------
        %s
        cluster_ids : None, int, or list of int
            If None, returns maps of all clusters.
            If int, returns maps of clusters with ids from 0 to given
            int.
            If list of ints, returns maps of clusters with ids in
            given list.
        calibrate : bool, default True
            If True, calibrates plots where calibration is available
            from the axes_manager. If False, plots are in pixels/channels.
        same_window : bool, default True
            If True, plots each center to the same window.  They are
            not scaled.
        title : None or str, default None
            Title of the matplotlib plot or label of the line in the legend
            when the dimension of loadings is 1 and ``same_window`` is ``True``.
        per_row : int, default 3
            The number of plots in each row, when the same_window parameter is
            True.

        See Also
        --------
        plot_cluster_labels

        """
        if self.axes_manager.signal_dimension > 2:
            raise NotImplementedError(
                "This method cannot plot factors of "
                "signals of dimension higher than 2."
            )
        cs = self._get_cluster_signals_factors(signal=signal)
        if same_window is None:
            same_window = True
        factors = cs.T

        if cluster_ids is None:
            cluster_ids = range(factors.shape[1])

        if title is None:
            title = self._get_plot_title("Cluster centers of", same_window=same_window)

        return self._plot_factors_or_pchars(
            factors,
            comp_ids=cluster_ids,
            calibrate=calibrate,
            same_window=same_window,
            comp_label=title,
            per_row=per_row,
        )

    plot_cluster_signals.__doc__ %= CLUSTER_SIGNALS_ARG

    def plot_cluster_labels(
        self,
        cluster_ids=None,
        calibrate=True,
        same_window=True,
        with_centers=False,
        cmap=plt.cm.gray,
        no_nans=False,
        per_row=3,
        axes_decor="all",
        title=None,
        **kwargs,
    ):
        """Plot cluster labels from a cluster analysis. In case of 1D navigation axis,
        each loading line can be toggled on and off by clicking on the legended
        line.

        Parameters
        ----------

        cluster_ids : None, int, or list of int
            if None (default), returns maps of all components using the
            number_of_cluster was defined when
            executing ``cluster``. Otherwise it raises a ValueError.
            if int, returns maps of cluster labels with ids from 0 to
            given int.
            if list of ints, returns maps of cluster labels with ids in
            given list.
        calibrate : bool
            if True, calibrates plots where calibration is available
            from the axes_manager. If False, plots are in pixels/channels.
        same_window : bool
            if True, plots each factor to the same window.  They are
            not scaled. Default is True.
        title : str
            Title of the matplotlib plot or label of the line in the legend
            when the dimension of labels is 1 and ``same_window`` is ``True``.
        with_centers : bool
            If True, also returns figure(s) with the cluster centers for the
            given cluster_ids.
        cmap : matplotlib.colors.Colormap
            The colormap used for the factor image, or for peak
            characteristics, the colormap used for the scatter plot of
            some peak characteristic.
        no_nans : bool
            If True, removes NaN's from the loading plots.
        per_row : int
            the number of plots in each row, when the same_window
            parameter is True.
        axes_decor : None or str {``'all'``, ``'ticks'``, ``'off'``}, default ``'all'``
            Controls how the axes are displayed on each image; default is 'all'
            If 'all', both ticks and axis labels will be shown
            If 'ticks', no axis labels will be shown, but ticks/labels will
            If 'off', all decorations and frame will be disabled
            If None, no axis decorations will be shown, but ticks/frame will

        See Also
        --------
        plot_cluster_signals, plot_cluster_results

        """
        if self.axes_manager.navigation_dimension > 2:
            raise NotImplementedError(
                "This method cannot plot labels of "
                "dimension higher than 2."
                "You can use "
                "`plot_cluster_results` instead."
            )
        if same_window is None:
            same_window = True
        labels = self.learning_results.cluster_labels.astype("uint")
        if with_centers:
            centers = self.learning_results.cluster_centers.T
        else:
            centers = None

        if cluster_ids is None:
            cluster_ids = range(labels.shape[0])

        if title is None:
            title = self._get_plot_title("Cluster labels of", same_window=same_window)

        return self._plot_loadings(
            labels,
            comp_ids=cluster_ids,
            with_factors=with_centers,
            factors=centers,
            same_window=same_window,
            comp_label=title,
            cmap=cmap,
            no_nans=no_nans,
            per_row=per_row,
            axes_decor=axes_decor,
        )

    def plot_cluster_distances(
        self,
        cluster_ids=None,
        calibrate=True,
        same_window=True,
        with_centers=False,
        cmap=plt.cm.gray,
        no_nans=False,
        per_row=3,
        axes_decor="all",
        title=None,
        **kwargs,
    ):
        """Plot the euclidian distances to the centroid of each cluster.

        In case of 1D navigation axis, each line can be toggled on and
        off by clicking on the corresponding line in the legend.

        Parameters
        ----------
        cluster_ids : None, int, or list of int
            if None (default), returns maps of all components using the
            number_of_cluster was defined when
            executing ``cluster``. Otherwise it raises a ValueError.
            if int, returns maps of cluster labels with ids from 0 to
            given int.
            if list of ints, returns maps of cluster labels with ids in
            given list.
        calibrate : bool
            if True, calibrates plots where calibration is available
            from the axes_manager. If False, plots are in pixels/channels.
        same_window : bool
            if True, plots each factor to the same window.  They are
            not scaled. Default is True.
        title : str
            Title of the matplotlib plot or label of the line in the legend
            when the dimension of distance is 1 and ``same_window`` is ``True``.
        with_centers : bool
            If True, also returns figure(s) with the cluster centers for the
            given cluster_ids.
        cmap : matplotlib.colors.Colormap
            The colormap used for the factor image, or for peak
            characteristics, the colormap used for the scatter plot of
            some peak characteristic.
        no_nans : bool
            If True, removes NaN's from the loading plots.
        per_row : int
            the number of plots in each row, when the same_window
            parameter is True.
        axes_decor : None or str {'all', 'ticks', 'off'}, optional
            Controls how the axes are displayed on each image; default is 'all'
            If 'all', both ticks and axis labels will be shown
            If 'ticks', no axis labels will be shown, but ticks/labels will
            If 'off', all decorations and frame will be disabled
            If None, no axis decorations will be shown, but ticks/frame will

        See Also
        --------
        plot_cluster_signals, plot_cluster_results, plot_cluster_labels

        """
        if self.axes_manager.navigation_dimension > 2:
            raise NotImplementedError(
                "This method cannot plot labels of "
                "dimension higher than 2."
                "You can use "
                "`plot_cluster_results` instead."
            )
        if same_window is None:
            same_window = True
        distances = self.learning_results.cluster_distances
        if with_centers:
            centers = self.learning_results.cluster_centers.T
        else:
            centers = None

        if cluster_ids is None:
            cluster_ids = range(distances.shape[0])

        if title is None:
            title = self._get_plot_title(
                "Cluster distances of", same_window=same_window
            )

        return self._plot_loadings(
            distances,
            comp_ids=cluster_ids,
            with_factors=with_centers,
            factors=centers,
            same_window=same_window,
            comp_label=title,
            cmap=cmap,
            no_nans=no_nans,
            per_row=per_row,
            axes_decor=axes_decor,
        )

    def plot_cluster_results(
        self,
        centers_navigator="smart_auto",
        labels_navigator="smart_auto",
        centers_dim=2,
        labels_dim=2,
    ):
        """Plot the cluster labels and centers.

        Unlike :meth:`~hyperspy.api.signals.BaseSignal.plot_cluster_labels` and
        :meth:`~hyperspy.api.signals.BaseSignal.plot_cluster_signals`, this
        method displays one component at a time.
        Therefore it provides a more compact visualization than then other
        two methods.  The labels and centers  are displayed in different
        windows and each has its own navigator/sliders to navigate them if
        they are multidimensional. The component index axis is synchronized
        between the two.

        Parameters
        ----------
        centers_navigator, labels_navigator : None, \
        {``"smart_auto"`` | ``"auto"`` | ``"spectrum"``} or \
        :class:`~hyperspy.api.signals.BaseSignal`, default ``"smart_auto"``
            ``"smart_auto"`` displays sliders if the navigation
            dimension is less than 3. For a description of the other options
            see ``plot`` documentation for details.
        labels_dim, centers_dims : int, default 2
            Currently HyperSpy cannot plot signals of dimension higher than
            two. Therefore, to visualize the clustering results when the
            centers or the labels have signal dimension greater than 2
            we can view the data as spectra(images) by setting this parameter
            to 1(2)

        See Also
        --------
        plot_cluster_signals, plot_cluster_labels

        """
        centers = self.get_cluster_signals()
        distances = self.get_cluster_distances()
        self.get_cluster_labels(merged=True).plot()
        _plot_x_results(
            factors=centers,
            loadings=distances,
            factors_navigator=centers_navigator,
            loadings_navigator=labels_navigator,
            factors_dim=centers_dim,
            loadings_dim=labels_dim,
        )


def _plot_x_results(
    factors, loadings, factors_navigator, loadings_navigator, factors_dim, loadings_dim
):
    factors.axes_manager._axes[0] = loadings.axes_manager._axes[0]
    if loadings.axes_manager.signal_dimension > 2:
        loadings.axes_manager._set_signal_dimension(loadings_dim)
    if factors.axes_manager.signal_dimension > 2:
        factors.axes_manager._set_signal_dimension(factors_dim)
    if (
        loadings_navigator == "smart_auto"
        and loadings.axes_manager.navigation_dimension < 3
    ):
        loadings_navigator = "slider"
    else:
        loadings_navigator = "auto"
    if factors_navigator == "smart_auto" and (
        factors.axes_manager.navigation_dimension < 3 or loadings_navigator is not None
    ):
        factors_navigator = None
    else:
        factors_navigator = "auto"
    loadings.plot(navigator=loadings_navigator)
    factors.plot(navigator=factors_navigator)


class SpecialSlicersSignal(SpecialSlicers):
    def __setitem__(self, i, j):
        """x.__setitem__(i, y) <==> x[i]=y"""
        if isinstance(j, BaseSignal):
            j = j.data
        array_slices = self.obj._get_array_slices(i, self.isNavigation)
        self.obj.data[array_slices] = j

    def __len__(self):
        return self.obj.axes_manager.signal_shape[0]


class BaseSetMetadataItems(t.HasTraits):
    def __init__(self, signal):
        for key, value in self.mapping.items():
            if signal.metadata.has_item(key):
                setattr(self, value, signal.metadata.get_item(key))
        self.signal = signal

    def store(self, *args, **kwargs):
        for key, value in self.mapping.items():
            if getattr(self, value) != t.Undefined:
                self.signal.metadata.set_item(key, getattr(self, value))


class BaseSignal(
    FancySlicing,
    MVA,
    MVATools,
):
    """

    Attributes
    ----------
    ragged : bool
        Whether the signal is ragged or not.
    isig
        Signal indexer/slicer.
    inav
        Navigation indexer/slicer.
    metadata : hyperspy.misc.utils.DictionaryTreeBrowser
        The metadata of the signal structured as documented in
        :ref:`metadata_structure`.
    original_metadata : hyperspy.misc.utils.DictionaryTreeBrowser
        All metadata read when loading the data.


    Examples
    --------
    General signal created from a numpy or cupy array.

    >>> data = np.ones((10, 10))
    >>> s = hs.signals.BaseSignal(data)

    """

    _dtype = "real"
    # When _signal_dimension=-1, the signal dimension of BaseSignal is defined
    # by the dimension of the array, and this is implemented by the default
    # value of navigate=False in BaseDataAxis
    _signal_dimension = -1
    _signal_type = ""
    _lazy = False
    _alias_signal_types = []
    _additional_slicing_targets = [
        "metadata.Signal.Noise_properties.variance",
    ]

    def __init__(self, data, **kwds):
        """
        Create a signal instance.

        Parameters
        ----------
        data : numpy.ndarray
           The signal data. It can be an array of any dimensions.
        axes : [dict/axes], optional
            List of either dictionaries or axes objects to define the axes (see
            the documentation of the :class:`~hyperspy.axes.AxesManager`
            class for more details).
        attributes : dict, optional
            A dictionary whose items are stored as attributes.
        metadata : dict, optional
            A dictionary containing a set of parameters
            that will to stores in the ``metadata`` attribute.
            Some parameters might be mandatory in some cases.
        original_metadata : dict, optional
            A dictionary containing a set of parameters
            that will to stores in the ``original_metadata`` attribute. It
            typically contains all the parameters that has been
            imported from the original data file.
        ragged : bool or None, optional
            Define whether the signal is ragged or not. Overwrite the
            ``ragged`` value in the ``attributes`` dictionary. If None, it does
            nothing. Default is None.
        """
        # the 'full_initialisation' keyword is private API to be used by the
        # _assign_subclass method. Purposely not exposed as public API.
        # Its purpose is to avoid creating new attributes, which breaks events
        # and to reduce overhead when changing 'signal_type'.
        if kwds.get("full_initialisation", True):
            self._create_metadata()
            self.models = ModelManager(self)
            self.learning_results = LearningResults()
            kwds["data"] = data
            self._plot = None
            self.inav = SpecialSlicersSignal(self, True)
            self.isig = SpecialSlicersSignal(self, False)
            self.events = Events()
            self.events.data_changed = Event(
                """
                Event that triggers when the data has changed

                The event trigger when the data is ready for consumption by any
                process that depend on it as input. Plotted signals automatically
                connect this Event to its `BaseSignal.plot()`.

                Note: The event only fires at certain specific times, not everytime
                that the `BaseSignal.data` array changes values.

                Parameters
                ----------
                    obj: The signal that owns the data.
                """,
                arguments=["obj"],
            )
            self._load_dictionary(kwds)

        if self._signal_dimension >= 0:
            # We don't explicitly set the signal_dimension of ragged because
            # we can't predict it in advance
            self.axes_manager._set_signal_dimension(self._signal_dimension)

    def _create_metadata(self):
        self._metadata = DictionaryTreeBrowser()
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
        self._original_metadata = DictionaryTreeBrowser()
        self.tmp_parameters = DictionaryTreeBrowser()

    def __repr__(self):
        if self.metadata._HyperSpy.Folding.unfolded:
            unfolded = "unfolded "
        else:
            unfolded = ""
        string = "<"
        string += self.__class__.__name__
        string += ", title: %s" % self.metadata.General.title
        string += ", %sdimensions: %s" % (
            unfolded,
            self.axes_manager._get_dimension_str(),
        )

        string += ">"

        return string

    def _binary_operator_ruler(self, other, op_name):
        exception_message = "Invalid dimensions for this operation."
        if isinstance(other, BaseSignal):
            # Both objects are signals
            oam = other.axes_manager
            sam = self.axes_manager
            if (
                sam.navigation_shape == oam.navigation_shape
                and sam.signal_shape == oam.signal_shape
            ):
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
                    ns = self._deepcopy_with_new_data(getattr(sdata, op_name)(odata))
                    ns.axes_manager._sort_axes()
                    return ns
            else:
                # Different navigation and/or signal shapes
                if not are_signals_aligned(self, other):
                    raise ValueError(exception_message)
                else:
                    # They are broadcastable but have different number of axes
                    ns, no = broadcast_signals(self, other)
                    sdata = ns.data
                    odata = no.data
                    if op_name in INPLACE_OPERATORS:
                        # This should raise a ValueError if the operation
                        # changes the shape of the object on the left.
                        self.data = getattr(sdata, op_name)(odata)
                        self.axes_manager._sort_axes()
                        return self
                    else:
                        ns.data = getattr(sdata, op_name)(odata)
                        return ns

        else:
            # Second object is not a Signal
            if op_name in INPLACE_OPERATORS:
                getattr(self.data, op_name)(other)
                return self
            else:
                return self._deepcopy_with_new_data(getattr(self.data, op_name)(other))

    def _unary_operator_ruler(self, op_name):
        return self._deepcopy_with_new_data(getattr(self.data, op_name)())

    def _check_signal_dimension_equals_one(self):
        if self.axes_manager.signal_dimension != 1:
            raise SignalDimensionError(self.axes_manager.signal_dimension, 1)

    def _check_signal_dimension_equals_two(self):
        if self.axes_manager.signal_dimension != 2:
            raise SignalDimensionError(self.axes_manager.signal_dimension, 2)

    def _deepcopy_with_new_data(
        self,
        data=None,
        copy_variance=False,
        copy_navigator=False,
        copy_learning_results=False,
    ):
        """Returns a deepcopy of itself replacing the data.

        This method has an advantage over the default :func:`copy.deepcopy`
        in that it does not copy the data, which can save memory.

        Parameters
        ----------
        data : None or :class:`numpy.ndarray`
        copy_variance : bool
            Whether to copy the variance of the signal to the new copy
        copy_navigator : bool
            Whether to copy the navigator of the signal to the new copy
        copy_learning_results : bool
            Whether to copy the learning_results of the signal to the new copy

        Returns
        -------
        ns : :class:`~hyperspy.signal.BaseSignal` (or subclass)
            The newly copied signal

        """
        old_np = None
        old_navigator = None
        old_learning_results = None
        try:
            old_data = self.data
            self.data = None
            old_plot = self._plot
            self._plot = None
            old_models = self.models._models
            if not copy_variance and "Noise_properties" in self.metadata.Signal:
                old_np = self.metadata.Signal.Noise_properties
                del self.metadata.Signal.Noise_properties
            if not copy_navigator and self.metadata.has_item("_HyperSpy.navigator"):
                old_navigator = self.metadata._HyperSpy.navigator
                del self.metadata._HyperSpy.navigator
            if not copy_learning_results:
                old_learning_results = self.learning_results
                del self.learning_results
            self.models._models = DictionaryTreeBrowser()
            ns = self.deepcopy()
            ns.data = data
            return ns
        finally:
            self.data = old_data
            self._plot = old_plot
            self.models._models = old_models
            if old_np is not None:
                self.metadata.Signal.Noise_properties = old_np
            if old_navigator is not None:
                self.metadata._HyperSpy.navigator = old_navigator
            if old_learning_results is not None:
                self.learning_results = old_learning_results

    def as_lazy(
        self, copy_variance=True, copy_navigator=True, copy_learning_results=True
    ):
        """
        Create a copy of the given Signal as a
        :class:`~hyperspy._signals.lazy.LazySignal`.

        Parameters
        ----------
        copy_variance : bool
            Whether or not to copy the variance from the original Signal to
            the new lazy version. Default is True.
        copy_navigator : bool
            Whether or not to copy the navigator from the original Signal to
            the new lazy version. Default is True.
        copy_learning_results : bool
            Whether to copy the learning_results from the original signal to
            the new lazy version. Default is True.

        Returns
        -------
        res : :class:`~hyperspy._signals.lazy.LazySignal`
            The same signal, converted to be lazy
        """
        res = self._deepcopy_with_new_data(
            self.data,
            copy_variance=copy_variance,
            copy_navigator=copy_navigator,
            copy_learning_results=copy_learning_results,
        )
        res._lazy = True
        res._assign_subclass()
        return res

    def _summary(self):
        string = "\n\tTitle: "
        string += self.metadata.General.title
        if self.metadata.has_item("Signal.signal_type"):
            string += "\n\tSignal type: "
            string += self.metadata.Signal.signal_type
        string += "\n\tData dimensions: "
        string += str(self.axes_manager.shape)
        string += "\n\tData type: "
        string += str(self.data.dtype)
        return string

    def _print_summary(self):
        print(self._summary())

    @property
    def data(self):
        """The underlying data structure as a :class:`numpy.ndarray` (or
        :class:`dask.array.Array`, if the Signal is lazy)."""
        return self._data

    @data.setter
    def data(self, value):
        # Object supporting __array_function__ protocol (NEP-18) or the
        # array API standard doesn't need to be cast to numpy array
        if not (
            hasattr(value, "__array_function__")
            or hasattr(value, "__array_namespace__")
        ):
            value = np.asanyarray(value)
        self._data = np.atleast_1d(value)

    @property
    def metadata(self):
        """The metadata of the signal."""
        return self._metadata

    @property
    def original_metadata(self):
        """The original metadata of the signal."""
        return self._original_metadata

    @property
    def ragged(self):
        """Whether the signal is ragged or not."""
        return self.axes_manager._ragged

    @ragged.setter
    def ragged(self, value):
        # nothing needs to be done!
        if self.ragged == value:
            return

        if value:
            if self.data.dtype != object:
                raise ValueError("The array is not ragged.")
            axes = [
                axis
                for axis in self.axes_manager.signal_axes
                if axis.index_in_array not in list(range(self.data.ndim))
            ]
            self.axes_manager.remove(axes)
            self.axes_manager._set_signal_dimension(0)
        else:
            if self._lazy:
                raise NotImplementedError(
                    "Conversion of a lazy ragged signal to its non-ragged "
                    "counterpart is not supported. Make the required "
                    "non-ragged dask array manually and make a new lazy "
                    "signal."
                )

            error = "The signal can't be converted to a non-ragged signal."
            try:
                # Check that we can actually make a non-ragged array
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # As of numpy 1.20, it raises a VisibleDeprecationWarning
                    # and in the future, it will raise an error
                    data = np.array(self.data.tolist())
            except Exception:
                raise ValueError(error)

            if data.dtype == object:
                raise ValueError(error)

            self.data = data
            # Add axes which were previously in the ragged dimension
            axes = [
                idx
                for idx in range(self.data.ndim)
                if idx not in self.axes_manager.navigation_indices_in_array
            ]
            for index in axes:
                axis = {"index_in_array": index, "size": self.data.shape[index]}
                self.axes_manager._append_axis(**axis)
            self.axes_manager._update_attributes()

        self.axes_manager._ragged = value

    def _load_dictionary(self, file_data_dict):
        """Load data from dictionary.

        Parameters
        ----------
        file_data_dict : dict
            A dictionary containing at least a 'data' keyword with an array of
            arbitrary dimensions. Additionally the dictionary can contain the
            following items:

            * data: the signal data. It can be an array of any dimensions.

            * axes: a dictionary to define the axes (see the documentation of
              the :class:`~hyperspy.axes.AxesManager` class for more details).
            * attributes: a dictionary whose items are stored as attributes.

            * metadata: a dictionary containing a set of parameters that will
              to stores in the `metadata` attribute. Some parameters might be
              mandatory in some cases.
            * original_metadata: a dictionary containing a set of parameters
              that will to stores in the `original_metadata` attribute. It
              typically contains all the parameters that has been
              imported from the original data file.
            * ragged: a bool, defining whether the signal is ragged or not.
              Overwrite the attributes['ragged'] entry

        """
        self.data = file_data_dict["data"]
        oldlazy = self._lazy
        attributes = file_data_dict.get("attributes", {})
        ragged = file_data_dict.get("ragged")
        if ragged is not None:
            attributes["ragged"] = ragged
        if "axes" not in file_data_dict:
            file_data_dict["axes"] = self._get_undefined_axes_list(
                attributes.get("ragged", False)
            )
        self.axes_manager = AxesManager(file_data_dict["axes"])
        # Setting `ragged` attributes requires the `axes_manager`
        for key, value in attributes.items():
            if hasattr(self, key):
                if isinstance(value, dict):
                    for k, v in value.items():
                        setattr(getattr(self, key), k, v)
                else:
                    setattr(self, key, value)
        if "models" in file_data_dict:
            self.models._add_dictionary(file_data_dict["models"])
        if "metadata" not in file_data_dict:
            file_data_dict["metadata"] = {}
        else:
            # Get all hspy object back from their dictionary representation
            _obj_in_dict2hspy(file_data_dict["metadata"], lazy=self._lazy)
        if "original_metadata" not in file_data_dict:
            file_data_dict["original_metadata"] = {}

        self.original_metadata.add_dictionary(file_data_dict["original_metadata"])
        self.metadata.add_dictionary(file_data_dict["metadata"])
        if "title" not in self.metadata.General:
            self.metadata.General.title = ""
        if self._signal_type or not self.metadata.has_item("Signal.signal_type"):
            self.metadata.Signal.signal_type = self._signal_type
        if "learning_results" in file_data_dict:
            self.learning_results.__dict__.update(file_data_dict["learning_results"])
        if self._lazy is not oldlazy:
            self._assign_subclass()

    # TODO: try to find a way to use dask ufuncs when called with lazy data (e.g.
    # np.log(s) -> da.log(s.data) wrapped.
    def __array__(self, dtype=None, copy=None):
        # The copy parameter was added in numpy 2.0
        if dtype is not None and dtype != self.data.dtype:
            if copy is not None and not copy:
                raise ValueError(
                    f"Converting array from {self.data.dtype} to "
                    f"{dtype} requires a copy."
                )
        if dtype:
            array = self.data.astype(dtype)
        else:
            array = self.data
        if copy:
            array = np.copy(array)

        return array

    def __array_wrap__(self, array, context=None, return_scalar=False):
        signal = self._deepcopy_with_new_data(array)
        if context is not None:
            # ufunc, argument of the ufunc, domain of the ufunc
            # In ufuncs with multiple outputs, domain indicates which output
            # is currently being prepared (eg. see modf).
            # In ufuncs with a single output, domain is 0
            uf, objs, huh = context

            def get_title(signal, i=0):
                g = signal.metadata.General
                if g.title:
                    return g.title
                else:
                    return "Untitled Signal %s" % (i + 1)

            title_strs = []
            i = 0
            for obj in objs:
                if isinstance(obj, BaseSignal):
                    title_strs.append(get_title(obj, i))
                    i += 1
                else:
                    title_strs.append(str(obj))

            signal.metadata.General.title = "%s(%s)" % (
                uf.__name__,
                ", ".join(title_strs),
            )

        return signal

    def squeeze(self):
        """Remove single-dimensional entries from the shape of an array
        and the axes. See :func:`numpy.squeeze` for more details.

        Returns
        -------
        s : signal
            A new signal object with single-entry dimensions removed

        Examples
        --------
        >>> s = hs.signals.Signal2D(np.random.random((2, 1, 1, 6, 8, 8)))
        >>> s
        <Signal2D, title: , dimensions: (6, 1, 1, 2|8, 8)>
        >>> s = s.squeeze()
        >>> s
        <Signal2D, title: , dimensions: (6, 2|8, 8)>
        """
        # We deepcopy everything but data
        self = self._deepcopy_with_new_data(self.data)
        for ax in (self.axes_manager.signal_axes, self.axes_manager.navigation_axes):
            for axis in reversed(ax):
                if axis.size == 1:
                    self._remove_axis(axis.index_in_axes_manager)
        self.data = self.data.squeeze()
        return self

    def _to_dictionary(
        self, add_learning_results=True, add_models=False, add_original_metadata=True
    ):
        """Returns a dictionary that can be used to recreate the signal.

        All items but `data` are copies.

        Parameters
        ----------
        add_learning_results : bool, optional
            Whether or not to include any multivariate learning results in
            the outputted dictionary. Default is True.
        add_models : bool, optional
            Whether or not to include any models in the outputted dictionary.
            Default is False
        add_original_metadata : bool
            Whether or not to include the original_medata in the outputted
            dictionary. Default is True.

        Returns
        -------
        dic : dict
            The dictionary that can be used to recreate the signal

        """
        dic = {
            "data": self.data,
            "axes": self.axes_manager._get_axes_dicts(),
            "metadata": copy.deepcopy(self.metadata.as_dictionary()),
            "tmp_parameters": self.tmp_parameters.as_dictionary(),
            "attributes": {"_lazy": self._lazy, "ragged": self.axes_manager._ragged},
        }
        if add_original_metadata:
            dic["original_metadata"] = copy.deepcopy(
                self.original_metadata.as_dictionary()
            )
        if add_learning_results and hasattr(self, "learning_results"):
            dic["learning_results"] = copy.deepcopy(self.learning_results.__dict__)
        if add_models:
            dic["models"] = self.models._models.as_dictionary()
        return dic

    def _get_undefined_axes_list(self, ragged=False):
        """Returns default list of axes construct from the data array shape."""
        axes = []
        for s in self.data.shape:
            axes.append(
                {
                    "size": int(s),
                }
            )
        # With ragged signal with navigation dimension 0 and signal dimension 0
        # we return an empty list to avoid getting a navigation axis of size 1,
        # which is incorrect, because it corresponds to the ragged dimension
        if ragged and len(axes) == 1 and axes[0]["size"] == 1:
            axes = []
        return axes

    def _get_current_data(self, axes_manager=None, fft_shift=False, as_numpy=False):
        if axes_manager is None:
            axes_manager = self.axes_manager
        indices = axes_manager._getitem_tuple
        if self._lazy:
            value = self._get_cache_dask_chunk(indices)
        else:
            value = self.data.__getitem__(indices)
        if as_numpy:
            value = to_numpy(value)
        value = np.atleast_1d(value)
        if fft_shift:
            value = np.fft.fftshift(value)
        return value

    @property
    def navigator(self):
        return self.metadata.get_item("_HyperSpy.navigator")

    @navigator.setter
    def navigator(self, navigator):
        self.metadata.set_item("_HyperSpy.navigator", navigator)

    def plot(self, navigator="auto", axes_manager=None, plot_markers=True, **kwargs):
        """%s
        %s
        %s
        %s
        """
        if self.axes_manager.ragged:
            raise RuntimeError("Plotting ragged signal is not supported.")
        if self._plot is not None:
            self._plot.close()
        if "power_spectrum" in kwargs:
            if not np.issubdtype(self.data.dtype, np.complexfloating):
                raise ValueError(
                    "The parameter `power_spectrum` required a "
                    "signal with complex data type."
                )

        if axes_manager is None:
            axes_manager = self.axes_manager
        if self.is_rgbx is True:
            if axes_manager.navigation_size < 2:
                navigator = None
            else:
                navigator = "slider"

        from hyperspy.defaults_parser import preferences

        if (
            "fig" not in kwargs.keys()
            and preferences.Plot.use_subfigure
            and axes_manager.navigation_dimension > 0
            and axes_manager.signal_dimension in [1, 2]
        ):
            # Create default subfigure
            fig = plt.figure(figsize=(15, 7), layout="constrained")
            subfigs = fig.subfigures(1, 2)
            kwargs["fig"] = subfigs[1]
            kwargs["navigator_kwds"] = dict(fig=subfigs[0])

        if axes_manager.signal_dimension == 0:
            if axes_manager.navigation_dimension == 0:
                # 0d signal without navigation axis: don't make a figure
                # and instead, we display the value
                return
            self._plot = mpl_he.MPL_HyperExplorer()
        elif axes_manager.signal_dimension == 1:
            # Hyperspectrum
            self._plot = mpl_hse.MPL_HyperSignal1D_Explorer()
        elif axes_manager.signal_dimension == 2:
            self._plot = mpl_hie.MPL_HyperImage_Explorer()
        else:
            raise ValueError(
                "Plotting is not supported for this view. "
                "Try e.g. 's.transpose(signal_axes=1).plot()' for "
                "plotting as a 1D signal, or "
                "'s.transpose(signal_axes=(1,2)).plot()' "
                "for plotting as a 2D signal."
            )

        self._plot.axes_manager = axes_manager
        self._plot.signal_data_function = partial(self._get_current_data, as_numpy=True)

        if self.metadata.has_item("Signal.quantity"):
            self._plot.quantity_label = self.metadata.Signal.quantity
        if self.metadata.General.title:
            title = self.metadata.General.title
            self._plot.signal_title = title
        elif self.tmp_parameters.has_item("filename"):
            self._plot.signal_title = self.tmp_parameters.filename

        def sum_wrapper(s, axis):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=UserWarning, module="hyperspy"
                )
                return s.sum(axis)

        def get_static_explorer_wrapper(*args, **kwargs):
            if np.issubdtype(navigator.data.dtype, np.complexfloating):
                return abs(navigator._get_current_data(as_numpy=True))
            else:
                return navigator._get_current_data(as_numpy=True)

        def get_1D_sum_explorer_wrapper(*args, **kwargs):
            # Sum over all but the first navigation axis.
            am = self.axes_manager
            navigator = sum_wrapper(self, am.signal_axes + am.navigation_axes[1:])
            return np.nan_to_num(to_numpy(navigator.data)).squeeze()

        def get_dynamic_image_explorer(*args, **kwargs):
            am = self.axes_manager
            nav_ind = am.indices[2:]  # image at first 2 nav indices
            slices = [slice(None)] * len(am.navigation_axes)
            slices[2:] = nav_ind
            new_nav = navigator.transpose(
                signal_axes=len(am.navigation_axes)
            )  # roll axes to signal axes
            ind = new_nav.isig.__getitem__(
                slices=slices
            )  # Get the value from the nav reverse because hyperspy
            return np.nan_to_num(to_numpy(ind.data)).squeeze()

        # function to disconnect when closing the navigator
        function_to_disconnect = None
        connected_event_copy = self.events.data_changed.connected.copy()

        if not isinstance(navigator, BaseSignal) and navigator == "auto":
            if self.navigator is not None:
                navigator = self.navigator
            elif self.axes_manager.navigation_dimension > 1 and np.any(
                np.array(
                    [not axis.is_uniform for axis in self.axes_manager.navigation_axes]
                )
            ):
                navigator = "slider"
            elif (
                self.axes_manager.navigation_dimension == 1
                and self.axes_manager.signal_dimension == 1
            ):
                navigator = "data"
            elif self.axes_manager.navigation_dimension > 0:
                if self.axes_manager.signal_dimension == 0:
                    navigator = self.deepcopy()
                else:
                    navigator = interactive(
                        f=sum_wrapper,
                        event=self.events.data_changed,
                        recompute_out_event=self.axes_manager.events.any_axis_changed,
                        s=self,
                        axis=self.axes_manager.signal_axes,
                    )
                    # Sets are not ordered, to retrieve the function to disconnect
                    # take the difference with the previous copy
                    function_to_disconnect = list(
                        self.events.data_changed.connected - connected_event_copy
                    )[0]
                if navigator.axes_manager.navigation_dimension == 1:
                    navigator = interactive(
                        f=navigator.as_signal1D,
                        event=navigator.events.data_changed,
                        recompute_out_event=navigator.axes_manager.events.any_axis_changed,
                        spectral_axis=0,
                    )
                else:
                    navigator = interactive(
                        f=navigator.as_signal2D,
                        event=navigator.events.data_changed,
                        recompute_out_event=navigator.axes_manager.events.any_axis_changed,
                        image_axes=(0, 1),
                    )
            else:
                navigator = None
        # Navigator properties
        if axes_manager.navigation_axes:
            # check first if we have a signal to avoid comparison of signal with
            # string
            if isinstance(navigator, BaseSignal):

                def is_shape_compatible(navigation_shape, shape):
                    return (
                        navigation_shape == shape
                        or navigation_shape[:2] == shape
                        or (navigation_shape[0],) == shape
                    )

                # Static navigator
                if is_shape_compatible(
                    axes_manager.navigation_shape, navigator.axes_manager.signal_shape
                ):
                    if len(axes_manager.navigation_shape) > 2:
                        self._plot.navigator_data_function = get_dynamic_image_explorer
                    else:
                        self._plot.navigator_data_function = get_static_explorer_wrapper
                # Static transposed navigator
                elif is_shape_compatible(
                    axes_manager.navigation_shape,
                    navigator.axes_manager.navigation_shape,
                ):
                    navigator = navigator.T
                    if len(axes_manager.navigation_shape) > 2:
                        self._plot.navigator_data_function = get_dynamic_image_explorer
                    else:
                        self._plot.navigator_data_function = get_static_explorer_wrapper
                else:
                    raise ValueError(
                        "The dimensions of the provided (or stored) navigator "
                        "are not compatible with this signal."
                    )
            elif isinstance(navigator, str):
                if navigator == "slider":
                    self._plot.navigator_data_function = "slider"
                elif navigator == "data":
                    if np.issubdtype(self.data.dtype, np.complexfloating):
                        self._plot.navigator_data_function = (
                            lambda axes_manager=None: to_numpy(abs(self.data))
                        )
                    else:
                        self._plot.navigator_data_function = (
                            lambda axes_manager=None: to_numpy(self.data)
                        )
                elif navigator == "spectrum":
                    self._plot.navigator_data_function = get_1D_sum_explorer_wrapper
            elif navigator is None:
                self._plot.navigator_data_function = None
            else:
                raise ValueError(
                    'navigator must be one of "spectrum","auto", '
                    '"slider", None, a Signal instance'
                )

        self._plot.plot(**kwargs)
        self.events.data_changed.connect(self.update_plot, [])

        # Disconnect event when closing signal
        p = (
            self._plot.signal_plot
            if self._plot.signal_plot
            else self._plot.navigator_plot
        )
        p.events.closed.connect(
            lambda: self.events.data_changed.disconnect(self.update_plot), []
        )
        # Disconnect events to the navigator when closing navigator
        if function_to_disconnect is not None:
            self._plot.navigator_plot.events.closed.connect(
                lambda: self.events.data_changed.disconnect(function_to_disconnect), []
            )
            self._plot.navigator_plot.events.closed.connect(
                lambda: self.axes_manager.events.any_axis_changed.disconnect(
                    function_to_disconnect
                ),
                [],
            )

        if plot_markers:
            if self.metadata.has_item("Markers"):
                self._plot_permanent_markers()

    plot.__doc__ %= (
        BASE_PLOT_DOCSTRING,
        BASE_PLOT_DOCSTRING_PARAMETERS,
        PLOT1D_DOCSTRING,
        PLOT2D_KWARGS_DOCSTRING,
    )

    def save(
        self, filename=None, overwrite=None, extension=None, file_format=None, **kwds
    ):
        """Saves the signal in the specified format.

        The function gets the format from the specified extension (see
        :ref:`supported-formats` in the User Guide for more information):

        * ``'hspy'`` for HyperSpy's HDF5 specification
        * ``'rpl'`` for Ripple (useful to export to Digital Micrograph)
        * ``'msa'`` for EMSA/MSA single spectrum saving.
        * ``'unf'`` for SEMPER unf binary format.
        * ``'blo'`` for Blockfile diffraction stack saving.
        * Many image formats such as ``'png'``, ``'tiff'``, ``'jpeg'``...

        If no extension is provided the default file format as defined
        in the `preferences` is used.
        Please note that not all the formats supports saving datasets of
        arbitrary dimensions, e.g. ``'msa'`` only supports 1D data, and
        blockfiles only supports image stacks with a `navigation_dimension` < 2.

        Each format accepts a different set of parameters. For details
        see the specific format documentation.

        Parameters
        ----------
        filename : str or None
            If None (default) and `tmp_parameters.filename` and
            `tmp_parameters.folder` are defined, the
            filename and path will be taken from there. A valid
            extension can be provided e.g. ``'my_file.rpl'``
            (see `extension` parameter).
        overwrite : None or bool
            If None, if the file exists it will query the user. If
            True(False) it does(not) overwrite the file if it exists.
        extension : None or str
            The extension of the file that defines the file format.
            Allowable string values are: {``'hspy'``, ``'hdf5'``, ``'rpl'``,
            ``'msa'``, ``'unf'``, ``'blo'``, ``'emd'``, and common image
            extensions e.g. ``'tiff'``, ``'png'``, etc.}
            ``'hspy'`` and ``'hdf5'`` are equivalent. Use ``'hdf5'`` if
            compatibility with HyperSpy versions older than 1.2 is required.
            If ``None``, the extension is determined from the following list in
            this order:

            i) the filename
            ii)  `Signal.tmp_parameters.extension`
            iii) ``'hspy'`` (the default extension)
        chunks : tuple or True or None (default)
            HyperSpy, Nexus and EMD NCEM format only. Define chunks used when
            saving. The chunk shape should follow the order of the array
            (``s.data.shape``), not the shape of the ``axes_manager``.
            If None and lazy signal, the dask array chunking is used.
            If None and non-lazy signal, the chunks are estimated automatically
            to have at least one chunk per signal space.
            If True, the chunking is determined by the the h5py ``guess_chunk``
            function.
        save_original_metadata : bool , default : False
            Nexus file only. Option to save hyperspy.original_metadata with
            the signal. A loaded Nexus file may have a large amount of data
            when loaded which you may wish to omit on saving
        use_default : bool , default : False
            Nexus file only. Define the default dataset in the file.
            If set to True the signal or first signal in the list of signals
            will be defined as the default (following Nexus v3 data rules).
        write_dataset : bool, optional
            Only for hspy and zspy files. If True, write the dataset, otherwise, don't
            write it. Useful to save attributes without having to write the
            whole dataset. Default is True. Note: for zspy files, only the default
            file writer (``DirectoryStore``) is supported.
        close_file : bool, optional
            Only for hdf5-based files and some zarr store. Close the file after
            writing. Default is True.
        file_format: string
            The file format of choice to save the file. If not given, it is inferred
            from the file extension.

        """
        if filename is None:
            if self.tmp_parameters.has_item(
                "filename"
            ) and self.tmp_parameters.has_item("folder"):
                filename = Path(
                    self.tmp_parameters.folder, self.tmp_parameters.filename
                )
                extension = (
                    self.tmp_parameters.extension if not extension else extension
                )
            elif self.metadata.has_item("General.original_filename"):
                filename = self.metadata.General.original_filename
            else:
                raise ValueError("File name not defined")

        if not isinstance(filename, MutableMapping):
            filename = Path(filename)
            if extension is not None:
                filename = filename.with_suffix(f".{extension}")
        io_save(filename, self, overwrite=overwrite, file_format=file_format, **kwds)

    def _replot(self):
        if self._plot is not None:
            if self._plot.is_active:
                self.plot()

    def update_plot(self):
        """
        If this Signal has been plotted, update the signal and navigator
        plots, as appropriate.
        """
        if self._plot is not None and self._plot.is_active:
            if self._plot.signal_plot is not None:
                self._plot.signal_plot.update()
            if self._plot.navigator_plot is not None:
                self._plot.navigator_plot.update()

    def get_dimensions_from_data(self):
        """Get the dimension parameters from the Signal's underlying data.
        Useful when the data structure was externally modified, or when the
        spectrum image was not loaded from a file
        """
        dc = self.data
        for axis in self.axes_manager._axes:
            axis.size = int(dc.shape[axis.index_in_array])

    def crop(self, axis, start=None, end=None, convert_units=False):
        """Crops the data in a given axis. The range is given in pixels.

        Parameters
        ----------
        axis : int or str
            Specify the data axis in which to perform the cropping
            operation. The axis can be specified using the index of the
            axis in `axes_manager` or the axis name.
        start : int, float, or None
            The beginning of the cropping interval. If type is ``int``,
            the value is taken as the axis index. If type is ``float`` the index
            is calculated using the axis calibration. If `start`/`end` is
            ``None`` the method crops from/to the low/high end of the axis.
        end : int, float, or None
            The end of the cropping interval. If type is ``int``,
            the value is taken as the axis index. If type is ``float`` the index
            is calculated using the axis calibration. If `start`/`end` is
            ``None`` the method crops from/to the low/high end of the axis.
        convert_units : bool
            Default is ``False``. If ``True``, convert the units using the
            :meth:`~hyperspy.axes.AxesManager.convert_units` method
            of the :class:`~hyperspy.axes.AxesManager`. If ``False``,
            does nothing.
        """
        axis = self.axes_manager[axis]
        i1, i2 = axis._get_index(start), axis._get_index(end)
        # To prevent an axis error, which may confuse users
        if i1 is not None and i2 is not None and not i1 != i2:
            raise ValueError("The `start` and `end` values need to be " "different.")

        # We take a copy to guarantee the continuity of the data
        self.data = self.data[
            (slice(None),) * axis.index_in_array + (slice(i1, i2), Ellipsis)
        ]

        axis.crop(i1, i2)
        self.get_dimensions_from_data()
        self.squeeze()
        self.events.data_changed.trigger(obj=self)
        if convert_units:
            self.axes_manager.convert_units(axis)

    def interpolate_on_axis(self, new_axis, axis=0, inplace=False, degree=1):
        """
        Replaces the given ``axis`` with the provided ``new_axis``
        and interpolates data accordingly using
        :func:`scipy.interpolate.make_interp_spline`.

        Parameters
        ----------
        new_axis : :class:`hyperspy.axes.UniformDataAxis`,
        :class:`hyperspy.axes.DataAxis`, :class:`hyperspy.axes.FunctionalDataAxis`
        or str
            Axis which replaces the one specified by the ``axis`` argument.
            If this new axis exceeds the range of the old axis,
            a warning is raised that the data will be extrapolated.
            If ``"uniform"``, convert the axis specified by the ``axis``
            parameter to a uniform axis with the same number of data points.
        axis : int or str, default 0
            Specifies the axis which will be replaced using the index of the
            axis in the `axes_manager`. The axis can be specified using the index of the
            axis in `axes_manager` or the axis name.
        inplace : bool, default False
            If ``True`` the data of `self` is replaced by the result and
            the axis is changed inplace. Otherwise `self` is not changed
            and a new signal with the changes incorporated is returned.
        degree: int, default 1
            Specifies the B-Spline degree of the used interpolator.

        Returns
        -------
        :class:`~.api.signals.BaseSignal` (or subclass)
            A copy of the object with the axis exchanged and the data interpolated.
            This only occurs when inplace is set to ``False``, otherwise nothing is returned.

        Examples
        --------
        >>> s = hs.data.luminescence_signal(uniform=False)
        >>> s2 = s.interpolate_on_axis("uniform", -1, inplace=False)
        >>> hs.plot.plot_spectra(
        ...     [s, s2],
        ...     legend=["FunctionalAxis", "Interpolated"],
        ...     drawstyle='steps-mid',
        ... )
        <Axes: xlabel='Energy (eV)', ylabel='Intensity'>

        Specifying a uniform axis:

        >>> s = hs.data.luminescence_signal(uniform=False)
        >>> new_axis = s.axes_manager[-1].copy()
        >>> new_axis.convert_to_uniform_axis()
        >>> s3 = s.interpolate_on_axis(new_axis, -1, inplace=False)
        >>> hs.plot.plot_spectra(
        ...     [s, s3],
        ...     legend=["FunctionalAxis", "Interpolated"],
        ...     drawstyle='steps-mid',
        ... )
        <Axes: xlabel='Energy (eV)', ylabel='Intensity'>
        """
        old_axis = self.axes_manager[axis]
        axis_idx = old_axis.index_in_array
        interpolator = make_interp_spline(
            old_axis.axis,
            self.data,
            axis=axis_idx,
            k=degree,
        )

        if new_axis == "uniform":
            if old_axis.is_uniform:
                raise ValueError(f"The axis {old_axis.name} is already uniform.")
            if inplace:
                old_axis.convert_to_uniform_axis(log_scale_error=False)
                new_axis = old_axis
            else:
                new_axis = old_axis.copy()
                new_axis.convert_to_uniform_axis(log_scale_error=False)
        else:
            if old_axis.navigate != new_axis.navigate:
                raise ValueError(
                    "The navigate attribute of new_axis differs from the to be replaced axis."
                )
            if (
                old_axis.low_value > new_axis.low_value
                or old_axis.high_value < new_axis.high_value
            ):
                _logger.warning(
                    "The specified new axis exceeds the range of the to be replaced old axis. "
                    "The data will be extrapolated if not specified otherwise via fill_value/bounds_error"
                )

        new_data = interpolator(new_axis.axis)

        if inplace:
            self.data = new_data
            s = self
        else:
            s = self._deepcopy_with_new_data(new_data)

        if new_axis is not old_axis:
            # user specifies a new_axis != "uniform"
            s.axes_manager.set_axis(new_axis, axis_idx)

        if not inplace:
            return s

    def swap_axes(self, axis1, axis2, optimize=False):
        """Swap two axes in the signal.

        Parameters
        ----------
        axis1%s
        axis2%s
        %s

        Returns
        -------
        s : :class:`~hyperspy.signal.BaseSignal` (or subclass)
            A copy of the object with the axes swapped.

        See Also
        --------
        rollaxis
        """
        axis1 = self.axes_manager[axis1].index_in_array
        axis2 = self.axes_manager[axis2].index_in_array
        s = self._deepcopy_with_new_data(self.data.swapaxes(axis1, axis2))
        am = s.axes_manager
        am._update_trait_handlers(remove=True)
        c1 = am._axes[axis1]
        c2 = am._axes[axis2]
        c1.slice, c2.slice = c2.slice, c1.slice
        c1.navigate, c2.navigate = c2.navigate, c1.navigate
        c1.is_binned, c2.is_binned = c2.is_binned, c1.is_binned
        am._axes[axis1] = c2
        am._axes[axis2] = c1
        am._update_attributes()
        am._update_trait_handlers(remove=False)
        if optimize:
            s._make_sure_data_is_contiguous()
        return s

    swap_axes.__doc__ %= (ONE_AXIS_PARAMETER, ONE_AXIS_PARAMETER, OPTIMIZE_ARG)

    def rollaxis(self, axis, to_axis, optimize=False):
        """Roll the specified axis backwards, until it lies in a given position.

        Parameters
        ----------
        axis %s The axis to roll backwards.
            The positions of the other axes do not change relative to one
            another.
        to_axis %s The axis is rolled until it lies before this other axis.
        %s

        Returns
        -------
        s : :class:`~hyperspy.signal.BaseSignal` (or subclass)
            Output signal.

        See Also
        --------
        :func:`numpy.roll`, swap_axes

        Examples
        --------
        >>> s = hs.signals.Signal1D(np.ones((5, 4, 3, 6)))
        >>> s
        <Signal1D, title: , dimensions: (3, 4, 5|6)>
        >>> s.rollaxis(3, 1)
        <Signal1D, title: , dimensions: (3, 4, 5|6)>
        >>> s.rollaxis(2, 0)
        <Signal1D, title: , dimensions: (5, 3, 4|6)>

        """
        axis = self.axes_manager[axis].index_in_array
        to_index = self.axes_manager[to_axis].index_in_array
        if axis == to_index:
            return self.deepcopy()
        new_axes_indices = rollelem(
            [axis_.index_in_array for axis_ in self.axes_manager._axes],
            index=axis,
            to_index=to_index,
        )

        s = self._deepcopy_with_new_data(self.data.transpose(new_axes_indices))
        s.axes_manager._axes = rollelem(
            s.axes_manager._axes, index=axis, to_index=to_index
        )
        s.axes_manager._update_attributes()
        if optimize:
            s._make_sure_data_is_contiguous()
        return s

    rollaxis.__doc__ %= (ONE_AXIS_PARAMETER, ONE_AXIS_PARAMETER, OPTIMIZE_ARG)

    @property
    def _data_aligned_with_axes(self):
        """Returns a view of `data` with is axes aligned with the Signal axes."""
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

    def _validate_rebin_args_and_get_factors(self, new_shape=None, scale=None):
        if new_shape is None and scale is None:
            raise ValueError("One of new_shape, or scale must be specified")
        elif new_shape is not None and scale is not None:
            raise ValueError(
                "Only one out of new_shape or scale should be specified. " "Not both."
            )
        elif new_shape:
            if len(new_shape) != len(self.data.shape):
                raise ValueError("Wrong new_shape size")
            for i, axis in enumerate(self.axes_manager._axes):
                # If the shape doesn't change, there is nothing to do
                # and we don't need to raise an error
                if axis.is_uniform is False and new_shape[i] != self.data.shape[i]:
                    raise NotImplementedError(
                        "Rebinning of non-uniform axes is not yet implemented. "
                        "An alternative is to interpolate the data on an uniform axis "
                        "using `interpolate_on_axis` before rebinning."
                    )
            new_shape_in_array = np.array(
                [
                    new_shape[axis.index_in_axes_manager]
                    for axis in self.axes_manager._axes
                ]
            )
            factors = np.array(self.data.shape) / new_shape_in_array
        else:
            if len(scale) != len(self.data.shape):
                raise ValueError("Wrong scale size")
            for i, axis in enumerate(self.axes_manager._axes):
                if axis.is_uniform is False and scale[i] != 1:
                    raise NotImplementedError(
                        "Rebinning of non-uniform axes is not yet implemented."
                    )
            factors = np.array(
                [scale[axis.index_in_axes_manager] for axis in self.axes_manager._axes]
            )
        return factors  # Factors are in array order

    def rebin(self, new_shape=None, scale=None, crop=True, dtype=None, out=None):
        """
        Rebin the signal into a smaller or larger shape, based on linear
        interpolation. Specify **either** ``new_shape`` or ``scale``. Scale of 1
        means no binning and scale less than one results in up-sampling.

        Parameters
        ----------
        %s
        %s

        Returns
        -------
        ~hyperspy.api.signals.BaseSignal
            The resulting cropped signal.

        Raises
        ------
        NotImplementedError
            If trying to rebin over a non-uniform axis.

        Examples
        --------
        >>> spectrum = hs.signals.Signal1D(np.ones([4, 4, 10]))
        >>> spectrum.data[1, 2, 9] = 5
        >>> print(spectrum)
        <Signal1D, title: , dimensions: (4, 4|10)>
        >>> print ('Sum =', sum(sum(sum(spectrum.data))))
        Sum = 164.0

        >>> scale = [2, 2, 5]
        >>> test = spectrum.rebin(scale)
        >>> print(test)
        <Signal1D, title: , dimensions: (2, 2|5)>
        >>> print('Sum =', sum(sum(sum(test.data))))
        Sum = 164.0

        >>> s = hs.signals.Signal1D(np.ones((2, 5, 10), dtype=np.uint8))
        >>> print(s)
        <Signal1D, title: , dimensions: (5, 2|10)>
        >>> print(s.data.dtype)
        uint8

        Use ``dtype=np.unit16`` to specify a dtype

        >>> s2 = s.rebin(scale=(5, 2, 1), dtype=np.uint16)
        >>> print(s2.data.dtype)
        uint16

        Use dtype="same" to keep the same dtype

        >>> s3 = s.rebin(scale=(5, 2, 1), dtype="same")
        >>> print(s3.data.dtype)
        uint8

        By default ``dtype=None``, the dtype is determined by the behaviour of
        numpy.sum, in this case, unsigned integer of the same precision as
        the platform integer

        >>> s4 = s.rebin(scale=(5, 2, 1))
        >>> print(s4.data.dtype) # doctest: +SKIP
        uint32

        """
        # TODO: Adapt so that it works if a non_uniform_axis exists, but is not
        # changed; for new_shape, a non_uniform_axis should be interpolated to a
        # linear grid
        factors = self._validate_rebin_args_and_get_factors(
            new_shape=new_shape,
            scale=scale,
        )
        s = out or self._deepcopy_with_new_data(None, copy_variance=True)
        data = array_rebin(self.data, scale=factors, crop=crop, dtype=dtype)

        if out:
            if out._lazy:
                out.data = data
            else:
                out.data[:] = data
        else:
            s.data = data
        s.get_dimensions_from_data()
        for axis, axis_src in zip(s.axes_manager._axes, self.axes_manager._axes):
            factor = factors[axis.index_in_array]
            if factor != 1:
                # Change scale, offset only when necessary
                axis.scale = axis_src.scale * factor
                axis.offset = axis_src.offset + (factor - 1) * axis_src.scale / 2
        if s.metadata.has_item("Signal.Noise_properties.variance"):
            if isinstance(s.metadata.Signal.Noise_properties.variance, BaseSignal):
                var = s.metadata.Signal.Noise_properties.variance
                s.metadata.Signal.Noise_properties.variance = var.rebin(
                    new_shape=new_shape, scale=scale, crop=crop, out=out, dtype=dtype
                )
        if out is None:
            return s
        else:
            out.events.data_changed.trigger(obj=out)

    rebin.__doc__ %= (REBIN_ARGS, OUT_ARG)

    def split(self, axis="auto", number_of_parts="auto", step_sizes="auto"):
        """Splits the data into several signals.

        The split can be defined by giving the `number_of_parts`, a homogeneous
        step size, or a list of customized step sizes. By default (``'auto'``),
        the function is the reverse of :func:`~hyperspy.api.stack`.

        Parameters
        ----------
        axis %s
            If ``'auto'`` and if the object has been created with
            :func:`~hyperspy.api.stack` (and ``stack_metadata=True``),
            this method will return the former list of signals (information
            stored in `metadata._HyperSpy.Stacking_history`).
            If it was not created with :func:`~hyperspy.api.stack`,
            the last navigation axis will be used.
        number_of_parts : str or int
            Number of parts in which the spectrum image will be split. The
            splitting is homogeneous. When the axis size is not divisible
            by the `number_of_parts` the remainder data is lost without
            warning. If `number_of_parts` and `step_sizes` is ``'auto'``,
            `number_of_parts` equals the length of the axis,
            `step_sizes` equals one, and the axis is suppressed from each
            sub-spectrum.
        step_sizes : str, list (of int), or int
            Size of the split parts. If ``'auto'``, the `step_sizes` equals one.
            If an int is given, the splitting is homogeneous.

        Examples
        --------
        >>> s = hs.signals.Signal1D(np.random.random([4, 3, 2]))
        >>> s
        <Signal1D, title: , dimensions: (3, 4|2)>
        >>> s.split()
        [<Signal1D, title: , dimensions: (3|2)>, <Signal1D, title: , dimensions: (3|2)>, <Signal1D, title: , dimensions: (3|2)>, <Signal1D, title: , dimensions: (3|2)>]
        >>> s.split(step_sizes=2)
        [<Signal1D, title: , dimensions: (3, 2|2)>, <Signal1D, title: , dimensions: (3, 2|2)>]
        >>> s.split(step_sizes=[1, 2])
        [<Signal1D, title: , dimensions: (3, 1|2)>, <Signal1D, title: , dimensions: (3, 2|2)>]

        Raises
        ------
        NotImplementedError
            If trying to split along a non-uniform axis.

        Returns
        -------
        list of :class:`~hyperspy.api.signals.BaseSignal`
            A list of the split signals

        """
        if number_of_parts != "auto" and step_sizes != "auto":
            raise ValueError(
                "You can define step_sizes or number_of_parts but not both."
            )

        shape = self.data.shape
        signal_dict = self._to_dictionary(add_learning_results=False)

        if axis == "auto":
            mode = "auto"
            if hasattr(self.metadata._HyperSpy, "Stacking_history"):
                stack_history = self.metadata._HyperSpy.Stacking_history
                axis_in_manager = stack_history.axis
                step_sizes = stack_history.step_sizes
            else:
                axis_in_manager = self.axes_manager[-1 + 1j].index_in_axes_manager
        else:
            mode = "manual"
            axis_in_manager = self.axes_manager[axis].index_in_axes_manager

        axis = self.axes_manager[axis_in_manager].index_in_array
        len_axis = self.axes_manager[axis_in_manager].size
        if self.axes_manager[axis].is_uniform is False:
            raise NotImplementedError(
                "Splitting of signals over a non-uniform axis is not implemented"
            )

        if number_of_parts == "auto" and step_sizes == "auto":
            step_sizes = 1
            number_of_parts = len_axis
        elif step_sizes == "auto":
            if number_of_parts > shape[axis]:
                raise ValueError("The number of parts is greater than the axis size.")

            step_sizes = [
                shape[axis] // number_of_parts,
            ] * number_of_parts

        if isinstance(step_sizes, numbers.Integral):
            step_sizes = [step_sizes] * int(len_axis / step_sizes)

        splitted = []
        cut_index = np.array([0] + step_sizes).cumsum()

        axes_dict = signal_dict["axes"]
        for i in range(len(cut_index) - 1):
            axes_dict[axis]["offset"] = self.axes_manager._axes[axis].index2value(
                cut_index[i]
            )
            axes_dict[axis]["size"] = cut_index[i + 1] - cut_index[i]
            data = self.data[
                (slice(None),) * axis
                + (slice(cut_index[i], cut_index[i + 1]), Ellipsis)
            ]
            signal_dict["data"] = data
            splitted += (self.__class__(**signal_dict),)

        if number_of_parts == len_axis or step_sizes == [1] * len_axis:
            for i, signal1D in enumerate(splitted):
                signal1D.data = signal1D.data[
                    signal1D.axes_manager._get_data_slice([(axis, 0)])
                ]
                signal1D._remove_axis(axis_in_manager)

        if mode == "auto" and hasattr(self.original_metadata, "stack_elements"):
            for i, spectrum in enumerate(splitted):
                se = self.original_metadata.stack_elements["element" + str(i)]
                spectrum._metadata = copy.deepcopy(se["metadata"])
                spectrum._original_metadata = copy.deepcopy(se["original_metadata"])
                spectrum.metadata.General.title = se.metadata.General.title

        return splitted

    split.__doc__ %= ONE_AXIS_PARAMETER

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

        See Also
        --------
        fold

        Notes
        -----
        WARNING: this private function does not modify the signal subclass
        and it is intended for internal use only. To unfold use the public
        :meth:`~hyperspy.signal.BaseSignal.unfold`,
        :meth:`~hyperspy.signal.BaseSignal.unfold_navigation_space`,
        :meth:`~hyperspy.signal.BaseSignal.unfold_signal_space` instead.
        It doesn't make sense to perform an unfolding when `dim` < 2

        """
        if self.data.squeeze().ndim < 2:
            return

        # We need to store the original shape and coordinates to be used
        # by the fold function only if it has not been already stored by a
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
        uname = ""
        uunits = ""
        to_remove = []
        for axis, dim in zip(self.axes_manager._axes, new_shape):
            if dim == 1:
                uname += "," + str(axis)
                uunits = "," + str(axis.units)
                to_remove.append(axis)
        ua = self.axes_manager._axes[unfolded_axis]
        ua.name = str(ua) + uname
        ua.units = str(ua.units) + uunits
        ua.size = self.data.shape[unfolded_axis]
        for axis in to_remove:
            self.axes_manager.remove(axis.index_in_axes_manager)
        self.data = self.data.squeeze()
        self._assign_subclass()

    def unfold(self, unfold_navigation=True, unfold_signal=True):
        """Modifies the shape of the data by unfolding the signal and
        navigation dimensions separately

        Parameters
        ----------
        unfold_navigation : bool
            Whether or not to unfold the navigation dimension(s) (default:
            ``True``)
        unfold_signal : bool
            Whether or not to unfold the signal dimension(s) (default:
            ``True``)

        Returns
        -------
        needed_unfolding : bool
            Whether or not one of the axes needed unfolding (and that
            unfolding was performed)

        Notes
        -----
        It doesn't make sense to perform an unfolding when the total number
        of dimensions is < 2.
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

        See Also
        --------
        unfold, fold

        Examples
        --------
        >>> import numpy as np
        >>> s = BaseSignal(np.random.random((64, 64, 1024)))
        >>> with s.unfolded():
        ...     # Do whatever needs doing while unfolded here
        ...     pass

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
            Whether or not the navigation space needed unfolding (and whether
            it was performed)
        """

        if self.axes_manager.navigation_dimension < 2:
            needed_unfolding = False
        else:
            needed_unfolding = True
            steady_axes = [
                axis.index_in_array for axis in self.axes_manager.signal_axes
            ]
            unfolded_axis = self.axes_manager.navigation_axes[0].index_in_array
            self._unfold(steady_axes, unfolded_axis)
            if self.metadata.has_item("Signal.Noise_properties.variance"):
                variance = self.metadata.Signal.Noise_properties.variance
                if isinstance(variance, BaseSignal):
                    variance.unfold_navigation_space()
        return needed_unfolding

    def unfold_signal_space(self):
        """Modify the shape of the data to obtain a signal space of
        dimension 1

        Returns
        -------
        needed_unfolding : bool
            Whether or not the signal space needed unfolding (and whether
            it was performed)
        """
        if self.axes_manager.signal_dimension < 2:
            needed_unfolding = False
        else:
            needed_unfolding = True
            steady_axes = [
                axis.index_in_array for axis in self.axes_manager.navigation_axes
            ]
            unfolded_axis = self.axes_manager.signal_axes[0].index_in_array
            self._unfold(steady_axes, unfolded_axis)
            self.metadata._HyperSpy.Folding.signal_unfolded = True
            if self.metadata.has_item("Signal.Noise_properties.variance"):
                variance = self.metadata.Signal.Noise_properties.variance
                if isinstance(variance, BaseSignal):
                    variance.unfold_signal_space()
        return needed_unfolding

    def fold(self):
        """If the signal was previously unfolded, fold it back"""
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
            self._assign_subclass()
            if self.metadata.has_item("Signal.Noise_properties.variance"):
                variance = self.metadata.Signal.Noise_properties.variance
                if isinstance(variance, BaseSignal):
                    variance.fold()

    def _make_sure_data_is_contiguous(self):
        if self.data.flags["C_CONTIGUOUS"] is False:
            _logger.info(
                "{0!r} data is replaced by its optimized copy, see "
                "optimize parameter of ``Basesignal.transpose`` "
                "for more information.".format(self)
            )
            self.data = np.ascontiguousarray(self.data, like=self.data)

    def _iterate_signal(self, iterpath=None):
        """Iterates over the signal data. It is faster than using the signal
        iterator, because it avoids making deepcopy of metadata and other
        attributes.

        Parameters
        ----------
        iterpath : None or str or iterable
            Any valid iterpath supported by the axes_manager.

        Returns
        -------
        numpy array when iterating over the navigation space
        """
        original_index = self.axes_manager.indices

        with self.axes_manager.switch_iterpath(iterpath):
            self.axes_manager.indices = tuple(
                [0 for _ in self.axes_manager.navigation_axes]
            )
            for _ in self.axes_manager:
                yield self._get_current_data()
            # restore original index
            self.axes_manager.indices = original_index

    def _cycle_signal(self):
        """Cycles over the signal data.

        It is faster than using the signal iterator.

        Warning! could produce a infinite loop.

        """
        if self.axes_manager.navigation_size < 2:
            while True:
                yield self._get_current_data()
            return  # pragma: no cover

        self._make_sure_data_is_contiguous()
        axes = [axis.index_in_array for axis in self.axes_manager.signal_axes]
        if axes:
            unfolded_axis = self.axes_manager.navigation_axes[0].index_in_array
            new_shape = [1] * len(self.data.shape)
            for axis in axes:
                new_shape[axis] = self.data.shape[axis]
            new_shape[unfolded_axis] = -1
        else:  # signal_dimension == 0
            new_shape = (-1, 1)
            axes = [1]
            unfolded_axis = 0
        # Warning! if the data is not contigous it will make a copy!!
        data = self.data.reshape(new_shape)
        getitem = [0] * len(data.shape)
        for axis in axes:
            getitem[axis] = slice(None)
        i = 0
        Ni = data.shape[unfolded_axis]
        while True:
            getitem[unfolded_axis] = i
            yield (data[tuple(getitem)])
            i += 1
            i = 0 if i == Ni else i

    def _remove_axis(self, axes):
        am = self.axes_manager
        axes = am[axes]
        if not np.iterable(axes):
            axes = (axes,)
        if am.navigation_dimension + am.signal_dimension >= len(axes):
            old_signal_dimension = am.signal_dimension
            am.remove(axes)
            if old_signal_dimension != am.signal_dimension:
                self._assign_subclass()
        if not self.axes_manager._axes and not self.ragged:
            # Create a "Scalar" axis because the axis is the last one left and
            # HyperSpy does not # support 0 dimensions
            add_scalar_axis(self)

    def _ma_workaround(self, s, function, axes, ar_axes, out):
        # TODO: Remove if and when numpy.ma accepts tuple `axis`

        # Basically perform unfolding, but only on data. We don't care about
        # the axes since the function will consume it/them.
        if not np.iterable(ar_axes):
            ar_axes = (ar_axes,)

        ar_axes = sorted(ar_axes)
        new_shape = list(self.data.shape)
        for index in ar_axes[1:]:
            new_shape[index] = 1
        new_shape[ar_axes[0]] = -1
        data = self.data.reshape(new_shape).squeeze()

        if out:
            data = np.atleast_1d(
                function(
                    data,
                    axis=ar_axes[0],
                )
            )
            if data.shape == out.data.shape:
                out.data[:] = data
                out.events.data_changed.trigger(obj=out)
            else:
                raise ValueError(
                    "The output shape %s does not match  the shape of "
                    "`out` %s" % (data.shape, out.data.shape)
                )
        else:
            s.data = function(
                data,
                axis=ar_axes[0],
            )
            s._remove_axis([ax.index_in_axes_manager for ax in axes])
            return s

    def _apply_function_on_data_and_remove_axis(
        self, function, axes, out=None, **kwargs
    ):
        axes = self.axes_manager[axes]
        if not np.iterable(axes):
            axes = (axes,)

        # Use out argument in numpy function when available for operations that
        # do not return scalars in numpy.
        np_out = not len(self.axes_manager._axes) == len(axes)
        ar_axes = tuple(ax.index_in_array for ax in axes)

        if len(ar_axes) == 0:
            # no axes is provided, so no operation needs to be done but we
            # still need to finished the execution of the function properly.
            if out:
                out.data[:] = self.data
                out.events.data_changed.trigger(obj=out)
                return
            else:
                return self
        elif len(ar_axes) == 1:
            ar_axes = ar_axes[0]

        s = out or self._deepcopy_with_new_data(None)

        if np.ma.is_masked(self.data):
            return self._ma_workaround(
                s=s, function=function, axes=axes, ar_axes=ar_axes, out=out
            )
        if out:
            if np_out:
                function(
                    self.data,
                    axis=ar_axes,
                    out=out.data,
                )
            else:
                data = np.atleast_1d(
                    function(
                        self.data,
                        axis=ar_axes,
                    )
                )
                if data.shape == out.data.shape:
                    out.data[:] = data
                else:
                    raise ValueError(
                        "The output shape %s does not match  the shape of "
                        "`out` %s" % (data.shape, out.data.shape)
                    )
            out.events.data_changed.trigger(obj=out)
        else:
            s.data = np.atleast_1d(
                function(
                    self.data,
                    axis=ar_axes,
                )
            )
            s._remove_axis([ax.index_in_axes_manager for ax in axes])
            return s

    def sum(self, axis=None, out=None, rechunk=False):
        """Sum the data over the given axes.

        Parameters
        ----------
        axis %s
        %s
        %s

        Returns
        -------
        ~hyperspy.signal.BaseSignal
            A new Signal containing the sum of the provided Signal along the
            specified axes.

        Notes
        -----
        If you intend to calculate the numerical integral of an unbinned signal,
        please use the :meth:`~.api.signals.BaseSignal.integrate1D` function
        instead. To avoid erroneous misuse of the `sum` function as integral,
        it raises a warning when working with an unbinned, non-uniform axis.

        See Also
        --------
        max, min, mean, std, var, indexmax, indexmin, valuemax, valuemin

        Examples
        --------
        >>> import numpy as np
        >>> s = BaseSignal(np.random.random((64, 64, 1024)))
        >>> s
        <BaseSignal, title: , dimensions: (|1024, 64, 64)>
        >>> s.sum(0)
        <Signal2D, title: , dimensions: (|64, 64)>

        """

        if axis is None:
            axis = self.axes_manager.navigation_axes

        axes = self.axes_manager[axis]
        if not np.iterable(axes):
            axes = (axes,)
        if any([not ax.is_uniform and not ax.is_binned for ax in axes]):
            warnings.warn(
                "You are summing over an unbinned, non-uniform axis. "
                "The result can not be used as an approximation of "
                "the integral of the signal. For this functionality, "
                "use integrate1D instead."
            )

        return self._apply_function_on_data_and_remove_axis(
            np.sum, axis, out=out, rechunk=rechunk
        )

    sum.__doc__ %= (MANY_AXIS_PARAMETER, OUT_ARG, RECHUNK_ARG)

    def max(self, axis=None, out=None, rechunk=False):
        """Returns a signal with the maximum of the signal along at least one
        axis.

        Parameters
        ----------
        axis %s
        %s
        %s

        Returns
        -------
        s : :class:`~hyperspy.signal.BaseSignal` (or subclass)
            A new Signal containing the maximum of the provided Signal over the
            specified axes

        See Also
        --------
        min, sum, mean, std, var, indexmax, indexmin, valuemax, valuemin

        Examples
        --------
        >>> import numpy as np
        >>> s = BaseSignal(np.random.random((64, 64, 1024)))
        >>> s
        <BaseSignal, title: , dimensions: (|1024, 64, 64)>
        >>> s.max(0)
        <Signal2D, title: , dimensions: (|64, 64)>

        """
        if axis is None:
            axis = self.axes_manager.navigation_axes
        return self._apply_function_on_data_and_remove_axis(
            np.max, axis, out=out, rechunk=rechunk
        )

    max.__doc__ %= (MANY_AXIS_PARAMETER, OUT_ARG, RECHUNK_ARG)

    def min(self, axis=None, out=None, rechunk=False):
        """Returns a signal with the minimum of the signal along at least one
        axis.

        Parameters
        ----------
        axis %s
        %s
        %s

        Returns
        -------
        s : :class:`~hyperspy.signal.BaseSignal` (or subclass)
            A new Signal containing the minimum of the provided Signal over the
            specified axes

        See Also
        --------
        max, sum, mean, std, var, indexmax, indexmin, valuemax, valuemin

        Examples
        --------
        >>> import numpy as np
        >>> s = BaseSignal(np.random.random((64, 64, 1024)))
        >>> s
        <BaseSignal, title: , dimensions: (|1024, 64, 64)>
        >>> s.min(0)
        <Signal2D, title: , dimensions: (|64, 64)>

        """
        if axis is None:
            axis = self.axes_manager.navigation_axes
        return self._apply_function_on_data_and_remove_axis(
            np.min, axis, out=out, rechunk=rechunk
        )

    min.__doc__ %= (MANY_AXIS_PARAMETER, OUT_ARG, RECHUNK_ARG)

    def mean(self, axis=None, out=None, rechunk=False):
        """Returns a signal with the average of the signal along at least one
        axis.

        Parameters
        ----------
        axis %s
        %s
        %s

        Returns
        -------
        s : :class:`~hyperspy.signal.BaseSignal` (or subclass)
            A new Signal containing the mean of the provided Signal over the
            specified axes

        See Also
        --------
        max, min, sum, std, var, indexmax, indexmin, valuemax, valuemin

        Examples
        --------
        >>> import numpy as np
        >>> s = BaseSignal(np.random.random((64, 64, 1024)))
        >>> s
        <BaseSignal, title: , dimensions: (|1024, 64, 64)>
        >>> s.mean(0)
        <Signal2D, title: , dimensions: (|64, 64)>

        """
        if axis is None:
            axis = self.axes_manager.navigation_axes
        return self._apply_function_on_data_and_remove_axis(
            np.mean, axis, out=out, rechunk=rechunk
        )

    mean.__doc__ %= (MANY_AXIS_PARAMETER, OUT_ARG, RECHUNK_ARG)

    def std(self, axis=None, out=None, rechunk=False):
        """Returns a signal with the standard deviation of the signal along
        at least one axis.

        Parameters
        ----------
        axis %s
        %s
        %s

        Returns
        -------
        s : :class:`~hyperspy.signal.BaseSignal` (or subclass)
            A new Signal containing the standard deviation of the provided
            Signal over the specified axes

        See Also
        --------
        max, min, sum, mean, var, indexmax, indexmin, valuemax, valuemin

        Examples
        --------
        >>> import numpy as np
        >>> s = BaseSignal(np.random.random((64, 64, 1024)))
        >>> s
        <BaseSignal, title: , dimensions: (|1024, 64, 64)>
        >>> s.std(0)
        <Signal2D, title: , dimensions: (|64, 64)>

        """
        if axis is None:
            axis = self.axes_manager.navigation_axes
        return self._apply_function_on_data_and_remove_axis(
            np.std, axis, out=out, rechunk=rechunk
        )

    std.__doc__ %= (MANY_AXIS_PARAMETER, OUT_ARG, RECHUNK_ARG)

    def var(self, axis=None, out=None, rechunk=False):
        """Returns a signal with the variances of the signal along at least one
        axis.

        Parameters
        ----------
        axis %s
        %s
        %s

        Returns
        -------
        s : :class:`~hyperspy.signal.BaseSignal` (or subclass)
            A new Signal containing the variance of the provided Signal over the
            specified axes

        See Also
        --------
        max, min, sum, mean, std, indexmax, indexmin, valuemax, valuemin

        Examples
        --------
        >>> import numpy as np
        >>> s = BaseSignal(np.random.random((64, 64, 1024)))
        >>> s
        <BaseSignal, title: , dimensions: (|1024, 64, 64)>
        >>> s.var(0)
        <Signal2D, title: , dimensions: (|64, 64)>

        """
        if axis is None:
            axis = self.axes_manager.navigation_axes
        return self._apply_function_on_data_and_remove_axis(
            np.var, axis, out=out, rechunk=rechunk
        )

    var.__doc__ %= (MANY_AXIS_PARAMETER, OUT_ARG, RECHUNK_ARG)

    def nansum(self, axis=None, out=None, rechunk=False):
        """%s"""
        if axis is None:
            axis = self.axes_manager.navigation_axes
        axes = self.axes_manager[axis]
        if not np.iterable(axes):
            axes = (axes,)
        if any([not ax.is_uniform for ax in axes]):
            warnings.warn(
                "You are summing over a non-uniform axis. The result "
                "can not be used as an approximation of the "
                "integral of the signal. For this functionaliy, "
                "use integrate1D instead."
            )
        return self._apply_function_on_data_and_remove_axis(
            np.nansum, axis, out=out, rechunk=rechunk
        )

    nansum.__doc__ %= NAN_FUNC.format("sum")

    def nanmax(self, axis=None, out=None, rechunk=False):
        """%s"""
        if axis is None:
            axis = self.axes_manager.navigation_axes
        return self._apply_function_on_data_and_remove_axis(
            np.nanmax, axis, out=out, rechunk=rechunk
        )

    nanmax.__doc__ %= NAN_FUNC.format("max")

    def nanmin(self, axis=None, out=None, rechunk=False):
        """%s"""
        if axis is None:
            axis = self.axes_manager.navigation_axes
        return self._apply_function_on_data_and_remove_axis(
            np.nanmin, axis, out=out, rechunk=rechunk
        )

    nanmin.__doc__ %= NAN_FUNC.format("min")

    def nanmean(self, axis=None, out=None, rechunk=False):
        """%s"""
        if axis is None:
            axis = self.axes_manager.navigation_axes
        return self._apply_function_on_data_and_remove_axis(
            np.nanmean, axis, out=out, rechunk=rechunk
        )

    nanmean.__doc__ %= NAN_FUNC.format("mean")

    def nanstd(self, axis=None, out=None, rechunk=False):
        """%s"""
        if axis is None:
            axis = self.axes_manager.navigation_axes
        return self._apply_function_on_data_and_remove_axis(
            np.nanstd, axis, out=out, rechunk=rechunk
        )

    nanstd.__doc__ %= NAN_FUNC.format("std")

    def nanvar(self, axis=None, out=None, rechunk=False):
        """%s"""
        if axis is None:
            axis = self.axes_manager.navigation_axes
        return self._apply_function_on_data_and_remove_axis(
            np.nanvar, axis, out=out, rechunk=rechunk
        )

    nanvar.__doc__ %= NAN_FUNC.format("var")

    def diff(self, axis, order=1, out=None, rechunk=False):
        """Returns a signal with the ``n``-th order discrete difference along
        given axis. `i.e.` it calculates the difference between consecutive
        values in the given axis: ``out[n] = a[n+1] - a[n]``. See
        :func:`numpy.diff` for more details.

        Parameters
        ----------
        axis %s
        order : int
            The order of the discrete difference.
        %s
        %s

        Returns
        -------
        ~hyperspy.api.signals.BaseSignal or None
            Note that the size of the data on the given ``axis`` decreases by
            the given ``order``. `i.e.` if ``axis`` is ``"x"`` and ``order`` is
            2, the `x` dimension is N, ``der``'s `x` dimension is N - 2.

        Notes
        -----
        If you intend to calculate the numerical derivative, please use the
        proper :meth:`~hyperspy.api.signals.BaseSignal.derivative` function
        instead. To avoid erroneous misuse of the ``diff`` function as derivative,
        it raises an error when when working with a non-uniform axis.

        See Also
        --------
        hyperspy.api.signals.BaseSignal.derivative,
        hyperspy.api.signals.BaseSignal.integrate1D,
        hyperspy.api.signals.BaseSignal.integrate_simpson

        Examples
        --------
        >>> import numpy as np
        >>> s = BaseSignal(np.random.random((64, 64, 1024)))
        >>> s
        <BaseSignal, title: , dimensions: (|1024, 64, 64)>
        >>> s.diff(0)
        <BaseSignal, title: , dimensions: (|1023, 64, 64)>

        """
        if not self.axes_manager[axis].is_uniform:
            raise NotImplementedError(
                "Performing a numerical difference on a non-uniform axis "
                "is not implemented. Consider using `derivative` instead."
            )
        s = out or self._deepcopy_with_new_data(None)
        data = np.diff(self.data, n=order, axis=self.axes_manager[axis].index_in_array)
        if out is not None:
            out.data[:] = data
        else:
            s.data = data
        axis2 = s.axes_manager[axis]
        new_offset = self.axes_manager[axis].offset + (order * axis2.scale / 2)
        axis2.offset = new_offset
        s.get_dimensions_from_data()
        if out is None:
            return s
        else:
            out.events.data_changed.trigger(obj=out)

    diff.__doc__ %= (ONE_AXIS_PARAMETER, OUT_ARG, RECHUNK_ARG)

    def derivative(self, axis, order=1, out=None, **kwargs):
        r"""Calculate the numerical derivative along the given axis,
        with respect to the calibrated units of that axis.

        For a function :math:`y = f(x)` and two consecutive values :math:`x_1`
        and :math:`x_2`:

        .. math::

            \frac{df(x)}{dx} = \frac{y(x_2)-y(x_1)}{x_2-x_1}

        Parameters
        ----------
        axis %s
        order: int
            The order of the derivative.
        %s
        **kwargs : dict
            All extra keyword arguments are passed to :func:`numpy.gradient`

        Returns
        -------
        :class:`~hyperspy.api.signals.BaseSignal`
            Note that the size of the data on the given ``axis`` decreases by
            the given ``order``. `i.e.` if ``axis`` is ``"x"`` and ``order`` is
            2, if the `x` dimension is N, then ``der``'s `x` dimension is N - 2.

        Notes
        -----
        This function uses numpy.gradient to perform the derivative. See its
        documentation for implementation details.

        See Also
        --------
        integrate1D, integrate_simpson

        """
        # rechunk was a valid keyword up to HyperSpy 1.6
        if "rechunk" in kwargs:
            # We miss the deprecation cycle for 2.0
            warnings.warn(
                "The `rechunk` parameter is not used since HyperSpy 1.7 and"
                "will be removed in HyperSpy 3.0.",
                VisibleDeprecationWarning,
            )
            del kwargs["rechunk"]
        n = order
        der_data = self.data
        axis = self.axes_manager[axis]
        if isinstance(axis, tuple):
            axis = axis[0]
        while n:
            der_data = np.gradient(
                der_data,
                axis.axis,
                axis=axis.index_in_array,
                **kwargs,
            )
            n -= 1
        if out:
            out.data = der_data
            out.events.data_changed.trigger(obj=out)
        else:
            return self._deepcopy_with_new_data(der_data)

    derivative.__doc__ %= (ONE_AXIS_PARAMETER, OUT_ARG)

    def integrate_simpson(self, axis, out=None, rechunk=False):
        """Calculate the integral of a Signal along an axis using
        `Simpson's rule <https://en.wikipedia.org/wiki/Simpson%%27s_rule>`_.

        Parameters
        ----------
        axis %s
        %s
        %s

        Returns
        -------
        s : :class:`~hyperspy.signal.BaseSignal` (or subclass)
            A new Signal containing the integral of the provided Signal along
            the specified axis.

        See Also
        --------
        hyperspy.api.signals.BaseSignal.derivative,
        hyperspy.api.signals.BaseSignal.integrate1D

        Examples
        --------
        >>> s = BaseSignal(np.random.random((64, 64, 1024)))
        >>> s
        <BaseSignal, title: , dimensions: (|1024, 64, 64)>
        >>> s.integrate_simpson(0)
        <Signal2D, title: , dimensions: (|64, 64)>

        """
        axis = self.axes_manager[axis]
        s = out or self._deepcopy_with_new_data(None)
        data = integrate.simpson(y=self.data, x=axis.axis, axis=axis.index_in_array)
        if out is not None:
            out.data[:] = data
            out.events.data_changed.trigger(obj=out)
        else:
            s.data = data
            s._remove_axis(axis.index_in_axes_manager)
            return s

    integrate_simpson.__doc__ %= (ONE_AXIS_PARAMETER, OUT_ARG, RECHUNK_ARG)

    def fft(self, shift=False, apodization=False, real_fft_only=False, **kwargs):
        """Compute the discrete Fourier Transform.

        This function computes the discrete Fourier Transform over the signal
        axes by means of the Fast Fourier Transform (FFT) as implemented in
        numpy.

        Parameters
        ----------
        shift : bool, optional
            If ``True``, the origin of FFT will be shifted to the centre
            (default is ``False``).
        apodization : bool or str
            Apply an
            `apodization window <https://mathworld.wolfram.com/ApodizationFunction.html>`_
            before calculating the FFT in order to suppress streaks.
            Valid string values are {``'hann'`` or ``'hamming'`` or ``'tukey'``}
            If ``True`` or ``'hann'``, applies a Hann window.
            If ``'hamming'`` or ``'tukey'``, applies Hamming or Tukey
            windows, respectively (default is ``False``).
        real_fft_only : bool, default False
            If ``True`` and data is real-valued, uses :func:`numpy.fft.rfftn`
            instead of :func:`numpy.fft.fftn`
        **kwargs : dict
            other keyword arguments are described in :func:`numpy.fft.fftn`

        Returns
        -------
        s : :class:`~hyperspy._signals.complex_signal.ComplexSignal`
            A Signal containing the result of the FFT algorithm

        Raises
        ------
        NotImplementedError
            If performing FFT along a non-uniform axis.

        Examples
        --------
        >>> import skimage
        >>> im = hs.signals.Signal2D(skimage.data.camera())
        >>> im.fft()
        <ComplexSignal2D, title: FFT of , dimensions: (|512, 512)>

        >>> # Use following to plot power spectrum of `im`:
        >>> im.fft(shift=True, apodization=True).plot(power_spectrum=True)

        Notes
        -----
        Requires a uniform axis. For further information see the documentation
        of :func:`numpy.fft.fftn`
        """

        if self.axes_manager.signal_dimension == 0:
            raise AttributeError("Signal dimension must be at least one.")
        if apodization is True:
            apodization = "hann"

        if apodization:
            im_fft = self.apply_apodization(window=apodization, inplace=False)
        else:
            im_fft = self
        ax = self.axes_manager
        axes = ax.signal_indices_in_array
        if any([not axs.is_uniform for axs in self.axes_manager[axes]]):
            raise NotImplementedError("Not implemented for non-uniform axes.")
        use_real_fft = real_fft_only and (self.data.dtype.kind != "c")

        if use_real_fft:
            fft_f = np.fft.rfftn
        else:
            fft_f = np.fft.fftn

        if shift:
            im_fft = self._deepcopy_with_new_data(
                np.fft.fftshift(fft_f(im_fft.data, axes=axes, **kwargs), axes=axes)
            )
        else:
            im_fft = self._deepcopy_with_new_data(fft_f(self.data, axes=axes, **kwargs))

        im_fft.change_dtype("complex")
        im_fft.metadata.General.title = "FFT of {}".format(
            im_fft.metadata.General.title
        )
        im_fft.metadata.set_item("Signal.FFT.shifted", shift)
        if hasattr(self.metadata.Signal, "quantity"):
            self.metadata.Signal.__delattr__("quantity")

        for axis in im_fft.axes_manager.signal_axes:
            axis.scale = 1.0 / axis.size / axis.scale
            axis.offset = 0.0
            try:
                units = _ureg.parse_expression(str(axis.units)) ** (-1)
                axis.units = "{:~}".format(units.units)
            except UndefinedUnitError:
                _logger.warning("Units are not set or cannot be recognized")
            if shift:
                axis.offset = -axis.high_value / 2.0
        return im_fft

    def ifft(self, shift=None, return_real=True, **kwargs):
        """
        Compute the inverse discrete Fourier Transform.

        This function computes the real part of the inverse of the discrete
        Fourier Transform over the signal axes by means of the Fast Fourier
        Transform (FFT) as implemented in numpy.

        Parameters
        ----------
        shift : bool or None, optional
            If ``None``, the shift option will be set to the original status
            of the FFT using the value in metadata. If no FFT entry is
            present in metadata, the parameter will be set to ``False``.
            If ``True``, the origin of the FFT will be shifted to the centre.
            If ``False``, the origin will be kept at (0, 0)
            (default is ``None``).
        return_real : bool, default True
            If ``True``, returns only the real part of the inverse FFT.
            If ``False``, returns all parts.
        **kwargs : dict
            other keyword arguments are described in :func:`numpy.fft.ifftn`

        Returns
        -------
        s : :class:`~hyperspy.signal.BaseSignal` (or subclass)
            A Signal containing the result of the inverse FFT algorithm

        Raises
        ------
        NotImplementedError
            If performing IFFT along a non-uniform axis.

        Examples
        --------
        >>> import skimage
        >>> im = hs.signals.Signal2D(skimage.data.camera())
        >>> imfft = im.fft()
        >>> imfft.ifft()
        <Signal2D, title: real(iFFT of FFT of ), dimensions: (|512, 512)>

        Notes
        -----
        Requires a uniform axis. For further information see the documentation
        of :func:`numpy.fft.ifftn`
        """

        if self.axes_manager.signal_dimension == 0:
            raise AttributeError("Signal dimension must be at least one.")
        ax = self.axes_manager
        axes = ax.signal_indices_in_array
        if any([not axs.is_uniform for axs in self.axes_manager[axes]]):
            raise NotImplementedError("Not implemented for non-uniform axes.")
        if shift is None:
            shift = self.metadata.get_item("Signal.FFT.shifted", False)

        if shift:
            im_ifft = self._deepcopy_with_new_data(
                np.fft.ifftn(
                    np.fft.ifftshift(self.data, axes=axes), axes=axes, **kwargs
                )
            )
        else:
            im_ifft = self._deepcopy_with_new_data(
                np.fft.ifftn(self.data, axes=axes, **kwargs)
            )

        im_ifft.metadata.General.title = "iFFT of {}".format(
            im_ifft.metadata.General.title
        )
        if im_ifft.metadata.has_item("Signal.FFT"):
            del im_ifft.metadata.Signal.FFT

        if return_real:
            im_ifft = im_ifft.real

        for axis in im_ifft.axes_manager.signal_axes:
            axis.scale = 1.0 / axis.size / axis.scale
            try:
                units = _ureg.parse_expression(str(axis.units)) ** (-1)
                axis.units = "{:~}".format(units.units)
            except UndefinedUnitError:
                _logger.warning("Units are not set or cannot be recognized")
            axis.offset = 0.0
        return im_ifft

    def integrate1D(self, axis, out=None, rechunk=False):
        """Integrate the signal over the given axis.

        The integration is performed using
        `Simpson's rule <https://en.wikipedia.org/wiki/Simpson%%27s_rule>`_ if
        `axis.is_binned` is ``False`` and simple summation over the given axis
        if ``True`` (along binned axes, the detector already provides
        integrated counts per bin).

        Parameters
        ----------
        axis %s
        %s
        %s

        Returns
        -------
        s : :class:`~hyperspy.signal.BaseSignal` (or subclass)
            A new Signal containing the integral of the provided Signal along
            the specified axis.

        See Also
        --------
        integrate_simpson, derivative

        Examples
        --------
        >>> s = BaseSignal(np.random.random((64, 64, 1024)))
        >>> s
        <BaseSignal, title: , dimensions: (|1024, 64, 64)>
        >>> s.integrate1D(0)
        <Signal2D, title: , dimensions: (|64, 64)>

        """
        axis = self.axes_manager[axis]
        if isinstance(axis, tuple):
            axis = axis[0]
        if axis.is_binned:
            return self.sum(axis=axis, out=out, rechunk=rechunk)
        else:
            return self.integrate_simpson(axis=axis, out=out, rechunk=rechunk)

    integrate1D.__doc__ %= (ONE_AXIS_PARAMETER, OUT_ARG, RECHUNK_ARG)

    def indexmin(self, axis, out=None, rechunk=False):
        """Returns a signal with the index of the minimum along an axis.

        Parameters
        ----------
        axis %s
        %s
        %s

        Returns
        -------
        s : :class:`~hyperspy.signal.BaseSignal` (or subclass)
            A new Signal containing the indices of the minimum along the
            specified axis. Note: the data `dtype` is always ``int``.

        See Also
        --------
        max, min, sum, mean, std, var, indexmax, valuemax, valuemin

        Examples
        --------
        >>> s = BaseSignal(np.random.random((64, 64, 1024)))
        >>> s
        <BaseSignal, title: , dimensions: (|1024, 64, 64)>
        >>> s.indexmin(0)
        <Signal2D, title: , dimensions: (|64, 64)>

        """
        return self._apply_function_on_data_and_remove_axis(
            np.argmin, axis, out=out, rechunk=rechunk
        )

    indexmin.__doc__ %= (ONE_AXIS_PARAMETER, OUT_ARG, RECHUNK_ARG)

    def indexmax(self, axis, out=None, rechunk=False):
        """Returns a signal with the index of the maximum along an axis.

        Parameters
        ----------
        axis %s
        %s
        %s

        Returns
        -------
        s : :class:`~hyperspy.signal.BaseSignal` (or subclass)
            A new Signal containing the indices of the maximum along the
            specified axis. Note: the data `dtype` is always ``int``.

        See Also
        --------
        max, min, sum, mean, std, var, indexmin, valuemax, valuemin

        Examples
        --------
        >>> s = BaseSignal(np.random.random((64,64,1024)))
        >>> s
        <BaseSignal, title: , dimensions: (|1024, 64, 64)>
        >>> s.indexmax(0)
        <Signal2D, title: , dimensions: (|64, 64)>

        """
        return self._apply_function_on_data_and_remove_axis(
            np.argmax, axis, out=out, rechunk=rechunk
        )

    indexmax.__doc__ %= (ONE_AXIS_PARAMETER, OUT_ARG, RECHUNK_ARG)

    def valuemax(self, axis, out=None, rechunk=False):
        """Returns a signal with the value of coordinates of the maximum along
        an axis.

        Parameters
        ----------
        axis %s
        %s
        %s

        Returns
        -------
        s : :class:`~hyperspy.signal.BaseSignal` (or subclass)
            A new Signal containing the calibrated coordinate values of the
            maximum along the specified axis.

        See Also
        --------
        hyperspy.api.signals.BaseSignal.max, hyperspy.api.signals.BaseSignal.min,
        hyperspy.api.signals.BaseSignal.sum, hyperspy.api.signals.BaseSignal.mean,
        hyperspy.api.signals.BaseSignal.std, hyperspy.api.signals.BaseSignal.var,
        hyperspy.api.signals.BaseSignal.indexmax, hyperspy.api.signals.BaseSignal.indexmin,
        hyperspy.api.signals.BaseSignal.valuemin

        Examples
        --------
        >>> import numpy as np
        >>> s = BaseSignal(np.random.random((64, 64, 1024)))
        >>> s
        <BaseSignal, title: , dimensions: (|1024, 64, 64)>
        >>> s.valuemax(0)
        <Signal2D, title: , dimensions: (|64, 64)>

        """
        idx = self.indexmax(axis)
        data = self.axes_manager[axis].index2value(idx.data)
        if out is None:
            idx.data = data
            return idx
        else:
            out.data[:] = data
            out.events.data_changed.trigger(obj=out)

    valuemax.__doc__ %= (ONE_AXIS_PARAMETER, OUT_ARG, RECHUNK_ARG)

    def valuemin(self, axis, out=None, rechunk=False):
        """Returns a signal with the value of coordinates of the minimum along
        an axis.

        Parameters
        ----------
        axis %s
        %s
        %s

        Returns
        -------
        :class:`~hyperspy.api.signals.BaseSignal` or subclass
            A new Signal containing the calibrated coordinate values of the
            minimum along the specified axis.

        See Also
        --------
        hyperspy.api.signals.BaseSignal.max, hyperspy.api.signals.BaseSignal.min,
        hyperspy.api.signals.BaseSignal.sum, hyperspy.api.signals.BaseSignal.mean,
        hyperspy.api.signals.BaseSignal.std, hyperspy.api.signals.BaseSignal.var,
        hyperspy.api.signals.BaseSignal.indexmax, hyperspy.api.signals.BaseSignal.indexmin,
        hyperspy.api.signals.BaseSignal.valuemax

        """
        idx = self.indexmin(axis)
        data = self.axes_manager[axis].index2value(idx.data)
        if out is None:
            idx.data = data
            return idx
        else:
            out.data[:] = data
            out.events.data_changed.trigger(obj=out)

    valuemin.__doc__ %= (ONE_AXIS_PARAMETER, OUT_ARG, RECHUNK_ARG)

    def get_histogram(
        self, bins="fd", range_bins=None, max_num_bins=250, out=None, **kwargs
    ):
        """Return a histogram of the signal data.

        More sophisticated algorithms for determining the bins can be used
        by passing a string as the ``bins`` argument. Other than the ``'blocks'``
        and ``'knuth'`` methods, the available algorithms are the same as
        :func:`numpy.histogram`.

        Note: The lazy version of the algorithm only supports ``"scott"``
        and ``"fd"`` as a string argument for ``bins``.

        Parameters
        ----------
        %s
        %s
        %s
        %s
        %s
        **kwargs
            other keyword arguments (weight and density) are described in
            :func:`numpy.histogram`.

        Returns
        -------
        hist_spec : :class:`~.api.signals.Signal1D`
            A 1D spectrum instance containing the histogram.

        Notes
        -----
        See :func:`numpy.histogram` for more details on the
        meaning of the returned values.

        See Also
        --------
        hyperspy.api.signals.BaseSignal.print_summary_statistics,
        numpy.histogram, dask.array.histogram

        Examples
        --------
        >>> s = hs.signals.Signal1D(np.random.normal(size=(10, 100)))
        >>> # Plot the data histogram
        >>> s.get_histogram().plot()
        >>> # Plot the histogram of the signal at the current coordinates
        >>> s.get_current_signal().get_histogram().plot()

        """
        from hyperspy import signals

        data = self.data[~np.isnan(self.data)].flatten()

        hist, bin_edges = histogram(
            data, bins=bins, max_num_bins=max_num_bins, range=range_bins, **kwargs
        )

        if out is None:
            hist_spec = signals.Signal1D(hist)
        else:
            hist_spec = out
            if hist_spec.data.shape == hist.shape:
                hist_spec.data[:] = hist
            else:
                hist_spec.data = hist

        if isinstance(bins, str) and bins == "blocks":
            hist_spec.axes_manager.signal_axes[0].axis = bin_edges[:-1]
            warnings.warn(
                "The option `bins='blocks'` is not fully supported in this "
                "version of HyperSpy. It should be used for plotting purposes "
                "only.",
                UserWarning,
            )
        else:
            hist_spec.axes_manager[0].scale = bin_edges[1] - bin_edges[0]
            hist_spec.axes_manager[0].offset = bin_edges[0]
            hist_spec.axes_manager[0].size = hist.shape[-1]

        _set_histogram_metadata(self, hist_spec, **kwargs)

        if out is None:
            return hist_spec
        else:
            out.events.data_changed.trigger(obj=out)

    get_histogram.__doc__ %= (
        HISTOGRAM_BIN_ARGS,
        HISTOGRAM_RANGE_ARGS,
        HISTOGRAM_MAX_BIN_ARGS,
        OUT_ARG,
        RECHUNK_ARG,
    )

    def map(
        self,
        function,
        show_progressbar=None,
        num_workers=None,
        inplace=True,
        ragged=None,
        navigation_chunks=None,
        output_signal_size=None,
        output_dtype=None,
        lazy_output=None,
        silence_warnings=False,
        **kwargs,
    ):
        """Apply a function to the signal data at all the navigation
        coordinates.

        The function must operate on numpy arrays. It is applied to the data at
        each navigation coordinate pixel-py-pixel. Any extra keyword arguments
        are passed to the function. The keywords can take different values at
        different coordinates. If the function takes an `axis` or `axes`
        argument, the function is assumed to be vectorized and the signal axes
        are assigned to `axis` or `axes`.  Otherwise, the signal is iterated
        over the navigation axes and a progress bar is displayed to monitor the
        progress.

        In general, only navigation axes (order, calibration, and number) are
        guaranteed to be preserved.

        Parameters
        ----------

        function : :std:term:`function`
            Any function that can be applied to the signal. This function should
            not alter any mutable input arguments or input data. So do not do
            operations which alter the input, without copying it first.
            For example, instead of doing `image *= mask`, rather do
            `image = image * mask`. Likewise, do not do `image[5, 5] = 10`
            directly on the input data or arguments, but make a copy of it
            first. For example via `image = copy.deepcopy(image)`.
        %s
        %s
        inplace : bool, default True
            If ``True``, the data is replaced by the result. Otherwise
            a new Signal with the results is returned.
        ragged : None or bool, default None
            Indicates if the results for each navigation pixel are of identical
            shape (and/or numpy arrays to begin with). If ``None``,
            the output signal will be ragged only if the original signal is ragged.
        navigation_chunks : str, None, int or tuple of int, default ``None``
            Set the navigation_chunks argument to a tuple of integers to split
            the navigation axes into chunks. This can be useful to enable
            using multiple cores with signals which are less that 100 MB.
            This argument is passed to :meth:`~._signals.lazy.LazySignal.rechunk`.
        output_signal_size : None, tuple
            Since the size and dtype of the signal dimension of the output
            signal can be different from the input signal, this output signal
            size must be calculated somehow. If both ``output_signal_size``
            and ``output_dtype`` is ``None``, this is automatically determined.
            However, if for some reason this is not working correctly, this
            can be specified via ``output_signal_size`` and ``output_dtype``.
            The most common reason for this failing is due to the signal size
            being different for different navigation positions. If this is the
            case, use ragged=True. None is default.
        output_dtype : None, numpy.dtype
            See docstring for output_signal_size for more information.
            Default None.
        %s
        silence_warnings : bool, str or tuple, list of str
            If ``True``, don't warn when one of the signal axes are non-uniform,
            or the scales of the signal axes differ. If ``"non-uniform"``,
            ``"scales"`` or ``"units"`` are in the list the warning will be
            silenced when using non-uniform axis, different scales or different
            units of the signal axes, respectively.
            If ``False``, all warnings will be added to the logger.
            Default is ``False``.
        **kwargs : dict
            All extra keyword arguments are passed to the provided function

        Notes
        -----
        If the function results do not have identical shapes, the result is an
        array of navigation shape, where each element corresponds to the result
        of the function (of arbitrary object type), called a "ragged array". As
        such, most functions are not able to operate on the result and the data
        should be used directly.

        This method is similar to Python's :func:`python:map` that can
        also be utilized with a :class:`~hyperspy.signal.BaseSignal`
        instance for similar purposes. However, this method has the advantage of
        being faster because it iterates the underlying numpy data array
        instead of the :class:`~hyperspy.signal.BaseSignal`.

        Currently requires a uniform axis.

        Examples
        --------
        Apply a Gaussian filter to all the images in the dataset. The sigma
        parameter is constant:

        >>> import scipy.ndimage
        >>> im = hs.signals.Signal2D(np.random.random((10, 64, 64)))
        >>> im.map(scipy.ndimage.gaussian_filter, sigma=2.5)

        Apply a Gaussian filter to all the images in the dataset. The signal
        parameter is variable:

        >>> im = hs.signals.Signal2D(np.random.random((10, 64, 64)))
        >>> sigmas = hs.signals.BaseSignal(np.linspace(2, 5, 10)).T
        >>> im.map(scipy.ndimage.gaussian_filter, sigma=sigmas)

        Rotate the two signal dimensions, with different amount as a function
        of navigation index. Delay the calculation by getting the output
        lazily. The calculation is then done using the compute method.

        >>> from scipy.ndimage import rotate
        >>> s = hs.signals.Signal2D(np.random.random((5, 4, 40, 40)))
        >>> s_angle = hs.signals.BaseSignal(np.linspace(0, 90, 20).reshape(5, 4)).T
        >>> s.map(rotate, angle=s_angle, reshape=False, lazy_output=True)
        >>> s.compute()

        Rotate the two signal dimensions, with different amount as a function
        of navigation index. In addition, the output is returned as a new
        signal, instead of replacing the old signal.

        >>> s = hs.signals.Signal2D(np.random.random((5, 4, 40, 40)))
        >>> s_angle = hs.signals.BaseSignal(np.linspace(0, 90, 20).reshape(5, 4)).T
        >>> s_rot = s.map(rotate, angle=s_angle, reshape=False, inplace=False)

        If you want some more control over computing a signal that isn't lazy
        you can always set lazy_output to True and then compute the signal setting
        the scheduler to 'threading', 'processes', 'single-threaded' or 'distributed'.

        Additionally, you can set the navigation_chunks argument to a tuple of
        integers to split the navigation axes into chunks. This can be useful if your
        signal is less that 100 mb but you still want to use multiple cores.

        >>> s = hs.signals.Signal2D(np.random.random((5, 4, 40, 40)))
        >>> s_angle = hs.signals.BaseSignal(np.linspace(0, 90, 20).reshape(5, 4)).T
        >>> s.map(
        ...    rotate, angle=s_angle, reshape=False, lazy_output=True,
        ...    inplace=True, navigation_chunks=(2,2)
        ... )
        >>> s.compute()

        """
        if lazy_output is None:
            lazy_output = self._lazy
        if ragged is None:
            ragged = self.ragged
        if isinstance(silence_warnings, str):
            silence_warnings = (silence_warnings,)

        if isinstance(silence_warnings, (list, tuple)):
            for key in silence_warnings:
                if isinstance(key, str):
                    if key not in ["non-uniform", "units", "scales"]:
                        raise ValueError(
                            f"'{key}' is not a valid string for the "
                            "`silence_warnings` parameter."
                        )
                else:
                    raise TypeError(
                        "`silence_warnings` must be a boolean or a list, tuple of string."
                    )

        def _silence_warnings(silence_warnings, key):
            # Return True when it should silence warning,
            # otherwise False
            # Warn only when `silence_warnings` is False
            # or when the key is not in `silence_warnings`
            return silence_warnings is True or (
                isinstance(silence_warnings, (tuple, list)) and key in silence_warnings
            )

        # Separate arguments to pass to the mapping function:
        # ndkwargs dictionary contains iterating arguments which must be signals.
        # kwargs dictionary contains non-iterating arguments
        self_nav_shape = self.axes_manager.navigation_shape
        ndkwargs = {}
        ndkeys = [key for key in kwargs if isinstance(kwargs[key], BaseSignal)]
        for key in ndkeys:
            nd_nav_shape = kwargs[key].axes_manager.navigation_shape
            if nd_nav_shape == self_nav_shape:
                ndkwargs[key] = kwargs.pop(key)
            elif nd_nav_shape == () or nd_nav_shape == (1,):
                # This really isn't an iterating signal.
                kwargs[key] = np.squeeze(kwargs[key].data)
            else:
                raise ValueError(
                    f"The size of the navigation_shape for the kwarg {key} "
                    f"(<{nd_nav_shape}> must be consistent "
                    f"with the size of the mapped signal "
                    f"<{self_nav_shape}>"
                )
        if any([not ax.is_uniform for ax in self.axes_manager.signal_axes]):
            if not _silence_warnings(silence_warnings, "non-uniform"):
                _logger.warning(
                    "At least one axis of the signal is non-uniform. Can your "
                    "`function` operate on non-uniform axes?",
                )
        else:
            # Check if the signal axes have inhomogeneous scales and/or units and
            # display in warning if yes.
            scale = set()
            units = set()
            for i in range(len(self.axes_manager.signal_axes)):
                scale.add(self.axes_manager.signal_axes[i].scale)
                units.add(self.axes_manager.signal_axes[i].units)
            if len(scale) != 1 and not _silence_warnings(silence_warnings, "scales"):
                _logger.warning(
                    "The function you applied does not take into account "
                    "the difference of scales in-between axes."
                )
            if len(units) != 1 and not _silence_warnings(silence_warnings, "units"):
                _logger.warning(
                    "The function you applied does not take into account "
                    "the difference of units in-between axes."
                )
        # If the function has an axis argument and the signal dimension is 1,
        # we suppose that it can operate on the full array and we don't
        # iterate over the coordinates.
        fargs = []
        try:
            # numpy ufunc operate element-wise on the inputs and we don't
            # except them to have an axis argument
            if not isinstance(function, np.ufunc):
                fargs = inspect.signature(function).parameters.keys()
            else:
                _logger.warning(
                    f"The function `{function.__name__}` can directly operate "
                    "on hyperspy signals and it is not necessary to use `map`."
                )
        except TypeError as error:
            # This is probably a Cython function that is not supported by
            # inspect.
            _logger.warning(error)

        # If the function has an `axes` or `axis` argument
        # we suppose that it can operate on the full array and we don't
        # iterate over the coordinates.
        # We use _map_all only when the user doesn't specify axis/axes
        if (
            not ndkwargs
            and not lazy_output
            and self.axes_manager.signal_dimension == 1
            and "axis" in fargs
            and "axis" not in kwargs.keys()
        ):
            kwargs["axis"] = self.axes_manager.signal_axes[-1].index_in_array
            result = self._map_all(function, inplace=inplace, **kwargs)
        elif (
            not ndkwargs
            and not lazy_output
            and "axes" in fargs
            and "axes" not in kwargs.keys()
        ):
            kwargs["axes"] = tuple(
                [axis.index_in_array for axis in self.axes_manager.signal_axes]
            )
            result = self._map_all(function, inplace=inplace, **kwargs)
        else:
            if show_progressbar is None:
                from hyperspy.defaults_parser import preferences

                show_progressbar = preferences.General.show_progressbar
            # Iteration over coordinates.
            result = self._map_iterate(
                function,
                iterating_kwargs=ndkwargs,  # function argument(s) (iterating)
                show_progressbar=show_progressbar,
                ragged=ragged,
                inplace=inplace,
                lazy_output=lazy_output,
                num_workers=num_workers,
                output_dtype=output_dtype,
                output_signal_size=output_signal_size,
                navigation_chunks=navigation_chunks,
                **kwargs,  # function argument(s) (non-iterating)
            )
        if not inplace:
            return result
        else:
            self.events.data_changed.trigger(obj=self)

    map.__doc__ %= (SHOW_PROGRESSBAR_ARG, LAZY_OUTPUT_ARG, NUM_WORKERS_ARG)

    def _map_all(self, function, inplace=True, **kwargs):
        """
        The function has to have either 'axis' or 'axes' keyword argument,
        and hence support operating on the full dataset efficiently and remove
        the signal axes.
        """
        old_shape = self.data.shape
        newdata = function(self.data, **kwargs)
        if inplace:
            self.data = newdata
            if self.data.shape != old_shape:
                self.axes_manager.remove(self.axes_manager.signal_axes)
            self._lazy = False
            self._assign_subclass()
            return None
        else:
            sig = self._deepcopy_with_new_data(newdata)
            if sig.data.shape != old_shape:
                sig.axes_manager.remove(sig.axes_manager.signal_axes)
            sig._lazy = False
            sig._assign_subclass()
            return sig

    def _map_iterate(
        self,
        function,
        iterating_kwargs=None,
        show_progressbar=None,
        ragged=False,
        inplace=True,
        output_signal_size=None,
        output_dtype=None,
        lazy_output=None,
        num_workers=None,
        navigation_chunks="auto",
        **kwargs,
    ):
        if lazy_output is None:
            lazy_output = self._lazy

        if not self._lazy:
            s_input = self.as_lazy()
            s_input.rechunk(nav_chunks=navigation_chunks)
        else:
            s_input = self

        # unpacking keyword arguments
        if iterating_kwargs is None:
            iterating_kwargs = {}
        elif isinstance(iterating_kwargs, (tuple, list)):
            iterating_kwargs = dict((k, v) for k, v in iterating_kwargs)

        nav_indexes = s_input.axes_manager.navigation_indices_in_array
        chunk_span = np.equal(s_input.data.chunksize, s_input.data.shape)
        chunk_span = [
            chunk_span[i] for i in s_input.axes_manager.signal_indices_in_array
        ]

        if not all(chunk_span):
            _logger.info(
                "The chunk size needs to span the full signal size, rechunking..."
            )

            old_sig = s_input.rechunk(inplace=False, nav_chunks=None)
        else:
            old_sig = s_input

        os_am = old_sig.axes_manager

        autodetermine = (
            output_signal_size is None or output_dtype is None
        )  # try to guess output dtype and sig size?
        if autodetermine and is_cupy_array(self.data):
            raise ValueError(
                "Autodetermination of `output_signal_size` and "
                "`output_dtype` is not supported for cupy array."
            )

        args, arg_keys = old_sig._get_iterating_kwargs(iterating_kwargs)

        if autodetermine:  # trying to guess the output d-type and size from one signal
            testing_kwargs = {}
            for ikey, key in enumerate(arg_keys):
                test_ind = (0,) * len(os_am.navigation_axes)
                # For discussion on if squeeze is necessary, see
                # https://github.com/hyperspy/hyperspy/pull/2981
                testing_kwargs[key] = np.squeeze(args[ikey][test_ind].compute())[()]
            testing_kwargs = {**kwargs, **testing_kwargs}
            test_data = np.array(
                old_sig.inav[(0,) * len(os_am.navigation_shape)].data.compute()
            )
            temp_output_signal_size, temp_output_dtype = guess_output_signal_size(
                test_data=test_data,
                function=function,
                ragged=ragged,
                **testing_kwargs,
            )
            if output_signal_size is None:
                output_signal_size = temp_output_signal_size
            if output_dtype is None:
                output_dtype = temp_output_dtype
        output_shape = self.axes_manager._navigation_shape_in_array + output_signal_size
        arg_pairs, adjust_chunks, new_axis, output_pattern = _get_block_pattern(
            (old_sig.data,) + args, output_shape
        )

        axes_changed = len(new_axis) != 0 or len(adjust_chunks) != 0

        mapped = da.blockwise(
            process_function_blockwise,
            output_pattern,
            *concat(arg_pairs),
            adjust_chunks=adjust_chunks,
            new_axes=new_axis,
            align_arrays=False,
            dtype=output_dtype,
            concatenate=True,
            arg_keys=arg_keys,
            function=function,
            output_dtype=output_dtype,
            nav_indexes=nav_indexes,
            output_signal_size=output_signal_size,
            **kwargs,
        )

        data_stored = False

        cm = ProgressBar if show_progressbar else dummy_context_manager

        if inplace:
            if (
                not self._lazy
                and not lazy_output
                and (mapped.shape == self.data.shape)
                and (mapped.dtype == self.data.dtype)
            ):
                # da.store is used to avoid unnecessary amount of memory usage.
                # By using it here, the contents in mapped is written directly to
                # the existing NumPy array, avoiding a potential doubling of memory use.
                with cm():
                    da.store(
                        mapped,
                        self.data,
                        dtype=mapped.dtype,
                        compute=True,
                        num_workers=num_workers,
                        lock=False,
                    )
                data_stored = True
            else:
                self.data = mapped
            sig = self
        else:
            sig = s_input._deepcopy_with_new_data(mapped)

        am = sig.axes_manager
        sig._lazy = lazy_output

        if ragged:
            axes_dicts = self.axes_manager._get_navigation_axes_dicts()
            sig.axes_manager.__init__(axes_dicts)
            sig.axes_manager._ragged = True
        elif axes_changed:
            am.remove(am.signal_axes[len(output_signal_size) :])
            for ind in range(len(output_signal_size) - am.signal_dimension, 0, -1):
                am._append_axis(size=output_signal_size[-ind], navigate=False)

        if not ragged:
            sig.axes_manager._ragged = False
            if output_signal_size == () and am.navigation_dimension == 0:
                add_scalar_axis(sig)
            sig.get_dimensions_from_data()
        sig._assign_subclass()

        if not lazy_output and not data_stored:
            with cm():
                sig.data = sig.data.compute(num_workers=num_workers)

        return sig

    def _get_iterating_kwargs(self, iterating_kwargs):
        nav_chunks = self.get_chunk_size(self.axes_manager.navigation_axes)
        args, arg_keys = (), ()
        for key in iterating_kwargs:
            if iterating_kwargs[key]._lazy:
                axes = iterating_kwargs[key].axes_manager.navigation_axes
                if iterating_kwargs[key].get_chunk_size(axes) != nav_chunks:
                    iterating_kwargs[key].rechunk(nav_chunks=nav_chunks, sig_chunks=-1)
                chunk_span = np.equal(
                    iterating_kwargs[key].data.chunksize,
                    iterating_kwargs[key].data.shape,
                )
                chunk_span = [
                    chunk_span[i]
                    for i in iterating_kwargs[key].axes_manager.signal_indices_in_array
                ]
                if not all(chunk_span):
                    iterating_kwargs[key].rechunk(nav_chunks=nav_chunks, sig_chunks=-1)
            else:
                iterating_kwargs[key] = iterating_kwargs[key].as_lazy()
                iterating_kwargs[key].rechunk(nav_chunks=nav_chunks, sig_chunks=-1)
            args += (iterating_kwargs[key].data,)
            arg_keys += (key,)
        return args, arg_keys

    def copy(self):
        """
        Return a "shallow copy" of this Signal using the
        standard library's :func:`~copy.copy` function. Note: this will
        return a copy of the signal, but it will not duplicate the underlying
        data in memory, and both Signals will reference the same data.

        See Also
        --------
        :meth:`~hyperspy.api.signals.BaseSignal.deepcopy`
        """
        try:
            backup_plot = self._plot
            self._plot = None
            return copy.copy(self)
        finally:
            self._plot = backup_plot

    def __deepcopy__(self, memo):
        dc = type(self)(**self._to_dictionary())
        if isinstance(dc.data, np.ndarray):
            dc.data = dc.data.copy()

        # uncomment if we want to deepcopy models as well:

        # dc.models._add_dictionary(
        #     copy.deepcopy(
        #         self.models._models.as_dictionary()))

        # The Signal subclass might change the view on init
        # The following code just copies the original view
        for oaxis, caxis in zip(self.axes_manager._axes, dc.axes_manager._axes):
            caxis.navigate = oaxis.navigate

        if dc.metadata.has_item("Markers"):
            temp_marker_dict = dc.metadata.Markers.as_dictionary()
            markers_dict = {}
            for key in temp_marker_dict:
                markers_dict[key] = markers_dict_to_markers(
                    temp_marker_dict[key],
                )
            dc.metadata.Markers = markers_dict
        return dc

    def deepcopy(self):
        """
        Return a "deep copy" of this Signal using the
        standard library's :func:`~copy.deepcopy` function. Note: this means
        the underlying data structure will be duplicated in memory.

        See Also
        --------
        :meth:`~hyperspy.api.signals.BaseSignal.copy`
        """
        return copy.deepcopy(self)

    def change_dtype(self, dtype, rechunk=False):
        """Change the data type of a Signal.

        Parameters
        ----------
        dtype : str or :class:`numpy.dtype`
            Typecode string or data-type to which the Signal's data array is
            cast. In addition to all the standard numpy :ref:`arrays.dtypes`,
            HyperSpy supports four extra dtypes for RGB images: ``'rgb8'``,
            ``'rgba8'``, ``'rgb16'``, and ``'rgba16'``. Changing from and to
            any ``rgb(a)`` `dtype` is more constrained than most other `dtype`
            conversions. To change to an ``rgb(a)`` `dtype`,
            the `signal_dimension` must be 1, and its size should be 3 (for
            ``rgb``) or 4 (for ``rgba``) `dtypes`. The original `dtype`
            should be ``uint8`` or ``uint16`` if  converting to ``rgb(a)8``
            or ``rgb(a))16``, and the `navigation_dimension` should be at
            least 2. After conversion, the `signal_dimension` becomes 2. The
            `dtype` of images with original `dtype` ``rgb(a)8`` or ``rgb(a)16``
            can only be changed to ``uint8`` or ``uint16``, and the
            `signal_dimension` becomes 1.
        %s


        Examples
        --------
        >>> s = hs.signals.Signal1D([1, 2, 3, 4, 5])
        >>> s.data
        array([1, 2, 3, 4, 5])
        >>> s.change_dtype('float')
        >>> s.data
        array([1., 2., 3., 4., 5.])
        """
        if not isinstance(dtype, np.dtype):
            if dtype in rgb_tools.rgb_dtypes:
                if self.axes_manager.signal_dimension != 1:
                    raise AttributeError(
                        "Only 1D signals can be converted " "to RGB images."
                    )
                if "8" in dtype and self.data.dtype.name != "uint8":
                    raise AttributeError(
                        "Only signals with dtype uint8 can be converted to "
                        "rgb8 images"
                    )
                elif "16" in dtype and self.data.dtype.name != "uint16":
                    raise AttributeError(
                        "Only signals with dtype uint16 can be converted to "
                        "rgb16 images"
                    )
                replot = self._plot is not None and self._plot.is_active
                if replot:
                    # Close the figure to avoid error with events
                    self._plot.close()
                self.data = rgb_tools.regular_array2rgbx(self.data)
                self.axes_manager.remove(-1)
                self.axes_manager._set_signal_dimension(2)
                self._assign_subclass()
                if replot:
                    self.plot()
                return
            else:
                dtype = np.dtype(dtype)
        if rgb_tools.is_rgbx(self.data) is True:
            ddtype = self.data.dtype.fields["B"][0]

            if ddtype != dtype:
                raise ValueError("It is only possibile to change to %s." % ddtype)
            replot = self._plot is not None and self._plot.is_active
            if replot:
                # Close the figure to avoid error with events
                self._plot.close()
            self.data = rgb_tools.rgbx2regular_array(self.data)
            self.axes_manager._append_axis(
                size=self.data.shape[-1],
                scale=1,
                offset=0,
                name="RGB index",
                navigate=False,
            )
            self.axes_manager._set_signal_dimension(1)
            self._assign_subclass()
            if replot:
                self.plot()
            return
        else:
            self.data = self.data.astype(dtype)
        self._assign_subclass()

    change_dtype.__doc__ %= RECHUNK_ARG

    def estimate_poissonian_noise_variance(
        self,
        expected_value=None,
        gain_factor=None,
        gain_offset=None,
        correlation_factor=None,
    ):
        r"""Estimate the Poissonian noise variance of the signal.

        The variance is stored in the
        `metadata.Signal.Noise_properties.variance` attribute.

        The Poissonian noise variance is equal to the expected value. With the
        default arguments, this method simply sets the variance attribute to
        the given `expected_value`. However, more generally (although then the
        noise is not strictly Poissonian), the variance may be proportional to
        the expected value. Moreover, when the noise is a mixture of white
        (Gaussian) and Poissonian noise, the variance is described by the
        following linear model:

            .. math::

                \mathrm{Var}[X] = (a * \mathrm{E}[X] + b) * c

        Where `a` is the `gain_factor`, `b` is the `gain_offset` (the Gaussian
        noise variance) and `c` the `correlation_factor`. The correlation
        factor accounts for correlation of adjacent signal elements that can
        be modeled as a convolution with a Gaussian point spread function.

        Parameters
        ----------
        expected_value : None or :class:`~hyperspy.signal.BaseSignal` (or subclass)
            If ``None``, the signal data is taken as the expected value. Note
            that this may be inaccurate where the value of `data` is small.
        gain_factor : None or float
            `a` in the above equation. Must be positive. If ``None``, take the
            value from `metadata.Signal.Noise_properties.Variance_linear_model`
            if defined. Otherwise, suppose pure Poissonian noise (*i.e.*
            ``gain_factor=1``). If not ``None``, the value is stored in
            `metadata.Signal.Noise_properties.Variance_linear_model`.
        gain_offset : None or float
            `b` in the above equation. Must be positive. If ``None``, take the
            value from `metadata.Signal.Noise_properties.Variance_linear_model`
            if defined. Otherwise, suppose pure Poissonian noise (*i.e.*
            ``gain_offset=0``). If not ``None``, the value is stored in
            `metadata.Signal.Noise_properties.Variance_linear_model`.
        correlation_factor : None or float
            `c` in the above equation. Must be positive. If ``None``, take the
            value from `metadata.Signal.Noise_properties.Variance_linear_model`
            if defined. Otherwise, suppose pure Poissonian noise (*i.e.*
            ``correlation_factor=1``). If not ``None``, the value is stored in
            `metadata.Signal.Noise_properties.Variance_linear_model`.

        """
        if expected_value is None:
            expected_value = self
        dc = expected_value.data if expected_value._lazy else expected_value.data.copy()
        if self.metadata.has_item("Signal.Noise_properties.Variance_linear_model"):
            vlm = self.metadata.Signal.Noise_properties.Variance_linear_model
        else:
            self.metadata.add_node("Signal.Noise_properties.Variance_linear_model")
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
        variance = self._estimate_poissonian_noise_variance(
            dc, gain_factor, gain_offset, correlation_factor
        )

        variance = BaseSignal(variance, attributes={"_lazy": self._lazy})
        variance.axes_manager = self.axes_manager
        variance.metadata.General.title = "Variance of " + self.metadata.General.title

        self.set_noise_variance(variance)

    @staticmethod
    def _estimate_poissonian_noise_variance(
        dc, gain_factor, gain_offset, correlation_factor
    ):
        variance = (dc * gain_factor + gain_offset) * correlation_factor
        variance = np.clip(variance, gain_offset * correlation_factor, np.inf)
        return variance

    def set_noise_variance(self, variance):
        """Set the noise variance of the signal.

        Equivalent to ``s.metadata.set_item("Signal.Noise_properties.variance", variance)``.

        Parameters
        ----------
        variance : None or float or :class:`~hyperspy.signal.BaseSignal` (or subclass)
            Value or values of the noise variance. A value of None is
            equivalent to clearing the variance.

        Returns
        -------
        None

        """
        if isinstance(variance, BaseSignal):
            if (
                variance.axes_manager.navigation_shape
                != self.axes_manager.navigation_shape
            ):
                raise ValueError(
                    "The navigation shape of the `variance` is "
                    "not equal to the navigation shape of the signal"
                )
        elif isinstance(variance, numbers.Number):
            pass
        elif variance is None:
            pass
        else:
            raise ValueError(
                "`variance` must be one of [None, float, "
                f"hyperspy.signal.BaseSignal], not {type(variance)}."
            )

        self.metadata.set_item("Signal.Noise_properties.variance", variance)

    def get_noise_variance(self):
        """Get the noise variance of the signal, if set.

        Equivalent to ``s.metadata.Signal.Noise_properties.variance``.

        Parameters
        ----------
        None

        Returns
        -------
        variance : None or float or :class:`~hyperspy.signal.BaseSignal` (or subclass)
            Noise variance of the signal, if set.
            Otherwise returns None.

        """
        if "Signal.Noise_properties.variance" in self.metadata:
            return self.metadata.Signal.Noise_properties.variance

        return None

    def get_current_signal(self, auto_title=True, auto_filename=True, as_numpy=False):
        """Returns the data at the current coordinates as a
        :class:`~hyperspy.signal.BaseSignal` subclass.

        The signal subclass is the same as that of the current object. All the
        axes navigation attributes are set to ``False``.

        Parameters
        ----------
        auto_title : bool
            If ``True``, the current indices (in parentheses) are appended to
            the title, separated by a space, otherwise the title of the signal
            is used unchanged.
        auto_filename : bool
            If ``True`` and `tmp_parameters.filename` is defined
            (which is always the case when the Signal has been read from a
            file), the filename stored in the metadata is modified by
            appending an underscore and the current indices in parentheses.
        as_numpy : bool or None
            Only with cupy array. If ``True``, return the current signal
            as numpy array, otherwise return as cupy array.

        Returns
        -------
        cs : :class:`~hyperspy.signal.BaseSignal` (or subclass)
            The data at the current coordinates as a Signal

        Examples
        --------
        >>> im = hs.signals.Signal2D(np.zeros((2, 3, 32, 32)))
        >>> im
        <Signal2D, title: , dimensions: (3, 2|32, 32)>
        >>> im.axes_manager.indices = (2, 1)
        >>> im.get_current_signal()
        <Signal2D, title:  (2, 1), dimensions: (|32, 32)>

        """
        metadata = self.metadata.deepcopy()

        # Check if marker update
        if self.metadata.has_item("Markers"):
            markers_dict = self.metadata.Markers
            marker_name_list = self.metadata.Markers.keys()
            new_markers_dict = metadata.Markers
            for marker_name in marker_name_list:
                marker = markers_dict[marker_name]
                new_marker = new_markers_dict[marker_name]
                kwargs = marker.get_current_kwargs(only_variable_length=False)
                new_marker.kwargs = kwargs

        class_ = assign_signal_subclass(
            dtype=self.data.dtype,
            signal_dimension=self.axes_manager.signal_dimension,
            signal_type=self._signal_type,
            lazy=False,
        )

        cs = class_(
            self._get_current_data(as_numpy=as_numpy),
            axes=self.axes_manager._get_signal_axes_dicts(),
            metadata=metadata.as_dictionary(),
        )

        # reinitialize the markers from dictionary
        if cs.metadata.has_item("Markers"):
            temp_marker_dict = cs.metadata.Markers.as_dictionary()
            markers_dict = {}
            for key, item in temp_marker_dict.items():
                markers_dict[key] = markers_dict_to_markers(item)
                markers_dict[key]._signal = cs
            cs.metadata.Markers = markers_dict

        if auto_filename is True and self.tmp_parameters.has_item("filename"):
            cs.tmp_parameters.filename = (
                self.tmp_parameters.filename + "_" + str(self.axes_manager.indices)
            )
            cs.tmp_parameters.extension = self.tmp_parameters.extension
            cs.tmp_parameters.folder = self.tmp_parameters.folder
        if auto_title is True:
            cs.metadata.General.title = (
                cs.metadata.General.title + " " + str(self.axes_manager.indices)
            )
        cs.axes_manager._set_axis_attribute_values("navigate", False)
        return cs

    def _get_navigation_signal(self, data=None, dtype=None):
        """Return a signal with the same axes as the navigation space.

        Parameters
        ----------
        data : None or :class:`numpy.ndarray`, optional
            If ``None``, the resulting Signal data is an array of the same
            `dtype` as the current one filled with zeros. If a numpy array,
            the array must have the correct dimensions.

        dtype : :class:`numpy.dtype`, optional
            The desired data-type for the data array when `data` is ``None``,
            e.g., ``numpy.int8``.  The default is the data type of the current
            signal data.
        """
        if data is not None:
            ref_shape = (
                self.axes_manager._navigation_shape_in_array
                if self.axes_manager.navigation_dimension != 0
                else (1,)
            )
            if data.shape != ref_shape:
                raise ValueError(
                    (
                        "data.shape %s is not equal to the current navigation "
                        "shape in array which is %s"
                    )
                    % (str(data.shape), str(ref_shape))
                )
        else:
            if dtype is None:
                dtype = self.data.dtype
            if self.axes_manager.navigation_dimension == 0:
                data = np.array(
                    [
                        0,
                    ],
                    dtype=dtype,
                )
            else:
                data = np.zeros(
                    self.axes_manager._navigation_shape_in_array, dtype=dtype
                )
        if self.axes_manager.navigation_dimension == 0:
            s = BaseSignal(data)
        elif self.axes_manager.navigation_dimension == 1:
            from hyperspy._signals.signal1d import Signal1D

            s = Signal1D(data, axes=self.axes_manager._get_navigation_axes_dicts())
        elif self.axes_manager.navigation_dimension == 2:
            from hyperspy._signals.signal2d import Signal2D

            s = Signal2D(data, axes=self.axes_manager._get_navigation_axes_dicts())
        else:
            s = BaseSignal(data, axes=self.axes_manager._get_navigation_axes_dicts()).T
        if isinstance(data, da.Array):
            s = s.as_lazy()
        return s

    def _get_signal_signal(self, data=None, dtype=None):
        """Return a signal with the same axes as the signal space.

        Parameters
        ----------
        data : None or :class:`numpy.ndarray`, optional
            If ``None``, the resulting Signal data is an array of the same
            `dtype` as the current one filled with zeros. If a numpy array,
            the array must have the correct dimensions.

        dtype : :class:`numpy.dtype`, optional
            The desired data-type for the data array when `data` is ``None``,
            e.g., ``numpy.int8``.  The default is the data type of the current
            signal data.
        """
        if data is not None:
            ref_shape = (
                self.axes_manager._signal_shape_in_array
                if self.axes_manager.signal_dimension != 0
                else (1,)
            )
            if data.shape != ref_shape:
                raise ValueError(
                    "data.shape %s is not equal to the current signal shape in"
                    " array which is %s" % (str(data.shape), str(ref_shape))
                )
        else:
            if dtype is None:
                dtype = self.data.dtype
            if self.axes_manager.signal_dimension == 0:
                data = np.array(
                    [
                        0,
                    ],
                    dtype=dtype,
                )
            else:
                data = np.zeros(self.axes_manager._signal_shape_in_array, dtype=dtype)

        if self.axes_manager.signal_dimension == 0:
            s = BaseSignal(data)
            s.set_signal_type(self.metadata.Signal.signal_type)
        else:
            s = self.__class__(data, axes=self.axes_manager._get_signal_axes_dicts())
        if isinstance(data, da.Array):
            s = s.as_lazy()
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

    def as_signal1D(self, spectral_axis, out=None, optimize=True):
        """Return the Signal as a spectrum.

        The chosen spectral axis is moved to the last index in the
        array and the data is made contiguous for efficient iteration over
        spectra. By default, the method ensures the data is stored optimally,
        hence often making a copy of the data. See
        :meth:`~hyperspy.api.signals.BaseSignal.transpose` for a more general
        method with more options.

        Parameters
        ----------
        spectral_axis %s
        %s
        %s

        See Also
        --------
        hyperspy.api.signals.BaseSignal.as_signal2D,
        hyperspy.api.signals.BaseSignal.transpose,
        hyperspy.api.transpose

        Examples
        --------
        >>> img = hs.signals.Signal2D(np.ones((3, 4, 5, 6)))
        >>> img
        <Signal2D, title: , dimensions: (4, 3|6, 5)>
        >>> img.as_signal1D(-1+1j)
        <Signal1D, title: , dimensions: (6, 5, 4|3)>
        >>> img.as_signal1D(0)
        <Signal1D, title: , dimensions: (6, 5, 3|4)>

        """
        sp = self.transpose(signal_axes=[spectral_axis], optimize=optimize)
        if out is None:
            return sp
        else:
            if out._lazy:
                out.data = sp.data
            else:
                out.data[:] = sp.data
            out.events.data_changed.trigger(obj=out)

    as_signal1D.__doc__ %= (
        ONE_AXIS_PARAMETER,
        OUT_ARG,
        OPTIMIZE_ARG.replace("False", "True"),
    )

    def as_signal2D(self, image_axes, out=None, optimize=True):
        """Convert a signal to a :class:`~hyperspy.api.signals.Signal2D`.

        The chosen image axes are moved to the last indices in the
        array and the data is made contiguous for efficient
        iteration over images.

        Parameters
        ----------
        image_axes : tuple (of int, str or :class:`~hyperspy.axes.DataAxis`)
            Select the image axes. Note that the order of the axes matters
            and it is given in the "natural" i.e. `X`, `Y`, `Z`... order.
        %s
        %s

        Raises
        ------
        DataDimensionError
            When `data.ndim` < 2

        See Also
        --------
        hyperspy.api.signals.BaseSignal.as_signal1D,
        hyperspy.api.signals.BaseSignal.transpose,
        hyperspy.api.transpose

        Examples
        --------
        >>> s = hs.signals.Signal1D(np.ones((2, 3, 4, 5)))
        >>> s
        <Signal1D, title: , dimensions: (4, 3, 2|5)>
        >>> s.as_signal2D((0, 1))
        <Signal2D, title: , dimensions: (5, 2|4, 3)>

        >>> s.to_signal2D((1, 2))
        <Signal2D, title: , dimensions: (2, 5|4, 3)>

        """
        if self.data.ndim < 2:
            raise DataDimensionError(
                "A Signal dimension must be >= 2 to be converted to a Signal2D"
            )
        im = self.transpose(signal_axes=image_axes, optimize=optimize)
        if out is None:
            return im
        else:
            if out._lazy:
                out.data = im.data
            else:
                out.data[:] = im.data
            out.events.data_changed.trigger(obj=out)

    as_signal2D.__doc__ %= (OUT_ARG, OPTIMIZE_ARG.replace("False", "True"))

    def _assign_subclass(self):
        mp = self.metadata
        self.__class__ = assign_signal_subclass(
            dtype=self.data.dtype,
            signal_dimension=self.axes_manager.signal_dimension,
            signal_type=mp.Signal.signal_type
            if "Signal.signal_type" in mp
            else self._signal_type,
            lazy=self._lazy,
        )
        if self._alias_signal_types:  # In case legacy types exist:
            mp.Signal.signal_type = self._signal_type  # set to default!
        self.__init__(self.data, full_initialisation=False)
        if self._lazy:
            self._make_lazy()

    def set_signal_type(self, signal_type=""):
        """Set the signal type and convert the current signal accordingly.

        The ``signal_type`` attribute specifies the type of data that the signal
        contains e.g. electron energy-loss spectroscopy data,
        photoemission spectroscopy data, etc.

        When setting ``signal_type`` to a "known" type, HyperSpy converts the
        current signal to the most appropriate
        :class:`~.api.signals.BaseSignal` subclass. Known signal types are
        signal types that have a specialized
        :class:`~.api.signals.BaseSignal` subclass associated, usually
        providing specific features for the analysis of that type of signal.

        HyperSpy ships only with generic signal types. External packages
        can register domain-specific signal types, usually associated with
        specific measurement techniques. To print a list of
        registered signal types in the current installation, call
        :func:`~.api.print_known_signal_types`, and see
        the developer guide for details on how to add new signal_types.
        A non-exhaustive list of HyperSpy extensions is maintained
        here: https://github.com/hyperspy/hyperspy-extensions-list. See also the
        `HyperSpy Website <https://hyperspy.org>`_ for an overview of the
        ecosystem.

        Parameters
        ----------
        signal_type : str, optional
            If no arguments are passed, the ``signal_type`` is set to undefined
            and the current signal converted to a generic signal subclass.
            Otherwise, set the signal_type to the given signal
            type or to the signal type corresponding to the given signal type
            alias. Setting the signal_type to a known signal type (if exists)
            is highly advisable. If none exists, it is good practice
            to set signal_type to a value that best describes the data signal
            type.

        See Also
        --------
        hyperspy.api.print_known_signal_types

        Examples
        --------

        Let's first print all known signal types (assuming the extensions
        `eXSpy <https://hyperspy.org/exspy>`_ and `holoSpy
        <https://hyperspy.org/holospy>`_ are installed):

        >>> s = hs.signals.Signal1D([0, 1, 2, 3])
        >>> s
        <Signal1D, title: , dimensions: (|4)>
        >>> hs.print_known_signal_types() # doctest: +SKIP
        +--------------------+---------------------+--------------------+----------+
        |    signal_type     |       aliases       |     class name     | package  |
        +--------------------+---------------------+--------------------+----------+
        | DielectricFunction | dielectric function | DielectricFunction |   exspy  |
        |      EDS_SEM       |                     |   EDSSEMSpectrum   |   exspy  |
        |      EDS_TEM       |                     |   EDSTEMSpectrum   |   exspy  |
        |        EELS        |       TEM EELS      |    EELSSpectrum    |   exspy  |
        |      hologram      |                     |   HologramImage    |  holospy |
        +--------------------+---------------------+--------------------+----------+

        We can set the ``signal_type`` using the ``signal_type``:

        >>> s.set_signal_type("EELS") # doctest: +SKIP
        >>> s # doctest: +SKIP
        <EELSSpectrum, title: , dimensions: (|4)>
        >>> s.set_signal_type("EDS_SEM") # doctest: +SKIP
        >>> s # doctest: +SKIP
        <EDSSEMSpectrum, title: , dimensions: (|4)>

        or any of its aliases:

        >>> s.set_signal_type("TEM EELS") # doctest: +SKIP
        >>> s # doctest: +SKIP
        <EELSSpectrum, title: , dimensions: (|4)>

        To set the ``signal_type`` to "undefined", simply call the method without
        arguments:

        >>> s.set_signal_type()
        >>> s
        <Signal1D, title: , dimensions: (|4)>

        """
        if signal_type is None:
            raise TypeError("The `signal_type` argument must be of string type.")

        self.metadata.Signal.signal_type = signal_type
        # _assign_subclass takes care of matching aliases with their
        # corresponding signal class
        self._assign_subclass()

    def set_signal_origin(self, origin):
        """Set the `signal_origin` metadata value.

        The `signal_origin` attribute specifies if the data was obtained
        through experiment or simulation.

        Parameters
        ----------
        origin : str
            Typically ``'experiment'`` or ``'simulation'``
        """
        self.metadata.Signal.signal_origin = origin

    def print_summary_statistics(self, formatter="%.3g", rechunk=False):
        """Prints the five-number summary statistics of the data, the mean, and
        the standard deviation.

        Prints the mean, standard deviation (std), maximum (max), minimum
        (min), first quartile (Q1), median, and third quartile. nans are
        removed from the calculations.

        Parameters
        ----------
        formatter : str
           The number formatter to use for the output
        %s

        See Also
        --------
        get_histogram

        """
        _mean, _std, _min, _q1, _q2, _q3, _max = self._calculate_summary_statistics(
            rechunk=rechunk
        )
        print(underline("Summary statistics"))
        print("mean:\t" + formatter % _mean)
        print("std:\t" + formatter % _std)
        print()
        print("min:\t" + formatter % _min)
        print("Q1:\t" + formatter % _q1)
        print("median:\t" + formatter % _q2)
        print("Q3:\t" + formatter % _q3)
        print("max:\t" + formatter % _max)

    print_summary_statistics.__doc__ %= RECHUNK_ARG

    def _calculate_summary_statistics(self, **kwargs):
        data = self.data
        data = data[~np.isnan(data)]
        _mean = np.nanmean(data)
        _std = np.nanstd(data)
        _min = np.nanmin(data)
        _q1 = np.percentile(data, 25)
        _q2 = np.percentile(data, 50)
        _q3 = np.percentile(data, 75)
        _max = np.nanmax(data)
        return _mean, _std, _min, _q1, _q2, _q3, _max

    @property
    def is_rgba(self):
        """
        Whether or not this signal is an RGB + alpha channel `dtype`.
        """
        return rgb_tools.is_rgba(self.data)

    @property
    def is_rgb(self):
        """
        Whether or not this signal is an RGB `dtype`.
        """
        return rgb_tools.is_rgb(self.data)

    @property
    def is_rgbx(self):
        """
        Whether or not this signal is either an RGB or RGB + alpha channel
        `dtype`.
        """
        return rgb_tools.is_rgbx(self.data)

    def add_marker(
        self,
        marker,
        plot_on_signal=True,
        plot_marker=True,
        permanent=False,
        plot_signal=True,
        render_figure=True,
    ):
        """
        Add one or several markers to the signal or navigator plot and plot
        the signal, if not yet plotted (by default)

        Parameters
        ----------
        marker : :mod:`~hyperspy.api.plot.markers` object or iterable
            The marker or iterable (list, tuple, ...) of markers to add.
            See the :ref:`plot.markers` section in the User Guide if you want
            to add a large number of markers as an iterable, since this will
            be much faster. For signals with navigation dimensions,
            the markers can be made to change for different navigation
            indices. See the examples for info.
        plot_on_signal : bool, default True
            If ``True``, add the marker to the signal.
            If ``False``, add the marker to the navigator
        plot_marker : bool, default True
            If ``True``, plot the marker.
        permanent : bool, default True
            If ``False``, the marker will only appear in the current
            plot. If ``True``, the marker will be added to the
            `metadata.Markers` list, and be plotted with
            ``plot(plot_markers=True)``. If the signal is saved as a HyperSpy
            HDF5 file, the markers will be stored in the HDF5 signal and be
            restored when the file is loaded.

        Examples
        --------
        >>> im = hs.data.wave_image()
        >>> m = hs.plot.markers.Rectangles(
        ...    offsets=[(1.0, 1.5)], widths=(0.5,), heights=(0.7,)
        ... )
        >>> im.add_marker(m)

        Add permanent marker:

        >>> rng = np.random.default_rng(1)
        >>> s = hs.signals.Signal2D(rng.random((100, 100)))
        >>> marker = hs.plot.markers.Points(offsets=[(50, 60)])
        >>> s.add_marker(marker, permanent=True, plot_marker=True)

        Removing a permanent marker:

        >>> rng = np.random.default_rng(1)
        >>> s = hs.signals.Signal2D(rng.integers(10, size=(100, 100)))
        >>> marker = hs.plot.markers.Points(offsets=[(10, 60)])
        >>> marker.name = "point_marker"
        >>> s.add_marker(marker, permanent=True)
        >>> del s.metadata.Markers.point_marker

        Adding many markers as a list:

        >>> rng = np.random.default_rng(1)
        >>> s = hs.signals.Signal2D(rng.integers(10, size=(100, 100)))
        >>> marker_list = []
        >>> for i in range(10):
        ...     marker = hs.plot.markers.Points(rng.random(2))
        ...     marker_list.append(marker)
        >>> s.add_marker(marker_list, permanent=True)

        """
        if not plot_marker and not permanent:
            warnings.warn("`plot_marker=False` and `permanent=False` does nothing")
            return

        if isiterable(marker):
            marker_list = marker
        else:
            marker_list = [marker]
        markers_dict = {}

        if permanent:
            # Make a list of existing marker to check if this is not already
            # added to the metadata.
            if not self.metadata.has_item("Markers"):
                self.metadata.add_node("Markers")
            marker_object_list = []
            for marker_tuple in list(self.metadata.Markers):
                marker_object_list.append(marker_tuple[1])
            name_list = self.metadata.Markers.keys()

        marker_name_suffix = 1
        for m in marker_list:
            if (m._signal is not None) and (m._signal is not self):
                raise ValueError("Markers can not be added to several signals")

            m._plot_on_signal = plot_on_signal
            m._signal = self

            if plot_marker:
                if self._plot is None or not self._plot.is_active:
                    self.plot()
                if m._plot_on_signal:
                    self._plot.signal_plot.add_marker(m)
                else:
                    if self._plot.navigator_plot is None:
                        raise ValueError(
                            "Attempted to plot marker on navigator_plot when none is"
                            "active."
                        )
                    self._plot.navigator_plot.add_marker(m)
                m.plot(render_figure=False)

            if permanent:
                for marker_object in marker_object_list:
                    if m is marker_object:
                        raise ValueError("Marker already added to signal")
                name = m.name if m.name != "" else m.__class__.__name__
                temp_name = name
                # If it already exists in the list, add a suffix
                while temp_name in name_list:
                    temp_name = name + str(marker_name_suffix)
                    marker_name_suffix += 1
                m.name = temp_name
                markers_dict[m.name] = m
                marker_object_list.append(m)
                name_list.append(m.name)

        if permanent:
            self.metadata.Markers.add_dictionary(markers_dict)

        if plot_marker and render_figure:
            self._render_figure()

    def _render_figure(self, plot=["signal_plot", "navigation_plot"]):
        for p in plot:
            if hasattr(self._plot, p):
                p = getattr(self._plot, p)
                p.render_figure()

    def _plot_permanent_markers(self):
        marker_name_list = self.metadata.Markers.keys()
        markers_dict = self.metadata.Markers.__dict__
        for marker_name in marker_name_list:
            marker = markers_dict[marker_name]["_dtb_value_"]
            if marker._plot_on_signal:
                self._plot.signal_plot.add_marker(marker)
            else:
                self._plot.navigator_plot.add_marker(marker)
            marker._signal = self
            marker.plot(render_figure=False)
        self._render_figure()

    def add_poissonian_noise(self, keep_dtype=True, random_state=None):
        """Add Poissonian noise to the data.

        This method works in-place.

        Parameters
        ----------
        keep_dtype : bool, default True
            This parameter is deprecated and will be removed in HyperSpy 3.0.
            Currently, it does not have any effect. This method always
            keeps the original dtype of the signal.
        random_state : None, int or numpy.random.Generator, default None
            Seed for the random generator.

        Notes
        -----
        This method uses :func:`numpy.random.poisson`
        (or :func:`dask.array.random.poisson` for lazy signals)
        to generate the Poissonian noise.

        """
        kwargs = {}
        random_state = check_random_state(random_state, lazy=self._lazy)
        if not keep_dtype:
            warnings.warn(
                "The `keep_dtype` parameter is deprecated and will be removed in HyperSpy 3.0.",
                VisibleDeprecationWarning,
            )

        if self._lazy:
            kwargs["chunks"] = self.data.chunks

        self.data[:] = random_state.poisson(lam=self.data, **kwargs)
        self.events.data_changed.trigger(obj=self)
        self.events.data_changed.trigger(obj=self)

    def add_gaussian_noise(self, std, random_state=None):
        """Add Gaussian noise to the data.

        The operation is performed in-place (*i.e.* the data of the signal
        is modified). This method requires the signal to have a float data type,
        otherwise it will raise a :exc:`TypeError`.

        Parameters
        ----------
        std : float
            The standard deviation of the Gaussian noise.
        random_state : None, int or numpy.random.Generator, default None
            Seed for the random generator.

        Notes
        -----
        This method uses :func:`numpy.random.normal` (or
        :func:`dask.array.random.normal` for lazy signals)
        to generate the noise.

        """

        if self.data.dtype.char not in np.typecodes["AllFloat"]:
            raise TypeError(
                "`s.add_gaussian_noise()` requires the data to have "
                f"a float datatype, but the current type is '{self.data.dtype}'. "
                "To fix this issue, you can change the type using the "
                "change_dtype method (e.g. s.change_dtype('float64'))."
            )

        kwargs = {}
        random_state = check_random_state(random_state, lazy=self._lazy)

        if self._lazy:
            kwargs["chunks"] = self.data.chunks

        noise = random_state.normal(loc=0, scale=std, size=self.data.shape, **kwargs)

        self.data += noise
        self.events.data_changed.trigger(obj=self)

        self.events.data_changed.trigger(obj=self)

    def transpose(self, signal_axes=None, navigation_axes=None, optimize=False):
        """Transposes the signal to have the required signal and navigation
        axes.

        Parameters
        ----------
        signal_axes : None, int, or iterable type
            The number (or indices) of axes to convert to signal axes
        navigation_axes : None, int, or iterable type
            The number (or indices) of axes to convert to navigation axes
        %s

        Notes
        -----
        With the exception of both axes parameters (`signal_axes` and
        `navigation_axes` getting iterables, generally one has to be ``None``
        (i.e. "floating"). The other one specifies either the required number
        or explicitly the indices of axes to move to the corresponding space.
        If both are iterables, full control is given as long as all axes
        are assigned to one space only.

        See Also
        --------
        T, as_signal2D, as_signal1D

        Examples
        --------
        >>> # just create a signal with many distinct dimensions
        >>> s = hs.signals.BaseSignal(np.random.rand(1,2,3,4,5,6,7,8,9))
        >>> s
        <BaseSignal, title: , dimensions: (|9, 8, 7, 6, 5, 4, 3, 2, 1)>

        >>> s.transpose() # swap signal and navigation spaces
        <BaseSignal, title: , dimensions: (9, 8, 7, 6, 5, 4, 3, 2, 1|)>

        >>> s.T # a shortcut for no arguments
        <BaseSignal, title: , dimensions: (9, 8, 7, 6, 5, 4, 3, 2, 1|)>

        >>> # roll to leave 5 axes in navigation space
        >>> s.transpose(signal_axes=5)
        <BaseSignal, title: , dimensions: (4, 3, 2, 1|9, 8, 7, 6, 5)>

        >>> # roll leave 3 axes in navigation space
        >>> s.transpose(navigation_axes=3)
        <BaseSignal, title: , dimensions: (3, 2, 1|9, 8, 7, 6, 5, 4)>

        >>> # 3 explicitly defined axes in signal space
        >>> s.transpose(signal_axes=[0, 2, 6])
        <BaseSignal, title: , dimensions: (8, 6, 5, 4, 2, 1|9, 7, 3)>

        >>> # A mix of two lists, but specifying all axes explicitly
        >>> # The order of axes is preserved in both lists
        >>> s.transpose(navigation_axes=[1, 2, 3, 4, 5, 8], signal_axes=[0, 6, 7])
        <BaseSignal, title: , dimensions: (8, 7, 6, 5, 4, 1|9, 3, 2)>

        """

        if self.axes_manager.ragged:
            raise RuntimeError("Signal with ragged dimension can't be " "transposed.")

        am = self.axes_manager
        ax_list = am._axes
        if isinstance(signal_axes, int):
            if navigation_axes is not None:
                raise ValueError(
                    "The navigation_axes are not None, even "
                    "though just a number was given for "
                    "signal_axes"
                )
            if len(ax_list) < signal_axes:
                raise ValueError("Too many signal axes requested")
            if signal_axes < 0:
                raise ValueError("Can't have negative number of signal axes")
            elif signal_axes == 0:
                signal_axes = ()
                navigation_axes = ax_list[::-1]
            else:
                navigation_axes = ax_list[:-signal_axes][::-1]
                signal_axes = ax_list[-signal_axes:][::-1]
        elif iterable_not_string(signal_axes):
            signal_axes = tuple(am[ax] for ax in signal_axes)
            if navigation_axes is None:
                navigation_axes = tuple(ax for ax in ax_list if ax not in signal_axes)[
                    ::-1
                ]
            elif iterable_not_string(navigation_axes):
                # want to keep the order
                navigation_axes = tuple(am[ax] for ax in navigation_axes)
                intersection = set(signal_axes).intersection(navigation_axes)
                if len(intersection):
                    raise ValueError(
                        "At least one axis found in both spaces:" " {}".format(
                            intersection
                        )
                    )
                if len(am._axes) != (len(signal_axes) + len(navigation_axes)):
                    raise ValueError("Not all current axes were assigned to a " "space")
            else:
                raise ValueError(
                    "navigation_axes has to be None or an iterable"
                    " when signal_axes is iterable"
                )
        elif signal_axes is None:
            if isinstance(navigation_axes, int):
                if len(ax_list) < navigation_axes:
                    raise ValueError("Too many navigation axes requested")
                if navigation_axes < 0:
                    raise ValueError("Can't have negative number of navigation axes")
                elif navigation_axes == 0:
                    navigation_axes = ()
                    signal_axes = ax_list[::-1]
                else:
                    signal_axes = ax_list[navigation_axes:][::-1]
                    navigation_axes = ax_list[:navigation_axes][::-1]
            elif iterable_not_string(navigation_axes):
                navigation_axes = tuple(am[ax] for ax in navigation_axes)
                signal_axes = tuple(ax for ax in ax_list if ax not in navigation_axes)[
                    ::-1
                ]
            elif navigation_axes is None:
                signal_axes = am.navigation_axes
                navigation_axes = am.signal_axes
            else:
                raise ValueError("The passed navigation_axes argument is not valid")
        else:
            raise ValueError("The passed signal_axes argument is not valid")
        # translate to axes idx from actual objects for variance
        idx_sig = [ax.index_in_axes_manager for ax in signal_axes]
        idx_nav = [ax.index_in_axes_manager for ax in navigation_axes]
        # From now on we operate with axes in array order
        signal_axes = signal_axes[::-1]
        navigation_axes = navigation_axes[::-1]
        # get data view
        array_order = tuple(ax.index_in_array for ax in navigation_axes)
        array_order += tuple(ax.index_in_array for ax in signal_axes)
        newdata = self.data.transpose(array_order)
        res = self._deepcopy_with_new_data(
            newdata, copy_variance=True, copy_learning_results=True
        )

        # reconfigure the axes of the axesmanager:
        ram = res.axes_manager
        ram._update_trait_handlers(remove=True)
        # _axes are ordered in array order
        ram._axes = [ram._axes[i] for i in array_order]
        for i, ax in enumerate(ram._axes):
            if i < len(navigation_axes):
                ax.navigate = True
            else:
                ax.navigate = False
        ram._update_attributes()
        ram._update_trait_handlers(remove=False)
        res._assign_subclass()

        var = res.get_noise_variance()
        if isinstance(var, BaseSignal):
            var = var.transpose(
                signal_axes=idx_sig, navigation_axes=idx_nav, optimize=optimize
            )
            res.set_noise_variance(var)

        if optimize:
            res._make_sure_data_is_contiguous()
        if res.metadata.has_item("Markers"):
            # The markers might fail if the navigation dimensions are changed
            # so the safest is simply to not carry them over from the
            # previous signal.
            del res.metadata.Markers

        return res

    transpose.__doc__ %= OPTIMIZE_ARG

    @property
    def T(self):
        """The transpose of the signal, with signal and navigation spaces
        swapped. Enables calling
        :meth:`~hyperspy.api.signals.BaseSignal.transpose` with the default
        parameters as a property of a Signal.
        """
        return self.transpose()

    def apply_apodization(
        self, window="hann", hann_order=None, tukey_alpha=0.5, inplace=False
    ):
        """
        Apply an `apodization window
        <https://mathworld.wolfram.com/ApodizationFunction.html>`_ to a Signal.

        Parameters
        ----------
        window : str, optional
            Select between {``'hann'`` (default), ``'hamming'``, or ``'tukey'``}
        hann_order : None or int, optional
            Only used if ``window='hann'``
            If integer `n` is provided, a Hann window of `n`-th order will be
            used. If ``None``, a first order Hann window is used.
            Higher orders result in more homogeneous intensity distribution.
        tukey_alpha : float, optional
            Only used if ``window='tukey'`` (default is 0.5). From the
            documentation of
            :func:`scipy.signal.windows.tukey`:

                - Shape parameter of the Tukey window, representing the
                  fraction of the window inside the cosine tapered region. If
                  zero, the Tukey window is equivalent to a rectangular window.
                  If one, the Tukey window is equivalent to a Hann window.
        inplace : bool, optional
            If ``True``, the apodization is applied in place, *i.e.* the signal
            data  will be substituted by the apodized one (default is
            ``False``).

        Returns
        -------
        out : :class:`~hyperspy.signal.BaseSignal` (or subclass), optional
            If ``inplace=False``, returns the apodized signal of the same
            type as the provided Signal.

        Examples
        --------
        >>> import hyperspy.api as hs
        >>> wave = hs.data.wave_image()
        >>> wave.apply_apodization('tukey', tukey_alpha=0.1).plot()
        """

        if window == "hanning" or window == "hann":
            if hann_order:

                def window_function(m):
                    return hann_window_nth_order(m, hann_order)
            else:

                def window_function(m):
                    return np.hanning(m)
        elif window == "hamming":

            def window_function(m):
                return np.hamming(m)
        elif window == "tukey":

            def window_function(m):
                return sp_signal.windows.tukey(m, tukey_alpha)
        else:
            raise ValueError("Wrong type parameter value.")

        windows_1d = []

        axes = np.array(self.axes_manager.signal_indices_in_array)

        for axis, axis_index in zip(self.axes_manager.signal_axes, axes):
            if isinstance(self.data, da.Array):
                chunks = self.data.chunks[axis_index]
                window_da = da.from_array(window_function(axis.size), chunks=(chunks,))
                windows_1d.append(window_da)
            else:
                windows_1d.append(window_function(axis.size))

        window_nd = outer_nd(*windows_1d).T

        # Prepare slicing for multiplication window_nd nparray with data with
        # higher dimensionality:
        if inplace:
            slice_w = []

            # Iterate over all dimensions of the data
            for i in range(self.data.ndim):
                if any(
                    i == axes
                ):  # If current dimension represents one of signal axis, all elements in window
                    # along current axis to be subscribed
                    slice_w.append(slice(None))
                else:  # If current dimension is navigation one, new axis is absent in window and should be created
                    slice_w.append(None)

            self.data = self.data * window_nd[tuple(slice_w)]
            self.events.data_changed.trigger(obj=self)
        else:
            return self * window_nd

    def _check_navigation_mask(self, mask):
        """
        Check the shape of the navigation mask.

        Parameters
        ----------
        mask : numpy array or BaseSignal.
            Mask to check the shape.

        Raises
        ------
        ValueError
            If shape doesn't match the shape of the navigation dimension.

        Returns
        -------
        None.

        """
        if isinstance(mask, BaseSignal):
            if mask.axes_manager.signal_dimension != 0:
                raise ValueError(
                    "The navigation mask signal must have the "
                    "`signal_dimension` equal to 0."
                )
            elif (
                mask.axes_manager.navigation_shape != self.axes_manager.navigation_shape
            ):
                raise ValueError(
                    "The navigation mask signal must have the "
                    "same `navigation_shape` as the current "
                    "signal."
                )
        if isinstance(mask, np.ndarray) and (
            mask.shape != self.axes_manager.navigation_shape
        ):
            raise ValueError(
                "The shape of the navigation mask array must "
                "match `navigation_shape`."
            )

    def _check_signal_mask(self, mask):
        """
        Check the shape of the signal mask.

        Parameters
        ----------
        mask : numpy array or BaseSignal.
            Mask to check the shape.

        Raises
        ------
        ValueError
            If shape doesn't match the shape of the signal dimension.

        Returns
        -------
        None.

        """
        if isinstance(mask, BaseSignal):
            if mask.axes_manager.navigation_dimension != 0:
                raise ValueError(
                    "The signal mask signal must have the "
                    "`navigation_dimension` equal to 0."
                )
            elif mask.axes_manager.signal_shape != self.axes_manager.signal_shape:
                raise ValueError(
                    "The signal mask signal must have the same "
                    "`signal_shape` as the current signal."
                )
        if isinstance(mask, np.ndarray) and (
            mask.shape != self.axes_manager.signal_shape
        ):
            raise ValueError(
                "The shape of signal mask array must match " "`signal_shape`."
            )

    def to_device(self):
        """
        Transfer data array from host to GPU device memory using cupy.asarray.
        Lazy signals are not supported by this method, see user guide for
        information on how to process data lazily using the GPU.

        Raises
        ------
        BaseException
            Raise expection if cupy is not installed.
        BaseException
            Raise expection if signal is lazy.

        Returns
        -------
        None.

        """
        if self._lazy:
            raise LazyCupyConversion

        if not CUPY_INSTALLED:
            raise BaseException("cupy is required.")
        else:  # pragma: no cover
            self.data = cp.asarray(self.data)

    def to_host(self):
        """
        Transfer data array from GPU device to host memory.

        Raises
        ------
        BaseException
            Raise expection if signal is lazy.

        Returns
        -------
        None.

        """
        if self._lazy:  # pragma: no cover
            raise LazyCupyConversion
        self.data = to_numpy(self.data)


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
        ("def %s(self, other):\n" % name)
        + ("   return self._binary_operator_ruler(other, '%s')\n" % name)
    )
    exec("%s.__doc__ = np.ndarray.%s.__doc__" % (name, name))
    exec("setattr(BaseSignal, '%s', %s)" % (name, name))
    # The following commented line enables the operators with swapped
    # operands. They should be defined only for commutative operators
    # but for simplicity we don't support this at all atm.

    # exec("setattr(BaseSignal, \'%s\', %s)" % (name[:2] + "r" + name[2:],
    # name))

# Implement unary arithmetic operations
for name in UNARY_OPERATORS:
    exec(
        ("def %s(self):" % name) + ("   return self._unary_operator_ruler('%s')" % name)
    )
    exec("%s.__doc__ = int.%s.__doc__" % (name, name))
    exec("setattr(BaseSignal, '%s', %s)" % (name, name))
