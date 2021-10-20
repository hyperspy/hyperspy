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

import logging
from multiprocessing import cpu_count

import dill
import numpy as np

from hyperspy.misc.utils import DictionaryTreeBrowser
from hyperspy.misc.utils import slugify
from hyperspy.external.progressbar import progressbar
from hyperspy.signal import BaseSignal
from hyperspy.samfire_utils.strategy import (LocalStrategy,
                                             GlobalStrategy)
from hyperspy.samfire_utils.local_strategies import ReducedChiSquaredStrategy
from hyperspy.samfire_utils.global_strategies import HistogramStrategy


_logger = logging.getLogger(__name__)


class StrategyList(list):

    def __init__(self, samf):
        super(StrategyList, self).__init__()
        self.samf = samf

    def append(self, thing):
        thing.samf = self.samf
        list.append(self, thing)

    def extend(self, iterable):
        for thing in iterable:
            self.append(thing)

    def remove(self, thing):
        if isinstance(thing, int):
            thing = self[thing]
        thing.samf = None
        list.remove(self, thing)

    def __repr__(self):
        signature = u"%3s | %4s | %s"
        ans = signature % ("A", "#", "Strategy")
        ans += u"\n"
        ans += signature % (u'-' * 2, u'-' * 4, u'-' * 25)
        if self:
            for n, s in enumerate(self):
                ans += u"\n"
                name = repr(s)
                a = u" x" if self.samf._active_strategy_ind == n else u""
                ans += signature % (a, str(n), name)
        return ans


class Samfire:

    """Smart Adaptive Multidimensional Fitting (SAMFire) object

    SAMFire is a more robust way of fitting multidimensional datasets. By
    extracting starting values for each pixel from already fitted pixels,
    SAMFire stops the fitting algorithm from getting lost in the parameter
    space by always starting close to the optimal solution.

    SAMFire only picks starting parameters and the order the pixels (in the
    navigation space) are fitted, and does not provide any new minimisation
    algorithms.

    Attributes
    ----------

    model : Model instance
        The complete model
    optional_components : list
        A list of components that can be switched off at some pixels if it
        returns a better Akaike's Information Criterion with correction (AICc)
    workers : int
        A number of processes that will perform the fitting parallely
    pool : samfire_pool instance
        A proxy object that manages either multiprocessing or ipyparallel pool
    strategies : strategy list
        A list of strategies that will be used to select pixel fitting order
        and calculate required starting parameters. Strategies come in two
        "flavours" - local and global. Local strategies spread the starting
        values to the nearest pixels and forces certain pixel fitting order.
        Global strategies look for clusters in parameter values, and suggests
        most frequent values. Global strategy do not depend on pixel fitting
        order, hence it is randomised.
    metadata : dictionary
        A dictionary for important samfire parameters
    active_strategy : strategy
        The currently active strategy from the strategies list
    update_every : int
        If segmenter strategy is running, updates the historams every time
        update_every good fits are found.
    plot_every : int
        When running, samfire plots results every time plot_every good fits are
        found.
    save_every : int
        When running, samfire saves results every time save_every good fits are
        found.

    Methods
    -------

    start
        start SAMFire
    stop
        stop SAMFire
    plot
        force plot of currently selected active strategy
    refresh_database
        refresh current active strategy database. No previous structure is
        preserved
    backup
        backs up the current version of the model
    change_strategy
        changes strategy to a new one. Certain rules apply
    append
        appends strategy to the strategies list
    extend
        extends strategies list
    remove
        removes strategy from strategies list
    update
        updates the current model with values, received from a worker
    log
        if _log exists, logs the arguments to the list.
    generate_values
        creates a generator to calculate values to be sent to the workers
    """

    __active_strategy_ind = 0
    _progressbar = None
    pool = None
    _figure = None
    optional_components = []
    running_pixels = []
    plot_every = 0
    save_every = np.nan
    _workers = None
    _args = None
    count = 0

    def __init__(self, model, workers=None, setup=True, **kwargs):
        # constants:
        if workers is None:
            workers = max(1, cpu_count() - 1)
        self.model = model
        self.metadata = DictionaryTreeBrowser()

        self._scale = 1.0
        # -1 -> done pixel, use
        # -2 -> done, ignore when diffusion
        #  0 -> bad fit/no info
        # >0 -> select when turn comes

        self.metadata.add_node('marker')
        self.metadata.add_node('goodness_test')

        marker = np.empty(self.model.axes_manager.navigation_shape[::-1])
        marker.fill(self._scale)

        self.metadata.marker = marker
        self.strategies = StrategyList(self)
        self.strategies.append(ReducedChiSquaredStrategy())
        self.strategies.append(HistogramStrategy())
        self._active_strategy_ind = 0
        self.update_every = max(10, workers * 2)  # some sensible number....
        from hyperspy.samfire_utils.fit_tests import red_chisq_test
        self.metadata.goodness_test = red_chisq_test(tolerance=1.0)
        self.metadata._gt_dump = None
        from hyperspy.samfire_utils.samfire_kernel import single_kernel
        self.single_kernel = single_kernel
        self._workers = workers
        if len(kwargs) or setup:
            self._setup(**kwargs)
        self.refresh_database()

    @property
    def active_strategy(self):
        """Returns the active strategy"""
        return self.strategies[self._active_strategy_ind]

    @active_strategy.setter
    def active_strategy(self, value):
        self.change_strategy(value)

    def _setup(self, **kwargs):
        """Set up SAMFire - configure models, set up pool if necessary"""
        from hyperspy.samfire_utils.samfire_pool import SamfirePool
        self._figure = None
        self.metadata._gt_dump = dill.dumps(self.metadata.goodness_test)
        self._enable_optional_components()

        if hasattr(self.model, '_suspend_auto_fine_structure_width'):
            self.model._suspend_auto_fine_structure_width = True

        if hasattr(self, '_log'):
            self._log = []

        if self._workers and self.pool is None:
            if 'num_workers' not in kwargs:
                kwargs['num_workers'] = self._workers
            if self.pool is None:
                self.pool = SamfirePool(**kwargs)
            self._workers = self.pool.num_workers
            self.pool.prepare_workers(self)

    def start(self, **kwargs):
        """Starts SAMFire.

        Parameters
        ----------
        **kwargs : key-word arguments
            Any key-word arguments to be passed to Model.fit() call
        """
        self._setup()
        if self._workers and self.pool is not None:
            self.pool.update_parameters()
        if 'min_function' in kwargs:
            kwargs['min_function'] = dill.dumps(kwargs['min_function'])
        if 'min_function_grad' in kwargs:
            kwargs['min_function_grad'] = dill.dumps(
                kwargs['min_function_grad'])
        self._args = kwargs
        num_of_strat = len(self.strategies)
        total_size = self.model.axes_manager.navigation_size - self.pixels_done
        self._progressbar = progressbar(total=total_size)
        try:
            while True:
                self._run_active_strategy()
                self.plot()
                if self.pixels_done == self.model.axes_manager.navigation_size:
                    # all pixels are done, no need to go to the next strategy
                    break
                if self._active_strategy_ind == num_of_strat - 1:
                    # last one just finished running
                    break
                self.change_strategy(self._active_strategy_ind + 1)
        except KeyboardInterrupt:
            if self.pool is not None:
                _logger.warning(
                    'Collecting already started pixels, please wait')
                self.pool.collect_results()

    def append(self, strategy):
        """appends the given strategy to the end of the strategies list

        Parameters
        ----------
        strategy : strategy instance
        """
        self.strategies.append(strategy)

    def extend(self, iterable):
        """extend the strategies list by the given iterable

        Parameters
        ----------
        iterable : an iterable of strategy instances
        """
        self.strategies.extend(iterable)

    def remove(self, thing):
        """removes given strategy from the strategies list

        Parameters
        ----------
        thing : int or strategy instance
            Strategy that is in current strategies list or its index.
        """
        self.strategies.remove(thing)

    @property
    def _active_strategy_ind(self):
        return self.__active_strategy_ind

    @_active_strategy_ind.setter
    def _active_strategy_ind(self, value):
        self.__active_strategy_ind = np.abs(int(value))

    def _run_active_strategy(self):
        if self.pool is not None:
            self.count = 0
            self.pool.run()
        else:
            self._run_active_strategy_one()

    @property
    def pixels_left(self):
        """Returns the number of pixels that are left to solve. This number can
        increase as SAMFire learns more information about the data.
        """
        return np.sum(self.metadata.marker > 0.)

    @property
    def pixels_done(self):
        """Returns the number of pixels that have been solved"""
        return np.sum(self.metadata.marker <= -self._scale)

    def _run_active_strategy_one(self):
        self.count = 0
        while self.pixels_left:
            ind = self._next_pixels(1)[0]
            vals = self.active_strategy.values(ind)
            self.running_pixels.append(ind)
            isgood = self.single_kernel(self.model,
                                        ind,
                                        vals,
                                        self.optional_components,
                                        self._args,
                                        self.metadata.goodness_test)
            self.running_pixels.remove(ind)
            self.count += 1
            if isgood:
                self._progressbar.update(1)
            self.active_strategy.update(ind, isgood)
            self.plot(on_count=True)
            self.backup(on_count=True)

    def backup(self, filename=None, on_count=True):
        """Backs-up the samfire results in a file

        Parameters
        ----------
        filename: {str, None}
            the filename. If None, a default value of "backup_"+signal_title is
            used
        on_count: bool
            if True (default), only saves on the required count of steps
        """
        if filename is None:
            title = self.model.signal.metadata.General.title
            filename = slugify('backup_' + title)
        # maybe add saving marker + strategies as well?
        if self.count % self.save_every == 0 or not on_count:
            self.model.save(filename,
                            name='samfire_backup', overwrite=True)
            self.model.signal.models.remove('samfire_backup')

    def update(self, ind, results=None, isgood=None):
        """Updates the current model with the results, received from the
        workers. Results are only stored if the results are good enough

        Parameters
        ----------
        ind : tuple
            contains the index of the pixel of the results
        results : {dict, None}
            dictionary of the results. If None, means we are updating in-place
            (e.g. refreshing the marker or strategies)
        isgood : {bool, None}
            if it is known if the results are good according to the
            goodness-of-fit test. If None, the pixel is tested
        """
        if results is not None and (isgood is None or isgood):
            self._swap_dict_and_model(ind, results)

        if isgood is None:
            isgood = self.metadata.goodness_test.test(self.model, ind)
        self.count += 1
        if isgood and self._progressbar is not None:
            self._progressbar.update(1)

        self.active_strategy.update(ind, isgood)
        if not isgood and results is not None:
            self._swap_dict_and_model(ind, results)

    def refresh_database(self):
        """Refreshes currently selected strategy without preserving any
        "ignored" pixels
        """
        # updates current active strategy database / prob.
        # Assume when chisq is not None, it's relevant

        # TODO: if no calculated pixels, request user input

        calculated_pixels = np.logical_not(np.isnan(self.model.red_chisq.data))
        # only include pixels that are good enough
        calculated_pixels = self.metadata.goodness_test.map(
            self.model,
            calculated_pixels)

        self.active_strategy.refresh(True, calculated_pixels)

    def change_strategy(self, new_strat):
        """Changes current strategy to a new one. Certain rules apply:
        diffusion -> diffusion : resets all "ignored" pixels
        diffusion -> segmenter : saves already calculated pixels to be ignored
            when(if) subsequently diffusion strategy is run

        Parameters
        ----------
        new_strat : {int | strategy}
            index of the new strategy from the strategies list or the
            strategy object itself
        """
        from numbers import Number
        if not isinstance(new_strat, Number):
            try:
                new_strat = self.strategies.index(new_strat)
            except ValueError:
                raise ValueError(
                    "The passed object is not in current strategies list")

        new_strat = np.abs(int(new_strat))
        if new_strat == self._active_strategy_ind:
            self.refresh_database()

        # copy previous "done" pixels to the new one, delete old database

        # TODO: make sure it's a number. Get index if object is passed?
        if new_strat >= len(self.strategies):
            raise ValueError('too big new strategy index')

        current = self.active_strategy
        new = self.strategies[new_strat]

        if isinstance(current, LocalStrategy) and isinstance(
                new, LocalStrategy):
            # forget ignore/done levels, keep just calculated or not
            new.refresh(True)
        else:
            if isinstance(current, LocalStrategy) and isinstance(
                    new, GlobalStrategy):
                # if diffusion->segmenter, set previous -1 to -2 (ignored for
                # the next diffusion)
                self.metadata.marker[
                    self.metadata.marker == -
                    self._scale] -= self._scale

            new.refresh(False)
        current.clean()
        if current.close_plot is not None:
            current.close_plot()
        self._active_strategy_ind = new_strat

    def generate_values(self, need_inds):
        """Returns an iterator that yields the index of the pixel and the
        value dictionary to be sent to the workers.

        Parameters
        ----------
        need_inds: int
            the number of pixels to be returned in the generator
        """
        if need_inds:
            # get pixel index
            for ind in self._next_pixels(need_inds):
                # get starting parameters / array of possible values
                value_dict = self.active_strategy.values(ind)
                value_dict['fitting_kwargs'] = self._args
                value_dict['signal.data'] = \
                    self.model.signal.data[ind + (...,)]
                if self.model.signal._lazy:
                    value_dict['signal.data'] = value_dict[
                        'signal.data'].compute()
                if self.model.signal.metadata.has_item(
                        'Signal.Noise_properties.variance'):
                    var = self.model.signal.metadata.Signal.Noise_properties.variance
                    if isinstance(var, BaseSignal):
                        dat = var.data[ind + (...,)]
                        value_dict['variance.data'] = dat.compute(
                        ) if var._lazy else dat
                if hasattr(self.model,
                           'low_loss') and self.model.low_loss is not None:
                    dat = self.model.low_loss.data[ind + (...,)]
                    value_dict['low_loss.data'] = dat.compute(
                    ) if self.model.low_loss._lazy else dat

                self.running_pixels.append(ind)
                self.metadata.marker[ind] = 0.
                yield ind, value_dict

    def _next_pixels(self, number):
        best = self.metadata.marker.max()
        inds = []
        if best > 0.0:
            ind_list = np.where(self.metadata.marker == best)
            while number and ind_list[0].size > 0:
                i = np.random.randint(len(ind_list[0]))
                ind = tuple([lst[i] for lst in ind_list])
                if ind not in self.running_pixels:
                    inds.append(ind)
                # removing the added indices
                ind_list = [np.delete(lst, i, 0) for lst in ind_list]
                number -= 1
        return inds

    def _swap_dict_and_model(self, m_ind, dict_, d_ind=None):
        if d_ind is None:
            d_ind = tuple([0 for _ in dict_['dof.data'].shape])
        m = self.model
        for k in dict_.keys():
            if k.endswith('.data'):
                item = k[:-5]
                getattr(m, item).data[m_ind], dict_[k] = \
                    dict_[k].copy(), getattr(m, item).data[m_ind].copy()
        for comp_name, comp in dict_['components'].items():
            # only active components are sent
            if self.model[comp_name].active_is_multidimensional:
                self.model[comp_name]._active_array[m_ind] = True
            self.model[comp_name].active = True

            for param_model in self.model[comp_name].parameters:
                param_dict = comp[param_model.name]
                param_model.map[m_ind], param_dict[d_ind] = \
                    param_dict[d_ind].copy(), param_model.map[m_ind].copy()

        for component in self.model:
            # switch off all that did not appear in the dictionary
            if component.name not in dict_['components'].keys():
                if component.active_is_multidimensional:
                    component._active_array[m_ind] = False

    def _enable_optional_components(self):
        if len(self.optional_components) == 0:
            return
        for c in self.optional_components:
            comp = self.model._get_component(c)
            if not comp.active_is_multidimensional:
                comp.active_is_multidimensional = True
        if not np.all([isinstance(a, int) for a in
                       self.optional_components]):
            new_list = []
            for op in self.optional_components:
                for ic, c in enumerate(self.model):
                    if c is self.model._get_component(op):
                        new_list.append(ic)
            self.optional_components = new_list

    def _request_user_input(self):
        from hyperspy.signals import Image
        from hyperspy.drawing.widgets import SquareWidget
        mark = Image(self.metadata.marker,
                     axes=self.model.axes_manager._get_navigation_axes_dicts())
        mark.metadata.General.title = 'SAMFire marker'

        def update_when_triggered():
            ind = self.model.axes_manager.indices[::-1]
            isgood = self.metadata.goodness_test.test(self.model, ind)
            self.active_strategy.update(ind, isgood, 0)
            mark.events.data_changed.trigger(mark)

        self.model.plot()
        self.model.events.fitted.connect(update_when_triggered, [])
        self.model._plot.signal_plot.events.closed.connect(
            lambda: self.model.events.fitted.disconnect(update_when_triggered),
            [])

        mark.plot(navigator='slider')

        w = SquareWidget(self.model.axes_manager)
        w.color = 'yellow'
        w.set_mpl_ax(mark._plot.signal_plot.ax)
        w.connect_navigate()

        def connect_other_navigation1(axes_manager):
            with mark.axes_manager.events.indices_changed.suppress_callback(
                    connect_other_navigation2):
                for ax1, ax2 in zip(mark.axes_manager.navigation_axes,
                                    axes_manager.navigation_axes[2:]):
                    ax1.value = ax2.value

        def connect_other_navigation2(axes_manager):
            with self.model.axes_manager.events.indices_changed.suppress_callback(
                    connect_other_navigation1):
                for ax1, ax2 in zip(self.model.axes_manager.navigation_axes[2:],
                                    axes_manager.navigation_axes):
                    ax1.value = ax2.value

        mark.axes_manager.events.indices_changed.connect(
            connect_other_navigation2, {'obj': 'axes_manager'})
        self.model.axes_manager.events.indices_changed.connect(
            connect_other_navigation1, {'obj': 'axes_manager'})

        self.model._plot.signal_plot.events.closed.connect(
            lambda: mark._plot.close, [])
        self.model._plot.signal_plot.events.closed.connect(
            lambda: self.model.axes_manager.events.indices_changed.disconnect(
                connect_other_navigation1), [])

    def plot(self, on_count=False):
        """(if possible) plots current strategy plot. Local strategies plot
        grayscale navigation signal with brightness representing order of the
        pixel selection. Global strategies plot a collection of histograms,
        one per parameter.

        Parameters
        ----------
        on_count : bool
            if True, only tries to plot every speficied count, otherwise
            (default) always plots if possible.
        """
        count_test = self.plot_every and (self.count % self.plot_every == 0)
        if not on_count or count_test:
            if self.strategies:
                try:
                    self._figure = self.active_strategy.plot(self._figure)
                except BaseException:
                    self._figure = None
                    self._figure = self.active_strategy.plot(self._figure)

    def log(self, *args):
        """If has a list named "_log", appends the arguments there
        """
        if hasattr(self, '_log') and isinstance(self._log, list):
            self._log.append(args)

    def __repr__(self):
        ans = u"<SAMFire of the signal titled: '"
        ans += self.model.signal.metadata.General.title
        ans += u"'>"
        return ans
