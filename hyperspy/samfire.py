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

from multiprocessing import cpu_count, Pool, Manager
from itertools import product

import numpy as np
import dill
import time

from hyperspy.misc.utils import DictionaryTreeBrowser
from hyperspy.misc.utils import slugify
from hyperspy._samfire_utils.strategy import diffusion_strategy, segmenter_strategy
from hyperspy._samfire_utils._strategies.diffusion.red_chisq import reduced_chi_squared_strategy
from hyperspy._samfire_utils._strategies.segmenter.histogram import histogram_strategy


class Samfire(object):

    """Smart Adaptive Multidimensional Fitting (SAMFire) object

    SAMFire is a more robust way of fitting multidimensional datasets.
    By extracting starting values for each pixel from already fitted pixels,
    SAMFire stops the fitting algorithm from getting lost in the parameter space
    by always starting close to the optimal solution.

    SAMFire only picks starting parameters and the order the pixels
    (in the navigation space) are fitted, and does not provide any new
    minimisation algorithms.

    Attributes
    ----------

    model : Model instance
        The complete model
    optional_components : list
        A list of components that can be switched off at some pixels if it returns a better
        Akaike's Information Criterion with correction (AICc)
    workers : int
        A number of processes that will perform the fitting parallely
    strategies : strategy list
        A list of strategies that will be used to select pixel fitting order and calculate required starting
        parameters. Strategies come in two "flavours" - diffusion and segmenter. Diffusion spreads the starting
        values to the nearest pixels and forces certain pixel fitting order. Segmenter looks for clusters in
        parameter values, and suggests most frequent values. Segmenter strategy does not depend on pixel
        fitting order, hence it is randomised.
    metadata : dictionary
        A dictionary for important samfire parameters
    active_strategy : int
        Index of the currently active strategy from the strategies list
    update_every : int
        If segmenter strategy is running, updates the historams every time update_every good fits are found.
    plot_every : int
        When running, samfire plots results every time plot_every good fits are found.
    save_every : int
        When running, samfire saves results every time save_every good fits are found.
    kernel : function
        The function that performs fitting and model selection in parallely-run cores when running.

    Methods
    -------

    start
        start SAMFire
    stop
        stop SAMFire
    plot
        force plot of currently selected active strategy
    refresh_database
        refresh current active strategy database. No previous structure is preserved
    change_strategy
        changes strategy to a new one. Certain rules apply
    append
        appends strategy to the strategies list
    extend
        extends strategies list
    remove
        removes strategy from strategies list

    """

    _workers = 0
    _active_strategy = 0

    class _strategy_list(list):

        def __init__(self, samf):
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
                    a = "->" if self.samf.active_strategy is n else ""
                    ans += signature % (a, str(n), name)
            return ans.encode('utf8')

    def __init__(self, model, marker=None, workers=None):
        # constants:
        if workers is None:
            workers = cpu_count() - 1

        self.workers = workers
        self.optional_components = []
        self._running_pixels = []
        self.pool = None
        self.model = model
        self.metadata = DictionaryTreeBrowser()
        self._figure = None

        self._scale = 1.0
        # -1 -> done pixel, use
        # -2 -> done, ignore when diffusion
        #  0 -> bad fit/no info
        # >0 -> select when turn comes

        self.metadata.add_node('marker')
        self.metadata.add_node('goodness_test')

        if marker is None:
            marker = np.empty(
                self.model.axes_manager.navigation_shape[::-1])
            marker.fill(self._scale)
        self.metadata.marker = marker
        self.strategies = Samfire._strategy_list(self)
        self.strategies.append(reduced_chi_squared_strategy())
        self.strategies.append(histogram_strategy())
        self.active_strategy = 0
        self._result_q = None
        self.update_every = max(
            10,
            self.workers *
            2)  # some sensible number....
        self.plot_every = self.update_every
        self.save_every = np.nan
        from hyperspy._samfire_utils.fit_tests import red_chisq_test
        self.metadata.goodness_test = red_chisq_test(tolerance=1.0)
        self._gt_dump = None
        from hyperspy._samfire_utils.samfire_kernel import multi_kernel, single_kernel
        self.multi_kernel = multi_kernel
        self.single_kernel = single_kernel
        self._max_time_in_seconds = 60
        self.refresh_database()

    def _setup(self):

        self._gt_dump = dill.dumps(self.metadata.goodness_test)
        self._enable_optional_components()

        if hasattr(self.model, '_suspend_auto_fine_structure_width'):
            self.model._suspend_auto_fine_structure_width = True

        if self.workers:
            if self._result_q is None:
                m = Manager()
                self._result_q = m.Queue()

            if self.pool is None:
                self.pool = Pool(processes=self.workers)

            # in case the pool was not "deleted" when terminating:
            if self.pool._state is not 0:
                self.pool.terminate()
                self.pool = Pool(processes=self.workers)

    def start(self, **kwargs):
        """Starts SAMFire.

        Parameters
        ----------
        **kwargs : key-word arguments
            Any key-word arguments to be passed to Model.fit() call
        """
        self._args = kwargs
        self._setup()
        num_of_strat = len(self.strategies)

        while True:
            self._run_active_strategy()
            self._plot(0)
            if self.active_strategy == num_of_strat - 1:
                # last one just finished running
                break

            self.change_strategy(self.active_strategy + 1)
        self.stop()

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
    def workers(self):
        return self._workers

    @workers.setter
    def workers(self, value):
        self._workers = np.abs(int(value))

    @property
    def active_strategy(self):
        return self._active_strategy

    @active_strategy.setter
    def active_strategy(self, value):
        self._active_strategy = np.abs(int(value))

    def _run_active_strategy(self):
        if self.workers:
            self._run_active_strategy_multi()
        else:
            self._run_active_strategy_one()

    def _run_active_strategy_multi(self):
        count = 0
        last_time = time.time()
        while np.any(self.metadata.marker > 0.) or len(self._running_pixels) > 0:
            if self._result_q.empty():
                if len(self._running_pixels) < self.workers:
                    self._add_jobs()
                if time.time() - last_time > self._max_time_in_seconds:
                    # print self._running_pixels
                    # print 'broke'
                    break
            else:
                count += 1
                (ind, results, isgood) = self._result_q.get()
                self._running_pixels.remove(ind)
                self._update(ind, count, results, isgood)

                self._plot(count)
                self._save(count)
                last_time = time.time()

    def _run_active_strategy_one(self):
        count = 0
        while np.any(self.metadata.marker > 0.):
            ind = self._next_pixels(1)[0]
            vals = self.strategies[self.active_strategy].values(ind)
            self._running_pixels.append(ind)
            isgood = self.single_kernel(self.model,
                                        ind,
                                        vals,
                                        self.optional_components,
                                        self._args,
                                        self.metadata.goodness_test)
            self._running_pixels.remove(ind)
            count += 1
            self.strategies[self.active_strategy].update(ind, isgood, count)
            self._plot(count)
            self._save(count)

    def _save(self, count):
        # maybe add saving marker + strategies as well?
        if count % self.save_every == 0:
            self.model.stash.save('samfire_backup')
            self.model.spectrum.save(slugify('backup_' + self.model.spectrum.metadata.General.title),
                                     overwrite=True)
            self.model.stash.remove('samfire_backup')

    def _update(self, ind, count, results=None, isgood=None):
        if results is not None and (isgood is None or isgood):
            self._swap_dict_and_model(ind, results)

        if isgood is None:
            isgood = self.metadata.goodness_test.test(self.model, ind)

        self.strategies[self.active_strategy].update(ind, isgood, count)
        if not isgood and results is not None:
            self._swap_dict_and_model(ind, results)

    def refresh_database(self):
        """Refreshes currently selected strategy without preserving any "ignored" pixels
        """
        # updates current active strategy database / prob.
        # Assume when chisq is not None, it's relevant

        # TODO: if no calculated pixels, request user input

        calculated_pixels = np.logical_not(np.isnan(self.model.red_chisq.data))
        # only include pixels that are good enough
        calculated_pixels = self.metadata.goodness_test.map(
            self.model,
            calculated_pixels)

        self.strategies[self.active_strategy].refresh(True, calculated_pixels)

    def change_strategy(self, new_strat):
        """Changes current strategy to a new one. Certain rules apply:
        diffusion -> diffusion : resets all "ignored" pixels
        diffusion -> segmenter : saves already calculated pixels to be ignored when(if) subsequently diffusion
                                 strategy is run

        Parameters
        ----------
            new_strat : int
                index of the new strategy from the strategies list
        """

        new_strat = np.abs(int(new_strat))
        if new_strat is self.active_strategy:
            self.refresh_database()

        # copy previous "done" pixels to the new one, delete old database

        # TODO: make sure it's a number. Get index if object is passed?
        if new_strat >= len(self.strategies):
            raise ValueError('too big new strategy index')

        current = self.strategies[self.active_strategy]
        new = self.strategies[new_strat]

        if isinstance(current, diffusion_strategy) and isinstance(new, diffusion_strategy):
            # forget ignore/done levels, keep just calculated or not
            new.refresh(True)
        else:
            if isinstance(current, diffusion_strategy) and isinstance(new, segmenter_strategy):
                # if diffusion->segmenter, set previous -1 to -2 (ignored for
                # the next diffusion)
                self.metadata.marker[
                    self.metadata.marker == -
                    self._scale] -= self._scale

            new.refresh(False)
        current.clean()
        if current.close_plot is not None:
            current.close_plot()
        self.active_strategy = new_strat

    def _add_jobs(self):
        # check that really need more jobs
        need_inds = self.workers - len(self._running_pixels)
        if need_inds:
            # get pixel index
            inds = self._next_pixels(need_inds)
            for ind in inds:
                # get starting parameters / array of possible values
                vals = self.strategies[self.active_strategy].values(ind)
                m = self.model.inav[ind[::-1]]
                m.stash.save('z')
                m_dict = m.spectrum._to_dictionary(False)
                self._dispatch_worker(ind, m_dict, vals)
                self._running_pixels.append(ind)
                self.metadata.marker[ind] = 0.

    def _next_pixels(self, number):
        best = self.metadata.marker.max()
        inds = []
        if best > 0.0:
            ind_list = np.where(self.metadata.marker == best)
            while number and ind_list[0].size > 0:
                i = np.random.randint(len(ind_list[0]))
                ind = tuple([lst[i] for lst in ind_list])
                if ind not in self._running_pixels:
                    inds.append(ind)
                # removing the added indices
                ind_list = [np.delete(lst, i, 0) for lst in ind_list]
                number -= 1
        return inds

    def _dispatch_worker(self, ind, m_dict, vals):
        run_args = (ind,
                    m_dict,
                    vals,
                    self.optional_components,
                    self._args,
                    self._result_q,
                    self._gt_dump
                    )
        # self.multi_kernel(*run_args)
        self.pool.apply_async(self.multi_kernel,
                              args=run_args)

    def _swap_dict_and_model(self, m_ind, dic, d_ind=None):
        if d_ind is None:
            d_ind = tuple([0 for _ in dic['chisq.data'].shape])
        self.model.chisq.data[m_ind], dic['chisq.data'] = dic[
            'chisq.data'].copy(), self.model.chisq.data[m_ind].copy()
        self.model.dof.data[m_ind], dic['dof.data'] = dic[
            'dof.data'].copy(), self.model.dof.data[m_ind].copy()

        for num, comp in enumerate(dic['components']):
            if self.model[num].active_is_multidimensional:
                self.model[num]._active_array[m_ind] = comp['active']
            self.model[num].active = comp['active']
            for (p_d, p_m) in product(comp['parameters'], self.model[num].parameters):
                if p_d['_id_name'] == p_m._id_name:
                    p_m.map[m_ind], p_d['map'][d_ind] = p_d[
                        'map'][d_ind].copy(), p_m.map[m_ind].copy()

    def stop(self):
        if self.workers:
            self.pool.terminate()
            del self.pool
            self.pool = None

    def _enable_optional_components(self):
        if len(self.optional_components) == 0:
            return
        else:
            for c in self.optional_components:
                comp = self.model._get_component(c)
                if not comp.active_is_multidimensional:
                    comp.active_is_multidimensional = True
            if not np.all([isinstance(a, int) for a in self.optional_components]):
                new_list = []
                for op in self.optional_components:
                    for ic, c in enumerate(self.model):
                        if c is self.model._get_component(op):
                            new_list.append(ic)
                self.optional_components = new_list

# TODO: request_user_input:
# requires instances to call back when the figure is closed, ... wait
# until implemented in hyperspy

#     def _request_user_input(self):
#         from hyperspy.signal import Signal
#         from hyperspy.drawing.utils import plot_signals
#         mark = Signal(self.metadata.marker)
#         for ax in mark.axes_manager._axes:
#             ax.navigate = True
#         mark.metadata.General.title = 'SAMFire marker'
#         plot_signals([self.model, mark])

    def _plot(self, count):
        if count % self.plot_every == 0:
            self.plot()

    def plot(self):
        """(if possible) plots current strategy plot.
        Diffusion strategies plot grayscale navigation signal with brightness representing order of the pixel
        selection.
        Segmenter strategies plot a collection of histograms, one per parameter
        """
        if self.strategies:
            self._figure = self.strategies[
                self.active_strategy].plot(
                self._figure)

    def __repr__(self):
        ans = u"<SAMFire of the signal titled: '"
        ans += self.model.spectrum.metadata.General.title
        ans += u"'>"
        return ans.encode('utf8')
