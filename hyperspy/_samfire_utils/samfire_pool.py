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


from multiprocessing import (cpu_count, Pool, Queue)
from multiprocessing.pool import Pool as Pool_type
import ipyparallel as ipp
import os
import logging

from hyperspy._samfire_utils.samfire_worker import create_worker

import time

_logger = logging.getLogger(__name__)


class samfire_pool:

    def __init__(self, num_workers=None, ipython_kwargs=None):
        if ipython_kwargs is None:
            ipython_kwargs = {}
        self.ipython_kwargs = {'timeout': 5.}
        self.ipython_kwargs.update(ipython_kwargs)
        self.pool = None
        self.samf = None
        self.ping = {}
        self.pid = {}
        self.num_workers = num_workers
        self.workers = {}
        self.timestep = 0.05
        self.timeout = 15.
        self.setup()

    @property
    def is_ipyparallel(self):
        return hasattr(self.pool, 'client')

    @property
    def is_multiprocessing(self):
        return isinstance(self.pool, Pool_type)

    @property
    def has_pool(self):
        return self.is_ipyparallel or self.is_multiprocessing and \
            self.pool._state is 0

    def setup(self):
        if not self.has_pool:
            try:
                ipyclient = ipp.Client(**self.ipython_kwargs)
                self.num_workers = min(self.num_workers, len(ipyclient))
                self.pool = ipyclient.load_balanced_view(
                    range(self.num_workers))
                self.results = []
            except OSError:
                self.num_workers = min(self.num_workers, cpu_count() - 1)
                self.pool(processes=self.num_workers)

    def prepare_workers(self, samfire):
        self.samf = samfire

        mall = samfire.model
        m = mall.inav[mall.axes_manager.indices]
        m.store('z')
        model_dict = m.spectrum._to_dictionary(False)
        model_dict['models'] = m.spectrum.models._models.as_dictionary()

        optional_names = {mall[c].name for c in samfire.optional_components}

        if self.is_ipyparallel:
            dv = self.pool.client[:self.num_workers]
            dv.block = True
            dv.execute("from hyperspy._samfire_utils.samfire_worker import "
                       "create_worker")
            dv.scatter('identity', range(self.num_workers), flatten=True)
            dv.execute('worker = create_worker(identity)')
            self.rworker = ipp.Reference('worker')
            dv.apply(lambda worker, m_dict: worker.create_model(m_dict, 'z'),
                     self.rworker, model_dict)
            dv.apply(lambda worker, ts: worker.setup_test(ts), self.rworker,
                     samfire._gt_dump)
            dv.apply(lambda worker, on: worker.set_optional_names(on),
                     self.rworker, optional_names)

            # self.pid = dv.apply_async(os.getpid).get_dict()

        if self.is_multiprocessing:
            self.shared_queue = Queue()
            self.result_queue = Queue()
            for i in range(self.num_workers):
                this_queue = Queue()
                self.workers[i] = this_queue
                self.pool.apply_async(create_worker, args=(i, this_queue,
                                                           self.shared_queue,
                                                           self.result_queue))
                this_queue.put(('setup_test', (samfire._gt_dump,)))
                this_queue.put(('create_model', (m_dict, 'z')))
                this_queue.put(('set_optional_names', (optional_names,)))

    def ping_workers(self, timeout=None):
        if self.samf is None:
            _logger.error('Have to add samfire to the pool first')
        else:
            if self.is_multiprocessing:
                for _id, this_queue in self.workers.items():
                    this_queue.put('ping')
                    self.ping[_id] = time.time()
            elif self.is_ipyparallel:
                for i in range(self.num_workers):
                    dv = self.pool.client[i]
                    self.results.append((dv.apply_async(lambda worker:
                                                        worker.ping(),
                                                        self.rworker),
                                         i))
                    self.ping[i] = time.time()
        time.sleep(0.5)
        self.collect_results(timeout)

    def __len__(self):
        if self.is_ipyparallel:
            return self.pool.client.queue_status()['unassigned']
        elif self.is_multiprocessing:
            return self.shared_queue.qlen()

    def add_jobs(self, needed_number=None):
        if needed_number is None:
            needed_number = self.need_pixels
        for ind, value_dict in self.samf._add_jobs(needed_number):
            if self.is_multiprocessing:
                self.shared_queue.put(('test', (ind, value_dict)))
            elif self.is_ipyparallel:
                def test_func(worker, ind, value_dict):
                    return worker.test(ind, value_dict)
                self.results.append((self.pool.apply_async(test_func,
                                                           self.rworker, ind,
                                                           value_dict), ind))

    def parse(self, value):
        keyword, the_rest = value
        samf = self.samf
        if keyword == 'pong':
            _id, pid, pong_time, message = the_rest
            self.ping[_id] = pong_time - self.ping[_id]
            self.pid[_id] = pid
            _logger.info('pong worker {} with time {} and message'
                         '"{}"'.format(_id, self.ping[_id], message))
        elif keyword == 'Error':
            _id, err_message = the_rest
            _logger.error('Error in worker {}\n{}'.format(_id, err_message))
        elif keyword == 'result':
            _id, ind, result, found = the_rest
            if ind in samf._running_pixels:
                samf._running_pixels.remove(ind)
                samf._update(ind, result, isgood)
                samf._plot()
                samf._save()
                if hasattr(samf, '_log') and isinstance(self._samf, list):
                    samf._log.append((ind, isgood, samf.count, _id))
        else:
            _logger.error('Unusual return from some worker. The value '
                          'is:\n{}'.format(value))

    def collect_results(self, timeout=None):
        if timeout is None:
            timeout = self.timeout
        found_something = False
        if self.is_ipyparallel:
            for res, ind in reversed(self.results):
                if res.ready():
                    try:
                        result = res.get(timeout=self.timeout)
                    except TimeoutError:
                        _logger.info('Ind {} failed to come back in {} '
                                     'seconds. Assuming failed'.format(
                                         ind, timeout))
                        result = ('result', (-1, ind, None, False))
                    self.parse(result)
                    self.results.remove((res, ind))
                    found_something = True
                else:
                    pass
        elif self.is_multiprocessing:
            while not self.result_queue.empty():
                try:
                    result = self.result_queue.get(block=True,
                                                   timeout=timeout)
                    self.parse(result)
                    found_something = True
                except TimeoutError:
                    _logger.info('Some ind failed to come back in {} '
                                 'seconds.'.format(self.timeout))

        return found_something

    @property
    def need_pixels(self):
        return min(self.samf.pixels_done * self.samf.metadata.marker.ndim,
                   self.num_workers - len(self))

    def run(self):
        while self.samf.pixels_left or self.samf.running_pixels:
            # bool if got something
            new_result = self.collect_results()
            need_number = self.need_pixels
            if need_number:
                self.add_jobs(need_number)
            if not need_number or not new_result:
                # did not spend much time, since no new results or added pixels
                self.sleep()

    def sleep(self, howlong=None):
        if howlong is None:
            howlong = self.timestep
        time.sleep(howlong)

    def stop(self):
        if self.is_multiprocessing:
            self.pool.close()
            # self.pool.terminate()
        elif self.is_ipyparallel:
            self.pool.client.clear()
