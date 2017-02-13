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


import time
import logging
from multiprocessing import Manager
import numpy as np
from dask.array import Array as dar

from hyperspy.utils.parallel_pool import ParallelPool
from hyperspy.samfire_utils.samfire_worker import create_worker

_logger = logging.getLogger(__name__)

def _walk_compute(athing):
    if isinstance(athing, dict):
        this = {}
        for key, val in athing.items():
            if isinstance(key, dar):
                raise ValueError('Dask arrays should not be used as keys')
            value = _walk_compute(val)
            this[key] = value
        return this
    elif isinstance(athing, list):
        return [_walk_compute(val) for val in athing]
    elif isinstance(athing, tuple):
        return tuple(_walk_compute(val) for val in athing)
    elif isinstance(athing, dar):
        _logger.debug('found a dask array!')
        return athing.compute()
    else:
        return athing


class SamfirePool(ParallelPool):
    """ Creates and manages a pool of SAMFire workers. For based on
    ParallelPool - either creates processes using multiprocessing, or connects
    and sets up ipyparallel load_balanced_view.

    Ipyparallel is managed directly, but multiprocessing pool is managed via
    three of Queues:
        - Shared by all (master and workers) for distributing "load-balanced"
        work.
        - Shared by all (master and workers) for sending results back to the
        master
        - Individual queues from master to each worker. For setting up and
        addressing individual workers in general. This one is checked with
        higher priority in workers.

    Methods
    -------

    prepare_workers
        given SAMFire object, populates the workers with the required
        information. In case of multiprocessing, starts worker listening to the
        queues.
    update_parameters
        updates various worker parameters
    ping_workers
        pings all workers. Stores the one-way trip time and the process_id
        (pid) of each worker if available
    add_jobs
        adds the requested number of jobs to the queue
    parse
        parses the messages, returned from the workers.
    colect_results
        collects all currently available results and parses them
    run
        runs the full procedure until no more pixels are left to run in the
        SAMFire
    stop
        stops the pool, (for ipyparallel) clears the memory
    setup
        sets up the ipyparallel or multiprocessing pool (collects to the
        client or creates the pool)
    sleep
        sleeps for the specified time, by default timestep

    Attributes
    ----------

    has_pool: Bool
        Boolean if the pool is available and active
    pool: {ipyparallel.load_balanced_view, multiprocessing.Pool}
        The pool object
    ipython_kwargs: dict
        The dictionary with Ipyparallel connection arguments.
    timeout: float
        Timeout for either pool when waiting for results
    num_workers: int
        The number of workers actually created (may be less than requested, but
        can't be more)
    timestep: float
        The timestep between "ticks" that the result queues are checked. Higher
        timestep means less frequent checking, which may reduce CPU load for
        difficult fits that take a long time to finish.
    ping: dict
        If recorded, stores one-way trip time of each worker
    pid: dict
        If available, stores the process-id of each worker
    """

    def __init__(self, **kwargs):
        """Creates a ParallelPool with additional methods for SAMFire. All
        arguments are passed to ParallelPool"""
        super(SamfirePool, self).__init__(**kwargs)
        self.samf = None
        self.ping = {}
        self.pid = {}
        self.workers = {}
        self.rworker = None
        self.result_queue = None
        self.shared_queue = None
        self._last_time = 0
        self.results = []

    def _timestep_set(self, value):
        value = np.abs(value)
        self._timestep = value
        if self.has_pool and self.is_multiprocessing:
            for this_queue in self.workers.values():
                this_queue.put(('change_timestep', (value,)))

    def prepare_workers(self, samfire):
        """Prepares the workers for work, in case of multiprocessing starts
        listening

        Parameters
        ----------
        samfire : samfire
            the SAMFire object that will be using the pool
        """
        _logger.debug('starting prepare_workers')
        self.samf = samfire
        mall = samfire.model
        model = mall.inav[mall.axes_manager.indices]
        model.store('z')
        m_dict = model.signal._to_dictionary(False)
        m_dict['models'] = model.signal.models._models.as_dictionary()

        m_dict = _walk_compute(m_dict)

        optional_names = {mall[c].name for c in samfire.optional_components}

        if self.is_ipyparallel:
            from ipyparallel import Reference as ipp_Reference
            _logger.debug('preparing ipyparallel workers')
            direct_view = self.pool.client[:self.num_workers]
            direct_view.block = True
            direct_view.execute("from hyperspy.samfire_utils.samfire_worker"
                                " import create_worker")
            direct_view.scatter('identity', range(self.num_workers),
                                flatten=True)
            direct_view.execute('worker = create_worker(identity)')
            self.rworker = ipp_Reference('worker')
            direct_view.apply(lambda worker, m_dict:
                              worker.create_model(m_dict, 'z'), self.rworker,
                              m_dict)
            direct_view.apply(lambda worker, ts: worker.setup_test(ts),
                              self.rworker, samfire.metadata._gt_dump)
            direct_view.apply(lambda worker, on: worker.set_optional_names(on),
                              self.rworker, optional_names)

        if self.is_multiprocessing:
            _logger.debug('preparing multiprocessing workers')
            manager = Manager()
            self.shared_queue = manager.Queue()
            self.result_queue = manager.Queue()
            for i in range(self.num_workers):
                this_queue = manager.Queue()
                self.workers[i] = this_queue
                this_queue.put(('setup_test', (samfire.metadata._gt_dump,)))
                this_queue.put(('create_model', (m_dict, 'z')))
                this_queue.put(('set_optional_names', (optional_names,)))
                self.pool.apply_async(create_worker, args=(i, this_queue,
                                                           self.shared_queue,
                                                           self.result_queue))

    def update_parameters(self):
        """Updates various worker parameters.
        
        Currently updates:
            - Optional components (that can be switched off by the worker)
            - Parameter boundaries
            - Goodness test"""
        samfire = self.samf
        optional_names = {samfire.model[c].name for c in
                          samfire.optional_components}
        boundaries = tuple(tuple((par.bmin, par.bmax) for par in
                                 comp.parameters) for comp in self.samf.model)
        if self.is_multiprocessing:
            for this_queue in self.workers.values():
                this_queue.put(('set_optional_names', (optional_names,)))
                this_queue.put(('setup_test', (samfire.metadata._gt_dump,)))
                this_queue.put(('set_parameter_boundaries', (boundaries,)))
        elif self.is_ipyparallel:
            direct_view = self.pool.client[:self.num_workers]
            direct_view.block = True
            direct_view.apply(lambda worker, on: worker.set_optional_names(on),
                              self.rworker, optional_names)
            direct_view.apply(lambda worker, ts: worker.setup_test(ts),
                              self.rworker, samfire.metadata._gt_dump)
            direct_view.apply(lambda worker, ts:
                              worker.set_parameter_boundaries(ts),
                              self.rworker, boundaries)

    def ping_workers(self, timeout=None):
        """Pings the workers and records one-way trip time and (if available)
        pid of the worker.

        Parameters
        ----------
        timeout: {None, flaot}
            the time to wait when collecting results after sending out the
            ping. If None, the default timeout is used
        """
        if self.samf is None:
            _logger.error('Have to add samfire to the pool first')
        else:
            if self.is_multiprocessing:
                for _id, this_queue in self.workers.items():
                    this_queue.put('ping')
                    self.ping[_id] = time.time()
            elif self.is_ipyparallel:
                for i in range(self.num_workers):
                    direct_view = self.pool.client[i]
                    self.results.append((direct_view.apply_async(lambda worker:
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
            return self.shared_queue.qsize()

    def add_jobs(self, needed_number=None):
        """Adds jobs to the job queue that is consumed by the workers.

        Parameters
        ----------
        needed_number: {None, int}
            The number of jobs to add. If None (default), adds `need_pixels`
        """
        if needed_number is None:
            needed_number = self.need_pixels
        for ind, value_dict in self.samf.generate_values(needed_number):
            if self.is_multiprocessing:
                self.shared_queue.put(('run_pixel', (ind, value_dict)))
            elif self.is_ipyparallel:
                def test_func(worker, ind, value_dict):
                    return worker.run_pixel(ind, value_dict)
                self.results.append((self.pool.apply_async(test_func,
                                                           self.rworker, ind,
                                                           value_dict), ind))

    def parse(self, value):
        """Parses the value, returned from the workers.

        Parameters
        ----------
        value: tuple of the form (keyword, the_rest)
            keyword currently can be one of ['pong', 'Error', 'result']. For
            each of the keywords, "the_rest" is a tuple of different elements,
            but generally the first one is always the worker_id that the result
            came from. In particular:
                - ('pong', (worker_id, pid, pong_time, optional_message_str))
                - ('Error', (worker_id, error_message_string))
                - ('result', (worker_id,
                              pixel_index,
                              result_dict,
                              bool_if_result_converged))
        """
        if value is None:
            keyword = 'Failed'
            _logger.debug('Got None')
        else:
            keyword, the_rest = value
        samf = self.samf
        if keyword == 'pong':
            _id, pid, pong_time, message = the_rest
            self.ping[_id] = pong_time - self.ping[_id]
            self.pid[_id] = pid
            _logger.info('pong worker %s with time %g and message'
                         '"%s"' % (str(_id), self.ping[_id], message))
        elif keyword == 'Error':
            _id, err_message = the_rest
            _logger.error('Error in worker %s\n%s' % (str(_id), err_message))
        elif keyword == 'result':
            _id, ind, result, isgood = the_rest
            _logger.debug('Got result from pixel {} and it is good:'
                          '{}'.format(ind, isgood))
            if ind in samf.running_pixels:
                samf.running_pixels.remove(ind)
                samf.update(ind, result, isgood)
                samf.plot(on_count=True)
                samf.backup()
                samf.log(ind, isgood, samf.count, _id)
        else:
            _logger.error('Unusual return from some worker. The value '
                          'is:\n%s' % str(value))

    def collect_results(self, timeout=None):
        """Collects and parses all results, currently not processed due to
        being in the queue.

        Parameters
        ----------
        timeout: {None, flaot}
            the time to wait when collecting results. If None, the default
            timeout is used

        """
        if timeout is None:
            timeout = self.timeout
        found_something = False
        if self.is_ipyparallel:
            # for res, ind in reversed(self.results):
            for res, ind in self.results:
                if res.ready():
                    try:
                        result = res.get(timeout=timeout)
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
        """Returns the number of pixels that should be added to the processing
        queue. At most is equal to the number of workers.
        """
        return min(self.samf.pixels_done * self.samf.metadata.marker.ndim,
                   self.num_workers - len(self))

    @property
    def _not_too_long(self):
        """Returns bool if it has been too long after receiving the last
        result, probably meaning some of the workers timed out or hung.
        """
        if not hasattr(self, '_last_time') or not isinstance(self._last_time,
                                                             float):
            self._last_time = time.time()
        return (time.time() - self._last_time) <= self.timeout

    def run(self):
        """Runs the full process of adding jobs to the processing queue,
        listening to the results and updating SAMFire as needed. Stops when
        timed out or no pixels are left to run.
        """
        while self._not_too_long and (self.samf.pixels_left or
                                      len(self.samf.running_pixels)):
            # bool if got something
            new_result = self.collect_results()
            need_number = self.need_pixels

            if need_number > 0:
                self.add_jobs(need_number)
            if not need_number or not new_result:
                # did not spend much time, since no new results or added pixels
                self.sleep()
            else:
                self._last_time = time.time()

    def stop(self):
        """Stops the appropriate pool and (if ipyparallel) clears the memory
        and history.
        """
        if self.is_multiprocessing:
            for queue in self.workers.values():
                queue.put('stop_listening')
            self.pool.close()
            # self.pool.terminate()
        elif self.is_ipyparallel:
            self.pool.client.clear()
