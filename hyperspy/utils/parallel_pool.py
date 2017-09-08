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
from multiprocessing import (cpu_count, Pool)
from multiprocessing.pool import Pool as Pool_type
import numpy as np

_logger = logging.getLogger(__name__)


class ParallelPool:
    """ Creates a ParallelPool by either looking for a ipyparallel client and
    then creating a load_balanced_view, or by creating a multiprocessing pool

    Methods
    -------
    setup
        sets up the requested pool
    sleep
        sleeps for the requested (or timeout) time

    Attributes
    ----------

    has_pool: Bool
        Boolean if the pool is available and active.
    pool: {ipyparallel.load_balanced_view, multiprocessing.Pool}
        The pool object.
    ipython_kwargs: dict
        The dictionary with Ipyparallel connection arguments.
    timeout: float
        Timeout for either pool when waiting for results.
    num_workers: int
        The number of workers actually created (may be less than requested, but
        can't be more).
    timestep: float
        Can be used as "ticks" to adjust CPU load when building upon this
        class.
    is_ipyparallel: bool
        If the pool is ipyparallel-based
    is_multiprocessing: bool
        If the pool is multiprocessing-based

    """

    _timestep = 0

    def __init__(self, num_workers=None, ipython_kwargs=None,
                 ipyparallel=None):
        """Creates the ParallelPool and sets it up.

        Parameters
        ----------
        num_workers: {None, int}
            the (max) number of workers to create. If less are available,
            smaller number is actually created.
        ipyparallel: {None, bool}
            which pool to set up. True - ipyparallel. False - multiprocessing.
            None - try ipyparallel, then multiprocessing if failed.
        ipython_kwargs: {None, dict}
            arguments that will be passed to the ipyparallel.Client when
            creating. Not None implies ipyparallel=True.
        """
        if ipython_kwargs is None:
            ipython_kwargs = {}
        else:
            ipyparallel = True
        self.timeout = 15.
        self.ipython_kwargs = {'timeout': self.timeout}
        self.ipython_kwargs.update(ipython_kwargs)
        self.pool = None
        if num_workers is None:
            num_workers = np.inf
        self.num_workers = np.abs(num_workers)
        self.timestep = 0.001
        self.setup(ipyparallel=ipyparallel)

    def _timestep_get(self):
        return self._timestep

    def _timestep_set(self, value):
        value = np.abs(value)
        self._timestep = value

    timestep = property(lambda s: s._timestep_get(),
                        lambda s, v: s._timestep_set(v))

    @property
    def is_ipyparallel(self):
        """Returns bool if the pool is ipyparallel-based"""
        return hasattr(self.pool, 'client')

    @property
    def is_multiprocessing(self):
        """Returns bool if the pool is multiprocessing-based"""
        return isinstance(self.pool, Pool_type)

    @property
    def has_pool(self):
        """Returns bool if the pool is ready and set-up"""
        return self.is_ipyparallel or self.is_multiprocessing and \
            self.pool._state is 0

    def _setup_ipyparallel(self):
        import ipyparallel as ipp
        _logger.debug('Calling _setup_ipyparallel')
        try:
            ipyclient = ipp.Client(**self.ipython_kwargs)
            self.num_workers = min(self.num_workers, len(ipyclient))
            self.pool = ipyclient.load_balanced_view(
                range(self.num_workers))
            return True
        except OSError:
            _logger.debug('Failed to find ipyparallel pool')
            return False

    def _setup_multiprocessing(self):
        _logger.debug('Calling _setup_multiprocessing')
        self.num_workers = min(self.num_workers, cpu_count() - 1)
        self.pool = Pool(processes=self.num_workers)
        return True

    def setup(self, ipyparallel=None):
        """Sets up the pool.

        Parameters
        ----------
        ipyparallel: {None, bool}
            if True, only tries to set up the ipyparallel pool. If False - only
            the multiprocessing. If None, first tries ipyparallel, and it does
            not succeed, then multiprocessing.
        """
        _logger.debug('Calling setup with ipyparallel={}'.format(ipyparallel))
        if not self.has_pool:
            if ipyparallel is True:
                if self._setup_ipyparallel():
                    return
                else:
                    raise ValueError('Could not connect to the ipyparallel'
                                     ' Client')
            elif ipyparallel is None:
                _ = self._setup_ipyparallel() or self._setup_multiprocessing()
                return
            elif ipyparallel is False:
                self._setup_multiprocessing()
            else:
                raise ValueError('ipyparallel has to be True, False or None '
                                 'type')

    def sleep(self, howlong=None):
        """Sleeps for the required number of seconds.

        Parameters
        ----------
        howlong: {None, float}
            How long the pool should sleep for in seconds. If None (default),
            sleeps for "timestep"
        """
        if howlong is None:
            howlong = self.timestep
        time.sleep(howlong)
