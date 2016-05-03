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
import ipyparallel as ipp
import numpy as np

_logger = logging.getLogger(__name__)


class ParallelPool:

    _timestep = 0

    def __init__(self, num_workers=None, ipython_kwargs=None,
                 ipyparallel=None):
        if ipython_kwargs is None:
            ipython_kwargs = {}
        else:
            ipyparallel = True
        self.timeout = 15.
        self.ipython_kwargs = {'timeout': self.timeout}
        self.ipython_kwargs.update(ipython_kwargs)
        self.pool = None
        self.num_workers = num_workers
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
        return hasattr(self.pool, 'client')

    @property
    def is_multiprocessing(self):
        return isinstance(self.pool, Pool_type)

    @property
    def has_pool(self):
        return self.is_ipyparallel or self.is_multiprocessing and \
            self.pool._state is 0

    def _setup_ipyparallel(self):
        _logger.debug('Calling _setup_ipyparallel')
        try:
            ipyclient = ipp.Client(**self.ipython_kwargs)
            self.num_workers = min(self.num_workers, len(ipyclient))
            self.pool = ipyclient.load_balanced_view(
                range(self.num_workers))
            self.results = []
            return True
        except OSError:
            return False

    def _setup_multiprocessing(self):
        _logger.debug('Calling _setup_multiprocessing')
        self.num_workers = min(self.num_workers, cpu_count() - 1)
        self.pool = Pool(processes=self.num_workers)
        return True

    def setup(self, ipyparallel=None):
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
        if howlong is None:
            howlong = self.timestep
        time.sleep(howlong)
