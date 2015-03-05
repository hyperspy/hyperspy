# -*- coding: utf-8 -*-
# Copyright 2007-2011 The HyperSpy developers
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

from multiprocessing import Pool as pool_mp
from IPython.parallel import Client as client_ip
from IPython.parallel import error


def pool(parallel, pool_type=None, ipython_timeout=1.):
    """
    Create a pool for multiprocessing

    Parameters
    ----------
    pool_type: 'ipython' or'mp'
        the type of pool
    ipython_timeout : float
        Timeout to be passed for ipython parallel Client.
    """
    if pool_type is None:
        try:
            c = client_ip(profile='hyperspy', timeout=ipython_timeout)
            pool = c[:parallel]
            pool_type = 'ipython'
        except (error.TimeoutError, IOError):
            pool_type = 'mp'
            pool = pool_mp(processes=parallel)
    elif pool_type == 'ipython':
        c = client_ip(profile='hyperspy', timeout=ipython_timeout)
        pool = c[:parallel]
        pool_type = 'ipython'
    else:
        pool_type = 'mp'
        pool = pool_mp(processes=parallel)
    return pool, pool_type


def close(pool, pool_type):
    """
    """
    if pool_type == 'mp':
        pool.close()
        pool.join()
