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

import numpy as np
import time

from hyperspy.signals import Signal1D, Signal2D


def get_plotting_speed(run_number=100, ndim=2, sdim=2, nshape=100, sshape=100):

    n = nshape**ndim*sshape**sdim
    shape = [nshape for dim in range(ndim)] + [sshape for dim in range(sdim)]
    data = np.arange(n).reshape(shape)

    if sdim == 1:
        s = Signal1D(data)
    elif sdim == 2:
        s = Signal2D(data)
    else:
        print('`sdim` should be 1 or 2.')

    s.plot()
    s._plot.signal_plot.figure.canvas.flush_events()

    t0 = time.time()
    for i in range(run_number):
        s.axes_manager.__next__()
    t1 = time.time()
    fps = run_number/(t1 - t0)
    print('Plot update speed: {} fps.'.format(fps))

    return fps


def main():
    run_number = 100
    ndim = 2
    sdim = 1
    nshape = 100
    sshape = 4000
    get_plotting_speed(run_number=run_number, ndim=ndim, sdim=sdim,
                       nshape=nshape, sshape=sshape)
