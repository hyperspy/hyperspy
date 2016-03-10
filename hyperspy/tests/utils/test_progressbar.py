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
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.import
# nose.tools


import nose.tools as nt
from hyperspy.external import progressbar


class TestProgressBar:

    def setUp(self):
        pass

    def test_progressbar_not_shown(self):
        pbar = progressbar.progressbar(maxval=2, disabled=True)
        for i in range(2):
            pbar.update(i)
        pbar.finish()

    def test_progressbar_shown(self):
        pbar = progressbar.progressbar(maxval=2, disabled=False)
        for i in range(2):
            pbar.update(i)
        pbar.finish()
