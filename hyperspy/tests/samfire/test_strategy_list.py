# Copyright 2007-2016 The HyperSpy developers
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
# along with HyperSpy. If not, see <http://www.gnu.org/licenses/>.


from hyperspy.samfire import (Samfire, StrategyList)
from hyperspy.misc.utils import DictionaryTreeBrowser


class TestStrategyList:

    def setup_method(self, method):
        self.w1 = DictionaryTreeBrowser()
        self.w2 = DictionaryTreeBrowser()

        for w in [self.w1, self.w2]:
            w.add_node('samf')
        self.samf = object()
        self.sl = StrategyList(self.samf)

    def test_append(self):
        assert not self.w1.samf is self.samf
        assert not self.w1 in self.sl
        self.sl.append(self.w1)
        assert self.w1.samf is self.samf
        assert self.w1 in self.sl

    def test_extend(self):
        self.sl.extend([self.w1, self.w2])
        assert self.w1 in self.sl
        assert self.w1.samf is self.samf
        assert self.w2 in self.sl
        assert self.w2.samf is self.samf

    def test_remove_int(self):
        self.sl.append(self.w1)
        self.sl.remove(0)
        assert not self.w1.samf is self.samf
        assert not self.w1 in self.sl

    def test_remove_object(self):
        self.sl.append(self.w1)
        self.sl.remove(self.w1)
        assert not self.w1.samf is self.samf
        assert not self.w1 in self.sl
