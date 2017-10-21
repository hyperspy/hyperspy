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

import pytest

from hyperspy.misc.utils import attrsetter
from hyperspy.misc.utils import DictionaryTreeBrowser


class DummyThing(object):

    def __init__(self):
        self.name = 'Dummy'
        self.another = None

    def multiply(self):
        self.another = self.__class__()


class TestAttrSetter:

    def setup_method(self, method):
        tree = DictionaryTreeBrowser(
            {
                "Node1": {"leaf11": 11,
                          "Node11": {"leaf111": 111},
                          },
                "Node2": {"leaf21": 21,
                          "Node21": {"leaf211": 211},
                          },
                "Leaf3": 3
            })
        self.tree = tree
        self.dummy = DummyThing()

    def test_dtb_settattr(self):
        t = self.tree
        attrsetter(t, 'Node1.leaf11', 119)
        assert t.Node1.leaf11 == 119
        attrsetter(t, 'Leaf3', 39)
        assert t.Leaf3 == 39

    def test_wrong_item(self):
        t = self.tree
        with pytest.raises(AttributeError):
            attrsetter(t, 'random.name.with.more.than.one', 13)

    def test_dummy(self):
        d = self.dummy
        d.multiply()
        attrsetter(d, 'another.name', 'New dummy')
        assert d.another.name == 'New dummy'
        d.another.multiply()
        attrsetter(d, 'another.another.name', 'super New dummy')
        assert d.another.another.name == 'super New dummy'
