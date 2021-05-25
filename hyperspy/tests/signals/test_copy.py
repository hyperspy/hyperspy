# Copyright 2007-2021 The HyperSpy developers
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


from hyperspy.signals import BaseSignal


def test_deepcopy():
    s = BaseSignal([0])
    s.metadata.test = [0]
    s.original_metadata.test = [0]
    s_deepcopy = s.deepcopy()
    s.metadata.test.append(1)
    s.original_metadata.test.append(1)
    assert s.metadata.test == [0, 1]
    assert s.original_metadata.test == [0, 1]
    assert s_deepcopy.metadata.test == [0]
    assert s_deepcopy.original_metadata.test == [0]

