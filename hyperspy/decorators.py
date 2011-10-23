# -*- coding: utf-8 -*-
# Copyright 2007-2011 The Hyperspy developers
#
# This file is part of  Hyperspy.
#
#  Hyperspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  Hyperspy.  If not, see <http://www.gnu.org/licenses/>.

# custom exceptions
from hyperspy import messages
from hyperspy.exceptions import NoInteractiveError
from hyperspy.defaults_parser import preferences
from hyperspy.gui.tools import SpectrumRangeSelector

def only_interactive(cm):
    def wrapper(*args, **kwargs):
        if preferences.General.interactive is True:
            cm(*args, **kwargs)
        else:
            raise NoInteractiveError
    return wrapper
    
def interactive_range_selector(cm):
    def wrapper(self, *args, **kwargs):
        if preferences.General.interactive is True and not args and not kwargs:
            range_selector = SpectrumRangeSelector(self)
            range_selector.on_close.append((cm, self))
            range_selector.edit_traits()
        else:
            cm(self, *args, **kwargs)
    return wrapper
