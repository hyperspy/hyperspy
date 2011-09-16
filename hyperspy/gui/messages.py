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

import enthought.traits.api as t
import enthought.traits.ui.api as tui
from enthought.traits.ui.menu import OKButton

class Message(t.HasTraits):
    text = t.Str
    
information_view = tui.View(tui.Item('text', show_label = False, 
                            style = 'readonly', springy = True, width = 300,), 
                            kind = 'modal', buttons = [OKButton,] )
                                
def information(text):
    message = Message()
    message.text = text
    message.edit_traits(view = information_view)
