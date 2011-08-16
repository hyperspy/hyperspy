# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 11:05:17 2010

@author: Francisco de la Peña
"""

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