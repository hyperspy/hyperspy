#!/usr/bin/python
# -*- coding: utf-8 -*-

import enthought.traits.api as t
import enthought.traits.ui.api as tui
from enthought.traits.ui.menu import OKButton, CancelButton, Action, MenuBar, Menu

from .. import file_io
from microscope import Microscope
from egerton_quantification import EgertonPanel

import messages
from ..interactive_ns import interactive_ns
from .. import Release

microscope = Microscope()

# File ###################################################
  

 
class LoadSpectrum(t.HasTraits):
    sp_file = t.File()
    traits_view = tui.View(
                            tui.Item('sp_file', label = 'File'),
                            buttons = [OKButton, CancelButton],
                            kind = 'modal')
        
file_view = tui.View(
            tui.Group(
                tui.Group(
                    tui.Item('f.hl_file'),
                    ),
                tui.Group(
                   tui.Item('micro.name', ),
                   tui.Item('micro.alpha'),
                   tui. Item('micro.beta'),
                   tui. Item('micro.E0'),
                   ),
                tui.Group(
                   tui. Item('micro.name', ),
                   tui. Item('micro.alpha'),
                   tui. Item('micro.beta'),
                   tui. Item('micro.E0'),)) 
                )
  
  
# Actions

menu_file_open = Action(name = "Open...",
                action = "open_file",
                toolip = "Open an SI file",)
                
menu_file_save = Action(name = "Save...",
                action = "save_file",
                toolip = "Save an SI file",)
                
menu_help_about = Action(name = "About",
                action = "notification_about",
                toolip = "",)
                
menu_tools_calibrate = Action(name = "Calibrate",
                action = "calibrate",
                toolip = "",)

menu_tools_egerton_quantification = Action(name = "Egerton Quantification",
                action = "egerton_quantification",
                toolip = "",)                
            
menu_edit_microscope = Action(name = "Microscope parameters",
                action = "edit_microscope",
                toolip = "",)
                
menu_edit_acquisition_parameters = Action(name = "Acquisition parameters",
                action = "edit_acquisition_parameters",
                toolip = "",)
                
# Menu
menubar = MenuBar()

# File
Menu(menu_file_open, menu_file_save, name = 'File')

# Edit

# Open

# Main Window ##################################################

    
class MainWindowHandler(tui.Handler):
        
    def open_file(self, *args, **kw):
        S = LoadSpectrum()
        S.edit_traits()
        if S.sp_file is not t.Undefined:
            s = file_io.load(S.sp_file)
            s.plot()
            interactive_ns['s'] = s
    
    def save_file(self, *args, **kw):
        pass
    
    def edit_microscope(self, *args, **kw):
        microscope.edit_traits()
        
    def egerton_quantification(self, *args, **kw):
        if interactive_ns.has_key('s'):
            ep = EgertonPanel(interactive_ns['s'])
            ep.edit_traits()
        
    def notification_about(self,*args, **kw):
        messages.information(Release.info)
                
    def calibrate(self,*args, **kw):
        pass
                
class MainWindow(t.HasTraits):
    
    traits_view = tui.View( #view contents,
                            # ...,
                            handler = MainWindowHandler(),
                            title = 'EELSLab',
                            width = 500,
                            menubar = MenuBar(
                            Menu(menu_file_open, menu_file_save,
                            name = 'File'),
                            Menu(menu_edit_microscope, 
                            menu_edit_acquisition_parameters,
                            name = 'Edit'),
                            Menu(menu_tools_calibrate, 
                            menu_tools_egerton_quantification, 
                            name = 'Tools'),
                            Menu(menu_help_about,
                            name = 'Help'),
                            ))


if __name__ == '__main__':
    window = MainWindow()
    window.configure_traits()
