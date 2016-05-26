# -*- coding: utf-8 -*-
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

#~ import traits.api as t
#~ import traitsui.api as tui
#~ from traitsui.menu import OKButton, CancelButton, Action, MenuBar, Menu
#~
#~ from hyperspy import io
#~ from egerton_quantification import EgertonPanel
#~
#~ import messages
#~ import tools
#~ from hyperspy.misc.interactive_ns import interactive_ns
#~ from hyperspy import Release
#~
#~ # File ###################################################
#~
#~
#~
#~ class LoadSignal1D(t.HasTraits):
#~ sp_file = t.File()
#~ traits_view = tui.View(
#~ tui.Item('sp_file', label = 'File'),
#~ buttons = [OKButton, CancelButton],
#~ kind = 'modal')
#~
#~ file_view = tui.View(
#~ tui.Group(
#~ tui.Group(
#~ tui.Item('f.hl_file'),
#~ ),
#~ tui.Group(
#~ tui.Item('micro.name', ),
#~ tui.Item('micro.alpha'),
#~ tui. Item('micro.beta'),
#~ tui. Item('micro.E0'),
#~ ),
#~ tui.Group(
#~ tui. Item('micro.name', ),
#~ tui. Item('micro.alpha'),
#~ tui. Item('micro.beta'),
#~ tui. Item('micro.E0'),))
#~ )
#~
#~
# ~ # Actions
#~
#~ menu_file_open = Action(name = "Open...",
#~ action = "open_file",
#~ toolip = "Open an SI file",)
#~
#~ menu_file_save = Action(name = "Save...",
#~ action = "save_file",
#~ toolip = "Save an SI file",)
#~
#~ menu_help_about = Action(name = "About",
#~ action = "notification_about",
#~ toolip = "",)
#~
#~ menu_tools_calibrate = Action(name = "Calibrate",
#~ action = "calibrate",
#~ toolip = "",)
#~
#~ menu_tools_egerton_quantification = Action(name = "Egerton Quantification",
#~ action = "egerton_quantification",
#~ toolip = "",)
#~
#~ menu_tools_savitzky_golay = Action(name = "Savitzky-Golay Smoothing",
#~ action = "savitzky_golay",
#~ toolip = "",)
#~
#~ menu_tools_lowess = Action(name = "Lowess Smoothing",
#~ action = "lowess",
#~ toolip = "",)
#~
#~ menu_edit_acquisition_parameters = Action(name = "Acquisition parameters",
#~ action = "edit_acquisition_parameters",
#~ toolip = "",)
#~
# ~ # Menu
#~ menubar = MenuBar()
#~
# ~ # File
#~ Menu(menu_file_open, menu_file_save, name = 'File')
#~
# ~ # Edit
#~
# ~ # Open
#~
#~ # Main Window ##################################################
#~
#~
#~ class MainWindowHandler(tui.Handler):
#~
#~ def open_file(self, *args, **kw):
#~ S = LoadSignal1D()
#~ S.edit_traits()
#~ if S.sp_file is not t.Undefined:
#~ s = io.load(S.sp_file)
#~ s.plot()
#~ interactive_ns['s'] = s
#~
#~ def save_file(self, *args, **kw):
#~ pass
#~
#~
#~ def egerton_quantification(self, *args, **kw):
#~ if interactive_ns.has_key('s'):
#~ ep = EgertonPanel(interactive_ns['s'])
#~ ep.edit_traits()
#~
#~ def savitzky_golay(self, *args, **kw):
#~ sg = tools.SavitzkyGolay()
#~ sg.edit_traits()
#~
#~ def lowess(self, *args, **kw):
#~ lw = tools.Lowess()
#~ lw.edit_traits()
#~
#~ def notification_about(self,*args, **kw):
#~ messages.information(Release.info)
#~
#~ def calibrate(self,*args, **kw):
#~ w = tools.Calibration()
#~ w.edit_traits()
#~
#~ class MainWindow(t.HasTraits):
#~
# ~ traits_view = tui.View( #view contents,
# ~ # ...,
#~ handler = MainWindowHandler(),
#~ title = 'HyperSpy',
#~ width = 500,
#~ menubar = MenuBar(
#~ Menu(
#~ menu_file_open,
#~ menu_file_save,
#~ name = 'File'),
#~ Menu(
#~ menu_edit_acquisition_parameters,
#~ name = 'Edit'),
#~ Menu(
#~ menu_tools_calibrate,
#~ menu_tools_egerton_quantification,
#~ menu_tools_savitzky_golay,
#~ menu_tools_lowess,
#~ name = 'Tools'),
#~ Menu(
#~ menu_help_about,
#~ name = 'Help'),
#~ ))
#~
#~
#~ if __name__ == '__main__':
#~ window = MainWindow()
#~ window.configure_traits()
