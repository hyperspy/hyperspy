#!/usr/bin/python
# This code was obtained from http://code.enthought.com/projects/traits/docs/html/tutorials/traits_ui_scientific_app.html
# The author is Gael Varoquaux

import wx

import matplotlib
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_wx import NavigationToolbar2Wx

from traits.api import Any, Instance
from traitsui.wx.editor import Editor
from traitsui.wx.basic_editor_factory import BasicEditorFactory

class _MPLFigureEditor(Editor):

    scrollable  = True

    def init(self, parent):
        self.control = self._create_canvas(parent)
        self.set_tooltip()

    def update_editor(self):
        pass

    def _create_canvas(self, parent):
        """ Create the MPL canvas. """
        # The panel lets us add additional controls.
        panel = wx.Panel(parent, -1, style=wx.CLIP_CHILDREN)
        sizer = wx.BoxSizer(wx.VERTICAL)
        panel.SetSizer(sizer)
        # matplotlib commands to create a canvas
        mpl_control = FigureCanvas(panel, -1, self.value)
        sizer.Add(mpl_control, 1, wx.LEFT | wx.TOP | wx.GROW)
        toolbar = NavigationToolbar2Wx(mpl_control)
        sizer.Add(toolbar, 0, wx.EXPAND)
        self.value.canvas.SetMinSize((10,10))
        return panel

class MPLFigureEditor(BasicEditorFactory):

    klass = _MPLFigureEditor


if __name__ == "__main__":
    # Create a window to demo the editor
    from enthought.traits.api import HasTraits
    from enthought.traits.ui.api import View, Item
    from numpy import sin, cos, linspace, pi

    class Test(HasTraits):

        figure = Instance(Figure, ())

        view = View(Item('figure', editor=MPLFigureEditor(),
                                show_label=False),
                        width=400,
                        height=300,
                        resizable=True)

        def __init__(self):
            super(Test, self).__init__()
            axes = self.figure.add_subplot(111)
            t = linspace(0, 2*pi, 200)
            axes.plot(sin(t)*(1+0.5*cos(11*t)), cos(t)*(1+0.5*cos(11*t)))

    Test().configure_traits()
