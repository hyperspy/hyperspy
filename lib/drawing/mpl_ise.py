# -*- coding: utf-8 -*-
# Copyright © 2007 Francisco Javier de la Peña
#
# This file is part of EELSLab.
#
# EELSLab is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# EELSLab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EELSLab; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  
# USA
import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
# Set the matplotlib cmap to gray (the default is jet)
plt.rcParams['image.cmap'] = 'gray'

import utils

class MPL_HyperImage_Explorer():
    def change_to_frame(self, i1 = 0, i2 = 0):
        shape = self.data_cube.squeeze().shape
        if len(shape) == 3:
            self.image_ax.images[0].set_array(self.data_cube[...,i1].T)
        elif len(shape) == 4:
            self.image_ax.images[0].set_array(self.data_cube[...,i1,i2].T)
        if self.auto_contrast is True:
            self.image_ax.images[0].autoscale()
        plt.draw()
        
    def update_figure(self):
        self.change_to_frame(self.coordinates.ix, self.coordinates.iy)
        
    def _on_figure_close(self):
        self.image_figure = None
        self.image_ax = None
        self.coordinates.reset()
        
    def create_image_figure(self, dc):
        if self.image_figure is not None:
            # Test if the figure really exists. If not call the reset function 
            # and start again. This is necessary because with some backends 
            # EELSLab fails to connect the close event to the function.
            try:
                self.image_figure.show()
            except:
                self._on_figure_close()
                self.create_image_figure(dc)
            return True
            
        self.image_figure = plt.figure()
        utils.on_window_close(self.image_figure, self._on_figure_close)
        if hasattr(self, 'title'):
            title = self.title
        else:
            title = 'Image'
        self.image_figure.canvas.set_window_title(title)
        self.image_ax = self.image_figure.add_subplot(111)
        self.image_ax.imshow(dc.T, interpolation = 'nearest')
        self.image_figure.canvas.draw()
        self.coordinates.on_coordinates_change.append(self.update_figure)
        plt.connect('key_press_event', self.coordinates.key_navigator)
        plt.show()
        return True
plt.show()