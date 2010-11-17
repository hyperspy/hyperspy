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
        
class Painter:
    '''
    '''
    
    def plot_maps(self):
        for component in self:
            if component.active:
                component.plot_maps()

    def plot(self,ix=None,iy=None, non_convolved = False,new=True):

        # Set the coordinates
        if (ix is not None and ix != self.ix) or (iy is not None and
                                                    iy != self.iy):
            self.set_coordinates(ix,iy)

        title = "EEL Spectrum. Coordinates: (" + str(self.ix) + ", " + \
        str(self.iy) + ")"

        # Plot Interactivity
        def key_navigator(event):
            if event.key == "up":
                self.plot(iy=self.iy+1,new=False)
            elif event.key == "down":
                self.plot(iy=self.iy-1,new=False)
            elif event.key == "right":
                self.plot(ix=self.ix+1,new=False)
            elif event.key == "left":
                self.plot(ix=self.ix-1,new=False)
            elif event.key == "d":
                self.plot(non_convolved=True,new=False)
            elif event.key == "c":
                self.plot(new=False)
            elif event.key == "l":
                plt.figure(3)
                plt.scatter(self.ll.energy_axis, self.ll.data_cube[:,
                self.ix, self.iy], s=1)
                plt.title(title)

        def mouse_navigator_x(event):
            if event.button ==1:
                self.plot(ix=round(event.ydata), new=False)
        def mouse_navigator_y(event):
            if event.button ==1:
                self.plot(iy=round(event.ydata), new=False)
        def mouse_navigator_xy(event):
            if event.button ==1:
                self.plot(ix=round(event.xdata),iy=round(event.ydata),
                new=False)

        # Plotting commands
        plt.figure(1)
        plt.cla()
        plt.plot(self.hl.energy_axis,self.__call__(non_convolved,
        onlyactive = True))
        plt.scatter(self.hl.energy_axis[self.channel_switches],
        self.hl.data_cube[self.channel_switches,self.ix,self.iy], s =1)
        plt.xlabel('Energy Loss (eV)')
        plt.ylabel('Counts')
        plt.title(title)
        if new:
            plt.connect('key_press_event', key_navigator)

        if self.hl.xdimension > 1 and self.hl.ydimension == 1:
            plt.figure(2)
            plt.cla()
            plt.imshow(np.transpose(self.hl.data_cube[:,:,self.iy]),
            interpolation='nearest')
            plt.axhline(self.ix,color='r')
            plt.xlabel("Channel")
            plt.ylabel("x")
            plt.title(title)
            if new:
                plt.connect('key_press_event', key_navigator)
                plt.connect('button_press_event', mouse_navigator_x)
        elif self.hl.ydimension > 1 and self.hl.xdimension == 1:
            spectra2D = plt.figure(2)
            plt.cla()
            plt.imshow(self.hl.data_cube[:,self.ix,:],interpolation='nearest')
            plt.axvline(self.iy,color='r')
            plt.xlabel("Channel")
            plt.ylabel("y")
            plt.title(title)
            if new:
                plt.connect('key_press_event', key_navigator)
                plt.connect('button_press_event', mouse_navigator_y)
        elif self.hl.ydimension > 1 and self.hl.xdimension > 1:
            plt.figure(2)
            if new :
                self.__navigator_ax = plt.subplot(111)
                if self.hl.image is None:
                    plt.imshow(np.transpose(np.sum(self.hl.data_cube,axis=0)),
                    interpolation='nearest')
                    new_title = "Pseudo HAADF "+title
                    plt.title(new_title)
                else :
                    plt.imshow(np.transpose(self.hl.image_array),
                    interpolation='nearest')
                    new_title = "HAADF "+title
                    plt.title(new_title)
                plt.colorbar()
                plt.xlabel("x")
                plt.ylabel("y")
            if not new:
                self.__navigator_ax.patches.remove(self.__fig_navigator_point)
            self.__fig_navigator_point  = plt.Rectangle((self.ix-0.5,self.iy
             - 0.5),1,1,fc = 'r',fill= False,lw = 2)
            self.__navigator_ax.add_patch(self.__fig_navigator_point)

            plt.draw()
            if new:
                plt.figure(1)
                plt.disconnect('key_press_event')
                plt.figure(2)
                plt.connect('key_press_event', key_navigator)
                plt.connect('button_press_event', mouse_navigator_xy)
##                plt.subplot(122)
##                plt.imshow(np.transpose(self.ll.thickness),
##                interpolation='nearest')
##                plt.colorbar()
##                new_title = "Thickness"
##                plt.xlabel("x")
##                plt.ylabel("y")
##                plt.title(new_title)
                plt.show()
