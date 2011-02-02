#!/usr/bin/python
# -*- coding: utf-8 -*-

import math

import wx
import scipy.optimize
import pylab
import numpy

from enthought.traits.api import HasTraits, File, String, Bool, List, ListBool
from enthought.traits.ui.api import View, Handler, Group, Item,SetEditor

from eelslab import *
from eelslab.edges_db import edges_dict

# Mine

class FilesAndOptions(HasTraits):
    core_loss_spectra = File(None)
    core_loss_HAADF = File(None)
 
    # First Step. Files declaration and Options 
    view = View(Group(Group(Item( name = 'core_loss_spectra'),                   
                       Item( name = 'core_loss_HADF'),
                       label = "Files",
                       show_border = True),
                 Group(
                        Item(name = 'correct_spatial_shift', enabled_when = '(low_loss_spectra is not None) and (core_loss_spectra is not None) and (core_loss_HADF is not None) and (low_loss_HADF is not None)'),
                       label = 'Options',
                       show_border = True),
                       show_border = True,),
                       kind = 'wizard',
                       title = 'Model Builder Wizard')


# Load the core loss to get the energy range
                                    


class Edges_Selector(HasTraits):
        selected = List
                 

###### Main #################################################################

class Main():

    # Methods definitions
    def filter_edges(self):
        for element in edges_dict:
            for shell in edges_dict[element]['subshells']:
                print element
                print shell
                print edges_dict[element]['subshells'][shell]['onset_energy']
                if self.start_energy <= edges_dict[element]['subshells'][shell]['onset_energy'] <= self.end_energy :
                    if not self.available_edges.has_key(element):
                        self.available_edges[element] = {}
                    self.available_edges[element][shell] = edges_dict[element]['subshells'][shell]
    
    def reposition_edges(self):    
        self.__picked_vline = None
        self.__cid_on_move = None
        
        # Picker function definitions
        def on_move(event):
            if event.inaxes :
                self.__picked_vline.set_xdata(event.xdata)
                self.__picked_vline.label._x = event.xdata
#                self.core_loss.sp_ax1.draw()
                pylab.figure(self.core_loss.sp_figure.number)
                pylab.draw()
            
        def unclick(event) :
            if self.__cid_on_move is not None:
                self.core_loss.sp_figure.canvas.mpl_disconnect(self.__cid_on_move)

        def on_vline_pick(event):
            if event.mouseevent.button == 1 :
                if event.artist in self.__vlines :
                    self.__picked_vline = event.artist
                    self.core_loss.sp_figure.canvas.mpl_connect('button_release_event',unclick)
                    self.__cid_on_move = self.core_loss.sp_figure.canvas.mpl_connect('motion_notify_event', on_move )
        
        # Plotting       
        self.__vlines = []
        self.core_loss.plot()
        for element in self.selected_edges :
            i = 0
            for shell in sorted(self.available_edges[element],reverse = True):
                self.available_edges[element][shell]['vline'] = self.core_loss.sp_ax1.axvline(x=self.available_edges[element][shell]['onset_energy'], picker = True )
                min, max = self.core_loss.sp_ax1.get_ylim()
                width = max - min
                self.__vlines.append(self.available_edges[element][shell]['vline'])
                self.available_edges[element][shell]['vline'].label = self.core_loss.sp_ax1.text( self.available_edges[element][shell]['onset_energy'],max*0.95-i*width*0.05 , element + '.' + shell )
                i += 1
        
        # Connecting
        self.core_loss.sp_figure.canvas.mpl_connect('button_release_event',unclick)
        self.core_loss.sp_figure.canvas.mpl_connect('pick_event',on_vline_pick)
        pylab.figure(self.core_loss.sp_figure.number)
        pylab.draw()
        pylab.show()
    
    def multi_edge_calibration(self):
        x_data = []
        y_data = []
        data_label = []
        calibration_bool = []
        def energy2channel(E,origin,dispersion):
            return ( E-origin ) / dispersion
        for element in self.selected_edges :
            for shell in self.available_edges[element]:
                file_data = self.available_edges[element][shell]['onset_energy']
                y_data.append(file_data)
                new_x_data = self.available_edges[element][shell]['vline']._x[0]
                x_data.append(energy2channel(new_x_data,self.file_origin,self.file_dispersion))
                data_label.append ( element+'.'+shell )
                if file_data == new_x_data :
                    calibration_bool.append(False)
                else :
                    calibration_bool.append(True)
        x_data = numpy.array(x_data)
        y_data = numpy.array(y_data)
        calibration_bool = numpy.array(calibration_bool)
            
        # define our (line) fitting function
        fitfunc = lambda p, x: p[0] + p[1] * x
        errfunc = lambda p, x, y: (y - fitfunc(p, x))
        pinit = [self.file_origin, self.file_dispersion]
        out = scipy.optimize.leastsq(errfunc, pinit,
                               args=(x_data[calibration_bool], y_data[calibration_bool]), full_output=1)

        pfinal = out[0]
        covar = out[1]
        self.dispersion = pfinal[1]
        self.origin = pfinal[0]
        self.originErr = math.sqrt( covar[0][0] )
        self.dispersionErr = math.sqrt( covar[1][1] )
        print "dispersion = ", self.dispersion, "+-", self.dispersionErr
        print "origin = ", self.origin, "+-", self.originErr
        self.core_loss.set_new_calibration(self.origin, self.dispersion)
    

    def __init__(self) :
        self.available_edges = {}
        __files = FilesAndOptions()
        __files.edit_traits()

        # Get information from the core loss
        exec('self.core_loss = Spectrum(\'%s\')' %__files.core_loss_spectra)
        self.start_energy = self.core_loss.energy_axis[0]
        self.end_energy = self.core_loss.energy_axis[-1]
        self.file_dispersion = self.core_loss.energyscale
        self.file_origin =  self.core_loss.energyorigin
        self.filter_edges()

        # Step 2. Select the edges ##################################
        available = self.available_edges.keys()
        step2 = Edges_Selector()
        step2_view = View(Item('selected',editor = SetEditor(values = available)),kind = 'wizard',title="Edges Selector")
        step2.edit_traits(view = step2_view)
        self.selected_edges = step2.selected
        self.reposition_edges()
        self.multi_edge_calibration()
        self.reposition_edges()
        wx.PySimpleApp().MainLoop()

a = Main()



