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

import copy
import sys
import os
import tempfile

import numpy as np
from components.edge import Edge
from components.power_law import PowerLaw
from interactive_ns import interactive_ns
from defaults_parser import defaults
from utils import two_area_powerlaw_estimation
from painter import Painter
from estimators import Estimators
from optimizers import Optimizers
from model_controls import Controls
import messages

class Model(list, Painter, Optimizers, Estimators, Controls):
    '''Build a fit a model
    
    Parameters
    ----------
    data : Spectrum instance
    auto_background : boolean
        If True, it adds automatically a powerlaw to the model and estimate the 
        parameters by the two-area method.
    auto_add_edges : boolen
        If True (default), it will automatically add the ionization edges as 
        defined in the Spectrum instance.
    '''
    
    ix, iy = 0, 0
    __firstimetouch = True

    def __init__(self, data, auto_background = True, auto_add_edges = True):
        self.divide_ll_by_I0 = False
        try:       
            self.Experiments = data
            self.hl = data.hl
            if hasattr(data, 'll'):
                self.ll = data.ll
            else:
                self.ll = None
        except:
            try:
                self.hl = data
                self.ll = None
            except:
                print "\n\nWarning!!"
                print "The data must be a Spectrum or Experiments"
                sys.exit()
        if self.divide_ll_by_I0 is True:
            if self.ll is not None:
                if hasattr(self.ll, 'I0') and self.ll.I0 is not None:
                    x, y = self.ll.I0.data_cube.shape
                    self.ll4dc = self.ll.data_cube / self.ll.I0.data_cube.reshape(
                    (1,x,y))
                else:
                    string = '''Deconvolution requires the calculation of I0.
                    You can use Spectrum.calculate_I0 on the LL SI or set 
                    divide_ll_by_I0 to False'''
                    message.warning_exit(string)
        elif self.ll is not None:
            self.ll4dc = self.ll.data_cube
            
        self.least_squares_fit_output = np.zeros((self.hl.xdimension,
        self.hl.ydimension)).tolist()
        self.free_parameters_boundaries = None
        self.channel_switches=np.array([True]*self.hl.energydimension)
#        self.spectrum = copy.deepcopy(self.hl)
#        self.spectrum.data_cube = np.zeros(self.hl.data_cube.shape)
#        self.model_cube = self.spectrum.data_cube
        
        self.model_cube = np.zeros(self.hl.data_cube.shape)

        if auto_background:
            bg = PowerLaw()
            interactive_ns['bg'] = bg
            self.append(bg)
            
        if self.hl.edges and auto_add_edges:
            self.extend(self.hl.edges)

        if self.ll is not None:
            self.convolved = True
        else:
            self.convolved = False 
        self.touch()         
    # TODO Redifine the method of the list, so touch is called everytime a 
    # component is added or removed
    def touch(self):
        '''Update the edges list
        
        This function must be called everytime that we add or remove components
        from the model.
        It creates the bookmarks self.edges and sef.__background_components and 
        configures the edges by setting the dispersion attribute and setting 
        the fine structure.
        
        Note:
        -----
        This is an annoyance that should dissappear soon
        '''
        self.edges = []
        self.__background_components = []
        for component in self:
            if isinstance(component,Edge):
                component.dispersion = self.hl.energyscale
                component.setfslist()
                if component.edge_position() < \
                self.hl.energy_axis[self.channel_switches][0]:
                    component.isbackground = True
                if component.isbackground is not True:
                    self.edges.append(component)
                else :
                    component.fs_state = False
                    component.fslist.free = False
                    component.backgroundtype = "edge"
                    self.__background_components.append(component)

            elif isinstance(component,PowerLaw) or component.isbackground is True:
                self.__background_components.append(component)

        self.set()
        if len(self.edges) == 0 :
            print "Warning : the model contains no edges"
        else :
            self.edges.sort(key = Edge.edge_position)
        if len(self.__background_components) > 1 :
            self.__backgroundtype = "mix"
        elif len(self.__background_components) == 0 :
            print "Warning : You did't define any model for the background."
        else :
            self.__backgroundtype = \
            self.__background_components[0].__repr__()
            if self.__firstimetouch and len(self.edges) != 0 :
                self.two_area_background_estimation()
                self.__firstimetouch = False
                
    def generate_cube(self):
        '''Generate a SI with the current model
        
        The SI is stored in self.model_cube
        '''
        for iy in range(self.model_cube.shape[2]):
            for ix in range(self.model_cube.shape[1]):
                print "x = %i\ty = %i" % (ix, iy)
                self.set_coordinates(ix, iy)
                self.model_cube[:,self.ix,self.iy] = self.__call__(
                non_convolved = not self.convolved, onlyactive = True)

    def resolve_fine_structure(self,preedge_safe_window_width = 
        defaults.preedge_safe_window_width, i1 = 0):
        '''Adjust the fine structure of all edges to avoid overlapping
        
        This function is called automatically everytime the position of an edge
        changes
        
        Parameters
        ----------
        preedge_safe_window_width : float
            minimum distance between the fine structure of an ionization edge 
            and that of the following one.
        '''

        while (self.edges[i1].fs_state is False or  
        self.edges[i1].active is False) and i1 < len(self.edges)-1 :
            i1+=1
        print "i1 = ", i1
        if i1 < len(self.edges)-1 :
            i2=i1+1
            while (self.edges[i2].fs_state is False or 
            self.edges[i2].active is False) and \
            i2 < len(self.edges)-1:
                i2+=1
            print "i2 = ", i2
            if self.edges[i2].fs_state is True:
                distance_between_edges = self.edges[i2].edge_position() - \
                self.edges[i1].edge_position()
                if self.edges[i1].fs_emax > distance_between_edges - \
                preedge_safe_window_width :
                    if (distance_between_edges - 
                    preedge_safe_window_width) <= \
                    defaults.min_distance_between_edges_for_fine_structure:
                        print " Automatically desactivating the fine \
                        structure of edge number",i2+1,"to avoid conflicts\
                         with edge number",i1+1
                        self.edges[i2].fs_state = False
                        self.edges[i2].fslist.free = False
                        self.resolve_fine_structure(i1 = i2)
                    else:
                        new_fs_emax = distance_between_edges - \
                        preedge_safe_window_width
                        print "Automatically changing the fine structure \
                        width of edge",i1+1,"from", \
                        self.edges[i1].fs_emax, "eV to", new_fs_emax, \
                        "eV to avoid conflicts with edge number", i2+1
                        self.edges[i1].fs_emax = new_fs_emax
                        self.resolve_fine_structure(i1 = i2)
                else:
                    self.resolve_fine_structure(i1 = i2)
        else:
            return
        
    def set_coordinates(self,ix=None,iy=None):
        '''Store the current parameters in the parameter matrix and change the
         active spectrum to the given coordinates
         
         ix, iy : int
        '''
        if ix is not None or iy is not None:
            # Set the parameters before changing the coordinates
            self.set() 
        if ix is not None:
            ix = int(ix) # Convert it to int, just in case...
            if ix >= 0 :
                if ix < self.hl.xdimension :
                    self.ix = ix
                else :
                    self.ix = ix - self.hl.xdimension
            elif ix < 0:
                self.ix = self.hl.xdimension + ix

        if iy is not None:
            iy = int(iy) # Convert it to int, just in case...
            if iy >= 0 :
                if iy < self.hl.ydimension :
                    self.iy = iy
                else :
                    self.iy = iy - self.hl.ydimension
            elif iy < 0:
                self.iy = self.hl.ydimension + iy
        if ix is not None or iy is not None:
            self.charge()

    
        
    def _set_p0(self):
        p0 = None
        for component in self:
            component.refresh_free_parameters()
            if component.active:
                for param in component.free_parameters:
                    if p0 is not None:
                        p0 = (p0 + [param.value,] if not isinstance(param.value, list) \
                        else p0 + param.value)
                    else:
                        p0 = ([param.value,] if not isinstance(param.value, list) \
                        else param.value)
        self.p0 = tuple(p0)
    
    def set_boundaries(self):
        '''Generate the boundary list.
        
        Necessary before fitting with a boundary aware optimizer
        '''
        self.free_parameters_boundaries = []
        for component in self:
            component.refresh_free_parameters()
            if component.active:
                for param in component.free_parameters:
                    if param._number_of_elements == 1:
                        self.free_parameters_boundaries.append((
                        param._bounds))
                    else:
                        self.free_parameters_boundaries.extend((
                        param._bounds))

    def set(self):
        ''' Store the parameters of the current coordinates into the 
        parameters array.
        
        If the parameters array has not being defined yet it creates it filling 
        it with the current parameters.'''
        for component in self:
            component.store_current_parameters_in_map(self.ix,self.iy,
            self.hl.xdimension,self.hl.ydimension)

    def charge(self, only_fixed = False):
        '''Charge the parameters for the current spectrum from the parameters 
        array
        
        Parameters
        ----------
        only_fixed : bool
            If True, only the fixed parameters will be charged.
        '''
        for component in self :
            component.charge_value_from_map(self.ix,self.iy, only_fixed = 
            only_fixed)

    def _charge_p0(self, p_std = None):
        '''Charge the free data for the current coordinates (x,y) from the
        p0 array.
        
        Parameters
        ----------
        p_std : array
            array containing the corresponding standard deviation
        '''
        comp_p_std = None
        counter = 0
        for component in self: # Cut the parameters list
            if component.active:
                if p_std is not None:
                    comp_p_std = p_std[counter: counter + component.nfree_param]
                component.charge(
                self.p0[counter: counter + component.nfree_param], True, 
                comp_p_std)
                counter += component.nfree_param

    # Defines the functions for the fitting process -------------------------
    def __call__(self,non_convolved=False, onlyactive=False) :
        '''Returns the corresponding model for the current coordinates
        
        Parameters
        ----------
        non_convolved : bool
            If True it will return the deconvolved model
        only_active : bool
            If True, only the active components will be used to build the model.
        
        Returns
        -------
        numpy array
        '''
            
        if self.convolved is False or non_convolved is True:
            sum_ = np.zeros(len(self.hl.energy_axis))
            if onlyactive :
                for component in self: # Cut the parameters list
                    if component.active:
                        np.add(sum_, component.function(self.hl.energy_axis),
                        sum_)
                return sum_
            else :
                for component in self: # Cut the parameters list
                    np.add(sum_, component.function(self.hl.energy_axis),
                     sum_)
                return sum_

        else: # convolved
            counter = 0
            sum_convolved = np.zeros(len(self.Experiments.convolution_axis))
            sum_ = np.zeros(len(self.hl.energy_axis))
            for component in self: # Cut the parameters list
                if onlyactive :
                    if component.active:
                        if component.convolved:
                            np.add(sum_convolved,
                            component.function(self.Experiments.convolution_axis),
                            sum_convolved)
                        else:
                            np.add(sum_,
                            component.function(self.hl.energy_axis), sum_)
                        counter+=component.nfree_param
                else :
                    if component.convolved:
                        np.add(sum_convolved,
                        component.function(self.Experiments.convolution_axis),
                        sum_convolved)
                    else:
                        np.add(sum_, component.function(self.hl.energy_axis),
                        sum_)
                    counter+=component.nfree_param
            return sum_ + np.convolve(self.ll4dc[: , self.ix, self.iy],
            sum_convolved, mode="valid")


    def set_energy_region(self,E1 = None,E2= None):
        '''Use only the selected area in the fitting routine.
        
        Parameters
        ----------
        E1 : None or float
        E2 : None or float
        
        Notes
        -----
        To use the full energy range call the function without arguments.
        '''
        if E1 is not None :
            if E1 > self.hl.energy_axis[0]:
                start_index = self.hl.energy2index(E1)
            else :
                start_index = None
        else :
            start_index = None
        if E2 is not None :
            if E2 < self.hl.energy_axis[-1]:
                stop_index = self.hl.energy2index(E2)
            else :
                stop_index = None
        else:
            stop_index = None
        self.backup_channel_switches = copy.copy(self.channel_switches)
        self.channel_switches[:] = False
        self.channel_switches[start_index:stop_index] = True

    def remove_data_range(self,E1 = None,E2= None):
        '''Do not use the data in the selected range in the fitting rountine
        
        Parameters
        ----------
        E1 : None or float
        E2 : None or float
        
        Notes
        -----
        To use the full energy range call the function without arguments.
        '''
        if E1 is not None :
            start_index = self.hl.energy2index(E1)
        else :
            start_index = None
        if E2 is not None :
            stop_index = self.hl.energy2index(E2)
        else:
            stop_index = None
        self.channel_switches[start_index:stop_index] = False

    def _model_function(self,param):

        if self.convolved:
            counter = 0
            sum_convolved = np.zeros(len(self.Experiments.convolution_axis))
            sum = np.zeros(len(self.hl.energy_axis))
            for component in self: # Cut the parameters list
                if component.active:
                    if component.convolved:
                        np.add(sum_convolved, component(param[\
                        counter:counter+component.nfree_param],
                        self.Experiments.convolution_axis), sum_convolved)
                    else:
                        np.add(sum, component(param[counter:counter + \
                        component.nfree_param],self.hl.energy_axis), sum)
                    counter+=component.nfree_param

            return (sum + np.convolve(self.ll4dc[:,self.ix,self.iy],
        sum_convolved,mode="valid"))[self.channel_switches]

        else :
            counter = 0
            first = True
            for component in self: # Cut the parameters list
                if component.active:
                    if first :
                        sum=component(param[counter:counter + \
                        component.nfree_param],self.hl.energy_axis)
                        first = False
                    else:
                        sum+=component(param[counter:counter + \
                        component.nfree_param],self.hl.energy_axis)
                    counter+=component.nfree_param
            return sum[self.channel_switches]

    def _jacobian(self,param, y, weights = None):
        if self.convolved:
            counter = 0
            grad_convolved = np.zeros(len(self.Experiments.convolution_axis))
            grad = np.zeros(len(self.hl.energy_axis))
            for component in self: # Cut the parameters list
                if component.active:
                    component.charge(param[counter:counter + \
                    component.nfree_param] , onlyfree = True)
                    if component.convolved:
                        for parameter in component.free_parameters :
                            par_grad = np.convolve(
                            parameter.grad(self.Experiments.convolution_axis), 
                            self.ll4dc[:,self.ix,self.iy], 
                            mode="valid")
                            if parameter._twins:
                                for parameter in parameter._twins:
                                    np.add(par_grad, np.convolve(
                                    parameter.grad(self.Experiments.convolution_axis), 
                                    self.ll4dc[:, self.ix, self.iy], 
                                    mode="valid"), par_grad)
                            grad = np.vstack((grad, par_grad))
                        counter += component.nfree_param

                    else:
                        for parameter in component.free_parameters :
                            par_grad = parameter.grad(self.hl.energy_axis)
                            if parameter._twins:
                                for parameter in parameter._twins:
                                    np.add(par_grad, parameter.grad(
                                    self.hl.energy_axis), par_grad)
                            grad = np.vstack((grad, par_grad))
                        counter += component.nfree_param
            if weights is None:
                return grad[1:, self.channel_switches]
            else:
                return grad[1:, self.channel_switches] * weights
        else :
            counter = 0
            grad = self.hl.energy_axis
            for component in self: # Cut the parameters list
                if component.active:
                    component.charge(param[counter:counter + \
                    component.nfree_param] , onlyfree = True)
                    for parameter in component.free_parameters :
                        par_grad = parameter.grad(self.hl.energy_axis)
                        if parameter._twins:
                            for parameter in parameter._twins:
                                np.add(par_grad, parameter.grad(
                                self.hl.energy_axis), par_grad)
                        grad = np.vstack((grad, par_grad))
                    counter+=component.nfree_param
            if weights is None:
                return grad[1:,self.channel_switches]
            else:
                return grad[1:,self.channel_switches] * weights
        
    def _function4odr(self,param,x):
        return self._model_function(param)
    
    def _jacobian4odr(self,param,x):
        return self._jacobian(param, x)

    def smart_fit(self, background_fit_E1 = None, **kwards):
        ''' Fits everything in a cascade style.'''

        # Fit background
        self.fit_background(background_fit_E1, **kwards)

        # Fit the edges
        for i in range(0,len(self.edges)) :
            self.fit_edge(i,background_fit_E1, **kwards)
                
    def fit_background(self,startenergy = None, kind = 'single', **kwards):
        '''Fit an EELS spectrum ionization edge by ionization edge from left 
        to right to optimize convergence.
        
        Parameters
        ----------
        startenergy : float
        kind : {'single', 
        '''
        ea = self.hl.energy_axis[self.channel_switches]

        print "Fitting the", self.__backgroundtype, "background"
        edges = copy.copy(self.edges)
        edge = edges.pop(0)
        if startenergy is None:
            startenergy = ea[0]
        i = 0
        while edge.edge_position() < startenergy or edge.active is False:
            i+=1
            edge = edges.pop(0)
        self.set_energy_region(startenergy,edge.edge_position() - \
        defaults.preedge_safe_window_width)
        active_edges = []
        for edge in self.edges[i:]:
            if edge.active:
                active_edges.append(edge)
        self.disable_edges(active_edges)
        if kind == 'single':
            self.fit(**kwards)
        if kind == 'multi':
            self.multifit(**kwards)
        self.channel_switches = copy.copy(self.backup_channel_switches)
        self.enable_edges(active_edges)
    def two_area_background_estimation(self, E1 = None, 
    E2 = None):
        '''
        Estimates the parameters of a power law background with the two
        area method.
        '''
        ea = self.hl.energy_axis[self.channel_switches]
        if E1 is None or E1 < ea[0]:
            E1 = ea[0]
        else:
            E1 = E1
        if E2 is None:
            i = 0
            while self.edges[i].edge_position() < E1 or \
            self.edges[i].active is False:
                i += 1
            E2 = self.edges[i].edge_position() - \
            defaults.preedge_safe_window_width
        else:
            E2 = E2           
        print \
        "Estimating the parameters of the background by the two area method"
        print "E1 = %s\t E2 = %s" % (E1, E2)
        bg = self.__background_components[0]
        bg.A.already_set_map = np.ones((self.hl.xdimension, 
        self.hl.ydimension))
        bg.r.already_set_map = np.ones((self.hl.xdimension, 
        self.hl.ydimension))
        estimation = two_area_powerlaw_estimation(self.hl, E1, E2)
        bg.r.map = estimation['r']
        bg.A.map = estimation['A']
        bg.charge_value_from_map(self.ix,self.iy)

    def fit_edge(self,edgenumber,startenergy = None, **kwards):
        backup_channel_switches = self.channel_switches.copy()
        ea = self.hl.energy_axis[self.channel_switches]
        if startenergy is None:
            startenergy = ea[0]
        preedge_safe_window_width = defaults.preedge_safe_window_width
        # Declare variables
        edge = self.edges[edgenumber]
        if edge.intensity.twin is not None or edge.active is False or \
        edge.edge_position() < startenergy or edge.edge_position() > ea[-1]:
            return 1
        print "Fitting edge ", edge.name 
        edgeenergy = edge.edge_position()
        last_index = len(self.edges) - 1
        i = 1
        twins = []
        print "Last edge index", last_index
        while edgenumber + i <= last_index and (
        self.edges[edgenumber+i].intensity.twin is not None or 
        self.edges[edgenumber+i].active is False):
            if self.edges[edgenumber+i].intensity.twin is not None:
                twins.append(self.edges[edgenumber+i])
            i+=1
        print "twins", twins
        print "next_edge_index", edgenumber + i
        if  (edgenumber + i) > last_index:
            nextedgeenergy = ea[-1]
        else:
            nextedgeenergy = self.edges[edgenumber+i].edge_position() - \
            preedge_safe_window_width

        # Backup the fsstate
        to_activate_fs = []
        for edge_ in [edge,] + twins:
            if edge_.fs_state is True and edge_.fslist.free is True:
                to_activate_fs.append(edge_)
        self.disable_fine_structure(to_activate_fs)
        
        # Smart Fitting

        print("Fitting region: %s-%s" % (startenergy,nextedgeenergy))

        # Without fine structure to determine delta
        edges_to_activate = []
        for edge_ in self.edges[edgenumber+1:]:
            if edge_.active is True and edge_.edge_position() >= nextedgeenergy:
                edge_.active = False
                edges_to_activate.append(edge_)
        print "edges_to_activate", edges_to_activate
        print "Fine structure to fit", to_activate_fs
        
        self.set_energy_region(startenergy, nextedgeenergy)
        if edge.freedelta is True:
            print "Fit without fine structure, delta free"
            edge.delta.free = True
            self.fit(**kwards)
            edge.delta.free = False
            print "delta = ", edge.delta.value
            self.touch()
        elif edge.intensity.free is True:
            print "Fit without fine structure"
            self.enable_fine_structure(to_activate_fs)
            self.remove_fine_structure_data(to_activate_fs)
            self.disable_fine_structure(to_activate_fs)
            self.fit(**kwards)

        if len(to_activate_fs) > 0:
            self.set_energy_region(startenergy, nextedgeenergy)
            self.enable_fine_structure(to_activate_fs)
            print "Fit with fine structure"
            self.fit(**kwards)
            
        self.enable_edges(edges_to_activate)
        # Recover the channel_switches. Remove it or make it smarter.
        self.channel_switches = backup_channel_switches

    def multifit(self, background_fit_E1 = None, mask = None,
     order = "normal", kind = "normal", fitter = "leastsq", 
     charge_only_fixed = False, grad = False, autosave = "pixel", **kwargs) :
        if autosave is not None:
            fd, autosave_fn = tempfile.mkstemp(prefix = 'eelslab_autosave-', 
            dir = '.', suffix = '.par')
            os.close(fd)
            autosave_fn = autosave_fn[:-4]
            text = "Autosaving each %s in file: %s.par" % (autosave, autosave_fn)
            messages.information(text)
            text = "When multifit finishes its job the file will be deleted"
            messages.warning(text)
        if mask is not None and \
        (np.shape(mask) != (self.hl.xdimension,self.hl.ydimension)):
            print " The mask must be an array with the same espatial \
            dimensions as the data cube"
            return 0
        if order == "normal":
            for y in np.arange(0,self.hl.ydimension) :
                for x in np.arange(0,self.hl.xdimension) :
                    if mask is None or mask[x,y] :
                        self.ix = x
                        self.iy = y
                        self.charge(only_fixed=charge_only_fixed)
                        print '-'*40
                        print "Fitting x=",self.ix," y=",self.iy
                        if kind  == "smart" :
                            self.smart_fit(background_fit_E1 = None,
                             fitter = fitter, **kwargs)
                        elif kind == "normal" :
                            self.fit(fitter = fitter, grad = grad, **kwargs)
                        if autosave == 'pixel':
                            self.save_parameters2file(autosave_fn)
                    if autosave == 'row':
                            self.save_parameters2file(autosave_fn)

        elif order == "zigzag":
            inverter = 1
            for x in range(0,self.hl.xdimension):
                inverter*=-1
                if inverter == 1 :
                    for y in range(0,self.hl.ydimension) :
                        if mask[x,y] is True:
                            self.ix = x
                            self.iy = y
                            print '-'*40
                            print "Fitting x=",self.ix," y=",self.iy
                            if kind  == "smart" :
                                self.smart_fit(background_fit_E1
                                = None, fitter = fitter)
                            elif kind == "normal" :
                                self.fit(fitter = fitter)
                if inverter == -1 :
                    for y in range(self.hl.ydimension - 1, 0, -1):
                        if mask[x,y] is True:
                            self.ix = x
                            self.iy = y
                            if kind  == "smart" :
                                self.smart_fit(background_fit_E1
                                = None, fitter = fitter)
                            elif kind == "normal" :
                                self.fit(fitter = fitter)
        messages.information(
        'Removing the temporary file %s' % (autosave_fn + 'par'))
        os.remove(autosave_fn + '.par')

    def generate_chisq(self, degrees_of_freedom = 'auto') :
        # Generate the model_cube if it doesn't exist
        if self.hl.variance is None:
            self.hl.estimate_variance()
        variance = self.hl.variance[self.channel_switches]
        differences = (self.model_cube - self.hl.data_cube)[self.channel_switches]
        self.chisq = np.sum(differences**2 / variance, 0)
        if degrees_of_freedom == 'auto' :
            self.red_chisq = self.chisq / \
            (np.sum(np.ones(self.hl.energydimension)[self.channel_switches]) \
            - len(self.p0) -1)
            print "Degrees of freedom set to auto"
            print "DoF = ", len(self.p0)
        elif type(degrees_of_freedom) is int :
            self.red_chisq = self.chisq / \
            (np.sum(np.ones(self.hl.energydimension)[self.channel_switches]) \
            - degrees_of_freedom -1)
        else:
            print "degrees_of_freedom must be on interger type."
            print "The red_chisq could not been calculated"
            
    def save(self, filename):
        for component in self:
            pass

    def save_parameters2file(self,filename):
        '''Save the parameters array in binary format'''

        for component in self:
            for param in component.parameters:
                try:
                    parameters_array=np.concatenate((parameters_array,
                    np.atleast_3d(param.map)),2)
                except:
                    parameters_array=np.atleast_3d(param.map)
        parameters_array.dump(filename+'.par')

    def load_parameters_from_file(self,filename):
        '''Loads the parameters array from  a binary file written with the
        'save_parameters2file' function'''
        
        parameters_array = np.load(filename)
        
        for component in self: # Cut the parameters list
            component.update_number_parameters()
            component.charge2map(parameters_array[:, :,
            counter:counter+component.nparam])
            counter+=component.nparam
        if parameters_array.shape[2] != counter:
            print "\nWarning, the number of parameters != number of parameters in file"
            print "%s != %s" % (parameters_array.shape[2], counter)
        print "\n%s parameters charged from %s" % (counter, filename)
        self.charge()
    
    def quantify(self):
        elements = {}
        for edge in self.edges:
            if edge.active and edge.intensity.twin is None:
                element = edge._Edge__element
                subshell = edge._Edge__subshell
                if element not in elements:
                    elements[element] = {}
                elements[element][subshell] = edge.intensity.value
        # Print absolute quantification
        print
        print "Absolute quantification:"
        print "Elem.\tAreal density (atoms/nm**2)"
        for element in elements:
            if len(elements[element]) == 1:
                for subshell in elements[element]:
                    print "%s\t%f" % (element, elements[element][subshell])
            else:
                for subshell in elements[element]:
                    print "%s_%s\t%f" % (element, subshell, 
                    elements[element][subshell])
