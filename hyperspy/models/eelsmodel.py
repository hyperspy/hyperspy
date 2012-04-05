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

import copy

import numpy as np
import traits.api as t

from hyperspy.model import Model
from hyperspy.components.eels_cl_edge import EELSCLEdge
from hyperspy.components import PowerLaw
from hyperspy.misc.interactive_ns import interactive_ns
from hyperspy.defaults_parser import preferences
import hyperspy.messages as messages
from hyperspy import components
from hyperspy.decorators import only_interactive
from hyperspy.exceptions import MissingParametersError
from hyperspy.signals.eels import EELSSpectrum
from hyperspy.gui.eels import TEMParametersUI
import hyperspy.gui.messages as messagesui


class EELSModel(Model):
    """Build a fit a model
    
    Parameters
    ----------
    spectrum : an Spectrum (or any Spectrum subclass) instance
    auto_background : boolean
        If True, and if spectrum is an EELS instance adds automatically a powerlaw to the model and estimate the 
        parameters by the two-area method.
    auto_add_edges : boolean
        If True, and if spectrum is an EELS instance, it will automatically add the ionization edges as 
        defined in the Spectrum instance.
    """
    
    def __init__(self, spectrum, auto_background = True, auto_add_edges = True, 
                 ll = None, *args, **kwargs):
        Model.__init__(self, spectrum, *args, **kwargs)
        self.ll = ll
        
        if auto_background is True:
            background = PowerLaw()
            background.name = 'background'
            interactive_ns['background'] = background
            self.append(background)
        if self.ll is not None:
            self.convolved = True
            if self.experiments.convolution_axis is None:
                self.experiments.set_convolution_axis()
        else:
            self.convolved = False
        if self.spectrum.subshells and auto_add_edges is True:
            self._add_edges_from_subshells_names()
            
    @property
    def spectrum(self):
        return self._spectrum
                
    @spectrum.setter
    def spectrum(self, value):
        if isinstance(value, EELSSpectrum):
            self._spectrum = value
            self.check_eels_parameters()
        else:
            raise WrongObjectError(str(type(value)), 'EELSSpectrum')
            
    def check_eels_parameters(self):
        must_exist = (
            'TEM.convergence_angle', 
            'TEM.beam_energy',
            'TEM.EELS.collection_angle',)
        missing_parameters = []
        for item in must_exist:
            exists = self.spectrum.mapped_parameters.has_item(item)
            if exists is False:
                missing_parameters.append(item)
        if missing_parameters:
            if preferences.General.interactive is True:
                par_str = "The following parameters are missing:\n"
                for par in missing_parameters:
                    par_str += '%s\n' % par
                par_str += 'Please set them in the following wizard'
                is_ok = messagesui.information(par_str)
                if is_ok:
                    self.define_eels_parameters()
                else:
                    raise MissingParametersError(missing_parameters)
            else:
                raise MissingParametersError(missing_parameters)
    
    @only_interactive            
    def define_eels_parameters(self, defined_parameters = None):
        if self.spectrum.mapped_parameters.has_item('TEM') is False:
            self.spectrum.mapped_parameters.add_node('TEM')
        if self.spectrum.mapped_parameters.has_item('TEM.EELS') is False:
            self.spectrum.mapped_parameters.TEM.add_node('EELS')
        tem_par = TEMParametersUI()
        mapping = {
            'TEM.convergence_angle' : 'tem_par.convergence_angle',
            'TEM.beam_energy' : 'tem_par.beam_energy',
            'TEM.EELS.collection_angle' : 'tem_par.collection_angle',}
        for key, value in mapping.iteritems():
            if self.spectrum.mapped_parameters.has_item(key):
                exec('%s = self.spectrum.mapped_parameters.%s' % (value, key))
        tem_par.edit_traits()
        mapping = {
            'TEM.convergence_angle' : tem_par.convergence_angle,
            'TEM.beam_energy' : tem_par.beam_energy,
            'TEM.EELS.collection_angle' : tem_par.collection_angle,}
        for key, value in mapping.iteritems():
            if value != t.Undefined:
                exec('self.spectrum.mapped_parameters.%s = %s' % (key, value))
        self.check_eels_parameters()
        
            
    def _touch(self):
        """Run model setup tasks
        
        This function must be called everytime that we add or remove components
        from the model.
        It creates the bookmarks self.edges and sef._background_components and 
        configures the edges by setting the energy_scale attribute and setting 
        the fine structure.
        """
        self._Model__touch()
        self.edges = []
        self._background_components = []
        for component in self:
            if isinstance(component,EELSCLEdge):
                component.set_microscope_parameters(
                E0 = self.spectrum.mapped_parameters.TEM.beam_energy, 
                alpha = self.spectrum.mapped_parameters.TEM.convergence_angle,
                beta = self.spectrum.mapped_parameters.TEM.EELS.collection_angle, 
                energy_scale = self.axis.scale)
                component.energy_scale = self.axis.scale
                component.setfslist()
                if component.edge_position() < \
                self.axis.axis[self.channel_switches][0]:
                    component.isbackground = True
                if component.isbackground is not True:
                    self.edges.append(component)
                else :
                    component.fs_state = False
                    component.fslist.free = False
                    component.backgroundtype = "edge"
                    self._background_components.append(component)

            elif isinstance(component,PowerLaw) or component.isbackground is True:
                self._background_components.append(component)

        if not self.edges:
            messages.warning("The model contains no edges")
        else:
            self.edges.sort(key = EELSCLEdge.edge_position)
            self.resolve_fine_structure()
        if len(self._background_components) > 1 :
            self._backgroundtype = "mix"
        elif not self._background_components:
            messages.warning("No background model has been defined")
        else :
            self._backgroundtype = \
            self._background_components[0].__repr__()
            if self._firstimetouch and self.edges:
                self.two_area_background_estimation()
                self._firstimetouch = False
        
    def _add_edges_from_subshells_names(self, e_shells = None, 
                                        copy2interactive_ns = True):
        """Create the Edge instances and configure them appropiately
        Parameters
        ----------
        e_shells : list of strings
        copy2interactive_ns : bool
            If True, variables with the format Element_Shell will be created in
            IPython's interactive shell
        """
        if e_shells is None:
            e_shells = list(self.spectrum.subshells)
        e_shells.sort()
        master_edge = EELSCLEdge(e_shells.pop())
        self.append(master_edge)
        interactive_ns[self[-1].__repr__()] = self[-1]
        element = self[-1].__repr__().split('_')[0]
        interactive_ns[element] = []
        interactive_ns[element].append(self[-1])
        while len(e_shells) > 0:
            next_element = e_shells[-1].split('_')[0]
            if next_element != element:
                self._add_edges_from_subshells_names(e_shells = e_shells)
            else:
                self.append(EELSCLEdge(e_shells.pop()))
                self[-1].intensity.twin = master_edge.intensity
                self[-1].delta.twin = master_edge.delta
                self[-1].freedelta = False
                if copy2interactive_ns is True:
                    interactive_ns[self[-1].__repr__()] = self[-1]
                    interactive_ns[element].append(self[-1])
                
    def resolve_fine_structure(self,preedge_safe_window_width = 
        preferences.EELS.preedge_safe_window_width, i1 = 0):
        """Adjust the fine structure of all edges to avoid overlapping
        
        This function is called automatically everytime the position of an edge
        changes
        
        Parameters
        ----------
        preedge_safe_window_width : float
            minimum distance between the fine structure of an ionization edge 
            and that of the following one.
        """

        while (self.edges[i1].fs_state is False or  
        self.edges[i1].active is False) and i1 < len(self.edges)-1 :
            i1+=1
        if i1 < len(self.edges)-1 :
            i2=i1+1
            while (self.edges[i2].fs_state is False or 
            self.edges[i2].active is False) and \
            i2 < len(self.edges)-1:
                i2+=1
            if self.edges[i2].fs_state is True:
                distance_between_edges = self.edges[i2].edge_position() - \
                self.edges[i1].edge_position()
                if self.edges[i1].fs_emax > distance_between_edges - \
                preedge_safe_window_width :
                    if (distance_between_edges - 
                    preedge_safe_window_width) <= \
                    preferences.EELS.min_distance_between_edges_for_fine_structure:
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
            
    def fit(self, *args, **kwargs):
        if 'kind' in kwargs and kwargs['kind'] == 'smart':
            self.smart_fit(*args, **kwargs)
        else:
            Model.fit(self, *args, **kwargs)
            
    def smart_fit(self, background_fit_E1 = None, **kwards):
        """ Fits everything in a cascade style."""

        # Fit background
        self.fit_background(background_fit_E1, **kwards)

        # Fit the edges
        for i in xrange(0,len(self.edges)) :
            self.fit_edge(i, background_fit_E1, **kwards)
            
    def fit_background(self,startenergy = None, kind = 'single', **kwards):
        """Fit an EELS spectrum ionization edge by ionization edge from left 
        to right to optimize convergence.
        
        Parameters
        ----------
        startenergy : float
        kind : {'single', 
        """
        ea = self.axis.axis[self.channel_switches]

        print "Fitting the", self._backgroundtype, "background"
        edges = copy.copy(self.edges)
        edge = edges.pop(0)
        if startenergy is None:
            startenergy = ea[0]
        i = 0
        while edge.edge_position() < startenergy or edge.active is False:
            i+=1
            edge = edges.pop(0)
        self.set_data_range_in_units(startenergy,edge.edge_position() - \
        preferences.EELS.preedge_safe_window_width)
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
        
    def two_area_background_estimation(self, E1 = None, E2 = None, 
        powerlaw = None):
        """Estimates the parameters of a power law background with the two
        area method.
        
        Parameters
        ----------
        E1 : float
        E2 : float
        powerlaw : PowerLaw component or None
            If None, it will try to guess the right component from the 
            background components of the model
        """
        ea = self.axis.axis[self.channel_switches]
        if E1 is None or E1 < ea[0]:
            E1 = ea[0]
        else:
            E1 = E1
        if E2 is None:
            if self.edges:
                i = 0
                while self.edges[i].edge_position() < E1 or \
                self.edges[i].active is False:
                    i += 1
                E2 = self.edges[i].edge_position() - \
                preferences.EELS.preedge_safe_window_width
            else:
                E2 = ea[-1]
        else:
            E2 = E2           
        print \
        "Estimating the parameters of the background by the two area method"
        print "E1 = %s\t E2 = %s" % (E1, E2)
        if powerlaw is None:
            for component in self._background_components:
                if isinstance(component, components.PowerLaw):
                    if powerlaw is None:
                        powerlaw = component
                    else:
                        message.warning('There are more than two power law '
                        'background components defined in this model, please '
                        'use the powerlaw keyword to specify one of them')
                        return
                        
        
        if powerlaw.estimate_parameters(self.spectrum, E1, E2, False) is True:
            self.charge()
        else:
            messages.warning(
            "The power law background parameters could not be estimated\n"
            "Try choosing a different energy range for the estimation")
            return

    def fit_edge(self, edgenumber, startenergy = None, **kwards):
        backup_channel_switches = self.channel_switches.copy()
        ea = self.axis.axis[self.channel_switches]
        if startenergy is None:
            startenergy = ea[0]
        preedge_safe_window_width = preferences.EELS.preedge_safe_window_width
        # Declare variables
        edge = self.edges[edgenumber]
        if edge.intensity.twin is not None or edge.active is False or \
        edge.edge_position() < startenergy or edge.edge_position() > ea[-1]:
            return 1
        print "Fitting edge ", edge.name 
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
        
        self.set_data_range_in_units(startenergy, nextedgeenergy)
        if edge.freedelta is True:
            print "Fit without fine structure, delta free"
            edge.delta.free = True
            self.fit(**kwards)
            edge.delta.free = False
            print "delta = ", edge.delta.value
            self.__touch()
        elif edge.intensity.free is True:
            print "Fit without fine structure"
            self.enable_fine_structure(to_activate_fs)
            self.remove_fine_structure_data(to_activate_fs)
            self.disable_fine_structure(to_activate_fs)
            self.fit(**kwards)

        if len(to_activate_fs) > 0:
            self.set_data_range_in_units(startenergy, nextedgeenergy)
            self.enable_fine_structure(to_activate_fs)
            print "Fit with fine structure"
            self.fit(**kwards)
            
        self.enable_edges(edges_to_activate)
        # Recover the channel_switches. Remove it or make it smarter.
        self.channel_switches = backup_channel_switches
        
    def quantify(self):
        elements = {}
        for edge in self.edges:
            if edge.active and edge.intensity.twin is None:
                element = edge._EELSCLEdge__element
                subshell = edge._EELSCLEdge__subshell
                if element not in elements:
                    elements[element] = {}
                elements[element][subshell] = edge.intensity.value
        # Print absolute quantification
        print
        print "Absolute quantification:"
        print "Elem.\tIntensity"
        for element in elements:
            if len(elements[element]) == 1:
                for subshell in elements[element]:
                    print "%s\t%f" % (element, elements[element][subshell])
            else:
                for subshell in elements[element]:
                    print "%s_%s\t%f" % (element, subshell, 
                    elements[element][subshell])
                    
    def remove_fine_structure_data(self, edges_list = None):
        """
        Remove the fine structure data from the fitting routine as defined in 
        the fs_emax parameter of each edge
        """
        if edges_list is None:
            edges_list = self.edges
        for edge in edges_list :
            if edge.isbackground is False and edge.fs_state is True:
                start = edge.edgeenergy + edge.delta.value
                stop = start + edge.fs_emax
                self.remove_data_range_in_units(start,stop)
       
    def enable_edges(self,edges_list = None):
        """
        Enable the edges listed in edges_list. If edges_list is None (default)
        all the edges with onset in the spectrum energy region will be enabled.
        """
        if edges_list is None :
            edges_list = self.edges
        for edge in edges_list :
            if edge.isbackground is False:
                edge.active = True
    def disable_edges(self,edges_list = None):
        """
        Disable the edges listed in edges_list. If edges_list is None (default)
        all the edges with onset in the spectrum energy region will be
        disabled.
        """
        if edges_list is None :
            edges_list = self.edges
        for edge in edges_list :
            if edge.isbackground is False:
                edge.active = False

    def enable_background(self):
        """
        Enable the background.
        """
        for component in self._background_components:
            component.active = True
    def disable_background(self):
        """
        Disable the background.
        """
        for component in self._background_components:
            component.active = False

    def enable_fine_structure(self,edges_list = None):
        """
        Enable the fine structure of the edges listed in edges_list.
        If edges_list is None (default) the fine structure of all the edges
        with onset in the spectrum energy region will be enabled.
        """
        if edges_list is None :
            edges_list = self.edges
        for edge in edges_list :
            if edge.isbackground is False:
                edge.fs_state = True
                edge.fslist.free = True
    def disable_fine_structure(self,edges_list = None):
        """
        Disable the fine structure of the edges listed in edges_list.
        If edges_list is None (default) the fine structure of all the edges
        with onset in the spectrum energy region will be disabled.
        """
        if edges_list is None :
            edges_list = self.edges
        for edge in edges_list :
            if edge.isbackground is False:
                edge.fs_state = False
                edge.fslist.free = False
                
    def set_all_edges_intensities_positive(self):
        """
        """

        for edge in self.edges:
            edge.intensity.ext_force_positive = True
            edge.intensity.ext_bounded = True
            
    def unset_all_edges_intensities_positive(self):
        """
        """

        for edge in self.edges:
            edge.intensity.ext_force_positive = False
            edge.intensity.ext_bounded = False
            
    def enable_freedelta(self,edges_list = None):
        """
        Enable the automatic unfixing of the delta parameter during a
        smart fit for the edges listed in edges_list.
        If edges_list is None (default) the delta of all the edges
        with onset in the spectrum energy region will be unfixed.
        """
        if edges_list is None :
            edges_list = self.edges
        for edge in edges_list :
            if edge.isbackground is False:
                edge.freedelta = True
    def disable_freedelta(self,edges_list = None):
        """
        Disable the automatic unfixing of the delta parameter during a
        smart fit for the edges listed in edges_list.
        If edges_list is None (default) the delta of all the edges
        with onset in the spectrum energy region will not be unfixed.
        Note that if their atribute edge.delta.free is True, the parameter
        will be free during the smart fit.
        """
        if edges_list is None :
            edges_list = self.edges
        for edge in edges_list :
            if edge.isbackground is False:
                edge.freedelta = True

    def fix_edges(self,edges_list = None):
        """
        Fixes all the parameters of the edges given in edges_list.
        If edges_list is None (default) all the edges will be fixed.
        """
        if edges_list is None :
            edges_list = self.edges
        for edge in edges_list :
            if edge.isbackground is False:
                edge.intensity.free = False
                edge.delta.free = False
                edge.fslist.free = False

    def unfix_edges(self,edges_list = None):
        """
        Unfixes all the parameters of the edges given in edges_list.
        If edges_list is None (default) all the edges will be unfixed.
        """
        if edges_list is None :
            edges_list = self.edges
        for edge in edges_list :
            if edge.isbackground is False:
                edge.intensity.free = True
                #edge.delta.free = True
                #edge.fslist.free = True
                
    def fix_fine_structure(self,edges_list = None):
        """
        Fixes all the parameters of the edges given in edges_list.
        If edges_list is None (default) all the edges will be fixed.
        """
        if edges_list is None :
            edges_list = self.edges
        for edge in edges_list :
            if edge.isbackground is False:
                edge.fslist.free = False

    def unfix_fine_structure(self,edges_list = None):
        """
        Unfixes all the parameters of the edges given in edges_list.
        If edges_list is None (default) all the edges will be unfixed.
        """
        if edges_list is None :
            edges_list = self.edges
        for edge in edges_list :
            if edge.isbackground is False:
                edge.fslist.free = True
