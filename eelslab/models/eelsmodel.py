#!/usr/bin/python
import copy

import numpy as np

from eelslab.model import Model
from eelslab.components.eels_cl_edge import EELSCLEdge
from eelslab.components import PowerLaw
from eelslab.misc.interactive_ns import interactive_ns
from eelslab.defaults_parser import defaults
import eelslab.messages as messages
import eelslab.misc.utils as utils


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
    
    def __init__(self, auto_background = True, auto_add_edges = True, 
                 *args, **kwargs):
        Model.__init__(self, *args, **kwargs)
        if auto_background is True:
            bg = PowerLaw()
            interactive_ns['bg'] = bg
            self.append(bg)
        if self.ll is not None:
            self.convolved = True
            if self.experiments.convolution_axis is None:
                self.experiments.set_convolution_axis()
        else:
            self.convolved = False
        if self.spectrum.edges and auto_add_edges is True:
            self.extend(self.spectrum.edges)
            
    def _touch(self):
        """Run model setup tasks
        
        This function must be called everytime that we add or remove components
        from the model.
        It creates the bookmarks self.edges and sef.__background_components and 
        configures the edges by setting the dispersion attribute and setting 
        the fine structure.
        """
        self.edges = []
        self.__background_components = []
        for component in self:
            if isinstance(component,EELSCLEdge):
                component.dispersion = self.spectrum.energyscale
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
                    self.__background_components.append(component)

            elif isinstance(component,PowerLaw) or component.isbackground is True:
                self.__background_components.append(component)

        if not self.edges:
            messages.warning("The model contains no edges")
        else:
            self.edges.sort(key = EELSCLEdge.edge_position)
            self.resolve_fine_structure()
        if len(self.__background_components) > 1 :
            self.__backgroundtype = "mix"
        elif not self.__background_components:
            messages.warning("No background model has been defined")
        else :
            self.__backgroundtype = \
            self.__background_components[0].__repr__()
            if self.__firstimetouch and self.edges:
                self.two_area_background_estimation()
                self.__firstimetouch = False
        
    def generate_edges(self, e_shells, copy2interactive_ns = True):
        """Create the Edge instances and configure them appropiately
        Parameters
        ----------
        e_shells : list of strings
        copy2interactive_ns : bool
            If True, variables with the format Element_Shell will be created in
            IPython's interactive shell
        """
        
        e_shells.sort()
        master_edge = EELSCLEdge(e_shells.pop())
        self.edges.append(master_edge)
        interactive_ns[self.edges[-1].__repr__()] = self.edges[-1]
        element = self.edges[-1].__repr__().split('_')[0]
        interactive_ns[element] = []
        interactive_ns[element].append(self.edges[-1])
        while len(e_shells) > 0:
            self.edges.append(EELSCLEdge(e_shells.pop()))
            self.edges[-1].intensity.twin = master_edge.intensity
            self.edges[-1].delta.twin = master_edge.delta
            self.edges[-1].freedelta = False
            if copy2interactive_ns is True:
                interactive_ns[self.edges[-1].__repr__()] = self.edges[-1]
                interactive_ns[element].append(self.edges[-1])
                
    def resolve_fine_structure(self,preedge_safe_window_width = 
        defaults.preedge_safe_window_width, i1 = 0):
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

        print "Fitting the", self.__backgroundtype, "background"
        edges = copy.copy(self.edges)
        edge = edges.pop(0)
        if startenergy is None:
            startenergy = ea[0]
        i = 0
        while edge.edge_position() < startenergy or edge.active is False:
            i+=1
            edge = edges.pop(0)
        self.set_data_region(startenergy,edge.edge_position() - \
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
        
    def two_area_background_estimation(self, E1 = None, E2 = None):
        """
        Estimates the parameters of a power law background with the two
        area method.
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
                defaults.preedge_safe_window_width
            else:
                E2 = ea[-1]
        else:
            E2 = E2           
        print \
        "Estimating the parameters of the background by the two area method"
        print "E1 = %s\t E2 = %s" % (E1, E2)

        try:
            estimation = utils.two_area_powerlaw_estimation(self.spectrum, E1, E2)
            bg = self.__background_components[0]
            bg.A.already_set_map = np.ones(
                (self.spectrum.xdimension,self.spectrum.ydimension))
            bg.r.already_set_map = np.ones(
                (self.spectrum.xdimension, self.spectrum.ydimension))
            bg.r.map = estimation['r']
            bg.A.map = estimation['A']
            bg.charge_value_from_map(self.coordinates.ix,self.coordinates.iy)
        except ValueError:
            messages.warning(
            "The power law background parameters could not be estimated\n"
            "Try choosing an energy range for the estimation")

    def fit_edge(self, edgenumber, startenergy = None, **kwards):
        backup_channel_switches = self.channel_switches.copy()
        ea = self.axis.axis[self.channel_switches]
        if startenergy is None:
            startenergy = ea[0]
        preedge_safe_window_width = defaults.preedge_safe_window_width
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
        
        self.set_data_region(startenergy, nextedgeenergy)
        if edge.freedelta is True:
            print "Fit without fine structure, delta free"
            edge.delta.free = True
            self.fit(**kwards)
            edge.delta.free = False
            print "delta = ", edge.delta.value
            self._touch()
        elif edge.intensity.free is True:
            print "Fit without fine structure"
            self.enable_fine_structure(to_activate_fs)
            self.remove_fine_structure_data(to_activate_fs)
            self.disable_fine_structure(to_activate_fs)
            self.fit(**kwards)

        if len(to_activate_fs) > 0:
            self.set_data_region(startenergy, nextedgeenergy)
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
                self.remove_data_range(start,stop)
       
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
        for component in self.__background_components:
            component.active = True
    def disable_background(self):
        """
        Disable the background.
        """
        for component in self.__background_components:
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
