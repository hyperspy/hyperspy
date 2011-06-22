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


class Controls:
    """
    """
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
    def _enable_ext_bounding(self,components = None):
        """
        """
        if components is None :
            components = self
        for component in components:
            for parameter in component.parameters:
                parameter.ext_bounded = True
    def _disable_ext_bounding(self,components = None):
        """
        """
        if components is None :
            components = self
        for component in components:
            for parameter in component.parameters:
                parameter.ext_bounded = False
                
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
    def set_current_values_to(self, components_list = None, mask = None):
        if components_list is None:
            components_list = []
            for comp in self:
                if comp.active:
                    components_list.append(comp)
        for comp in components_list:
            for parameter in comp.parameters:
                parameter.set_current_value_to(mask = mask)
                
