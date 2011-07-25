#!/usr/bin/python

from eelslab.model import Model
from eelslab.components.edge import Edge
from eelslab.interactive_ns import interactive_ns


class EELSModel(Model):
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
        master_edge = Edge(e_shells.pop())
        self.edges.append(master_edge)
        interactive_ns[self.edges[-1].__repr__()] = self.edges[-1]
        element = self.edges[-1].__repr__().split('_')[0]
        interactive_ns[element] = []
        interactive_ns[element].append(self.edges[-1])
        while len(e_shells) > 0:
            self.edges.append(Edge(e_shells.pop()))
            self.edges[-1].intensity.twin = master_edge.intensity
            self.edges[-1].delta.twin = master_edge.delta
            self.edges[-1].freedelta = False
            if copy2interactive_ns is True:
                interactive_ns[self.edges[-1].__repr__()] = self.edges[-1]
                interactive_ns[element].append(self.edges[-1])
