# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import copy
import itertools

import numpy as np
from hyperspy.utils import markers


class SpectrumLabelPosition():
    '''
    A class to get the positions of labels in spectrums such as EELS, EDX,
    XRF. The main method is the get_markers method which the user supplies a
    dictionary of edge labels specifying its energy positions, and it returns
    markers for labelling them.
    '''

    edge_label_style = {'ha' : 'center', 'va' : 'center',
                        'bbox' : dict(facecolor='white', alpha=0.2)}
    colour_list_label = ['black', 'darkblue', 'darkgreen',
                         'darkcyan', 'darkmagenta', 'dimgray',
                         'brown', 'deeppink', 'olive',
                         'crimson']

    def __init__(self, signal):
        self.signal = signal
        self.axis = self.signal.axes_manager.signal_axes[0]
        self._ele_col_dict = {}
        self.color_cycle = itertools.cycle(self.colour_list_label)

        self._set_active_figure_properties()

    def _set_active_figure_properties(self):
        # set the properties which depend on the figure
        self.figure = self.signal._plot.signal_plot.figure
        self.figsize = self.figure.get_size_inches()
        self.smin, self.smax = self.signal._plot.signal_plot.ax.get_ylim()

        self.sig_index = self._get_current_signal_index()
        self.text_width, self.text_height = self._estimate_textbox_dimension()

    def _get_current_signal_index(self):
        # if it is a hyperspectrum, get the correct active figure
        if self.signal._plot.pointer is not None:
            sig_index = self.signal._plot.pointer.indices[0]
        else:
            sig_index = 0

        return sig_index

    def _check_signal_figure_changed(self):
        # check if the spectrum is changed
        # reset its properties if changed
        current_sig_index = self._get_current_signal_index()
        current_figsize = self.figure.get_size_inches()
        if (current_sig_index != self.sig_index) or \
            not np.isclose(current_figsize, self.figsize).all():
            return True
        else:
            return False

    def _get_bbox_from_text_artist(self, text):
        """
        Parameters
        ----------
        text : matplotlib.text.Text instance

        """
        # get the bbox object of the textbox
        ax = text.axes
        fig = text.figure
        r = text.figure.canvas.get_renderer()
        fig.draw(r)
        extent = text.get_bbox_patch().get_window_extent()
        bbox_patch = extent.transformed(ax.transData.inverted())

        return bbox_patch

    def _estimate_textbox_dimension(self, dummy_text='My_g8'):
        # get the dimension of a typical textbox in the current figure
        dummy_style = copy.deepcopy(self.edge_label_style)
        dummy_style['bbox']['alpha'] = 0
        dummy_style['alpha'] = 0
        tx = markers.text(
            x=(self.axis.low_value+self.axis.high_value)/2,
            y=(self.smin+self.smax)/2,
            text=self._text_parser(dummy_text),
            **dummy_style
            )
        self.signal.add_marker(tx)

        dummybb = self._get_bbox_from_text_artist(tx.marker)
        tx.close()

        text_width = dummybb.width
        text_height = dummybb.height

        return text_width, text_height

    def get_markers(self, labels):
        '''Get the markers (vertical line segment and text box) for labelling
        the edges

        Parameters
        ----------
        labels : dictionary
            A dictionary with the labels as keys and their energies as values.
            E.g. for EELS edges it could be {'Mn_L2': 651.0, 'Cr_L3': 575.0}.

        Returns
        -------
        vls : list
            A list contains HyperSpy's vertical line segment marker
        txs : list
            A list contains HyperSpy's text marker
        '''

        xytext = self._get_textbox_pos(labels)

        vls = []
        txs = []
        for xyt in xytext:
            vl = markers.vertical_line_segment(
                x=xyt[0],
                y1=xyt[1],
                y2=xyt[2],
                color=xyt[4]
                )
            tx = markers.text(
                x=xyt[0],
                y=xyt[1],
                text=self._text_parser(xyt[3]), color=xyt[4],
                **self.edge_label_style
                )

            vl.events.closed.connect(self.signal._edge_marker_closed)
            tx.events.closed.connect(self.signal._edge_marker_closed)
            vl.auto_update = True
            tx.auto_update = True

            vls.append(vl)
            txs.append(tx)

        return vls, txs

    def _get_textbox_pos(self, edges, offset=None, step=None, lb=None,
                         ub=None):
        # get the information on placing the textbox and its properties
        if offset is None:
            offset = self.text_height
        if step is None:
            step = self.text_height
        if lb is None:
            lb = self.smin + offset
        if ub is None:
            ub = self.smax - offset

        if not self._ele_col_dict:
            self._ele_col_dict = self._element_colour_dict(edges)

        mid = (self.smax + self.smin) / 2
        itop = 1
        ibtm = 1

        xytext = []
        for edge in edges:
            energy = edges[edge]

            yval = self.signal.isig[float(energy)].data[self.sig_index]
            if yval <= mid: # from top
                y = ub - itop*step
                if y <= lb:
                    itop = 1
                    y = ub - itop*step
                itop += 1
            else: # from bottom
                y = lb + ibtm*step
                if y >= ub:
                    ibtm = 1
                    y = lb + ibtm*step
                ibtm += 1

            try:
                c = self._ele_col_dict[edge.split('_')[0]]
            except KeyError:
                self._ele_col_dict[edge.split('_')[0]] = next(self.color_cycle)
                c =  self._ele_col_dict[edge.split('_')[0]]

            xytext.append((energy, y, yval, edge, c))

        return xytext

    def _element_colour_dict(self, edges):
        # assign a colour to each element of the edges
        if isinstance(edges, dict):
            edges = edges.keys()

        elements = self._unique_element_of_edges(edges)

        d = {}
        for element in elements:
            d[element] = next(self.color_cycle)

        return d

    def _unique_element_of_edges(self, edges):
        # get the unique elements present in a sequence of edges
        elements = set()
        for edge in edges:
            element, _ = edge.split('_')
            elements.update([element])

        return elements

    def _text_parser(self, text_edge):
        # format the edge labels for LaTeX
        element, subshell = text_edge.split('_')

        if subshell[-1].isdigit():
            formatted = f"{element} {subshell[0]}$_{subshell[-1]}$"
        else:
            formatted = f"{element} {subshell[0]}"

        return formatted
