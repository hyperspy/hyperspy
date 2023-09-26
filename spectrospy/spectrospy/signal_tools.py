import numpy as np
import traits.api as t

from hyperspy.exceptions import SignalDimensionError
from hyperspy.ui_registry import add_gui_method
from hyperspy.signal_tools import SpanSelectorInSignal1D
from spectrospy.misc.eels.tools import get_edges_near_energy, get_info_from_edges


@add_gui_method(toolkey="spectrospy.EELSSpectrum.print_edges_table")
class EdgesRange(SpanSelectorInSignal1D):
    units = t.Unicode()
    edges_list = t.Tuple()
    only_major = t.Bool()
    order = t.Unicode('closest')
    complementary = t.Bool(True)

    def __init__(self, signal, interactive=True):
        if signal.axes_manager.signal_dimension != 1:
            raise SignalDimensionError(
                signal.axes_manager.signal_dimension, 1)

        if interactive:
            super().__init__(signal)
        else:
            # ins non-interactive mode, don't initialise the span selector
            self.signal = signal
            self.axis = self.signal.axes_manager.signal_axes[0]

        self.active_edges = list(self.signal._edge_markers["names"])
        self.active_complementary_edges = []
        self.units = self.axis.units
        self.btns = []

        if self.signal._edge_markers["lines"] is None:
            self.signal._initialise_markers()

        self._get_edges_info_within_energy_axis()

        self.signal.axes_manager.events.indices_changed.connect(
            self._on_navigation_indices_changed, [])
        self.signal._plot.signal_plot.events.closed.connect(
            lambda: self.signal.axes_manager.events.indices_changed.disconnect(
            self._on_navigation_indices_changed), [])

    def _get_edges_info_within_energy_axis(self):
        mid_energy = (self.axis.low_value + self.axis.high_value) / 2
        rng = self.axis.high_value - self.axis.low_value
        self.edge_all = np.asarray(get_edges_near_energy(mid_energy, rng,
                                                         order=self.order))
        info = get_info_from_edges(self.edge_all)

        energy_all = []
        relevance_all = []
        description_all = []
        for d in info:
            onset = d['onset_energy (eV)']
            relevance = d['relevance']
            threshold = d['threshold']
            edge_ = d['edge']
            description = threshold + '. '*(threshold !='' and edge_ !='') + edge_

            energy_all.append(onset)
            relevance_all.append(relevance)
            description_all.append(description)

        self.energy_all = np.asarray(energy_all)
        self.relevance_all = np.asarray(relevance_all)
        self.description_all = np.asarray(description_all)

    def _on_navigation_indices_changed(self):
        self.signal._plot.signal_plot.update()

    def update_table(self):
        if self.span_selector is not None:
            energy_mask = (self.ss_left_value <= self.energy_all) & \
                (self.energy_all <= self.ss_right_value)
            if self.only_major:
                relevance_mask = self.relevance_all == 'Major'
            else:
                relevance_mask = np.ones(len(self.edge_all), bool)

            mask = energy_mask & relevance_mask
            self.edges_list = tuple(self.edge_all[mask])
            energy = tuple(self.energy_all[mask])
            relevance = tuple(self.relevance_all[mask])
            description = tuple(self.description_all[mask])
        else:
            self.edges_list = ()
            energy, relevance, description = (), (), ()

        self._keep_valid_edges()

        return self.edges_list, energy, relevance, description

    def _keep_valid_edges(self):
        edge_all = list(self.signal._edge_markers["names"])
        for edge in edge_all:
            if (edge not in self.edges_list):
                if edge in self.active_edges:
                    self.active_edges.remove(edge)
                elif edge in self.active_complementary_edges:
                    self.active_complementary_edges.remove(edge)
                self.signal._remove_edge_labels([edge], render_figure=False)
            elif (edge not in self.active_edges):
                self.active_edges.append(edge)

        self._on_complementary()
        self._update_labels()

    def update_active_edge(self, change):
        state = change['new']
        edge = change['owner'].description

        if state:
            self.active_edges.append(edge)
        else:
            if edge in self.active_edges:
                self.active_edges.remove(edge)
            if edge in self.active_complementary_edges:
                self.active_complementary_edges.remove(edge)
            self.signal._remove_edge_labels([edge], render_figure=False)
        self._on_complementary()
        self._update_labels()

    def _on_complementary(self):
        if self.complementary:
            self.active_complementary_edges = \
                self.signal._get_complementary_edges(self.active_edges,
                                                    self.only_major)
        else:
            self.active_complementary_edges = []

    def check_btn_state(self):
        edges = [btn.description for btn in self.btns]
        for btn in self.btns:
            edge = btn.description
            if btn.value is False:
                if edge in self.active_edges:
                    self.active_edges.remove(edge)
                    self.signal._remove_edge_labels([edge])
                if edge in self.active_complementary_edges:
                    btn.value = True

            if btn.value is True and self.complementary:
                comp = self.signal._get_complementary_edges(self.active_edges,
                                                           self.only_major)
                for cedge in comp:
                    if cedge in edges:
                        pos = edges.index(cedge)
                        self.btns[pos].value = True

    def _update_labels(self, active=None, complementary=None):
        # update selected and/or complementary edges
        if active is None:
            active = self.active_edges
        if complementary is None:
            complementary = self.active_complementary_edges

        edges_on_signal = set(self.signal._edge_markers["names"])
        edges_to_show = set(set(active).union(complementary))
        edge_keep = edges_on_signal.intersection(edges_to_show)
        edge_remove = edges_on_signal.difference(edge_keep)
        edge_add = edges_to_show.difference(edge_keep)

        if edge_remove:
            # Remove edges out
            self.signal._remove_edge_labels(edge_remove, render_figure=False)
        if edge_add:
            # Add the new edges
            self.signal._add_edge_labels(edge_add, render_figure=False)
        if edge_remove or edge_add:
            # Render figure only once
            self.signal._render_figure(plot=['signal_plot'])

    def _clear_markers(self):
        # Used in hyperspy_gui_ipywidgets
        self.signal._remove_edge_labels()
        self.active_edges = []
        self.active_complementary_edges = []
