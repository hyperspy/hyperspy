# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np

from hyperspy.roi import (BaseInteractiveROI, SpanROI, RectangularROI,
                          roi_dict_to_roi)


class ROIsManager:

    """ Class to manager widgets (add, store and remove) to a figure.

    Attributes
    ----------
    widgets : list
        List of widgets

    Methods
    -------
    add_widgets: add widget by using a list of object or a single object.
    remove_widget: remove a widget from the plot and from the widget_list.
    add_widgets_interactively: add widget interactively: use the control `key` 
        to widget and the `delete` key to remove widget.
    """

    def __init__(self, signal):
        self.signal = signal
        self._index = 0
        self._rois_list = []

        # Default name of the metadata node to store widget
        self.metadata_node_name = 'ROIs'
        # TODO: fix issue with overlaying span

    def add_ROIs(self, roi):
        """
        Add one or several ROIs.

        Parameters
        ----------
        roi : ROI or dictionary or iterable of ROIs, dictionary
            Dictionary can be obtained from a ROI using the `to_dictionary` 
            method.
        """
        if isinstance(roi, list):
            for _roi in roi:
                self._add_ROI(self.signal, _roi)
        else:
            self._add_ROI(self.signal, roi)

    def _add_ROI(self, signal, roi):
        axes = None
        if isinstance(roi, BaseInteractiveROI):
            pass
        elif isinstance(roi, dict):
            if roi['on_signal']:
                axes = self.signal.axes_manager.signal_axes
            else:
                axes = self.signal.axes_manager.navigation_axes
            roi = roi_dict_to_roi(roi)
        else:
            raise ValueError("Please check the 'roi' parameter.")
        roi.add_widget(signal, axes=axes)

    def create_ROI_interactively(self, figure):
        """
        Create ROI interactively: the ROI will be created when cliking on the
        figure.

        Parameters
        ----------
        figure : Figure object
            The figure on which the ROI will be created      

        """
        if hasattr(figure, "xaxis"):
            # for Signal2DFigure, add RectangularROI as default ROI
            raise ValueError("Currently not implemented.")
            roi = RectangularROI(None, None, None, None)
            axes = [figure.xaxis, figure.yaxis]
        else:
            # for Signal1DFigure, add SpanROI as default ROI
            roi = SpanROI(None, None)
            axes = figure.axis
        roi.add_widget(self.signal, axes=axes, set_initial_value=False)

    def remove_roi(self, roi):
        """
        Remove a roi. The roi can be selected either by providing
        the roi itself or its index in the `_rois_list`.
        """
        if isinstance(roi, int):
            roi = self._rois_list[roi]
        elif not isinstance(roi, BaseInteractiveROI):
            return
        roi._get_widget().disconnect()
        self._index = 0
        self._rois_list.remove(roi)

    def add_ROIs_to_metadata(self, metadata_name=None):
        """ Add all widgets to metadata. """
        if len(self) == 0:
            return
        if metadata_name is None:
            metadata_name = 'ROIs'
        if not self.signal.metadata.has_item(metadata_name):
            self.signal.metadata.add_node(metadata_name)
        metadata_node = self.signal.metadata.get_item(metadata_name)
        for i, roi in enumerate(self._rois_list):
            roi_dict = roi.to_dictionary()
            name = roi.name if roi.name == "" else "ROI_{}".format(i)
            metadata_node.set_item(name, roi_dict)

    def __getitem__(self, y):
        """x.__getitem__(y) <==> x[y]

        """
        if isinstance(y, int):
            return self._rois_list[y]
        elif isinstance(y, str) or not np.iterable(y):
            return self._roi_getter(y)
        else:
            return [self._roi_getter(roi) for roi in y]

    def _roi_getter(self, y):
        if y in self._rois_list:
            return y
        if isinstance(y, str):
            for roi in self._rois_list:
                if y == roi.name:
                    return roi
            raise ValueError("There is no roi named %s" % y)

    def __getslice__(self, i=None, j=None):
        """x.__getslice__(i, j) <==> x[i:j]

        """
        return self._rois_list[i:j]

    def __next__(self):
        """
        Standard iterator method, updates the index and returns the next
        roi 

        Returns
        -------
        val : tuple of ints
            Returns a tuple containing the coordinates of the current
            iteration.

        """
        if self._index is None:
            self._index = 0
        elif self._index >= len(self._rois_list) - 1:
            raise StopIteration
        else:
            self._index += 1
        return self._rois_list[self._index]

    def __iter__(self):
        # Reset the _index that can have a value != None due to
        # a previous iteration that did not hit a StopIteration
        self._index = None
        return self

    def __len__(self):
        return len(self._rois_list)

    def append(self, roi):
        self._rois_list.append(roi)
