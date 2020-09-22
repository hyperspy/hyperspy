# -*- coding: utf-8 -*-
# Copyright 2007-2011 The HyperSpy developers
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
from scipy.signal import argrelextrema

from hyperspy.external.astroML.histtools import histogram


class HistogramSegmenter(object):
    """Historam Segmenter strategy of the SAMFire. Uses histograms to estimate
    parameter distribusions, and then passes the most frequent values as
    the starting parameter estimates.
    """

    def __init__(self, bins='freedman'):
        self.database = None
        self.bins = bins
        self._min_points = 4

    def most_frequent(self):
        """Calculates the most frequent values in the currently stored
        histograms of the database. Does to by looking for local maxima in the
        frequences.
        """
        freq = {}
        for c_n, comp in self.database.items():
            comp_dict = {}
            for p_n, (hist, bin_edges) in comp.items():
                # calculate frequent values
                maxima_hist_ind = argrelextrema(
                    np.append(
                        0,
                        hist),
                    np.greater,
                    mode='wrap')
                middles_of_maxima = 0.5 * \
                    (bin_edges[maxima_hist_ind] +
                     bin_edges[([i - 1 for i in maxima_hist_ind[0]],)])
                comp_dict[p_n] = middles_of_maxima.tolist()
            freq[c_n] = comp_dict
        return freq

    # MUCH LATER: return boundaries of the n-dimensional domains, projected to
    # the parameter axes, to be used as fitting boundaries.

    def update(self, value_dict):
        """Recalculates the database, given value dictionary (with all values!)

        Parameters
        ----------
        value_dict : dict
            dictionary of all already calculated values in the form of
            {component_name: {parameter_name: values, ...}, ...}
        """
        # recalculate with values. All values are passed, not just new
        self.database = {}
        for component_name, component in value_dict.items():
            comp_dict = {}
            for par_name, par in component.items():
                if par.size <= self._min_points:
                    comp_dict[par_name] = np.histogram(par,
                                                       max(10,
                                                           self._min_points))
                else:
                    comp_dict[par_name] = histogram(par, bins=self.bins)
            self.database[component_name] = comp_dict
