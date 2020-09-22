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


def make_sure_ind(inds, req_len=None):
    """Given an object, constructs a tuple of floats the required length.
    Either removes items that cannot be cast as floats, or adds the last valid
    item until the required length is reached.

    Parameters
    ----------
    inds : sequence
        the sequence to be constructed into tuple of floats
    req_len : {None, number}
        The required length of the output

    Returns
    -------
    indices : tuple of floats
    """
    try:
        number = len(inds)   # for error catching
        val = ()
        for i in inds:
            try:
                val = val + (float(i),)
            except TypeError:
                pass
        number = len(val)
    except TypeError:
        val = (float(inds),)
        number = len(val)
    if req_len:
        if req_len < number:
            val = val[:req_len]
        else:
            val = val + tuple([val[-1] for _ in range(number, req_len)])
    return val


def nearest_indices(shape, ind, radii):
    """Returns the slices to slice a given size array to get the required size
    rectangle around the given index. Deals nicely with boundaries.

    Parameters
    ----------
    shape : tuple
        the shape of the original (large) array
    ind : tuple
        the index of interest in the large array (centre)
    radii : tuple of floats
        the distances of interests in all dimensions around the centre index.

    Returns
    -------
    slices : tuple of slices
        The slices to slice the large array to get the required region.
    center : tuple of ints
        The index of the original centre (ind) position in the new (sliced)
        array.
    """
    par = ()
    center = ()
    for cent, i in enumerate(ind):
        top = min(i + radii[cent] + 1, shape[cent])
        bot = max(0, i - radii[cent])
        center = center + (int(i - bot),)
        par = par + (slice(int(bot), int(top)),)
    return par, center


class SamfireStrategy(object):
    """A SAMFire strategy base class.
    """

    samf = None
    close_plot = None
    name = ""

    def update(self, ind, isgood):
        """Updates the database and marker with the given pixel results

        Parameters
        ----------
        ind : tuple
            the index with new results
        isgood : bool
            if the fit was successful.
        """
        count = self.samf.count
        if isgood:
            self._update_marker(ind)
        self._update_database(ind, count)

    def __repr__(self):
        return self.name

    def remove(self):
        """Removes this strategy from its SAMFire
        """
        self.samf.strategies.remove(self)


class LocalStrategy(SamfireStrategy):
    """A SAMFire strategy that operates in "pixel space" - i.e calculates the
    starting point estimates based on the local averages of the pixels.
    Requires some weighting method (e.g. reduced chi-squared).
    """

    _radii = None
    _radii_changed = True
    _untruncated = None
    _mask_all = None
    _weight = None
    _samf = None

    def __init__(self, name):
        self.name = name
        self.decay_function = lambda x: np.exp(-x)

    @property
    def samf(self):
        """The SAMFire that owns this strategy.
        """
        return self._samf

    @samf.setter
    def samf(self, value):
        if value is not None and self.weight is not None:
            self._weight.model = value.model
        self._samf = value

    @property
    def weight(self):
        """A Weight object, able to assign significance weights to separate
        pixels or maps, given the model.
        """
        return self._weight

    @weight.setter
    def weight(self, value):
        if self._weight is not None:
            self._weight.model = None
        self._weight = value
        if value is not None and self.samf is not None:
            value.model = self.samf.model

    def clean(self):
        """Purges the currently saved values.
        """
        self._untruncated = None
        self._mask_all = None
        self._radii_changed = True

    @property
    def radii(self):
        """A tuple of >=0 floats that show the "radii of relevance"
        """
        return self._radii

    @radii.setter
    def radii(self, value):
        m_sh = None
        if self.samf is not None:
            m_sh = len(self.samf.metadata.marker.shape)
        value = make_sure_ind(value, m_sh)
        if value != self._radii:
            self._radii_changed = True
            self._radii = value

    def _update_database(self, ind, count):
        """Dummy method for compatibility
        """
        pass

    def refresh(self, overwrite, given_pixels=None):
        """Refreshes the marker - recalculates with the current values from
        scratch.

        Parameters
        ----------
        overwrite : Bool
            If True, all but the given_pixels will be recalculated. Used when
            part of already calculated results has to be refreshed.
            If False, only use pixels with marker == -scale (by default -1) to
            propagate to pixels with marker >= 0. This allows "ignoring" pixels
            with marker < -scale (e.g. -2).
        given_pixels : boolean numpy array
            Pixels with True value are assumed as correctly calculated.
        """
        marker = self.samf.metadata.marker
        shape = marker.shape
        scale = self.samf._scale

        if overwrite:
            if given_pixels is None:
                calc_pixels = marker < 0
            else:
                calc_pixels = given_pixels
            todo_pixels = np.logical_not(calc_pixels)
        else:
            calc_pixels = marker == -scale
            todo_pixels = marker >= 0

            if given_pixels is not None:
                calc_pixels = np.logical_and(calc_pixels, given_pixels)
                todo_pixels = np.logical_or(
                    todo_pixels,
                    np.logical_xor(
                        marker == -
                        scale,
                        calc_pixels))

        done_number = np.sum(calc_pixels)
        todo_number = np.sum(todo_pixels)

        marker[todo_pixels] = 0.
        marker[calc_pixels] = -scale

        weights_all = self.decay_function(self.weight.map(calc_pixels))

        if done_number <= todo_number:
            # most efficient to propagate FROM fitted pixels
            ind_list = np.where(calc_pixels)
            for iindex in range(ind_list[0].size):
                ind = [one_list[iindex] for one_list in ind_list]

                distances, slices, _, mask = self._get_distance_array(
                    shape, ind)

                mask = np.logical_and(mask, todo_pixels[slices])

                weight = weights_all[tuple(ind)]
                distance_f = self.decay_function(distances)
                marker[slices][mask] += weight * distance_f[mask]
        else:
            # most efficient to propagate TO unknown pixels
            ind_list = np.where(todo_pixels)
            for iindex in range(ind_list[0].size):
                ind = [one_list[iindex] for one_list in ind_list]

                distances, slices, centre, mask = self._get_distance_array(
                    shape, ind)

                mask = np.logical_and(mask, calc_pixels[slices])

                weight = weights_all[slices]
                distance_f = self.decay_function(distances)
                marker[tuple(ind)] = np.sum(weight[mask] * distance_f[mask])

    def _get_distance_array(self, shape, ind):
        """Calculatex the array of distances (withing radii) from the given
        pixel. Deals with borders well.

        Parameters
        ----------
        shape : tuple
            the shape of the original array
        ind : tuple
            the index to calculate the distances from

        Returns
        -------
        ans : numpy array
            the array of distances
        slices : tuple of slices
            slices to slice the original marker to get the correct part of the
            array
        centre : tuple
            the centre index in the sliced array
        mask : boolean numpy array
            a binary mask for the values to consider
        """
        radii = make_sure_ind(self.radii, len(ind))
        # This should be unnecessary.......................
        if self._untruncated is not None and self._untruncated.ndim != len(
                ind):
            self._untruncated = None
            self._mask_all = None
        if self._radii_changed or \
           self._untruncated is None or \
           self._mask_all is None:
            par = []
            for radius in radii:
                radius_top = np.ceil(radius)
                par.append(np.abs(np.arange(-radius_top, radius_top + 1)))
            meshg = np.array(np.meshgrid(*par, indexing='ij'))
            self._untruncated = np.sqrt(np.sum(meshg ** 2.0, axis=0))
            distance_mask = np.array(
                [c / float(radii[i]) for i, c in enumerate(meshg)])
            self._mask_all = np.sum(distance_mask ** 2.0, axis=0)
            self._radii_changed = False

        slices_return, centre = nearest_indices(shape, ind, np.ceil(radii))

        slices_temp = ()
        for radius, cent, _slice in zip(np.ceil(radii), centre, slices_return):
            slices_temp += (slice(int(radius - cent),
                                  int(_slice.stop - _slice.start +
                                      radius - cent)),)
        ans = self._untruncated[slices_temp].copy()
        mask = self._mask_all[slices_temp].copy()

        # don't give values outside the radius - less prone to mistakes
        ans[mask > 1.0] = np.nan
        mask_to_send = mask <= 1.0  # within radius
        # don't want to add anything to the pixel itself
        mask_to_send[centre] = False
        return ans, slices_return, centre, mask_to_send

    def _update_marker(self, ind):
        """Updates the marker with the spatially decaying envelope around
        calculated pixels.

        Parameters
        ----------
        ind : tuple
            the index of the pixel to "spread" the envelope around.
        """
        marker = self.samf.metadata.marker
        shape = marker.shape
        distances, slices, _, mask = self._get_distance_array(shape, ind)
        mask = np.logical_and(mask, marker[slices] >= 0)

        weight = self.decay_function(self.weight.function(ind))
        distance_f = self.decay_function(distances)
        marker[slices][mask] += weight * distance_f[mask]

        scale = self.samf._scale
        for i in self.samf.running_pixels:
            marker[i] = 0
        marker[ind] = -scale

    def values(self, ind):
        """Returns the current starting value estimates for the given pixel.
        Calculated as the weighted local average. Only returns components that
        are active, and parameters that are free.

        Parameters
        ----------
        ind : tuple
            the index of the pixel of interest.

        Returns
        -------
        values : dict
            A dictionary of estimates, structured as
            {component_name: {parameter_name: value, ...}, ...}
            for active components and free parameters.

        """
        marker = self.samf.metadata.marker
        shape = marker.shape
        model = self.samf.model

        distances, slices, _, mask_dist = self._get_distance_array(
            shape, ind)

        # only use pixels that are calculated and "active"
        mask_dist_calc = np.logical_and(
            mask_dist,
            marker[slices] == -
            self.samf._scale)

        distance_f = self.decay_function(distances)
        weights_values = self.decay_function(
            self.weight.map(
                mask_dist_calc,
                slices))
        ans = {}
        for component in model:
            if component.active_is_multidimensional:
                mask = np.logical_and(
                    component._active_array[slices],
                    mask_dist_calc)
            else:
                if component.active:
                    mask = mask_dist_calc
                else:  # not multidim and not active, skip
                    continue
            comp_dict = {}
            weight = distance_f[mask] * weights_values[mask]
            if weight.size:  # should never happen that np.sum(weight) == 0
                for par in component.parameters:
                    if par.free:
                        comp_dict[par.name] = np.average(
                            par.map[slices][mask]['values'],
                            weights=weight,
                            axis=0)
                ans[component.name] = comp_dict
        return ans

    def plot(self, fig=None):
        """Plots the current marker in a flat image

        Parameters
        ----------
        fig : {Image, None}
            if an already plotted image, then updates. Otherwise creates a new
            one.

        Returns
        -------
        fig: Image
            the resulting image. If passed again, will be updated
            (computationally cheaper operation).
        """
        marker = self.samf.metadata.marker.copy()

        if marker.ndim > 2:
            marker = np.sum(
                marker, tuple([i for i in range(0, marker.ndim - 2)]))
        elif marker.ndim < 2:
            marker = np.atleast_2d(marker)

        from hyperspy.signals import Signal2D
        if not isinstance(
                fig, Signal2D) or fig._plot.signal_plot.figure is None:
            fig = Signal2D(marker)
            fig.plot()
            self.close_plot = fig._plot.signal_plot.close
        else:
            fig.data = marker
            fig._plot.signal_plot.update()
        return fig


class GlobalStrategy(SamfireStrategy):
    """A SAMFire strategy that operates in "parameter space" - i.e the pixel
    positions are not important, and only parameter value distributions are
    segmented to be used as starting point estimators.
    """

    segmenter = None
    _saved_values = None

    def clean(self):
        """Purges the currently saved values (not the database).
        """
        self._saved_values = None

    def __init__(self, name):
        self.name = name

    def refresh(self, overwrite, given_pixels=None):
        """Refreshes the database (i.e. constructs it again from scratch)
        """
        scale = self.samf._scale
        mark = self.samf.metadata.marker
        mark = np.where(np.isnan(mark), np.inf, mark)
        if overwrite:
            if given_pixels is None:
                good_pixels = mark < 0
            else:
                good_pixels = given_pixels
            self.samf.metadata.marker[good_pixels] = -scale
        else:
            good_pixels = mark < 0
            if given_pixels is not None:
                good_pixels = np.logical_and(good_pixels, given_pixels)
        self.samf.metadata.marker[~good_pixels] = scale
        self._update_database(None, 0)  # to force to update

    def _update_marker(self, ind):
        """Updates the SAMFire marker in the given pixel
        """
        scale = self.samf._scale
        self.samf.metadata.marker[ind] = -scale

    def _package_values(self):
        """Packages he current values to be sent to the segmenter
        """
        model = self.samf.model
        mask_calc = self.samf.metadata.marker < 0
        ans = {}
        for component in model:
            if component.active_is_multidimensional:
                mask = np.logical_and(component._active_array, mask_calc)
            else:
                if component.active:
                    mask = mask_calc
                else:  # not multidim and not active, skip
                    continue
            component_dict = {}
            for par in component.parameters:
                if par.free:
                    # only keeps active values and ravels
                    component_dict[par.name] = par.map[mask]['values']
            ans[component.name] = component_dict
        return ans

    def _update_database(self, ind, count):
        """
        Updates the database with current values
        """
        if not count % self.samf.update_every:
            self._saved_values = None
            self.segmenter.update(self._package_values())

    def values(self, ind=None):
        """Returns the saved most frequent values that should be used for
        prediction
        """
        if self._saved_values is None:
            self._saved_values = self.segmenter.most_frequent()
        return self._saved_values

    def plot(self, fig=None):
        """Plots the current database of histograms

        Parameters
        ----------
        fig: {None, HistogramTilePlot}
            If given updates the plot.
        """
        from hyperspy.drawing.tiles import HistogramTilePlot

        kwargs = {'color': '#4C72B0'}
        dbase = self.segmenter.database
        if dbase is None or not len(dbase):
            return fig

        if not isinstance(fig, HistogramTilePlot):
            fig = HistogramTilePlot()
            fig.plot(dbase, **kwargs)
            self.close_plot = fig.close
        else:
            fig.update(dbase, **kwargs)
        return fig
