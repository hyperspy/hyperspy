# -*- coding: utf-8 -*-
# Copyright 2007-2015 The HyperSpy developers
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
# import matplotlib.pyplot as plt


def make_sure_ind(inds, req_len=None):
    try:
        v = len(inds)   # for error catching
        val = ()
        for i in inds:
            try:
                val = val + (float(i),)
            except TypeError:
                pass
        v = len(val)
    except TypeError:
        val = (float(inds),)
        v = len(val)
    if req_len:
        if req_len < v:
            val = val[:req_len]
        else:
            val = val + tuple([val[-1] for _ in range(v, req_len)])
    return val


def nearest_indices(shape, ind, radii):
    par = ()
    center = ()

    for c, i in enumerate(ind):
        top = min(i + radii[c] + 1, shape[c])
        bot = max(0, i - radii[c])
        center = center + (i - bot,)
        par = par + (slice(int(bot), int(top)),)
    return par, center


class strategy(object):

    samf = None
    close_plot = None
    name = ""

    def update(self, ind, isgood, count):
        if isgood:
            self._update_marker(ind)
        self._update_database(ind, count)

    def __repr__(self):
        return self.name.encode('utf8')


class diffusion_strategy(strategy):

    _radii = None
    _radii_changed = True
    _untruncated = None
    _mask_all = None
    decay_function = None
    _weight = None
    _samf = None

    def __init__(self, name):
        self.name = name

    @property
    def samf(self):
        return self._samf

    @samf.setter
    def samf(self, value):
        if value is not None and self.weight is not None:
            self._weight.model = value.model
        self._samf = value

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value):
        if self._weight is not None:
            self._weight.model = None
        self._weight = value
        if value is not None and self.samf is not None:
            value.model = self.samf.model

    def clean(self):
        self._untruncated = None
        self._mask_all = None
        self._radii_changed = True

    @property
    def radii(self):
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
        pass

    def refresh(self, overwrite, given_pixels=None):

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
            for ii in xrange(ind_list[0].size):
                ind = [one_list[ii] for one_list in ind_list]

                distances, slices, centre, mask = self._get_distance_array(
                    shape, ind)

                mask = np.logical_and(mask, todo_pixels[slices])

                weight = weights_all[tuple(ind)]
                distance_f = self.decay_function(distances)
                marker[slices][mask] += weight * distance_f[mask]
        else:
            # most efficient to propagate TO unknown pixels
            ind_list = np.where(todo_pixels)
            for ii in xrange(ind_list[0].size):
                ind = [one_list[ii] for one_list in ind_list]

                distances, slices, centre, mask = self._get_distance_array(
                    shape, ind)

                mask = np.logical_and(mask, calc_pixels[slices])

                weight = weights_all[slices]
                distance_f = self.decay_function(distances)
                marker[tuple(ind)] = np.sum(weight[mask] * distance_f[mask])

    def _get_distance_array(self, shape, ind):
        radii = make_sure_ind(self.radii, len(ind))
        # This should be unnecessary.......................
        if self._untruncated is not None and self._untruncated.ndim != len(ind):
            self._untruncated = None
            self._mask_all = None
        if self._radii_changed or self._untruncated is None or self._mask_all is None:
            par = []
            for r in radii:
                rc = np.ceil(r)
                par.append(np.abs(np.arange(-rc, rc + 1)))
            mg = np.array(np.meshgrid(*par, indexing='ij'))
            self._untruncated = np.sqrt(np.sum(mg ** 2.0, axis=0))
            distance_mask = np.array(
                [c / float(radii[i]) for i, c in enumerate(mg)])
            self._mask_all = np.sum(distance_mask ** 2.0, axis=0)
            self._radii_changed = False

        slices_return, centre = nearest_indices(shape, ind, np.ceil(radii))

        slices_temp = ()
        for r, c, s in zip(np.ceil(radii), centre, slices_return):
            slices_temp += (slice(r - c, s.stop - s.start + r - c),)
        ans = self._untruncated[slices_temp].copy()
        mask = self._mask_all[slices_temp].copy()

        # don't give values outside the radius - less prone to mistakes
        ans[mask > 1.0] = np.nan
        mask_to_send = mask <= 1.0  # within radius
        # don't want to add anything to the pixel itself
        mask_to_send[centre] = False
        return ans, slices_return, centre, mask_to_send

    def _update_marker(self, ind):
        marker = self.samf.metadata.marker
        shape = marker.shape
        distances, slices, centre, mask = self._get_distance_array(shape, ind)
        mask = np.logical_and(mask, marker[slices] >= 0)

        weight = self.decay_function(self.weight.function(ind))
        distance_f = self.decay_function(distances)
        marker[slices][mask] += weight * distance_f[mask]

        scale = self.samf._scale
        for i in self.samf._running_pixels:
            marker[i] = 0
        marker[ind] = -scale

    def values(self, ind):
        marker = self.samf.metadata.marker
        shape = marker.shape
        m = self.samf.model

        distances, slices, centre, mask_dist = self._get_distance_array(
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
        for component in m:
            if component.active_is_multidimensional:
                mask = np.logical_and(
                    component._active_array[slices],
                    mask_dist_calc)
            else:
                if component.active:
                    mask = mask_dist_calc
                else:  # not multidim and not active, skip
                    continue
            c = {}
            weight = distance_f[mask] * weights_values[mask]
            if weight.size:  # should never happen that np.sum(weight) == 0
                for par in component.parameters:
                    if par.free:
                        c[par.name] = np.average(
                            par.map[slices][mask]['values'], weights=weight, axis=0)
                ans[component.name] = c
        return ans

    def plot(self, fig):

        kwargs = {'interpolation': 'nearest'}
        marker = self.samf.metadata.marker.copy()

        if marker.ndim > 2:
            marker = np.sum(
                marker, tuple([i for i in range(0, marker.ndim - 2)]))
        elif marker.ndim < 2:
            marker = np.atleast_2d(marker)

        from hyperspy.signals import Image
        if not isinstance(fig, Image):
            fig = Image(marker)
            fig.plot()
            self.close_plot = fig._plot.signal_plot.close
        else:
            fig.data = marker
            fig._plot.signal_plot.update()
        return fig


class segmenter_strategy(strategy):

    segmenter = None
    _saved_values = None

    def clean(self):
        self._saved_values = None

    def __init__(self, name):
        self.name = name

    def refresh(self, overwrite, given_pixels=None):
        scale = self.samf._scale
        if overwrite:
            if given_pixels is None:
                good_pixels = self.samf.metadata.marker < 0
            else:
                good_pixels = given_pixels
            self.samf.metadata.marker[good_pixels] = -scale
        else:
            good_pixels = self.samf.metadata.marker < 0
            if given_pixels is not None:
                good_pixels = np.logical_and(good_pixels, given_pixels)
        self.samf.metadata.marker[~good_pixels] = scale
        self._update_database(None, 0)  # to force to update

    def _update_marker(self, ind):
        scale = self.samf._scale
        self.samf.metadata.marker[ind] = -scale

    def _package_values(self):
        m = self.samf.model
        mask_calc = self.samf.metadata.marker < 0
        ans = {}
        for component in m:
            if component.active_is_multidimensional:
                mask = np.logical_and(component._active_array, mask_calc)
            else:
                if component.active:
                    mask = mask_calc
                else:  # not multidim and not active, skip
                    continue
            c = {}
            for par in component.parameters:
                if par.free:
                    # only keeps active values and ravels
                    c[par.name] = par.map[mask]['values']
            ans[component.name] = c
        return ans

    def _update_database(self, ind, count):
        if not count % self.samf.update_every:
            self._saved_values = None
            self.segmenter.update(self._package_values())

    def values(self, ind=None):
        if self._saved_values is None:
            self._saved_values = self.segmenter.most_frequent()
        return self._saved_values

    def plot(self, fig):
        from hyperspy.drawing.tiles import HistogramTilePlot

        kwargs = {'color': '#4C72B0'}
        db = self.segmenter.database
        if db is None or not len(db):
            return fig

        if not isinstance(fig, HistogramTilePlot):
            fig = HistogramTilePlot()
            fig.plot(db, **kwargs)
            self.close_plot = fig.close
        else:
            fig.update(db, **kwargs)

        return fig
