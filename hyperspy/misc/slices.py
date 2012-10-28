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

import numpy as np
from scipy.interpolate import griddata

class Slice:
    """A consistent class for slices"""
    def __init__(self, sli, axe):
        self._orig_sli = sli
        self._axe = axe

        #Init params
        self.start = 0
        self.stop = axe.size
        self.step = 1

        #Init flags
        self.is_array = False

        #Update params
        self._update()

    def gen(self, to_values=False):
        #By default returns indexes
        if to_values:
            i2v = self._axe.index2value
            if self.is_array:
                return [i2v(el) if not isinstance(el, float) else el
                        for el in self._orig_sli]
            else:
                return slice(i2v(self.start) if not isinstance(self.start, float) 
                             else self.start,
                             i2v(self.stop) if not isinstance(self.stop, float) 
                             else self.stop,
                             i2v(self.step) if not isinstance(self.step, float) 
                             else self.step)
        else:
            v2i = self._axe.value2index
            if self.is_array:
                return [v2i(el) if isinstance(el, float) else el
                        for el in self._orig_sli]
            else:
                return slice(v2i(self.start) if isinstance(self.start, float) 
                             else self.start,
                             v2i(self.stop) if isinstance(self.stop, float) 
                             else self.stop,
                             v2i(self.step) if isinstance(self.step, float) 
                             else self.step)

    def _update(self):
        #Don't do too much (value2index or index2value):
        #    Original values will be needed when interpolation is implemented
        sli = self._orig_sli
        try:
            #Suppose "sli" is a slice instance...
            if sli.start is not None:
                self.start = sli.start

            if sli.stop is not None:
                self.stop = sli.stop

            if sli.step is not None and abs(sli.step) >=1 :
                self.step = int(round(sli.step)) #No non-int values for now 
            else:
                self.step = 1 #idem

            #No need for explicit size calculation from now on...
            #self.size = int(round(abs((self.stop-self.start)/self.step)))


        except (AttributeError, TypeError):
            try:
                #Suppose "sli" is a tuple-like instance
                self.start = None
                self.stop = None
                self.step = None
                self.is_array = True
            except TypeError:
                #Suppose "sli" is an int/float
                #Build a slice from it to keep dim(s.data) constant
                self.start = self._orig_sli
                self.stop = self.start + 1
                self.step = None

class Slices:
    def __init__(self, slices):
        try:
            len(slices)
        except TypeError:
            slices = (slices,)
        self._orig_slices = slices
        self.nav_indexes = None
        self.spec_indexes = None 
        
        self.slices = None 

        self.has_nav = True
        self.has_spec = True
        self.XYZ_ordering = True

    
    def __call__(self, signal):
        self._axes_manager = signal.axes_manager
        self._data = signal.data

        self._process_slices()
        self.apply()

    def _get_nav(self):
        self.nav_indexes =  np.array([el.index_in_array for el in
                    self._axes_manager.navigation_axes])

    def _get_spec(self):
        self.spec_indexes =  np.array([el.index_in_array for el in
                    self._axes_manager.signal_axes])

    def _process_slices(self):
        self._get_nav()
        self._get_spec()
        if self.XYZ_ordering:
            cut = slice(None, None, -1)
        else:
            cut = slice(None, None, 1)

        if self.has_nav and self.has_spec:
            idx = np.append(self.nav_indexes[cut], self.spec_indexes[cut])
        elif self.has_nav and not self.has_spec:
            idx = self.nav_indexes[cut]
        elif self.has_spec and not self.has_nav:
            idx = self.spec_indexes[cut]
        else:
            idx = None

        axe = self._axes_manager.axes
        slices = np.append(self._orig_slices,
                [slice(None,)]*max(0, (len(idx)-len(self._orig_slices))))

        self.slices = [Slice(sli, axe[i]).gen() for sli, i in zip(slices, idx)]

    def apply(self):
        return self._data[self.slices]

    def set_XYZ_ordering(self, bool):
        if bool:
            self.XYZ_ordering = True
        else:
            self.XYZ_ordering = False

    def has_nav(self, bool):
        if bool:
            self.has_nav = True
        else:
            self.has_nav = False
            if not self.has_spec:
                self.has_spec = True

    def has_spec(self, bool):
        if bool:
            self.has_spec = True
        else:
            self.has_spec = False
            if not self.has_nav:
                self.has_nav = True
