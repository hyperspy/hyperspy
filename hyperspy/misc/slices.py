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

class Slice:
    """A consistent class for slices"""
    def __init__(self, sli, axe):
        self._orig_sli = sli
        self._axe = axe
        self._orig_offset = axe.offset
        #Init params
        self.start = 0
        self.stop = axe.size
        self.step = 1
        self.offset = axe.offset
        #Init flags
        self.is_array = False
        #Update params
        self._update()

    def gen(self, to_values=False):
        #Generate actual slices
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
        #Don't do too much (no value2index or index2value):
        #Original values will be needed once interpolation is implemented
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
                self.step = None #idem
        except (AttributeError, TypeError):
            try:
                #Suppose "sli" is a tuple-like instance
                self.start = sli[0]
                self.stop = None
                self.step = None
                self.is_array = True
            except TypeError:
                #Suppose "sli" is an int/float
                self.start = self._orig_sli
                if isinstance(self.start, float):
                    self.stop = self._axe.index2value(self.start) + 1
                else:
                    self.stop = self.start + 1
                self.step = None
        if isinstance(self.start, float):
            self.offset = self.start
        else:
            self.offset = self._axe.index2value(self.start)

class SliceSignal:
    def __init__(self, slices):
        try:
            len(slices)
        except TypeError:
            slices = (slices,)
        self._orig_slices = slices

        self.nav_indexes = None
        self.signal_indexes = None 

        self._signal = None
        
        self.idx = None
        self._Slices = None
        self.slices = None
        self.offset = None
        
        self.has_nav = True
        self.has_signal = True
        self.XYZ_ordering = True

    
    def __call__(self, signal):
        self._signal = signal.deepcopy()
        self._process_slices()
        return self.apply()

    def _get_nav(self):
        self.nav_indexes =  np.array([el.index_in_array for el in
                    self._signal.axes_manager.navigation_axes])

    def _get_signal(self):
        self.signal_indexes =  np.array([el.index_in_array for el in
                    self._signal.axes_manager.signal_axes])

    def _process_slices(self):
        #Change axes order and get processed slices from the "Slice" class
        self._get_nav()
        self._get_signal()
        if self.XYZ_ordering:
            cut = slice(None, None, -1)
        else:
            cut = slice(None, None, 1)

        if self.has_nav and self.has_signal:
            idx = np.append(self.nav_indexes[cut], self.signal_indexes[cut])
        elif self.has_nav and not self.has_signal:
            idx = self.nav_indexes[cut]
        elif self.has_signal and not self.has_nav:
            idx = self.signal_indexes[cut]
        else:
            idx = None

        axe = self._signal.axes_manager.axes
        slices = np.append(self._orig_slices,
                [slice(None,)]*max(0, (len(idx)-len(self._orig_slices))))
        self.idx = idx
        self._Slices = [Slice(slices[i], axe[i]) for  i in idx]
        self.slices = [sli.gen() for sli in self._Slices]
        self.offset = [sli.offset for sli in self._Slices]

    def _clean_axes(self):
        #Update axe sizes and offsets
        for i, (slice_len,j) in enumerate(zip(self._signal.data.shape, self.idx)):
            self._signal.axes_manager.axes[i].size = slice_len
            self._signal.axes_manager.axes[i].offset = self.offset[j]
        #Remove len = 1 axes
        for slice_len, axe in zip(self._signal.data.shape, self._signal.axes_manager.axes):
            if slice_len < 2:
                self._signal.axes_manager.axes.remove(axe)
        self._signal.data = self._signal.data.squeeze()
                     

    def update_slices(self, slices):
        self.__init__(slices)

    def apply(self):
        self._signal.data = self._signal.data[self.slices]
        self._clean_axes()
        return self._signal

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
            if not self.has_signal:
                self.has_signal = True

    def has_signal(self, bool):
        if bool:
            self.has_signal = True
        else:
            self.has_signal = False
            if not self.has_nav:
                self.has_nav = True
