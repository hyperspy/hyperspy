from operator import attrgetter
from hyperspy.misc.utils import attrsetter

import numpy as np


class SpecialSlicers(object):

    def __init__(self, obj, isNavigation):
        self.isNavigation = isNavigation
        self.obj = obj

    def __getitem__(self, slices):
        return self.obj._slicer(slices, self.isNavigation)


class FancySlicing(object):

    def _get_array_slices(self, slices, isNavigation=None):
        try:
            len(slices)
        except TypeError:
            slices = (slices,)
        _orig_slices = slices

        has_nav = True if isNavigation is None else isNavigation
        has_signal = True if isNavigation is None else not isNavigation

        nav_idx = [el.index_in_array for el in
                   self.axes_manager.navigation_axes]
        signal_idx = [el.index_in_array for el in
                      self.axes_manager.signal_axes]

        if not has_signal:
            idx = nav_idx
        elif not has_nav:
            idx = signal_idx
        else:
            idx = nav_idx + signal_idx

        # Add support for Ellipsis
        if Ellipsis in _orig_slices:
            _orig_slices = list(_orig_slices)
            # Expand the first Ellipsis
            ellipsis_index = _orig_slices.index(Ellipsis)
            _orig_slices.remove(Ellipsis)
            _orig_slices = (_orig_slices[:ellipsis_index] + [slice(None), ] *
                            max(0, len(idx) - len(_orig_slices)) +
                            _orig_slices[ellipsis_index:])
            # Replace all the following Ellipses by :
            while Ellipsis in _orig_slices:
                _orig_slices[_orig_slices.index(Ellipsis)] = slice(None)
            _orig_slices = tuple(_orig_slices)

        if len(_orig_slices) > len(idx):
            raise IndexError("too many indices")

        slices = np.array([slice(None,)] *
                          len(self.axes_manager._axes))

        slices[idx] = _orig_slices + (slice(None),) * max(
            0, len(idx) - len(_orig_slices))

        array_slices = []
        for slice_, axis in zip(slices, self.axes_manager._axes):
            if (isinstance(slice_, slice) or
                    len(self.axes_manager._axes) < 2):
                array_slices.append(axis._get_array_slices(slice_))
            else:
                if isinstance(slice_, float):
                    slice_ = axis.value2index(slice_)
                array_slices.append(slice_)
        return array_slices

    def _slicer(self, slices, isNavigation=None):
        array_slices = self._get_array_slices(slices, isNavigation)
        _obj = self._deepcopy_with_new_data(self.data[array_slices])
        for slice_, axis in zip(array_slices, _obj.axes_manager._axes):
            if (isinstance(slice_, slice) or
                    len(self.axes_manager._axes) < 2):
                axis._slice_me(slice_)
            else:
                _obj._remove_axis(axis.index_in_axes_manager)
        if hasattr(self, "_additional_slicing_targets"):
            for ta in self._additional_slicing_targets:
                try:
                    t = attrgetter(ta)(self)
                    if hasattr(t, '_slicer'):
                        attrsetter(
                            _obj,
                            ta,
                            t._slicer(
                                slices,
                                isNavigation))
                except AttributeError:
                    pass
        _obj.get_dimensions_from_data()

        return _obj
