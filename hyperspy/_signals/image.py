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


from hyperspy.signal import Signal


class Image(Signal):

    """
    """
    _record_by = "image"

    def __init__(self, *args, **kw):
        super(Image, self).__init__(*args, **kw)
        self.axes_manager.set_signal_dimension(2)

    def to_spectrum(self):
        """Returns the image as a spectrum.

        See Also
        --------
        as_spectrum : a method for the same purpose with more options.
        signals.Image.to_spectrum : performs the inverse operation on images.

        """
        return self.as_spectrum(0 + 3j)

    def plot_orthoview(image):
        """
        Plot an orthogonal view of a 3D images

        Parameters
        ---------
        image: signals.Image
            A 3D image.
        """
        if len(image.axes_manager.shape) != 3:
            raise ValueError("Image must have 3 dimension.")

        image.metadata.General.title = 'xy'
        image.axes_manager.set_signal_dimension(0)

        im_xz = image.deepcopy()
        im_xz = im_xz.rollaxis(2, 1)
        im_xz.metadata.General.title = 'xz'
        im_xz.axes_manager.set_signal_dimension(0)

        im_xz.axes_manager._axes[2] = image.axes_manager._axes[2]
        im_xz.axes_manager._axes[1] = image.axes_manager._axes[0]
        im_xz.axes_manager._axes[0] = image.axes_manager._axes[1]

        im_yz = image.deepcopy()
        im_yz = im_yz.rollaxis(0, 2)
        im_yz = im_yz.rollaxis(1, 0)
        im_yz.metadata.General.title = 'yz'
        im_yz.axes_manager.set_signal_dimension(0)

        im_yz.axes_manager._axes = image.axes_manager._axes[::-1]

        image.axes_manager[0].index = (image.axes_manager[0].high_index -
                                       image.axes_manager[0].low_index)/2
        image.axes_manager[1].index = (image.axes_manager[1].high_index -
                                       image.axes_manager[1].low_index)/2
        image.axes_manager[2].index = (image.axes_manager[2].high_index -
                                       image.axes_manager[2].low_index)/2

        im_xz.axes_manager._update_attributes()
        im_yz.axes_manager._update_attributes()
        image.plot()
        im_xz.plot()
        im_yz.plot()
