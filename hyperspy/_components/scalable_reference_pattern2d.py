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


from hyperspy.component import Component
from skimage import transform as tf
import numpy as np


class ScalableReferencePattern2D(Component):
    """Fixed 2-dimensional pattern component which is scaled by a 2D affine
    transformation of the form:

        X = d11*x + d12*y
        Y = d21*x + d21*y

    The fixed two-dimensional pattern is defined by a single image which must
    be passed to the ScalableReferencePattern2D constructor, e.g.:

    .. code-block:: ipython

        In [1]: im = load('my_image_data.hdf5')
        In [2] : ref = components.ScalableFixedPattern(im.inav[11,30]))

    Attributes
    ----------

    D : list
        List containing matrix components for affine matrix

    order : 1, 3
        Interpolation order used when applying image transformation

    """

    def __init__(self, image,
                 d11=1., d12=0.,
                 d21=1., d22=1.,
                 order=3
                 ):

        Component.__init__(self, ['d11', 'd12',
                                  'd21', 'd22'])

        self.signal = image
        self.order = order
        self.d11.value = d11
        self.d12.value = d12
        self.d21.value = d21
        self.d22.value = d22

    def function(self, x, y):

        image = self.signal.data
        order = self.order
        d11 = self.d11.value
        d12 = self.d12.value
        d21 = self.d21.value
        d22 = self.d22.value

        D = np.array([[d11, d12, 0.],
                      [d21, d22, 0.],
                      [0., 0., 1.]])

        shifty, shiftx = np.array(image.shape[:2]) / 2

        shift = tf.SimilarityTransform(translation=[-shiftx, -shifty])
        tform = tf.AffineTransform(matrix=D)
        shift_inv = tf.SimilarityTransform(translation=[shiftx, shifty])

        transformed = tf.warp(image, (shift + (tform + shift_inv)).inverse,
                              order=order)

        return transformed
