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


class ScalableFixedPattern2D(Component):
    """Fixed 2-dimensional pattern component with interpolation support.

        f(x,y) = a*s(b*x-x0) + c

    +------------+-----------+
    | Parameter  | Attribute |
    +------------+-----------+
    +------------+-----------+
    |     a      |  yscale   |
    +------------+-----------+
    |     b      |  xscale   |
    +------------+-----------+
    |    x0      |  shift    |
    +------------+-----------+


    The fixed two-dimensional pattern is defined by a single image which must
    be passed to the ScalableFixedPattern2D constructor, e.g.:

    .. code-block:: ipython

        In [1]: s = load('my_spectrum.hdf5')
        In [2] : my_fixed_pattern = components.ScalableFixedPattern(s))

    Attributes
    ----------

    yscale, xscale, shift : Float
    interpolate : Bool
        If False no interpolation is performed and only a y-scaled spectrum is
        returned.

    Methods
    -------

    prepare_interpolator : method to fine tune the interpolation

    """

    def __init__(self, image,
                 xscale=1.,
                 yscale=1.,
                 shear=0.,
                 rotation=0.,
                 scale=1.,
                 ):

        Component.__init__(self, ['xscale',
                                  'yscale',
                                  'shear',
                                  'rotation',
                                  'scale',
                                  ])

        self.image = image
        self.xscale.value = xscale
        self.yscale.value = yscale
        self.shear.value = shear
        self.rotation.value = rotation
        self.scale.value = scale

    def function(self, x, y):

        image = self.image.data
        sx = self.xscale.value
        sy = self.yscale.value
        sxy = self.shear.value
        rot = self.rotation.value
        sz = self.scale.value

        shifty, shiftx = np.array(image.shape[:2]) / 2

        shift = tf.SimilarityTransform(translation=[-shiftx, -shifty])
        tform = tf.AffineTransform(rotation=rot, scale=(sx, sy), shear=sxy)
        shift_inv = tf.SimilarityTransform(translation=[shiftx, shifty])

        transformed = sz * tf.warp(image, (shift + (tform + shift_inv)).inverse,
                                   order=3)

        return transformed
