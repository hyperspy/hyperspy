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
import numpy.testing as nt
from collections import OrderedDict
import hyperspy.api as hs


def test_rec_param():
    wave_image = hs.signals.ElectronWaveImage(
        np.exp(1j * (np.indices((3, 3)).sum(axis=0) + 4)))
    rec_param = np.arange(5)
    rec_param_dict = OrderedDict([('sb_pos_x0', rec_param[0]), ('sb_pos_y0', rec_param[1]),
                                      ('sb_pos_x1', rec_param[2]), ('sb_pos_y1', rec_param[3]),
                                      ('sb_size', rec_param[4])])

    wave_image.metadata.Signal.add_node('holo_rec_param')
    wave_image.metadata.Signal.holo_rec_param.add_dictionary(rec_param_dict)

    nt.assert_equal(wave_image.rec_param, rec_param)


if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)
