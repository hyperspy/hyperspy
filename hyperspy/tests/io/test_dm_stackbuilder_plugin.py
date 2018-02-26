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


import os

from hyperspy.io import load
from numpy.testing import assert_allclose


my_path = os.path.dirname(__file__)


class TestStackBuilder:

    def test_load_stackbuilder_imagestack(self):
        image_stack = load(os.path.join(my_path, "dm_stackbuilder_plugin",
                                        "test_stackbuilder_imagestack.dm3"))
        data_dimensions = image_stack.data.ndim
        am = image_stack.axes_manager
        axes_dimensions = am.signal_dimension + am.navigation_dimension
        assert data_dimensions == axes_dimensions
        md = image_stack.metadata
        assert md.Acquisition_instrument.TEM.acquisition_mode == "STEM"
        assert_allclose(md.Acquisition_instrument.TEM.beam_current, 0.0)
        assert_allclose(md.Acquisition_instrument.TEM.beam_energy, 200.0)
        assert_allclose(md.Acquisition_instrument.TEM.camera_length, 15.0)
        assert_allclose(
            md.Acquisition_instrument.TEM.dwell_time, 0.03000005078125)
        assert_allclose(md.Acquisition_instrument.TEM.magnification, 200000.0)
        assert md.Acquisition_instrument.TEM.microscope == "JEM-ARM200F"
        assert md.General.date == "2015-05-17"
        assert md.General.original_filename == "test_stackbuilder_imagestack.dm3"
        assert md.General.title == "stackbuilder_test4_16x2"
        assert md.General.time == "17:00:16"
        assert md.Sample.description == "DWNC"
        assert md.Signal.quantity == "Electrons (Counts)"
        assert md.Signal.signal_type == ""
        assert md.Signal.binned == False
        assert_allclose(md.Signal.Noise_properties.Variance_linear_model.gain_factor,
                        0.15674974)
        assert_allclose(md.Signal.Noise_properties.Variance_linear_model.gain_offset,
                        2228741.5)
