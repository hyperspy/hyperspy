# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
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
import pytest
import scipy.constants as constants

import hyperspy.api as hs
from hyperspy.datasets.example_signals import reference_hologram


# Set parameters outside the tests
img_size = 256
IMG_SIZE3X = 128
IMG_SIZE3Y = 64
FRINGE_DIRECTION = -np.pi / 6
FRINGE_SPACING = 5.23
FRINGE_DIRECTION3 = np.pi / 3
FRINGE_SPACING3 = 6.11
LS = np.linspace(-img_size / 2, img_size / 2 - 1, img_size)
X_START = int(IMG_SIZE3X / 9)
# larger fringes require larger crop
X_STOP = IMG_SIZE3X - 1 - int(IMG_SIZE3X / 9)
Y_START = int(IMG_SIZE3Y / 9)
Y_STOP = IMG_SIZE3Y - 1 - int(IMG_SIZE3Y / 9)


class TestStatistics:
    def setup_method(self, method):
        # Set the stack
        self.ref_holo = hs.stack([reference_hologram()] * 2)
        self.ref_holo = hs.stack([self.ref_holo] * 3)

        # Parameters measured using Gatan HoloWorks:
        self.REF_FRINGE_SPACING = 3.48604
        self.REF_FRINGE_SAMPLING = 3.7902

        # Measured using the definition of fringe contrast from the centre of image
        self.REF_FRINGE_CONTRAST = 0.0736

        # Prepare test data and derived statistical parameters
        self.ref_carrier_freq = 1.0 / self.REF_FRINGE_SAMPLING
        self.ref_carrier_freq_nm = 1.0 / self.REF_FRINGE_SPACING

        ht = self.ref_holo.metadata.Acquisition_instrument.TEM.beam_energy
        momentum = (
            2
            * constants.m_e
            * constants.elementary_charge
            * ht
            * 1000
            * (
                1
                + constants.elementary_charge
                * ht
                * 1000
                / (2 * constants.m_e * constants.c ** 2)
            )
        )
        wavelength = constants.h / np.sqrt(momentum) * 1e9  # in nm
        self.ref_carrier_freq_mrad = self.ref_carrier_freq_nm * 1000 * wavelength

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("fringe_contrast_algorithm", ["fourier", "statistical"])
    def test_single_values(self, lazy, fringe_contrast_algorithm):
        if lazy:
            self.ref_holo.as_lazy()

        stats = self.ref_holo.statistics(
            high_cf=True,
            single_values=True,
            fringe_contrast_algorithm=fringe_contrast_algorithm,
        )

        # Fringe contrast in experimental conditions can be only an estimate
        # therefore tolerance is 10%:
        np.testing.assert_allclose(
            stats["Fringe contrast"], self.REF_FRINGE_CONTRAST, rtol=0.1
        )

        np.testing.assert_allclose(
            stats["Fringe sampling (px)"], self.REF_FRINGE_SAMPLING, rtol=1e-5
        )
        np.testing.assert_allclose(
            stats["Fringe spacing (nm)"], self.REF_FRINGE_SPACING, rtol=1e-5
        )
        np.testing.assert_allclose(
            stats["Carrier frequency (1 / nm)"], self.ref_carrier_freq_nm, rtol=1e-5
        )
        np.testing.assert_allclose(
            stats["Carrier frequency (1/px)"], self.ref_carrier_freq, rtol=1e-5
        )
        np.testing.assert_allclose(
            stats["Carrier frequency (mrad)"], self.ref_carrier_freq_mrad, rtol=1e-5
        )

    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("fringe_contrast_algorithm", ["fourier", "statistical"])
    def test_no_single_values(self, lazy, fringe_contrast_algorithm):
        if lazy:
            self.ref_holo.as_lazy()

        stats = self.ref_holo.statistics(
            high_cf=True,
            single_values=False,
            fringe_contrast_algorithm=fringe_contrast_algorithm,
        )

        ref_fringe_contrast_stack = np.repeat(self.REF_FRINGE_CONTRAST, 6).reshape(
            (3, 2)
        )
        ref_fringe_sampling_stack = np.repeat(self.REF_FRINGE_SAMPLING, 6).reshape(
            (3, 2)
        )
        ref_fringe_spacing_stack = np.repeat(self.REF_FRINGE_SPACING, 6).reshape((3, 2))
        ref_carrier_freq_nm_stack = np.repeat(self.ref_carrier_freq_nm, 6).reshape(
            (3, 2)
        )
        ref_carrier_freq_stack = np.repeat(self.ref_carrier_freq, 6).reshape((3, 2))
        ref_carrier_freq_mrad_stack = np.repeat(self.ref_carrier_freq_mrad, 6).reshape(
            (3, 2)
        )

        # Fringe contrast in experimental conditions can be only an estimate
        # therefore tolerance is 10%:
        np.testing.assert_allclose(
            stats["Fringe contrast"].data, ref_fringe_contrast_stack, rtol=0.1
        )

        np.testing.assert_allclose(
            stats["Fringe sampling (px)"].data, ref_fringe_sampling_stack, rtol=1e-5
        )
        np.testing.assert_allclose(
            stats["Fringe spacing (nm)"].data, ref_fringe_spacing_stack, rtol=1e-5
        )
        np.testing.assert_allclose(
            stats["Carrier frequency (1 / nm)"].data,
            ref_carrier_freq_nm_stack,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            stats["Carrier frequency (1/px)"].data, ref_carrier_freq_stack, rtol=1e-5
        )
        np.testing.assert_allclose(
            stats["Carrier frequency (mrad)"].data,
            ref_carrier_freq_mrad_stack,
            rtol=1e-5,
        )
