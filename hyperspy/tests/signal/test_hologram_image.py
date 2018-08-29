# -*- coding: utf-8 -*-
#  Copyright 2007-2016 The HyperSpy developers
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

import gc

import numpy as np
import numpy.testing as nt
from numpy.testing import assert_allclose
import pytest
import scipy.constants as constants

import hyperspy.api as hs
from hyperspy.decorators import lazifyTestClass
from scipy.interpolate import interp2d
from hyperspy.datasets.example_signals import reference_hologram


@pytest.mark.parametrize('lazy', [True, False])
def test_set_microscope_parameters(lazy):
    holo_image = hs.signals.HologramImage(np.ones((3, 3)))
    if lazy:
        holo_image = holo_image.as_lazy()
    holo_image.set_microscope_parameters(
        beam_energy=300., biprism_voltage=80.5, tilt_stage=2.2)
    assert (holo_image.metadata.Acquisition_instrument.TEM.beam_energy == 300.)
    assert (
        holo_image.metadata.Acquisition_instrument.TEM.Biprism.voltage == 80.5)
    assert (holo_image.metadata.Acquisition_instrument.TEM.Stage.tilt_alpha == 2.2)


def calc_holo(x, y, phase_ref, FRINGE_SPACING, FRINGE_DIRECTION):
    return 2 * (1 + np.cos(phase_ref + FRINGE_SPACING / (2 * np.pi) * (
        x * np.cos(FRINGE_DIRECTION) + y * np.sin(FRINGE_DIRECTION))))


def calc_phaseref(x, y, z, img_sizex, img_sizey):
    return np.pi * x * (1 - z) / img_sizex + np.pi * y * z / img_sizey


img_size = 1024
IMG_SIZE3X = 512
IMG_SIZE3Y = 256
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


@pytest.mark.parametrize('parallel,lazy', [(True, False), (False, False),
                                           (None, True)])
def test_reconstruct_phase_single(parallel, lazy):

    x, y = np.meshgrid(LS, LS)
    phase_ref = calc_phaseref(x, 0, 0, img_size / 2.2, 1)
    holo = calc_holo(x, y, phase_ref, FRINGE_SPACING, FRINGE_DIRECTION)
    ref = calc_holo(x, y, 0, FRINGE_SPACING, FRINGE_DIRECTION)
    holo_image = hs.signals.HologramImage(holo)
    ref_image = hs.signals.HologramImage(ref)
    if lazy:
        ref_image = ref_image.as_lazy()
        holo_image = holo_image.as_lazy()
    wave_image = holo_image.reconstruct_phase(
        ref, store_parameters=True, parallel=parallel)
    sb_pos_cc = wave_image.metadata.Signal.Holography.Reconstruction_parameters.sb_position * (-1) + \
        [img_size, img_size]

    sb_size_cc = wave_image.metadata.Signal.Holography.Reconstruction_parameters.sb_size
    sb_smoothness_cc = wave_image.metadata.Signal.Holography.Reconstruction_parameters.sb_smoothness
    sb_units_cc = wave_image.metadata.Signal.Holography.Reconstruction_parameters.sb_units
    wave_image_cc = holo_image.reconstruct_phase(
        ref_image,
        sb_position=sb_pos_cc,
        sb_size=sb_size_cc,
        sb_smoothness=sb_smoothness_cc,
        sb_unit=sb_units_cc,
        parallel=parallel)
    X_START = int(wave_image.axes_manager.signal_shape[0] / 10)
    X_STOP = int(wave_image.axes_manager.signal_shape[0] * 9 / 10)
    wave_crop = wave_image.data[X_START:X_STOP, X_START:X_STOP]
    wave_cc_crop = wave_image_cc.data[X_START:X_STOP, X_START:X_STOP]

    # asserts that waves from different
    assert_allclose(wave_crop, np.conj(wave_cc_crop), rtol=1e-3)
    # sidebands are complex conjugate; this also tests possibility of
    # reconstruction with given sideband parameters

    # Cropping the reconstructed and original phase images and comparing:
    X_START = int(img_size / 10)
    X_STOP = img_size - 1 - int(img_size / 10)
    phase_new_crop = wave_image.unwrapped_phase(
        parallel=parallel).data[X_START:X_STOP, X_START:X_STOP]
    phase_ref_crop = phase_ref[X_START:X_STOP, X_START:X_STOP]
    assert_allclose(phase_new_crop, phase_ref_crop, atol=1E-2)


@pytest.mark.parametrize('parallel,lazy', [(True, False), (False, False),
                                           (None, True)])
def test_reconstruct_phase_nonstandard(parallel, lazy):
    # 2. Testing reconstruction with non-standard output size for stacked
    # images:
    gc.collect()
    x2, z2, y2 = np.meshgrid(LS, np.array([0, 1]), LS)
    phase_ref2 = calc_phaseref(x2, y2, z2, img_size / 2.2, img_size / 2.2)
    holo2 = calc_holo(x2, y2, phase_ref2, FRINGE_SPACING, FRINGE_DIRECTION)
    ref2 = calc_holo(x2, y2, 0, FRINGE_SPACING, FRINGE_DIRECTION)
    holo_image2 = hs.signals.HologramImage(holo2)
    ref_image2 = hs.signals.HologramImage(ref2)
    del x2, z2, y2
    gc.collect()
    if lazy:
        ref_image2 = ref_image2.as_lazy()
        holo_image2 = holo_image2.as_lazy()

    sb_position2 = ref_image2.estimate_sideband_position(
        sb='upper', parallel=parallel)
    sb_position2_lower = ref_image2.estimate_sideband_position(
        sb='lower', parallel=parallel)
    sb_position2_left = ref_image2.estimate_sideband_position(
        sb='left', parallel=parallel)
    sb_position2_right = ref_image2.estimate_sideband_position(
        sb='right', parallel=parallel)
    assert sb_position2 == sb_position2_left
    assert sb_position2_lower == sb_position2_right

    sb_size2 = ref_image2.estimate_sideband_size(
        sb_position2, parallel=parallel)
    output_shape = (np.int(sb_size2.inav[0].data * 2),
                    np.int(sb_size2.inav[0].data * 2))

    wave_image2 = holo_image2.reconstruct_phase(
        reference=ref_image2,
        sb_position=sb_position2,
        sb_size=sb_size2,
        sb_smoothness=sb_size2 * 0.05,
        output_shape=output_shape,
        parallel=parallel)
    # a. Reconstruction with parameters provided as ndarrays should be
    # identical to above:
    wave_image2a = holo_image2.reconstruct_phase(
        reference=ref_image2.data,
        sb_position=sb_position2.data,
        sb_size=sb_size2.data,
        sb_smoothness=sb_size2.data * 0.05,
        output_shape=output_shape,
        parallel=parallel)
    assert wave_image2 == wave_image2a
    del wave_image2a
    gc.collect()
    # interpolate reconstructed phase to compare with the input (reference
    # phase):
    interp_x = np.arange(output_shape[0])
    phase_interp0 = interp2d(
        interp_x,
        interp_x,
        wave_image2.inav[0].unwrapped_phase(parallel=parallel).data,
        kind='cubic')
    phase_new0 = phase_interp0(
        np.linspace(0, output_shape[0], img_size),
        np.linspace(0, output_shape[0], img_size))

    phase_interp1 = interp2d(
        interp_x,
        interp_x,
        wave_image2.inav[1].unwrapped_phase(parallel=parallel).data,
        kind='cubic')
    phase_new1 = phase_interp1(
        np.linspace(0, output_shape[0], img_size),
        np.linspace(0, output_shape[0], img_size))

    X_START = int(img_size / 10)
    X_STOP = img_size - 1 - int(img_size / 10)
    phase_new_crop0 = phase_new0[X_START:X_STOP, X_START:X_STOP]
    phase_new_crop1 = phase_new1[X_START:X_STOP, X_START:X_STOP]
    phase_ref_crop0 = phase_ref2[0, X_START:X_STOP, X_START:X_STOP]
    phase_ref_crop1 = phase_ref2[1, X_START:X_STOP, X_START:X_STOP]
    assert_allclose(phase_new_crop0, phase_ref_crop0, atol=0.05)
    assert_allclose(phase_new_crop1, phase_ref_crop1, atol=0.01)


@pytest.mark.parametrize('parallel,lazy', [(True, False), (False, False),
                                           (None, True)])
def test_reconstruct_phase_multi(parallel, lazy):

    x3, z3, y3 = np.meshgrid(
        np.linspace(-IMG_SIZE3X / 2, IMG_SIZE3X / 2, IMG_SIZE3X),
        np.arange(6), np.linspace(-IMG_SIZE3Y / 2, IMG_SIZE3Y / 2, IMG_SIZE3Y))
    phase_ref3 = calc_phaseref(x3, y3, z3 / 6, IMG_SIZE3X * 2, IMG_SIZE3Y * 2)
    newshape = (2, 3, IMG_SIZE3X, IMG_SIZE3Y)
    holo3 = calc_holo(x3, y3, phase_ref3, FRINGE_SPACING3,
                      FRINGE_DIRECTION3).reshape(newshape)
    ref3 = calc_holo(x3, y3, 0, FRINGE_SPACING3,
                     FRINGE_DIRECTION3).reshape(newshape)
    del x3, z3, y3
    gc.collect()
    holo_image3 = hs.signals.HologramImage(holo3)
    ref_image3 = hs.signals.HologramImage(ref3)

    if lazy:
        ref_image3 = ref_image3.as_lazy()
        holo_image3 = holo_image3.as_lazy()

    wave_image3 = holo_image3.reconstruct_phase(
        ref_image3.inav[0, 0], sb='upper', parallel=parallel)

    # Cropping the reconstructed and original phase images and comparing:
    phase3_new_crop = wave_image3.unwrapped_phase(parallel=parallel)
    phase3_new_crop.crop(2, Y_START, Y_STOP)
    phase3_new_crop.crop(3, X_START, X_STOP)
    phase3_ref_crop = phase_ref3.reshape(newshape)[:, :, X_START:X_STOP,
                                                   Y_START:Y_STOP]
    assert_allclose(phase3_new_crop.data, phase3_ref_crop, atol=2E-2)

    # 3a. Testing reconstruction with input parameters in 'nm' and with multiple parameter input,
    # but reference ndim=0:

    sb_position3 = ref_image3.estimate_sideband_position(
        sb='upper', parallel=parallel)
    f_sampling = np.divide(1, [
        a * b
        for a, b in zip(ref_image3.axes_manager.signal_shape,
                        (ref_image3.axes_manager.signal_axes[0].scale,
                         ref_image3.axes_manager.signal_axes[1].scale))
    ])
    sb_size3 = ref_image3.estimate_sideband_size(
        sb_position3, parallel=parallel) * np.mean(f_sampling)
    sb_smoothness3 = sb_size3 * 0.05
    sb_units3 = 'nm'

    wave_image3a = holo_image3.reconstruct_phase(
        ref_image3.inav[0, 0],
        sb_position=sb_position3,
        sb_size=sb_size3,
        sb_smoothness=sb_smoothness3,
        sb_unit=sb_units3,
        parallel=parallel)
    phase3a_new_crop = wave_image3a.unwrapped_phase(parallel=parallel)
    phase3a_new_crop.crop(2, Y_START, Y_STOP)
    phase3a_new_crop.crop(3, X_START, X_STOP)
    assert_allclose(phase3a_new_crop.data, phase3_ref_crop, atol=2E-2)
    # a. Mismatch of navigation dimensions of object and reference
    # holograms, except if reference hologram ndim=0
    with pytest.raises(ValueError):
        holo_image3.reconstruct_phase(ref_image3.inav[0, :])
    reference4a = ref_image3.inav[0, :]
    reference4a.set_signal_type('signal2d')
    with pytest.raises(ValueError):
        holo_image3.reconstruct_phase(reference=reference4a)
    #   b. Mismatch of signal shapes of object and reference holograms
    with pytest.raises(ValueError):
        holo_image3.reconstruct_phase(
            ref_image3.inav[:, :].isig[Y_START:Y_STOP, X_START:X_STOP])

    #   c. Mismatch of signal shape of sb_position
    sb_position_mismatched = hs.signals.Signal2D(np.arange(9).reshape((3, 3)))
    with pytest.raises(ValueError):
        holo_image3.reconstruct_phase(
            sb_position=sb_position_mismatched)
    #   d. Mismatch of navigation dimensions of reconstruction parameters
    sb_position_mismatched = hs.signals.Signal1D(np.arange(16).reshape((8, 2)))
    sb_size_mismatched = hs.signals.BaseSignal(np.arange(9)).T
    sb_smoothness_mismatched = hs.signals.BaseSignal(np.arange(9)).T
    with pytest.raises(ValueError):
        holo_image3.reconstruct_phase(
            sb_position=sb_position_mismatched)
    with pytest.raises(ValueError):
        holo_image3.reconstruct_phase(
            sb_size=sb_size_mismatched)
    with pytest.raises(ValueError):
        holo_image3.reconstruct_phase(
            sb_smoothness=sb_smoothness_mismatched)

    #   e. Beam energy is not assigned, while 'mrad' units selected
    with pytest.raises(AttributeError):
        holo_image3.reconstruct_phase(
            sb_size=40, sb_unit='mrad')


def _generate_parameters():
    parameters = []
    for parallel in [False, True]:
        for lazy in [False, True]:
            for single_values in [False, True]:
                for fcalgo in ['fourier', 'statistical']:
                    parameters.append([parallel,
                                       lazy,
                                       single_values,
                                       fcalgo])
    return parameters


@pytest.mark.parametrize('parallel, lazy, single_values, fringe_contrast_algorithm',
                         _generate_parameters())
def test_statistics(parallel, lazy, single_values, fringe_contrast_algorithm):
    # Parameters measured using Gatan HoloWorks:
    REF_FRINGE_SPACING = 3.48604
    REF_FRINGE_SAMPLING = 3.7902

    # Measured using the definition of fringe contrast from the centre of image
    REF_FRINGE_CONTRAST = 0.0736

    RTOL = 1e-5

    # 0. Prepare test data and derived statistical parameters

    ref_carrier_freq = 1. / REF_FRINGE_SAMPLING
    ref_carrier_freq_nm = 1. / REF_FRINGE_SPACING

    ref_holo = hs.stack([reference_hologram(), reference_hologram()])
    ref_holo = hs.stack([ref_holo, ref_holo, ref_holo])

    ht = ref_holo.metadata.Acquisition_instrument.TEM.beam_energy
    momentum = 2 * constants.m_e * constants.elementary_charge * ht * \
        1000 * (1 + constants.elementary_charge * ht *
                1000 / (2 * constants.m_e * constants.c ** 2))
    wavelength = constants.h / np.sqrt(momentum) * 1e9  # in nm
    ref_carrier_freq_mrad = ref_carrier_freq_nm * 1000 * wavelength

    if lazy:
        ref_holo.as_lazy()

    # 1. Test core functionality

    stats = ref_holo.statistics(high_cf=True,
                                parallel=parallel,
                                single_values=single_values,
                                fringe_contrast_algorithm=fringe_contrast_algorithm)
    if single_values:
        # Fringe contrast in experimental conditions can be only an estimate
        # therefore tolerance is 10%:
        assert_allclose(
            stats['Fringe contrast'],
            REF_FRINGE_CONTRAST,
            rtol=0.1)

        assert_allclose(
            stats['Fringe sampling (px)'],
            REF_FRINGE_SAMPLING,
            rtol=RTOL)
        assert_allclose(
            stats['Fringe spacing (nm)'],
            REF_FRINGE_SPACING,
            rtol=RTOL)
        assert_allclose(
            stats['Carrier frequency (1 / nm)'],
            ref_carrier_freq_nm,
            rtol=RTOL)
        assert_allclose(
            stats['Carrier frequency (1/px)'],
            ref_carrier_freq,
            rtol=RTOL)
        assert_allclose(
            stats['Carrier frequency (mrad)'],
            ref_carrier_freq_mrad,
            rtol=RTOL)
    else:
        ref_fringe_contrast_stack = np.repeat(
            REF_FRINGE_CONTRAST, 6).reshape((3, 2))
        ref_fringe_sampling_stack = np.repeat(
            REF_FRINGE_SAMPLING, 6).reshape((3, 2))
        ref_fringe_spacing_stack = np.repeat(
            REF_FRINGE_SPACING, 6).reshape((3, 2))
        ref_carrier_freq_nm_stack = np.repeat(
            ref_carrier_freq_nm, 6).reshape((3, 2))
        ref_carrier_freq_stack = np.repeat(ref_carrier_freq, 6).reshape((3, 2))
        ref_carrier_freq_mrad_stack = np.repeat(
            ref_carrier_freq_mrad, 6).reshape((3, 2))

        # Fringe contrast in experimental conditions can be only an estimate
        # therefore tolerance is 10%:
        assert_allclose(
            stats['Fringe contrast'].data,
            ref_fringe_contrast_stack,
            rtol=0.1)

        assert_allclose(
            stats['Fringe sampling (px)'].data,
            ref_fringe_sampling_stack,
            rtol=RTOL)
        assert_allclose(
            stats['Fringe spacing (nm)'].data,
            ref_fringe_spacing_stack,
            rtol=RTOL)
        assert_allclose(
            stats['Carrier frequency (1 / nm)'].data,
            ref_carrier_freq_nm_stack,
            rtol=RTOL)
        assert_allclose(
            stats['Carrier frequency (1/px)'].data,
            ref_carrier_freq_stack,
            rtol=RTOL)
        assert_allclose(
            stats['Carrier frequency (mrad)'].data,
            ref_carrier_freq_mrad_stack,
            rtol=RTOL)

    # 2. Test raises:
    holo_raise = hs.signals.HologramImage(np.random.random(20).reshape((5, 4)))

    # 2a. Test raise for absent units:
    with pytest.raises(ValueError):
        holo_raise.statistics(sb_position=(1, 1))
    holo_raise.axes_manager.signal_axes[0].units = 'nm'
    holo_raise.axes_manager.signal_axes[1].units = 'nm'

    # 2b. Test raise for absent beam_energy:
    with pytest.raises(AttributeError):
        holo_raise.statistics(sb_position=(1, 1))
    holo_raise.set_microscope_parameters(beam_energy=300.)

    # 2c. Test raise for wrong value of `fringe_contrast_algorithm`
    with pytest.raises(ValueError):
        holo_raise.statistics(
            sb_position=(
                1,
                1),
            fringe_contrast_algorithm='pure_guess')


if __name__ == '__main__':

    import pytest
    pytest.main(__name__)
