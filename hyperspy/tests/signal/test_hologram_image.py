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

import numpy as np

import numpy.testing as nt
from numpy.testing import assert_allclose
import pytest

import hyperspy.api as hs
from hyperspy.decorators import lazifyTestClass
from scipy.interpolate import interp2d


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
    assert (holo_image.metadata.Acquisition_instrument.TEM.tilt_stage == 2.2)


def calc_holo(x, y, phase_ref, fringe_spacing, fringe_direction):
    return 2 * (1 + np.cos(phase_ref + fringe_spacing / (2 * np.pi) * (
        x * np.cos(fringe_direction) + y * np.sin(fringe_direction))))


def calc_phaseref(x, y, z, img_sizex, img_sizey):
    return np.pi * x * (1 - z) / img_sizex + np.pi * y * z / img_sizey


img_size = 1024
img_size3x = 768
img_size3y = 512
fringe_direction = -np.pi / 6
fringe_spacing = 5.23
fringe_direction3 = np.pi / 3
fringe_spacing3 = 6.11
ls = np.linspace(-img_size / 2, img_size / 2 - 1, img_size)

x, y = np.meshgrid(ls, ls)
phase_ref = calc_phaseref(x, 0, 0, img_size / 2.2, 1)
holo = calc_holo(x, y, phase_ref, fringe_spacing, fringe_direction)
ref = calc_holo(x, y, 0, fringe_spacing, fringe_direction)

x2, z2, y2 = np.meshgrid(ls, np.array([0, 1]), ls)
phase_ref2 = calc_phaseref(x2, y2, z2, img_size / 2.2, img_size / 2.2)
holo2 = calc_holo(x2, y2, phase_ref2, fringe_spacing, fringe_direction)
ref2 = calc_holo(x2, y2, 0, fringe_spacing, fringe_direction)

x3, z3, y3 = np.meshgrid(
    np.linspace(-img_size3x / 2, img_size3x / 2, img_size3x),
    np.arange(6), np.linspace(-img_size3y / 2, img_size3y / 2, img_size3y))
phase_ref3 = calc_phaseref(x3, y3, z3 / 6, img_size3x * 2, img_size3y * 2)
newshape = (2, 3, img_size3x, img_size3y)
holo3 = calc_holo(x3, y3, phase_ref3, fringe_spacing3,
                  fringe_direction3).reshape(newshape)
ref3 = calc_holo(x3, y3, 0, fringe_spacing3,
                 fringe_direction3).reshape(newshape)
x_start = int(img_size3x / 9)
# larger fringes require larger crop
x_stop = img_size3x - 1 - int(img_size3x / 9)
y_start = int(img_size3y / 9)
y_stop = img_size3y - 1 - int(img_size3y / 9)

@pytest.mark.parametrize('parallel,lazy', [(True, False), (False, False),
                                           (None, True)])
def test_reconstruct_phase_single(parallel, lazy):
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
    x_start = int(wave_image.axes_manager.signal_shape[0] / 10)
    x_stop = int(wave_image.axes_manager.signal_shape[0] * 9 / 10)
    wave_crop = wave_image.data[x_start:x_stop, x_start:x_stop]
    wave_cc_crop = wave_image_cc.data[x_start:x_stop, x_start:x_stop]

    # asserts that waves from different
    assert_allclose(wave_crop, np.conj(wave_cc_crop), rtol=1e-3)
    # sidebands are complex conjugate; this also tests possibility of
    # reconstruction with given sideband parameters

    # Cropping the reconstructed and original phase images and comparing:
    x_start = int(img_size / 10)
    x_stop = img_size - 1 - int(img_size / 10)
    phase_new_crop = wave_image.unwrapped_phase(
        parallel=parallel).data[x_start:x_stop, x_start:x_stop]
    phase_ref_crop = phase_ref[x_start:x_stop, x_start:x_stop]
    assert_allclose(phase_new_crop, phase_ref_crop, atol=1E-2)


@pytest.mark.parametrize('parallel,lazy', [(True, False), (False, False),
                                           (None, True)])
def test_reconstruct_phase_nonstandard(parallel, lazy):
    # 2. Testing reconstruction with non-standard output size for stacked
    # images:
    holo_image2 = hs.signals.HologramImage(holo2)
    ref_image2 = hs.signals.HologramImage(ref2)

    if lazy:
        ref_image2 = ref_image2.as_lazy()
        holo_image2 = holo_image2.as_lazy()

    sb_position2 = ref_image2.estimate_sideband_position(
        sb='upper', parallel=parallel)
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

    x_start = int(img_size / 10)
    x_stop = img_size - 1 - int(img_size / 10)
    phase_new_crop0 = phase_new0[x_start:x_stop, x_start:x_stop]
    phase_new_crop1 = phase_new1[x_start:x_stop, x_start:x_stop]
    phase_ref_crop0 = phase_ref2[0, x_start:x_stop, x_start:x_stop]
    phase_ref_crop1 = phase_ref2[1, x_start:x_stop, x_start:x_stop]
    assert_allclose(phase_new_crop0, phase_ref_crop0, atol=0.05)
    assert_allclose(phase_new_crop1, phase_ref_crop1, atol=0.01)


@pytest.mark.parametrize('parallel,lazy', [(True, False), (False, False),
                                           (None, True)])
def test_reconstruct_phase_multi(parallel, lazy):

    holo_image3 = hs.signals.HologramImage(holo3)
    ref_image3 = hs.signals.HologramImage(ref3)

    if lazy:
        ref_image3 = ref_image3.as_lazy()
        holo_image3 = holo_image3.as_lazy()

    wave_image3 = holo_image3.reconstruct_phase(
        ref_image3.inav[0, 0], sb='upper', parallel=parallel)

    # Cropping the reconstructed and original phase images and comparing:
    phase3_new_crop = wave_image3.unwrapped_phase(parallel=parallel)
    phase3_new_crop.crop(2, y_start, y_stop)
    phase3_new_crop.crop(3, x_start, x_stop)
    phase3_ref_crop = phase_ref3.reshape(newshape)[:, :, x_start:x_stop,
                                                   y_start:y_stop]
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
    phase3a_new_crop.crop(2, y_start, y_stop)
    phase3a_new_crop.crop(3, x_start, x_stop)
    assert_allclose(phase3a_new_crop.data, phase3_ref_crop, atol=2E-2)

def test_reconstruct_phase_raises():
    holo_image3 = hs.signals.HologramImage(holo3)
    ref_image3 = hs.signals.HologramImage(ref3)
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
            ref_image3.inav[:, :].isig[y_start:y_stop, x_start:x_stop])

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


if __name__ == '__main__':

    import pytest
    pytest.main(__name__)
