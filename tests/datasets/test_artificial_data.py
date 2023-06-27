# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import pytest

import hyperspy.datasets.artificial_data as ad


@pytest.mark.parametrize("add_noise", (True, False))
def test_get_low_loss_eels_signal(add_noise):
    s = ad.get_low_loss_eels_signal(add_noise=add_noise)
    assert s.metadata.Signal.signal_type == 'EELS'


def test_get_core_loss_eels_signal():
    s = ad.get_core_loss_eels_signal(add_powerlaw=False)
    assert s.metadata.Signal.signal_type == 'EELS'
    s1 = ad.get_core_loss_eels_signal(add_powerlaw=True)
    assert s1.metadata.Signal.signal_type == 'EELS'
    assert s1.data.sum() > s.data.sum()

    s2 = ad.get_core_loss_eels_signal(add_noise=True, random_state=10)
    s3 = ad.get_core_loss_eels_signal(add_noise=True, random_state=10)
    assert (s2.data == s3.data).all()

@pytest.mark.parametrize("add_noise", (True, False))
def test_get_core_loss_eels_model(add_noise):
    m = ad.get_core_loss_eels_model(add_powerlaw=False, add_noise=add_noise)
    assert m.signal.metadata.Signal.signal_type == 'EELS'
    m1 = ad.get_core_loss_eels_model(add_powerlaw=True, add_noise=add_noise)
    assert m1.signal.metadata.Signal.signal_type == 'EELS'
    assert m1.signal.data.sum() > m.signal.data.sum()


@pytest.mark.parametrize("add_noise", (True, False))
def test_get_low_loss_eels_line_scan_signal(add_noise):
    s = ad.get_low_loss_eels_line_scan_signal(add_noise=add_noise)
    assert s.metadata.Signal.signal_type == 'EELS'


@pytest.mark.parametrize("add_powerlaw", (True, False))
@pytest.mark.parametrize("add_noise", (True, False))
def test_get_core_loss_eels_line_scan_signal(add_powerlaw, add_noise):
    s = ad.get_core_loss_eels_line_scan_signal(add_powerlaw, add_noise)
    assert s.metadata.Signal.signal_type == 'EELS'


def test_get_atomic_resolution_tem_signal2d():
    s = ad.get_atomic_resolution_tem_signal2d()
    assert s.axes_manager.signal_dimension == 2

@pytest.mark.parametrize("navigation_dimension",(0,1,2,3))
@pytest.mark.parametrize("uniform",(True,False))
@pytest.mark.parametrize("add_baseline",(True,False))
@pytest.mark.parametrize("add_noise",(True,False))
def test_get_luminescence_signal(navigation_dimension, uniform, add_baseline, add_noise):
    #Creating signal
    s = ad.get_luminescence_signal(navigation_dimension,
                                                uniform,
                                                add_baseline,
                                                add_noise)
    #Checking that dimension initialisation works
    assert tuple([10 for i in range(navigation_dimension)]+[1024]) == s.data.shape
    #Verifying that both functional and uniform data axis work
    sax = s.axes_manager.signal_axes[0]
    assert sax.is_uniform == uniform
    #Verifying that baseline works
    if add_baseline:
        assert s.data.min()>340
    #Verification that noise works
    #Case of baseline + energy axis is discarded because of
    #jacobian transformation
    if not(add_baseline and not uniform):
        #Verify that adding noise works
        noisedat = s.isig[:100].data
        assert (noisedat.std()>0.1)==add_noise
