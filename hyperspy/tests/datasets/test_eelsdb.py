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

import warnings

import pytest
import requests
from requests.exceptions import SSLError

from hyperspy.misc.eels.eelsdb import eelsdb


def _eelsdb(**kwargs):
    try:
        ss = eelsdb(**kwargs)
    except SSLError:
        warnings.warn(
            "The https://eelsdb.eu certificate seems to be invalid. "
            "Consider notifying the issue to the EELSdb webmaster.")
        ss = eelsdb(verify_certificate=False, **kwargs)
    except Exception as e:
        # e.g. failures such as ConnectionError or MaxRetryError
        pytest.skip(f"Skipping eelsdb test due to {e}")
    return ss


def eelsdb_down():
    try:
        _ = requests.get('https://api.eelsdb.eu', verify=True)
        return False
    except SSLError:
        _ = requests.get('https://api.eelsdb.eu', verify=False)
        return False
    except requests.exceptions.ConnectionError:
        return True


@pytest.mark.skipif(eelsdb_down(), reason="Unable to connect to EELSdb")
def test_eelsdb_eels():
    ss = _eelsdb(
        title="Boron Nitride Multiwall Nanotube",
        formula="BN",
        spectrum_type="coreloss",
        edge="K",
        min_energy=370,
        max_energy=1000,
        min_energy_compare="gt",
        max_energy_compare="lt",
        resolution="0.7 eV",
        resolution_compare="lt",
        max_n=2,
        order="spectrumMin",
        order_direction='DESC',
        monochromated=False,
        )

    assert len(ss) == 2
    md = ss[0].metadata
    assert md.General.author == "Odile Stephan"
    assert (
        md.Acquisition_instrument.TEM.Detector.EELS.collection_angle == 24)
    assert md.Acquisition_instrument.TEM.convergence_angle == 15
    assert md.Acquisition_instrument.TEM.beam_energy == 100
    assert md.Signal.signal_type == "EELS"
    assert "perpendicular" in md.Sample.description
    assert "parallel" in ss[1].metadata.Sample.description
    assert md.Sample.chemical_formula == "BN"
    assert md.Acquisition_instrument.TEM.microscope == "STEM-VG"


@pytest.mark.skipif(eelsdb_down(), reason="Unable to connect to EELSdb")
def test_eelsdb_xas():
    ss = _eelsdb(spectrum_type="xrayabs", max_n=1,)
    assert len(ss) == 1
    md = ss[0].metadata
    assert md.Signal.signal_type == "XAS"


@pytest.mark.skipif(eelsdb_down(), reason="Unable to connect to EELSdb")
def test_eelsdb_corrupted_file():
    ss = _eelsdb(
        spectrum_type='coreloss', element='Cu', edge='K', formula='Cu4O3'
        )
    assert len(ss) == 1
    assert ss[0].metadata.General.title == 'O-K edge in Cu4O3'


@pytest.mark.skipif(eelsdb_down(), reason="Unable to connect to EELSdb")
def test_eelsdb_elements_no():
    title = "Zero-loss c-FEG Hitachi Disp 0.214 eV"
    ss = _eelsdb(author='Luc Lajaunie', title=title)
    assert len(ss) == 1
    assert ss[0].metadata.General.title == title


@pytest.mark.skipif(eelsdb_down(), reason="Unable to connect to EELSdb")
def test_eelsdb_txt_file():
    _ = _eelsdb(title="Porous cobalt coating with He-filled nanopores")
