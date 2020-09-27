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

import pytest

import hyperspy.api as hs

imagecodecs = pytest.importorskip("imagecodecs", reason="skipping test_phenom tests, requires imagecodecs")

DIRPATH = os.path.dirname(__file__)
ELID2VERSION0 = os.path.join(DIRPATH, 'phenom_data', 'Elid2Version0.elid')
ELID2VERSION1 = os.path.join(DIRPATH, 'phenom_data', 'Elid2Version1.elid')
ELID2VERSION2 = os.path.join(DIRPATH, 'phenom_data', 'Elid2Version2.elid')


@pytest.mark.parametrize(('pathname'), [ELID2VERSION0, ELID2VERSION1, ELID2VERSION2])
def test_elid(pathname):
    s = hs.load(pathname)
    assert len(s) == 11

    assert s[0].data.shape == (16, 20)
    assert s[0].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'y', 'scale': 0.9757792598920122, 'offset': 0.0, 'size': 16, 'units': 'µm', 'navigate': True},
        'axis-1': {'name': 'x', 'scale': 0.9757792598920122, 'offset': 0.0, 'size': 20, 'units': 'µm', 'navigate': True}
    }
    assert s[0].metadata['Acquisition_instrument']['SEM']['Stage']['x'] == -2.586744298575455
    assert s[0].metadata['Acquisition_instrument']['SEM']['Stage']['y'] == -0.7322168400784014
    assert s[0].metadata['Acquisition_instrument']['SEM']['beam_energy'] == 15.0
    assert s[0].metadata['Acquisition_instrument']['SEM']['microscope'] == 'MVE027364-0026-L'
    assert s[0].metadata['General']['date'] == '2019-08-07'
    assert s[0].metadata['General']['original_filename'] == os.path.split(pathname)[1]
    assert s[0].metadata['General']['time'] == '09:37:31'
    assert s[0].metadata['General']['title'] == 'Image 1'
    assert s[0].metadata['Signal']['binned'] == False
    assert s[0].metadata['Signal']['signal_type'] == ''
    assert s[0].original_metadata['acquisition']['scan']['dwellTime']['value'] == '200'
    assert s[0].original_metadata['acquisition']['scan']['dwellTime']['unit'] == 'ns'
    assert s[0].original_metadata['acquisition']['scan']['fieldSize'] == 0.000019515585197840245
    assert s[0].original_metadata['acquisition']['scan']['highVoltage']['value'] == '-15'
    assert s[0].original_metadata['acquisition']['scan']['highVoltage']['unit'] == 'kV'
    assert s[0].original_metadata['pixelWidth']['value'] == '975.7792598920121'
    assert s[0].original_metadata['pixelWidth']['unit'] == 'nm'
    assert s[0].original_metadata['pixelHeight']['value'] == '975.7792598920121'
    assert s[0].original_metadata['pixelHeight']['unit'] == 'nm'
    assert s[0].original_metadata['samplePosition']['x'] == '-0.002586744298575455'
    assert s[0].original_metadata['samplePosition']['y'] == '-0.0007322168400784014'
    assert s[0].original_metadata['workingDistance']['value'] == '8.141749999999993'
    assert s[0].original_metadata['workingDistance']['unit'] == 'mm'
    assert s[0].original_metadata['instrument']['softwareVersion'] == '5.4.5.rc1.bb8fbe3.23039'
    assert s[0].original_metadata['instrument']['type'] == 'PhenomXL'
    assert s[0].original_metadata['instrument']['uniqueID'] == 'MVE027364-0026-L'

    assert s[1].metadata['General']['title'] == 'Image 1, Spot 1'
    assert s[1].data.shape == (2048,)
    assert s[1].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'Energy', 'scale': 0.00988676802994421, 'offset': -0.03634370080990722, 'size': 2048, 'units': 'keV', 'navigate': False}
    }
    assert s[1].data.tolist()[0:300] == [
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,15,16,19,30,52,61,98,125,
        145,129,114,69,45,22,14,11,18,17,30,26,29,19,16,20,26,29,35,51,59,103,
        139,157,209,220,179,113,99,65,49,31,36,39,42,35,48,37,55,50,45,46,49,
        40,49,54,35,49,57,63,71,64,75,76,92,98,83,81,94,118,120,160,215,325,
        363,368,429,403,376,254,204,173,136,124,102,89,97,84,83,75,83,71,85,
        101,81,72,87,84,90,93,84,68,93,91,82,86,112,85,84,100,110,118,132,118,
        125,138,128,135,143,143,136,148,227,301,538,1077,1946,3319,5108,7249,
        9032,10755,11441,10804,9219,7245,5335,3568,2213,1455,825,543,338,283,
        196,160,123,104,105,92,88,109,89,88,82,95,88,91,87,108,86,85,59,77,72,
        58,66,69,64,76,56,67,58,60,59,71,56,57,62,50,67,59,59,52,45,60,53,57,
        59,39,43,55,54,40,43,37,39,41,52,39,53,41,48,40,41,45,36,45,32,40,44,
        43,55,50,45,59,45,44,66,52,67,74,83,90,92,114,130,131,114,100,100,106,
        103,84,87,77,76,82,83,78,81,63,49,54,64,45,41,40,41,38,50,39,45,44,42,
        44,31,36,38,37,55,40,32,34,32,34,37,27,28,45,35,24,40,22,29,33,33,44,34]
    assert s[1].original_metadata['acquisition']['scan']['detectors']['EDS']['live_time'] == 5.7203385
    assert s[1].original_metadata['acquisition']['scan']['detectors']['EDS']['real_time'] == 10.162500000000001
    assert s[1].original_metadata['acquisition']['scan']['detectors']['EDS']['fast_peaking_time'] == 100e-9
    assert s[1].original_metadata['acquisition']['scan']['detectors']['EDS']['slow_peaking_time'] == 11.2e-6

    assert s[2].metadata['General']['title'] == 'Image 1, Region 2'
    assert s[2].data.shape == (2048,)
    assert s[2].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'Energy', 'scale': 0.00988676802994421, 'offset': -0.03634370080990722, 'size': 2048, 'units': 'keV', 'navigate': False}
    }
    assert s[2].original_metadata['acquisition']['scan']['detectors']['EDS']['live_time'] == 6.5802053
    assert s[2].original_metadata['acquisition']['scan']['detectors']['EDS']['real_time'] == 10.177700000000003

    assert s[3].metadata['General']['title'] == 'Image 1, Map 3'
    assert s[3].data.shape == (16, 16, 2048)
    assert s[3].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'y', 'scale': 1.2197240748650153, 'offset': 0.0, 'size': 16, 'units': 'µm', 'navigate': True},
        'axis-1': {'name': 'x', 'scale': 1.2197240748650153, 'offset': 0.0, 'size': 16, 'units': 'µm', 'navigate': True},
        'axis-2': {'name': 'X-ray energy', 'scale': 0.00988676802994421, 'offset': -0.03634370080990722, 'size': 2048, 'units': 'keV', 'navigate': False}
    }
    assert s[3].original_metadata['acquisition']['scan']['detectors']['EDS']['live_time'] == 4.047052
    assert s[3].original_metadata['acquisition']['scan']['detectors']['EDS']['real_time'] == 3.0005599999999997

    assert s[4].metadata['General']['title'] == 'Image 1, Line 4'
    assert s[4].data.shape == (64, 2048)
    assert s[4].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'i', 'scale': 1.0, 'offset': 0.0, 'size': 64, 'units': 'points', 'navigate': True},
        'axis-1': {'name': 'X-ray energy', 'scale': 0.00988676802994421, 'offset': -0.03634370080990722, 'size': 2048, 'units': 'keV', 'navigate': False}
    }
    assert s[4].original_metadata['acquisition']['scan']['detectors']['EDS']['live_time'] == 5.504343599999998
    assert s[4].original_metadata['acquisition']['scan']['detectors']['EDS']['real_time'] == 6.410299999999996

    assert s[5].metadata['General']['title'] == 'Image 1, Map 6'
    assert s[5].data.shape == (16, 16, 2048)
    assert s[5].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'y', 'scale': 1.2197240748650153, 'offset': 0.0, 'size': 16, 'units': 'µm', 'navigate': True},
        'axis-1': {'name': 'x', 'scale': 1.2197240748650153, 'offset': 0.0, 'size': 16, 'units': 'µm', 'navigate': True},
        'axis-2': {'name': 'X-ray energy', 'scale': 0.009886797201840245, 'offset': -0.04478043655810262, 'size': 2048, 'units': 'keV', 'navigate': False}
    }
    assert s[5].original_metadata['acquisition']['scan']['detectors']['EDS']['live_time'] == 4.5919591
    assert s[5].original_metadata['acquisition']['scan']['detectors']['EDS']['real_time'] == 3.00056

    assert s[6].metadata['General']['title'] == 'Image 1, Difference 3 - 6'
    assert s[6].data.shape == (2048,)
    assert s[6].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'Energy', 'scale': 0.00988676802994421, 'offset': -0.03634370080990722, 'size': 2048, 'units': 'keV', 'navigate': False}
    }

    assert s[7].metadata['General']['title'] == '385test - spectrum'
    assert s[7].data.shape == (24, 32)
    assert s[7].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'y', 'scale': 1.0, 'offset': 0.0, 'size': 24, 'units': 'points', 'navigate': True},
        'axis-1': {'name': 'x', 'scale': 1.0, 'offset': 0.0, 'size': 32, 'units': 'points', 'navigate': True}
    }
    assert not 'acquisition' in s[7].original_metadata

    assert s[8].metadata['General']['title'] == '385test - spectrum, MSA 1'
    assert s[8].data.shape == (2048,)
    assert s[8].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'Energy', 'scale': 0.0098868, 'offset': -0.0363437, 'size': 2048, 'units': 'keV', 'navigate': False}
    }
    assert s[8].original_metadata['acquisition']['scan']['detectors']['EDS']['live_time'] == 0.0
    assert s[8].original_metadata['acquisition']['scan']['detectors']['EDS']['real_time'] == 5.066

    assert s[9].metadata['General']['title'] == 'Image 1'
    assert s[9].data.shape == (35, 40)
    assert s[9].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'y', 'scale': 0.8120422280865187, 'offset': 0.0, 'size': 35, 'units': 'µm', 'navigate': True},
        'axis-1': {'name': 'x', 'scale': 0.8120422280865187, 'offset': 0.0, 'size': 40, 'units': 'µm', 'navigate': True}
    }
    assert not 'EDS' in s[9].original_metadata['acquisition']['scan']['detectors']

    assert s[10].metadata['General']['title'] == 'Image 1, Map 1'
    assert s[10].data.shape == (16, 16, 2048)
    assert s[10].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'y', 'scale': 2.0301055702162967, 'offset': 0.0, 'size': 16, 'units': 'µm', 'navigate': True},
        'axis-1': {'name': 'x', 'scale': 2.0301055702162967, 'offset': 0.0, 'size': 16, 'units': 'µm', 'navigate': True},
        'axis-2': {'name': 'X-ray energy', 'scale': 0.009886797201840245, 'offset': -0.04478043655810262, 'size': 2048, 'units': 'keV', 'navigate': False}
    }
    assert s[10].original_metadata['acquisition']['scan']['detectors']['EDS']['live_time'] == 4.821238
    assert s[10].original_metadata['acquisition']['scan']['detectors']['EDS']['real_time'] == 3.0005600000000006
