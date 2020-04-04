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


DIRPATH = os.path.dirname(__file__)
FILE1 = os.path.join(DIRPATH, 'phenom_data', 'Elid2Version0.elid')
FILE2 = os.path.join(DIRPATH, 'phenom_data', 'Elid2Version1.elid')


def test_load1():
    s = hs.load(FILE1)

    assert len(s) == 11

    assert s[0].data.shape == (1024, 1024)
    assert s[0].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'y', 'scale': 0.9757792598920122, 'offset': 0.0, 'size': 1024, 'units': 'µm', 'navigate': True},
        'axis-1': {'name': 'x', 'scale': 0.9757792598920122, 'offset': 0.0, 'size': 1024, 'units': 'µm', 'navigate': True}
    }
    assert s[0].metadata['Acquisition_instrument']['SEM']['Stage']['x'] == -2.586744298575455
    assert s[0].metadata['Acquisition_instrument']['SEM']['Stage']['y'] == -0.7322168400784014
    assert s[0].metadata['Acquisition_instrument']['SEM']['beam_energy'] == 15.0
    assert s[0].metadata['Acquisition_instrument']['SEM']['microscope'] == 'MVE027364-0026-L'
    assert s[0].metadata['General']['date'] == '2019-08-07'
    assert s[0].metadata['General']['original_filename'] == 'Elid2Version0.elid'
    assert s[0].metadata['General']['time'] == '09:37:31'
    assert s[0].metadata['General']['title'] == 'Image 1'
    assert s[0].metadata['Signal']['binned'] == False
    assert s[0].metadata['Signal']['signal_type'] == 'image'
    assert s[0].original_metadata['acquisition']['scan']['dwellTime']['value'] == '200'
    assert s[0].original_metadata['acquisition']['scan']['dwellTime']['unit'] == 'ns'
    assert s[0].original_metadata['acquisition']['scan']['fieldSize'] == 0.0009991979621294205
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
    assert s[1].original_metadata['acquisition']['scan']['detectors']['EDS']['live_time'] == 5.7203385
    assert s[1].original_metadata['acquisition']['scan']['detectors']['EDS']['real_time'] == 10.162500000000001

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
        'axis-0': {'name': 'y', 'scale': 62.44987263308878, 'offset': 0.0, 'size': 16, 'units': 'µm', 'navigate': True},
        'axis-1': {'name': 'x', 'scale': 62.44987263308878, 'offset': 0.0, 'size': 16, 'units': 'µm', 'navigate': True},
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
        'axis-0': {'name': 'y', 'scale': 62.44987263308878, 'offset': 0.0, 'size': 16, 'units': 'µm', 'navigate': True},
        'axis-1': {'name': 'x', 'scale': 62.44987263308878, 'offset': 0.0, 'size': 16, 'units': 'µm', 'navigate': True},
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
    assert s[7].data.shape == (1047, 1047)
    assert s[7].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'y', 'scale': 1.0, 'offset': 0.0, 'size': 1047, 'units': 'points', 'navigate': True},
        'axis-1': {'name': 'x', 'scale': 1.0, 'offset': 0.0, 'size': 1047, 'units': 'points', 'navigate': True}
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
    assert s[9].data.shape == (1024, 1024)
    assert s[9].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'y', 'scale': 0.8120422280865187, 'offset': 0.0, 'size': 1024, 'units': 'µm', 'navigate': True},
        'axis-1': {'name': 'x', 'scale': 0.8120422280865187, 'offset': 0.0, 'size': 1024, 'units': 'µm', 'navigate': True}
    }
    assert not 'EDS' in s[9].original_metadata['acquisition']['scan']['detectors']

    assert s[10].metadata['General']['title'] == 'Image 1, Map 1'
    assert s[10].data.shape == (16, 16, 2048)
    assert s[10].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'y', 'scale': 51.9707025975372, 'offset': 0.0, 'size': 16, 'units': 'µm', 'navigate': True},
        'axis-1': {'name': 'x', 'scale': 51.9707025975372, 'offset': 0.0, 'size': 16, 'units': 'µm', 'navigate': True},
        'axis-2': {'name': 'X-ray energy', 'scale': 0.009886797201840245, 'offset': -0.04478043655810262, 'size': 2048, 'units': 'keV', 'navigate': False}
    }
    assert s[10].original_metadata['acquisition']['scan']['detectors']['EDS']['live_time'] == 4.821238
    assert s[10].original_metadata['acquisition']['scan']['detectors']['EDS']['real_time'] == 3.0005600000000006


def dont_test_load2():
    s = hs.load(FILE2)

    assert len(s) == 22

    assert s[0].metadata['General']['date'] == '2020-01-21'
    assert s[0].metadata['General']['original_filename'] == 'Elid2Version1.elid'
    assert s[0].metadata['General']['time'] == '09:18:42'
    assert s[1].metadata['General']['title'] == 'Image 1, Spot 1'
    assert s[1].data.shape == (2048,)
    assert s[1].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'Energy', 'scale': 0.009957494815722499, 'offset': -0.043546104722405436, 'size': 2048, 'units': 'keV', 'navigate': False}
    }

    assert s[2].metadata['General']['title'] == 'Image 2'
    assert s[2].data.shape == (600, 960)
    assert s[2].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'y', 'scale': 0.7810477081594434, 'offset': 0.0, 'size': 600, 'units': 'µm', 'navigate': True},
        'axis-1': {'name': 'x', 'scale': 0.7810477081594434, 'offset': 0.0, 'size': 960, 'units': 'µm', 'navigate': True}
    }

    assert s[3].metadata['General']['title'] == 'Image 2, Spot 1'
    assert s[3].data.shape == (2048,)
    assert s[3].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'Energy', 'scale': 0.009957494815722499, 'offset': -0.043546104722405436, 'size': 2048, 'units': 'keV', 'navigate': False}
    }

    assert s[4].metadata['General']['title'] == 'Image 2, Spot 2'
    assert s[4].data.shape == (2048,)
    assert s[4].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'Energy', 'scale': 0.009957494815722499, 'offset': -0.043546104722405436, 'size': 2048, 'units': 'keV', 'navigate': False}
    }

    assert s[5].metadata['General']['title'] == 'Image 3'
    assert s[5].data.shape == (600, 960)
    assert s[5].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'y', 'scale': 0.7810477081594434, 'offset': 0.0, 'size': 600, 'units': 'µm', 'navigate': True},
        'axis-1': {'name': 'x', 'scale': 0.7810477081594434, 'offset': 0.0, 'size': 960, 'units': 'µm', 'navigate': True}
    }

    assert s[6].metadata['General']['title'] == 'Image 3, Spot 1'
    assert s[6].data.shape == (2048,)
    assert s[6].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'Energy', 'scale': 0.009957494815722499, 'offset': -0.043546104722405436, 'size': 2048, 'units': 'keV', 'navigate': False}
    }

    assert s[7].metadata['General']['title'] == 'HgS'
    assert s[7].data.shape == (600, 960)
    assert s[7].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'y', 'scale': 0.7810477081594434, 'offset': 0.0, 'size': 600, 'units': 'µm', 'navigate': True},
        'axis-1': {'name': 'x', 'scale': 0.7810477081594434, 'offset': 0.0, 'size': 960, 'units': 'µm', 'navigate': True}
    }

    assert s[8].metadata['General']['title'] == 'HgS, Spot 1'
    assert s[8].data.shape == (2048,)
    assert s[8].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'Energy', 'scale': 0.009957494815722499, 'offset': -0.043546104722405436, 'size': 2048, 'units': 'keV', 'navigate': False}
    }

    assert s[9].metadata['General']['title'] == 'PbMo'
    assert s[9].data.shape == (600, 960)
    assert s[9].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'y', 'scale': 0.7810477081594434, 'offset': 0.0, 'size': 600, 'units': 'µm', 'navigate': True},
        'axis-1': {'name': 'x', 'scale': 0.7810477081594434, 'offset': 0.0, 'size': 960, 'units': 'µm', 'navigate': True}
    }

    assert s[10].metadata['General']['title'] == 'PbMo, Spot 1'
    assert s[10].data.shape == (2048,)
    assert s[10].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'Energy', 'scale': 0.009957494815722499, 'offset': -0.043546104722405436, 'size': 2048, 'units': 'keV', 'navigate': False}
    }

    assert s[11].metadata['General']['title'] == 'BaTi'
    assert s[11].data.shape == (600, 960)
    assert s[11].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'y', 'scale': 0.7810477081594434, 'offset': 0.0, 'size': 600, 'units': 'µm', 'navigate': True},
        'axis-1': {'name': 'x', 'scale': 0.7810477081594434, 'offset': 0.0, 'size': 960, 'units': 'µm', 'navigate': True}
    }

    assert s[12].metadata['General']['title'] == 'BaTi, Spot 1'
    assert s[12].data.shape == (2048,)
    assert s[12].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'Energy', 'scale': 0.009957494815722499, 'offset': -0.043546104722405436, 'size': 2048, 'units': 'keV', 'navigate': False}
    }

    assert s[13].metadata['General']['title'] == 'HgS'
    assert s[13].data.shape == (600, 960)
    assert s[13].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'y', 'scale': 0.7810477081594434, 'offset': 0.0, 'size': 600, 'units': 'µm', 'navigate': True},
        'axis-1': {'name': 'x', 'scale': 0.7810477081594434, 'offset': 0.0, 'size': 960, 'units': 'µm', 'navigate': True}
    }

    assert s[14].metadata['General']['title'] == 'HgS, Spot 1'
    assert s[14].data.shape == (2048,)
    assert s[14].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'Energy', 'scale': 0.009957494815722499, 'offset': -0.043546104722405436, 'size': 2048, 'units': 'keV', 'navigate': False}
    }

    assert s[15].metadata['General']['title'] == 'Image 8'
    assert s[15].data.shape == (600, 960)
    assert s[15].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'y', 'scale': 0.7810477081594434, 'offset': 0.0, 'size': 600, 'units': 'µm', 'navigate': True},
        'axis-1': {'name': 'x', 'scale': 0.7810477081594434, 'offset': 0.0, 'size': 960, 'units': 'µm', 'navigate': True}
    }

    assert s[16].metadata['General']['title'] == 'Image 8, Spot 1'
    assert s[16].data.shape == (2048,)
    assert s[16].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'Energy', 'scale': 0.009957494815722499, 'offset': -0.043546104722405436, 'size': 2048, 'units': 'keV', 'navigate': False}
    }

    assert s[17].metadata['General']['title'] == 'Image 9'
    assert s[17].data.shape == (600, 960)
    assert s[17].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'y', 'scale': 0.7810477081594434, 'offset': 0.0, 'size': 600, 'units': 'µm', 'navigate': True},
        'axis-1': {'name': 'x', 'scale': 0.7810477081594434, 'offset': 0.0, 'size': 960, 'units': 'µm', 'navigate': True}
    }

    assert s[18].metadata['General']['title'] == 'Image 9, Spot 1'
    assert s[18].data.shape == (2048,)
    assert s[18].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'Energy', 'scale': 0.009957494815722499, 'offset': -0.043546104722405436, 'size': 2048, 'units': 'keV', 'navigate': False}
    }

    assert s[19].metadata['General']['title'] == 'Image 9, Spot 2'
    assert s[19].data.shape == (2048,)
    assert s[19].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'Energy', 'scale': 0.009957494815722499, 'offset': -0.043546104722405436, 'size': 2048, 'units': 'keV', 'navigate': False}
    }

    assert s[20].metadata['General']['title'] == 'Image 10'
    assert s[20].data.shape == (600, 960)
    assert s[20].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'y', 'scale': 0.7810477081594434, 'offset': 0.0, 'size': 600, 'units': 'µm', 'navigate': True},
        'axis-1': {'name': 'x', 'scale': 0.7810477081594434, 'offset': 0.0, 'size': 960, 'units': 'µm', 'navigate': True}
    }

    assert s[21].metadata['General']['title'] == 'Image 10, Spot 1'
    assert s[21].data.shape == (2048,)
    assert s[21].axes_manager.as_dictionary() == {
        'axis-0': {'name': 'Energy', 'scale': 0.009957494815722499, 'offset': -0.043546104722405436, 'size': 2048, 'units': 'keV', 'navigate': False}
    }
