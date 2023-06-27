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

import gc
from pathlib import Path

import pytest
import numpy as np

import hyperspy.api as hs
from hyperspy import __version__ as hs_version


def teardown_module(module):
    """
    Run a garbage collection cycle at the end of the test of this module
    to avoid any memory issue when continuing running the test suite.
    """
    gc.collect()


TESTS_FILE_PATH = Path(__file__).resolve().parent / 'JEOL_files'
TESTS_FILE_PATH2 = TESTS_FILE_PATH / 'InvalidFrame'

test_files = ['rawdata.ASW',
              'View000_0000000.img',
              'View000_0000001.map',
              'View000_0000002.map',
              'View000_0000003.map',
              'View000_0000004.map',
              'View000_0000005.map',
              'View000_0000006.pts'
              ]

test_files2 = ['dummy2.ASW',
               'Dummy-Data_0000000.img',
               'Dummy-Data_0000001.map',
               'Dummy-Data_0000002.map',
               'Dummy-Data_0000003.map',
               'Dummy-Data_0000004.map',
               'Dummy-Data_0000005.map',
               'Dummy-Data_0000006.map',
               'Dummy-Data_0000007.pts',
               'Dummy-Data_0000008.apb',
               'Dummy-Data_0000009.map',
               'Dummy-Data_0000010.map',
               'Dummy-Data_0000011.map',
               'Dummy-Data_0000012.map',
               'Dummy-Data_0000013.map',
               'Dummy-Data_0000014.map',
               'Dummy-Data_0000015.pts',
               'Dummy-Data_0000016.apb',
               'Dummy-Data_0000017.map',
               'Dummy-Data_0000018.map',
               'Dummy-Data_0000019.map',
               'Dummy-Data_0000020.map',
               'Dummy-Data_0000021.map',
               'Dummy-Data_0000022.map',
               'Dummy-Data_0000023.pts',
               'Dummy-Data_0000024.APB',
]


def test_load_project():
    # test load all elements of the project rawdata.ASW
    filename = TESTS_FILE_PATH / test_files[0]
    s = hs.load(filename)
    # first file is always a 16bit image of the work area
    assert s[0].data.dtype == np.uint8
    assert s[0].data.shape == (512, 512)
    assert s[0].axes_manager.signal_dimension == 2
    assert s[0].axes_manager[0].units == 'µm'
    assert s[0].axes_manager[0].name == 'x'
    assert s[0].axes_manager[1].units == 'µm'
    assert s[0].axes_manager[1].name == 'y'
    # 1 to 16 files are a 16bit image of work area and elemental maps
    for map in s[:-1]:
        assert map.data.dtype == np.uint8
        assert map.data.shape == (512, 512)
        assert map.axes_manager.signal_dimension == 2
        assert map.axes_manager[0].units == 'µm'
        assert map.axes_manager[0].name == 'x'
        assert map.axes_manager[1].units == 'µm'
        assert map.axes_manager[1].name == 'y'
    # last file is the datacube
    assert s[-1].data.dtype == np.uint8
    assert s[-1].data.shape == (512, 512, 4096)
    assert s[-1].axes_manager.signal_dimension == 1
    assert s[-1].axes_manager.navigation_dimension == 2
    assert s[-1].axes_manager[0].units == 'µm'
    assert s[-1].axes_manager[0].name == 'x'
    assert s[-1].axes_manager[1].units == 'µm'
    assert s[-1].axes_manager[1].name == 'y'
    assert s[-1].axes_manager[2].units == 'keV'
    np.testing.assert_allclose(s[-1].axes_manager[2].offset, -0.000789965-0.00999866*96)
    np.testing.assert_allclose(s[-1].axes_manager[2].scale, 0.00999866)
    assert s[-1].axes_manager[2].name == 'Energy'

    # check scale (image)
    filename = TESTS_FILE_PATH / 'Sample' / '00_View000' / test_files[1]
    s1 = hs.load(filename)
    np.testing.assert_allclose(s[0].axes_manager[0].scale, s1.axes_manager[0].scale)
    assert s[0].axes_manager[0].units == s1.axes_manager[0].units
    # check scale (pts)
    filename = TESTS_FILE_PATH / 'Sample' / '00_View000' / test_files[7]
    s2 = hs.load(filename)
    np.testing.assert_allclose(s[6].axes_manager[0].scale, s2.axes_manager[0].scale)
    assert s[6].axes_manager[0].units == s2.axes_manager[0].units


def test_load_image():
    # test load work area haadf image
    filename = TESTS_FILE_PATH / 'Sample' / '00_View000' / test_files[1]
    s = hs.load(filename)
    assert s.data.dtype == np.uint8
    assert s.data.shape == (512, 512)
    assert s.axes_manager.signal_dimension == 2
    assert s.axes_manager[0].units == 'µm'
    np.testing.assert_allclose(s.axes_manager[0].scale, 0.00869140587747097)
    assert s.axes_manager[0].name == 'x'
    assert s.axes_manager[1].units == 'µm'
    np.testing.assert_allclose(s.axes_manager[1].scale, 0.00869140587747097)
    assert s.axes_manager[1].name == 'y'


@pytest.mark.parametrize('SI_dtype', [np.int8, np.uint8])
def test_load_datacube(SI_dtype):
    # test load eds datacube
    filename = TESTS_FILE_PATH / 'Sample' / '00_View000' / test_files[7]
    s = hs.load(filename, SI_dtype=SI_dtype, cutoff_at_kV=5)
    assert s.data.dtype == SI_dtype
    assert s.data.shape == (512, 512, 596)
    assert s.axes_manager.signal_dimension == 1
    assert s.axes_manager.navigation_dimension == 2
    assert s.axes_manager[0].units == 'µm'
    np.testing.assert_allclose(s.axes_manager[0].scale, 0.00869140587747097)
    assert s.axes_manager[0].name == 'x'
    assert s.axes_manager[1].units == 'µm'
    np.testing.assert_allclose(s.axes_manager[1].scale, 0.00869140587747097)
    assert s.axes_manager[1].name == 'y'
    assert s.axes_manager[2].units == 'keV'
    np.testing.assert_allclose(s.axes_manager[2].offset, -0.000789965-0.00999866*96)
    np.testing.assert_allclose(s.axes_manager[2].scale, 0.00999866)
    assert s.axes_manager[2].name == 'Energy'


def test_load_datacube_rebin_energy():
    filename = TESTS_FILE_PATH / 'Sample' / '00_View000' / test_files[7]
    s = hs.load(filename, cutoff_at_kV=0.1)
    s_sum = s.sum()

    ref_data = hs.signals.Signal1D(
        np.array([   3,   23,   77,  200,  487,  984, 1599, 2391])
        )
    np.testing.assert_allclose(s_sum.data[88:96], ref_data.data)

    rebin_energy = 8
    s2 = hs.load(filename, rebin_energy=rebin_energy)
    s2_sum = s2.sum()

    np.testing.assert_allclose(s2_sum.data[11:12], ref_data.data.sum())

    with pytest.raises(ValueError, match='must be a multiple'):
        _ = hs.load(filename, rebin_energy=10)


def test_load_datacube_cutoff_at_kV():
    gc.collect()
    cutoff_at_kV = 10.
    filename = TESTS_FILE_PATH / 'Sample' / '00_View000' / test_files[7]
    s = hs.load(filename, cutoff_at_kV=None)
    s2 = hs.load(filename, cutoff_at_kV=cutoff_at_kV)

    assert s2.axes_manager[-1].size == 1096
    np.testing.assert_allclose(s2.axes_manager[2].scale, 0.00999866)
    np.testing.assert_allclose(s2.axes_manager[2].offset, -0.9606613)

    np.testing.assert_allclose(s.sum().isig[:cutoff_at_kV].data, s2.sum().data)


def test_load_datacube_downsample():
    downsample = 8
    filename = TESTS_FILE_PATH / test_files[0]
    s = hs.load(filename, downsample=1)[-1]
    s2 = hs.load(filename, downsample=downsample)[-1]

    s_sum = s.sum(-1).rebin(scale=(downsample, downsample))
    s2_sum = s2.sum(-1)

    assert s2.axes_manager[-1].size == 4096
    np.testing.assert_allclose(s2.axes_manager[2].scale, 0.00999866)
    np.testing.assert_allclose(s2.axes_manager[2].offset, -0.9606613)

    for axis in s2.axes_manager.navigation_axes:
        assert axis.size == 64
        np.testing.assert_allclose(axis.scale, 0.069531247)
        np.testing.assert_allclose(axis.offset, 0.0)

    np.testing.assert_allclose(s_sum.data, s2_sum.data)

    with pytest.raises(ValueError, match='must be a multiple'):
        _ = hs.load(filename, downsample=10)[-1]

    downsample = [8, 16]
    s = hs.load(filename, downsample=downsample)[-1]
    assert s.axes_manager['x'].size * downsample[0] == 512
    assert s.axes_manager['y'].size * downsample[1] == 512

    with pytest.raises(ValueError, match='must be a multiple'):
        _ = hs.load(filename, downsample=[256, 100])[-1]

    with pytest.raises(ValueError, match='must be a multiple'):
        _ = hs.load(filename, downsample=[100, 256])[-1]


def test_load_datacube_frames():
    rebin_energy = 2048
    filename = TESTS_FILE_PATH / 'Sample' / '00_View000' / test_files[7]
    s = hs.load(filename, sum_frames=True, rebin_energy=rebin_energy)
    assert s.data.shape == (512, 512, 2)
    s_frame = hs.load(filename, sum_frames=False, rebin_energy=rebin_energy)
    assert s_frame.data.shape == (14, 512, 512, 2)
    np.testing.assert_allclose(s_frame.sum(axis='Frame').data, s.data)
    np.testing.assert_allclose(s_frame.sum(axis=['x', 'y', 'Energy']).data,
                               np.array([22355, 21975, 22038, 21904, 21846,
                                         22115, 22021, 21917, 22123, 21919,
                                         22141, 22024, 22086, 21797]))


@pytest.mark.parametrize('filename_as_string', [True, False])
def test_load_eds_file(filename_as_string):
    filename = TESTS_FILE_PATH / 'met03.EDS'
    if filename_as_string:
        filename = str(filename)
    s = hs.load(filename)
    assert isinstance(s, hs.signals.EDSTEMSpectrum)
    assert s.data.shape == (2048,)
    axis = s.axes_manager[0]
    assert axis.name == 'Energy'
    assert axis.size == 2048
    assert axis.offset == -0.00176612
    assert axis.scale == 0.0100004

    # delete timestamp from metadata since it's runtime dependent
    del s.metadata.General.FileIO.Number_0.timestamp

    md_dict = s.metadata.as_dictionary()
    assert md_dict['General'] == {'original_filename': 'met03.EDS',
                                  'time': '14:14:51',
                                  'date': '2018-06-25',
                                  'title': 'EDX',
                                  'FileIO': {
                                    '0': {
                                        'operation': 'load',
                                        'hyperspy_version': hs_version,
                                        'io_plugin':
                                            'hyperspy.io_plugins.jeol'
                                    }
                                  }
                                  }
    TEM_dict = md_dict['Acquisition_instrument']['TEM']
    assert TEM_dict == {'beam_energy': 200.0,
                        'Detector': {'EDS': {'azimuth_angle': 90.0,
                                             'detector_type': 'EX24075JGT',
                                             'elevation_angle': 22.299999237060547,
                                             'energy_resolution_MnKa': 138.0,
                                             'live_time': 30.0}},
                        'Stage': {'tilt_alpha': 0.0}}


def test_shift_jis_encoding():
    # See https://github.com/hyperspy/hyperspy/issues/2812
    filename = TESTS_FILE_PATH / '181019-BN.ASW'
    # make sure we can open the file
    with open(filename, "br"):
        pass
    try:
        _ = hs.load(filename)
    except FileNotFoundError:
        # we don't have the other files required to open the data
        pass


def test_number_of_frames():
    dir1 = TESTS_FILE_PATH / 'Sample' / '00_View000'
    dir2 = TESTS_FILE_PATH / 'InvalidFrame' / 'Sample' / '00_Dummy-Data'

    test_list = [  # dir, file, num_frames, num_valid_frames
        [ dir1, test_files[7], 14, 14 ],
        [ dir2, test_files2[8], 1, 0 ],
        [ dir2, test_files2[16], 2, 1 ],
        [ dir2, test_files2[24], 1, 1 ]
    ]

    for item in test_list:
        dirname, filename, frames, valid = item
        fname = str(dirname / filename)

        # Count number of frames including incomplete frame
        data = hs.load(fname, sum_frames = False, only_valid_data = False,
                       downsample=[32,32], rebin_energy=512, SI_dtype=np.int32)
        assert data.axes_manager["Frame"].size == frames

        # Count number of valid frames
        data = hs.load(fname, sum_frames = False, only_valid_data = True,
                       downsample=[32,32], rebin_energy=512, SI_dtype=np.int32)
        assert data.axes_manager["Frame"].size == valid


def test_em_image_in_pts():
    dir1 = TESTS_FILE_PATH
    dir2 = TESTS_FILE_PATH / 'InvalidFrame'
    dir2p = dir2 / 'Sample' / '00_Dummy-Data'

    # no SEM/STEM image
    s = hs.load(dir1 / test_files[0],
                read_em_image=False, only_valid_data=False,
                cutoff_at_kV=1)
    assert len(s) == 7

    s = hs.load(dir1 / test_files[0],
                read_em_image=True, only_valid_data=False,
                cutoff_at_kV=1)
    assert len(s) == 7

    # with SEM/STEM image
    s = hs.load(dir2 / test_files2[0],
                read_em_image=False, only_valid_data=False,
                cutoff_at_kV=1)
    assert len(s) == 22
    s = hs.load(dir2 / test_files2[0],
                read_em_image=True, only_valid_data=False,
                cutoff_at_kV=1)
    assert len(s) == 25
    assert s[8].metadata.General.title == "S(T)EM Image extracted from " + s[8].metadata.General.original_filename
    assert s[8].data[38,15] == 87
    assert s[8].data[38,16] == 0

    # integrate SEM/STEM image along frame axis
    s = hs.load(dir2p / test_files2[16], read_em_image=True,
                only_valid_data=False, sum_frames=True, cutoff_at_kV=1,
                frame_list=[0,0,0,1])
    assert(s[1].data[0,0] == 87*4)
    assert(s[1].data[63,63] == 87*3)

    s = hs.load(dir2p / test_files2[16], read_em_image=True,
                only_valid_data=False, sum_frames=False, cutoff_at_kV=1)
    s2 = hs.load(dir2p / test_files2[16], read_em_image=True,
                 only_valid_data=False, sum_frames=True, cutoff_at_kV=1)
    s1 = [s[0].data.sum(axis=0), s[1].data.sum(axis=0)]
    assert np.array_equal(s1[0], s2[0].data)
    assert np.array_equal(s1[1], s2[1].data)


def test_pts_lazy():
    dir2 = TESTS_FILE_PATH / 'InvalidFrame'
    dir2p = dir2 / 'Sample' / '00_Dummy-Data'
    s = hs.load(dir2p / test_files2[16], read_em_image=True,
                only_valid_data=False, sum_frames=False, lazy=True)
    s1 = [s[0].data.sum(axis=0).compute(),
          s[1].data.sum(axis=0).compute()]
    s2 = hs.load(dir2p / test_files2[16], read_em_image=True,
                only_valid_data=False, sum_frames=True, lazy=False)
    assert np.array_equal(s1[0], s2[0].data)
    assert np.array_equal(s1[1], s2[1].data)


def test_pts_frame_shift():
    file = TESTS_FILE_PATH2 / 'Sample' / '00_Dummy-Data' / test_files2[16]

    # without frame shift
    ref = hs.load(file, read_em_image=True, only_valid_data=False,
                  sum_frames=False, lazy=False)
    #         x, y, en
    points=[[24,23,106],[21,16,106]]
    values=[3,1]
    targets=np.asarray([[2,3,106],[20,3,100],[4,20,100]],dtype=np.int16)

    # check values before shift
    d0 = np.zeros(len(points), dtype=np.int16)
    d1 = np.zeros(len(points), dtype=np.int16)
    d2 = np.zeros(len(points), dtype=np.int16)
    for frame in range(len(points)):
        p = points[frame]
        d0[frame] = ref[0].data[frame, p[1], p[0], p[2]]
        assert d0[frame] == values[frame]

    for target in targets:
        sfts = np.zeros((ref[0].axes_manager['Frame'].size,3),dtype=np.int16)
        for frame in range(ref[0].axes_manager['Frame'].size):
            origin = points[frame]
            sfts[frame] = np.asarray(target) - np.asarray(origin)
        shifts = sfts[:,[1,0,2]]

        # test frame shifts for dense (normal) loading
        s0 = hs.load(file, read_em_image=True,
                     only_valid_data=False, sum_frames=False,
                     frame_shifts = shifts, lazy=False)


        for frame in range(s0[0].axes_manager['Frame'].size):
            origin = points[frame]
            sfts0 = s0[0].original_metadata.jeol_pts_frame_shifts[frame]
            pos = [origin[0]+sfts0[1], origin[1]+sfts0[0], origin[2]+sfts0[2]]
            d1[frame] = s0[0].data[frame, pos[1], pos[0], pos[2]]
            assert d1[frame] == d0[frame]

	# test frame shifts for lazy loading
        s1 = hs.load(file, read_em_image=True,
                     only_valid_data=False, sum_frames=False,
                     frame_shifts=shifts, lazy=True)
        dt = s1[0].data.compute()
        for frame in range(s0[0].axes_manager['Frame'].size):
            origin = points[frame]
            sfts0 = s0[0].original_metadata.jeol_pts_frame_shifts[frame]
            pos = [origin[0]+sfts0[1], origin[1]+sfts0[0], origin[2]+sfts0[2]]
            d2[frame] = dt[frame, pos[1], pos[0], pos[2]]
            assert d2[frame] == d0[frame]
