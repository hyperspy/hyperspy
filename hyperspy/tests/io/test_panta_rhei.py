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


# The EMD format is a hdf5 standard proposed at Lawrence Berkeley
# National Lab (see http://emdatasets.com/ for more information).
# NOT to be confused with the FEI EMD format which was developed later.

from pathlib import Path
import numpy as np

import hyperspy.api as hs
from hyperspy.misc.test_utils import assert_deep_almost_equal

my_path = Path(__file__).parent


class TestLoadingPrzFiles:
    def test_metadata_prz_v5(self):
        md = {'General': {'title': 'AD', 'original_filename': 'panta_rhei_sample_v5.prz'},
              'Signal': {'signal_type': ''},
              'Acquisition_instrument': {'TEM': {'beam_energy': 200.0,
                                                 'acquisition_mode': 'STEM',
                                                 'magnification': 10000000,
                                                 'camera_length': 0.02,
                                                 }}}
        am = {'axis-0': {'_type': 'UniformDataAxis',
                         'name': 'Y',
                         'units': 'm',
                         'navigate': False,
                         'is_binned': False,
                         'size': 16,
                         'scale': 7.795828292907633e-09,
                         'offset': 0.0},
              'axis-1': {'_type': 'UniformDataAxis',
                         'name': 'X',
                         'units': 'm',
                         'navigate': False,
                         'is_binned': False,
                         'size': 16,
                         'scale': 7.795828292907633e-09,
                         'offset': 0.0}}

        s = hs.load(my_path / 'panta_rhei_files' / 'panta_rhei_sample_v5.prz')

        md_file = s.metadata.as_dictionary()
        md_file.pop('_HyperSpy')
        md_file['General'].pop('FileIO')
        assert_deep_almost_equal(md_file, md)
        assert_deep_almost_equal(s.axes_manager.as_dictionary(), am)
        assert (s.data.shape == (16, 16))
        assert (s.data.max() == 40571)
        assert (s.data.min() == 36193)
        np.testing.assert_almost_equal(s.data.std(), 1025.115644550)


def test_save_load_cycle(tmp_path):
    fname = tmp_path / 'test_file.prz'

    s = hs.load(my_path / 'panta_rhei_files' / 'panta_rhei_sample_v5.prz')
    s.save(fname)
    assert fname.is_file()

    s2 = hs.load(fname)
    np.testing.assert_allclose(s2.data, s.data)


def test_save_load_cycle_new_signal_1D_nav1(tmp_path):
    fname = tmp_path / 'test_file_new_signal_1D_nav1.prz'
    data = np.arange(20).reshape(2, 10)
    s = hs.signals.Signal1D(data)
    s.save(fname)
    assert fname.is_file()

    s2 = hs.load(fname)
    np.testing.assert_allclose(s2.data, s.data)
    assert isinstance(s2, s.__class__)


def test_save_load_cycle_new_signal_1D_nav2(tmp_path):
    fname = tmp_path / 'test_file_new_signal1D_nav2.prz'
    data = np.arange(100).reshape(2, 5, 10)
    s = hs.signals.Signal2D(data)
    s.save(fname)
    assert fname.is_file()

    s2 = hs.load(fname)
    np.testing.assert_allclose(s2.data, s.data)
    assert isinstance(s2, s.__class__)


def test_save_load_cycle_new_signal_2D(tmp_path):
    fname = tmp_path / 'test_file_new_signal2D.prz'
    data = np.arange(100).reshape(2, 5, 10)
    s = hs.signals.Signal2D(data)
    s.save(fname)
    assert fname.is_file()

    s2 = hs.load(fname)
    np.testing.assert_allclose(s2.data, s.data)
    assert isinstance(s2, s.__class__)


def test_save_load_cycle_new_signal_EELS(tmp_path):
    fname = tmp_path / 'test_file_new_signal2D.prz'
    data = np.arange(100).reshape(2, 5, 10)
    s = hs.signals.EELSSpectrum(data)
    s.save(fname)
    assert fname.is_file()

    s2 = hs.load(fname)
    np.testing.assert_allclose(s2.data, s.data)
    assert isinstance(s2, s.__class__)


def test_metadata_STEM(tmp_path):
    fname = tmp_path / 'test_file_new_signal_metadata_STEM.prz'
    data = np.arange(20).reshape(2, 10)
    s = hs.signals.EELSSpectrum(data)
    # Set some metadata
    md = {
        'Acquisition_instrument': {
            'TEM': {
                'beam_energy': 200.0,
                'acquisition_mode': 'STEM',
                'magnification': 500000,
                'camera_length': 200,
                'convergence_angle': 20,
                'Detector': {
                    'EELS': {'collection_angle': 60, 'aperture_size': 5},
                    },
                },
            },
        }

    s.metadata.add_dictionary(md)
    s.metadata.General.add_dictionary({'date':'2022-07-08', 'time':'16:00'})
    s.save(fname)
    assert fname.is_file()

    s2 = hs.load(fname)
    np.testing.assert_allclose(s2.data, s.data)
    assert isinstance(s2, s.__class__)

    assert_deep_almost_equal(
        s2.metadata.Acquisition_instrument.as_dictionary(),
        s.metadata.Acquisition_instrument.as_dictionary()
        )


def test_metadata_TEM(tmp_path):
    fname = tmp_path / 'test_file_new_signal_metadata_TEM.prz'
    data = np.arange(20).reshape(2, 10)
    s = hs.signals.EELSSpectrum(data)
    # Set some metadata
    md = {
        'Acquisition_instrument': {
            'TEM': {
                'beam_energy': 200.0,
                'acquisition_mode': 'TEM',
                'magnification': 500000,
                'camera_length': 200,
                'Detector': {
                    'EELS': {'collection_angle': 60, 'aperture_size': 5},
                    },
                },
            },
        }

    s.metadata.add_dictionary(md)
    s.metadata.General.add_dictionary({'date':'2022-07-08', 'time':'16:00'})
    s.save(fname)
    assert fname.is_file()

    s2 = hs.load(fname)
    np.testing.assert_allclose(s2.data, s.data)
    assert isinstance(s2, s.__class__)

    assert_deep_almost_equal(
        s2.metadata.Acquisition_instrument.as_dictionary(),
        s.metadata.Acquisition_instrument.as_dictionary()
        )
