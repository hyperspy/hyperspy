from pathlib import Path
from numpy.ma.testutils import assert_almost_equal

from hyperspy.io import load
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
                                                 'Detector': {'EELS': {'spectrometer': 'CEOS CEFID'}}}}}
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
        
        s = load(my_path / 'panta_rhei_files' / 'panta_rhei_sample_v5.prz')
        
        md_file = s.metadata.as_dictionary()
        md_file.pop('_HyperSpy')
        md_file['General'].pop('FileIO')
        assert_deep_almost_equal(md_file, md)
        assert_deep_almost_equal(s.axes_manager.as_dictionary(), am)
        assert (s.data.shape == (16, 16))
        assert (s.data.max() == 40571)
        assert (s.data.min() == 36193)
        assert_almost_equal(s.data.std(), 1025.115644550)
