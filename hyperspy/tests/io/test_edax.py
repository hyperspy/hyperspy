from nose.tools import assert_equal
import gzip
import hashlib
import os.path

from hyperspy.io import load

my_path = os.path.dirname(__file__)


def unzip_data():
    with gzip.open(os.path.join(
                   my_path,
                   "edax_files",
                   "spd_map.spd.gz")) as f_in:
        with open(os.path.join(
                my_path,
                "edax_files",
                "spd_map.spd"), 'wb') as f_out:
            f_out.write(f_in.read())
    if hashlib.md5(open('spd_map.spd', 'rb').read()).hexdigest() != \
            'a0c29793146c9e7438fa9b2e1ca05046':
        raise ValueError('Something went wrong with decompressing the test '
                         'file. Please try again.')
    print('Success!')

spc_parameters = {
    'Acquisition_instrument': {
        'SEM': {
            'Detector': {
                'EDS': {
                    'azimuth_angle': 0.0,
                    'elevation_angle': 34.0,
                    'energy_resolution_MnKa': 129.31299,
                    'live_time': 50.000004
                }
            },
            'beam_energy': 22.0,
            'tilt_stage': 0.0}
    },
    'General': {
        'original_filename': 'EDS Spot 1 Det 1.spc',
        'title': 'EDS Spectrum'},
    'Sample': {
        'elements': ['Al', 'C', 'Ce', 'Cu', 'F', 'Ho', 'Mg', 'O']
    },
    'Signal': {
        'binned': True,
        'record_by': 'spectrum',
        'signal_origin': '',
        'signal_type': 'EDS_SEM'
    },
    '_HyperSpy': {
        'Folding': {
            'original_axes_manager': None,
            'original_shape': None,
            'signal_unfolded': False,
            'unfolded': False
        }
    }
}

spd_parameters = {
    'Acquisition_instrument': {
        'SEM': {
            'Detector': {
                'EDS': {
                    'azimuth_angle': 0.0,
                    'elevation_angle': 34.0,
                    'energy_resolution_MnKa': 126.60252,
                    'live_time': 2621.4399
                }
            },
            'beam_energy': 22.0,
            'tilt_stage': 0.0}
    },
    'General': {'original_filename': 'spd_map.spd',
                'title': 'EDS Spectrum Image'},
    'Sample': {
        'elements': ['Ce', 'Co', 'Cr', 'Fe', 'Gd', 'La', 'Mg', 'O', 'Sr']},
    'Signal': {
        'binned': True,
        'record_by': 'spectrum',
        'signal_origin': '',
        'signal_type': 'EDS_SEM'
    },
    '_HyperSpy': {
        'Folding': {
            'original_axes_manager': None,
            'original_shape': None,
            'signal_unfolded': False,
            'unfolded': False}
    }
}


class TestSpcSpectrum:

    def setUp(self):
        self.spc = load(os.path.join(
            my_path,
            "edax_files",
            "single_spect.spc"))

    def test_data(self):
        assert_equal(
            [0, 0, 0, 0, 0, 0, 1, 2, 3, 3, 10, 4, 10, 10, 45, 87, 146, 236,
             312, 342], self.spc.data.tolist()[:20])

    def test_parameters(self):
        print(spc_parameters)
        print(self.spc.metadata.as_dictionary())
        assert_equal(
            spc_parameters,
            self.spc.metadata.as_dictionary())


class TestSpdMap:

    def setUp(self):
        unzip_data()
        self.spd = load(os.path.join(
            my_path,
            "edax_files",
            "spd_map.spd"))

    def test_data(self):
        assert_equal(
            [65.82,
             67.872,
             65.626,
             68.762,
             71.395,
             74.996,
             78.132,
             78.055,
             77.861,
             84.598,
             83.088,
             85.372,
             89.786,
             93.464,
             93.387,
             97.452,
             109.96,
             111.08,
             119.64,
             128.77,
             138.38,
             152.35,
             176.01,
             192.12,
             222.12,
             254.22,
             281.55,
             328.33,
             348.92,
             375.33,
             385.51,
             389.54,
             378.77,
             353.8,
             328.91,
             290.07,
             246.09,
             202.73,
             176.47,
             137.64,
             119.56,
             106.4,
             92.496,
             96.213,
             94.664,
             101.13,
             114.57,
             118.82,
             131.68,
             145.04,
             165.44,
             187.51,
             207.49,
             238.04,
             269.71,
             301.46,
             348.65,
             409.36,
             475.3,
             554.51,
             631.64,
             715.19,
             793.44,
             847.99,
             872.97,
             862.59,
             834.87,
             778.49,
             688.63,
             599.39,
             495.39,
             403.48,
             312.88,
             237.34,
             184.14,
             129.86,
             101.59,
             80.107,
             58.657,
             49.442], self.s.data.tolist())

    def test_parameters(self):
        assert_equal(
            example2_parameters,
            self.s.original_metadata.as_dictionary())
