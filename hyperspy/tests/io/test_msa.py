import os.path
import tempfile

from hyperspy.io import load
from hyperspy.misc.test_utils import assert_deep_almost_equal

my_path = os.path.dirname(__file__)

example1_TEM = {'Detector': {'EELS': {'collection_angle': 3.4,
                                      'collection_angle_units': "mR",
                                      'dwell_time': 100.0,
                                      'dwell_time_units': "ms"}},
                'beam_current': 12.345,
                'beam_current_units': "nA",
                'beam_energy': 120.0,
                'beam_energy_units': "kV",
                'convergence_angle': 1.5,
                'convergence_angle_units': "mR"}

example1_metadata = {'Acquisition_instrument': {'TEM': example1_TEM},
                     'General': {'original_filename': "example1.msa",
                                 'title': "NIO EELS OK SHELL",
                                 'date': "1991-10-01",
                                 'time': "12:00:00"},
                     'Sample': {'thickness': 50.0,
                                'thickness_units': "nm"},
                     'Signal': {'binned': True,
                                # bit silly...
                                'quantity': "Counts (Intensity)",
                                'signal_type': 'EELS'},
                     '_HyperSpy': {'Folding': {'original_axes_manager': None,
                                               'unfolded': False,
                                               'original_shape': None,
                                               'signal_unfolded': False}}}
minimum_md_om = {
    'COMMENT': 'File created by HyperSpy version 1.1.2+dev',
    'DATATYPE': 'Y',
    'DATE': '',
    'FORMAT': 'EMSA/MAS Spectral Data File',
    'NCOLUMNS': 1.0,
    'NPOINTS': 1.0,
    'OFFSET': 0.0,
    'OWNER': '',
    'SIGNALTYPE': '',
    'TIME': '',
    'TITLE': '',
    'VERSION': '1.0',
    'XLABEL': '',
    'XPERCHAN': 1.0,
    'XUNITS': ''}


example1_parameters = {
    'BEAMDIAM -nm': 100.0,
    'BEAMKV   -kV': 120.0,
    'CHOFFSET': -168.0,
    'COLLANGLE-mR': 3.4,
    'CONVANGLE-mR': 1.5,
    'DATATYPE': 'XY',
    'DATE': '01-OCT-1991',
    'DWELLTIME-ms': 100.0,
    'ELSDET': 'SERIAL',
    'EMISSION -uA': 5.5,
    'FORMAT': 'EMSA/MAS Spectral Data File',
    'MAGCAM': 100.0,
    'NCOLUMNS': 1.0,
    'NPOINTS': 20.0,
    'OFFSET': 520.13,
    'OPERMODE': 'IMAG',
    'OWNER': 'EMSA/MAS TASK FORCE',
    'PROBECUR -nA': 12.345,
    'SIGNALTYPE': 'ELS',
    'THICKNESS-nm': 50.0,
    'TIME': '12:00',
    'TITLE': 'NIO EELS OK SHELL',
    'VERSION': '1.0',
    'XLABEL': 'Energy',
    'XPERCHAN': 3.1,
    'XUNITS': 'eV',
    'YLABEL': 'Counts',
    'YUNITS': 'Intensity'}

example2_TEM = {'Detector': {'EDS': {'EDS_det': "SIWLS",
                                     'azimuth_angle': 90.0,
                                     'azimuth_angle_units': "dg",
                                     'elevation_angle': 20.0,
                                     'elevation_angle_units': 'dg',
                                     'live_time': 100.0,
                                     'live_time_units': "s",
                                     'real_time': 150.0,
                                     'real_time_units': "s"}},
                'beam_current': 12.345,
                'beam_current_units': "nA",
                'beam_energy': 120.0,
                'beam_energy_units': "kV",
                'tilt_stage': 45.0,
                'tilt_stage_units': "dg"}

example2_metadata = {'Acquisition_instrument': {'TEM': example2_TEM},
                     'General': {'original_filename': "example2.msa",
                                 'title': "NIO Windowless Spectra OK NiL",
                                 'date': "1991-10-01",
                                 'time': "12:00:00"},
                     'Sample': {'thickness': 50.0,
                                'thickness_units': "nm"},
                     'Signal': {'binned': False,
                                'quantity': "X-RAY INTENSITY (Intensity)",
                                'signal_type': 'EDS'},
                     '_HyperSpy': {'Folding': {'original_axes_manager': None,
                                               'unfolded': False,
                                               'original_shape': None,
                                               'signal_unfolded': False}}}

example2_parameters = {
    'ALPHA-1': '3.1415926535',
    'AZIMANGLE-dg': 90.0,
    'BEAMDIAM -nm': 100.0,
    'BEAMKV   -kV': 120.0,
    'CHOFFSET': -20.0,
    'COMMENT': 'The next two lines are User Defined Keywords and values',
    'DATATYPE': 'Y',
    'DATE': '01-OCT-1991',
    'EDSDET': 'SIWLS',
    'ELEVANGLE-dg': 20.0,
    'EMISSION -uA': 5.5,
    'FORMAT': 'EMSA/MAS Spectral Data File',
    'LIVETIME  -s': 100.0,
    'MAGCAM': 100.0,
    'NCOLUMNS': 5.0,
    'NPOINTS': 80.0,
    'OFFSET': 200.0,
    'OPERMODE': 'IMAG',
    'OWNER': 'EMSA/MAS TASK FORCE',
    'PROBECUR -nA': 12.345,
    'REALTIME  -s': 150.0,
    'RESTMASS': '511.030',
    'SIGNALTYPE': 'EDS',
    'SOLIDANGL-sR': '0.13',
    'TACTLYR  -cm': 0.3,
    'TAUWIND  -cm': 2e-06,
    'TBEWIND  -cm': 0.0,
    'TDEADLYR -cm': 1e-06,
    'THICKNESS-nm': 50.0,
    'TIME': '12:00',
    'TITLE': 'NIO Windowless Spectra OK NiL',
    'VERSION': '1.0',
    'XLABEL': 'X-RAY ENERGY',
    'XPERCHAN': 10.0,
    'XPOSITION': 123.0,
    'XTILTSTGE-dg': 45.0,
    'XUNITS': 'eV',
    'YLABEL': 'X-RAY INTENSITY',
    'YPOSITION': 456.0,
    'YTILTSTGE-dg': 20.0,
    'YUNITS': 'Intensity',
    'ZPOSITION': 0.0}


class TestExample1:

    def setup_method(self, method):
        self.s = load(os.path.join(
            my_path,
            "msa_files",
            "example1.msa"))

    def test_data(self):
        assert (
            [4066.0,
             3996.0,
             3932.0,
             3923.0,
             5602.0,
             5288.0,
             7234.0,
             7809.0,
             4710.0,
             5015.0,
             4366.0,
             4524.0,
             4832.0,
             5474.0,
             5718.0,
             5034.0,
             4651.0,
             4613.0,
             4637.0,
             4429.0,
             4217.0] == self.s.data.tolist())

    def test_parameters(self):
        assert (
            example1_parameters ==
            self.s.original_metadata.as_dictionary())

    def test_metadata(self):
        assert_deep_almost_equal(self.s.metadata.as_dictionary(),
                                 example1_metadata)

    def test_write_load_cycle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fname2 = os.path.join(tmpdir, "example1-export.msa")
            self.s.save(fname2)
            s2 = load(fname2)
            assert (s2.metadata.General.original_filename ==
                    "example1-export.msa")
            s2.metadata.General.original_filename = "example1.msa"
            assert_deep_almost_equal(self.s.metadata.as_dictionary(),
                                     s2.metadata.as_dictionary())


class TestExample2:

    def setup_method(self, method):
        self.s = load(os.path.join(
            my_path,
            "msa_files",
            "example2.msa"))

    def test_data(self):
        assert (
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
             49.442] == self.s.data.tolist())

    def test_parameters(self):
        assert (
            example2_parameters ==
            self.s.original_metadata.as_dictionary())

    def test_metadata(self):
        assert_deep_almost_equal(self.s.metadata.as_dictionary(),
                                 example2_metadata)

    def test_write_load_cycle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fname2 = os.path.join(tmpdir, "example2-export.msa")
            self.s.save(fname2)
            s2 = load(fname2)
            assert (s2.metadata.General.original_filename ==
                    "example2-export.msa")
            s2.metadata.General.original_filename = "example2.msa"
            assert_deep_almost_equal(self.s.metadata.as_dictionary(),
                                     s2.metadata.as_dictionary())


def test_minimum_metadata_example():
    s = load(os.path.join(my_path, "msa_files", "minimum_metadata.msa"))
    assert minimum_md_om == s.original_metadata.as_dictionary()
