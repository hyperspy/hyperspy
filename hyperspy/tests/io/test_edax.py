import nose.tools as nt

import gzip
import hashlib
import os.path
import os
import shutil
import numpy as np

from hyperspy.io import load
from hyperspy import signals

test_files = ['Live Map 2_Img.ipr',
              'single_spect.spc',
              'spd_map.spc',
              'spd_map.spd']
my_path = os.path.dirname(__file__)


class TestSpcSpectrum:

    def setUp(self):
        print('testing single spc spectrum...')
        self.spc = load(os.path.join(
            my_path,
            "edax_files",
            test_files[1]))

    def test_data(self):
        nt.assert_equal(np.uint32, self.spc.data.dtype)     # test datatype
        nt.assert_equal((4096,), self.spc.data.shape)       # test data shape
        nt.assert_equal(
            [0, 0, 0, 0, 0, 0, 1, 2, 3, 3, 10, 4, 10, 10, 45, 87, 146, 236,
             312, 342], self.spc.data[:20].tolist())   # test 1st 20 datapoints

    def test_parameters(self):
        elements = self.spc.metadata.as_dictionary()['Sample']['elements']
        sem_dict = self.spc.metadata.as_dictionary()[
            'Acquisition_instrument']['SEM']
        eds_dict = sem_dict['Detector']['EDS']
        signal_dict = self.spc.metadata.as_dictionary()['Signal']

        # Testing SEM parameters
        nt.assert_almost_equal(22, sem_dict['beam_energy'])
        nt.assert_almost_equal(0, sem_dict['tilt_stage'])

        # Testing EDS parameters
        nt.assert_almost_equal(0, eds_dict['azimuth_angle'])
        nt.assert_almost_equal(34, eds_dict['elevation_angle'])
        nt.assert_almost_equal(129.31299, eds_dict['energy_resolution_MnKa'],
                               places=5)
        nt.assert_almost_equal(50.000004, eds_dict['live_time'], places=6)

        # Testing elements
        nt.assert_set_equal({'Al', 'C', 'Ce', 'Cu', 'F', 'Ho', 'Mg', 'O'},
                            set(elements))

        # Testing HyperSpy parameters
        nt.assert_equal(True, signal_dict['binned'])
        nt.assert_equal('EDS_SEM', signal_dict['signal_type'])
        nt.assert_is_instance(self.spc, signals.EDSSEMSpectrum)

    def test_axes(self):
        spc_ax_manager = {'axis-0': {'name': 'Energy',
                                     'navigate': False,
                                     'offset': 0.0,
                                     'scale': 0.01,
                                     'size': 4096,
                                     'units': 'keV'}}
        nt.assert_dict_equal(spc_ax_manager,
                             self.spc.axes_manager.as_dictionary())


class TestSpdMap:

    @classmethod
    def setup_class(self):
        print('testing spd map...')
        spd_fname = os.path.join(my_path,
                                 "edax_files",
                                 test_files[3])

        if not os.path.isfile(spd_fname):
            with gzip.open(os.path.join(my_path,
                                        "edax_files",
                                        test_files[3] + ".gz")) as f_in:
                with open(spd_fname, 'wb') as f_out:
                    f_out.write(f_in.read())
                print('Successfully decompressed test map data!')

        if hashlib.md5(open(spd_fname, 'rb').read()).hexdigest() != \
                'a0c29793146c9e7438fa9b2e1ca05046':
            raise ValueError('Something went wrong with decompressing the test'
                             ' file. Please try again.')
        self.spd = load(os.path.join(my_path,
                                     "edax_files",
                                     test_files[3]))

    @classmethod
    def teardown_class(self):
        spd_fname = os.path.join(my_path,
                                 "edax_files",
                                 test_files[3])

        # hack to release memmap object to allow deleting uncompressed spd map
        self.spd.data._mmap.close()

        os.remove(spd_fname)

    def test_data(self):
        nt.assert_equal(np.uint16, self.spd.data.dtype)     # test d_type
        nt.assert_equal((200, 256, 2500), self.spd.data.shape)  # test d_shape
        nt.assert_equal([[[0, 0, 0, 0, 0],              # test random data
                          [0, 0, 1, 0, 1],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]],
                         [[0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]],
                         [[0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1],
                          [0, 1, 1, 0, 0],
                          [0, 0, 0, 0, 0]],
                         [[0, 1, 0, 0, 0],
                          [0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 1, 0]],
                         [[0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 1],
                          [0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0]]],
                        self.spd.data[15:20, 15:20, 15:20].tolist())

    def test_parameters(self):
        elements = self.spd.metadata.as_dictionary()['Sample']['elements']
        sem_dict = self.spd.metadata.as_dictionary()[
            'Acquisition_instrument']['SEM']
        eds_dict = sem_dict['Detector']['EDS']
        signal_dict = self.spd.metadata.as_dictionary()['Signal']

        # Testing SEM parameters
        nt.assert_almost_equal(22, sem_dict['beam_energy'])
        nt.assert_almost_equal(0, sem_dict['tilt_stage'])

        # Testing EDS parameters
        nt.assert_almost_equal(0, eds_dict['azimuth_angle'])
        nt.assert_almost_equal(34, eds_dict['elevation_angle'])
        nt.assert_almost_equal(126.60252, eds_dict['energy_resolution_MnKa'],
                               places=5)
        nt.assert_almost_equal(2621.4399, eds_dict['live_time'], places=4)

        # Testing elements
        nt.assert_set_equal({'Ce', 'Co', 'Cr', 'Fe', 'Gd', 'La', 'Mg', 'O',
                             'Sr'}, set(elements))

        # Testing HyperSpy parameters
        nt.assert_equal(True, signal_dict['binned'])
        nt.assert_equal('EDS_SEM', signal_dict['signal_type'])
        nt.assert_is_instance(self.spd, signals.EDSSEMSpectrum)

    def test_axes(self):
        spd_ax_manager = {'axis-0': {'name': 'y',
                                     'navigate': True,
                                     'offset': 0.0,
                                     'scale': 0.014227345585823059,
                                     'size': 200,
                                     'units': '$\\mu m$'},
                          'axis-1': {'name': 'x',
                                     'navigate': True,
                                     'offset': 0.0,
                                     'scale': 0.014235896058380604,
                                     'size': 256,
                                     'units': '$\\mu m$'},
                          'axis-2': {'name': 'Energy',
                                     'navigate': False,
                                     'offset': 0.0,
                                     'scale': 0.0050000000000000001,
                                     'size': 2500,
                                     'units': 'keV'}}
        nt.assert_dict_equal(spd_ax_manager,
                             self.spd.axes_manager.as_dictionary())

    def test_ipr_reading(self):
        ipr_header = self.spd.original_metadata['ipr_header']
        nt.assert_almost_equal(0.014235896, ipr_header['mppX'])
        nt.assert_almost_equal(0.014227346, ipr_header['mppY'])

    def test_spc_reading(self):
        # Test to make sure that spc metadata matches spd metadata
        spc_header = self.spd.original_metadata['spc_header']

        elements = self.spd.metadata.as_dictionary()['Sample']['elements']
        sem_dict = self.spd.metadata.as_dictionary()[
            'Acquisition_instrument']['SEM']
        eds_dict = sem_dict['Detector']['EDS']

        nt.assert_almost_equal(spc_header.azimuth,
                               eds_dict['azimuth_angle'])
        nt.assert_almost_equal(spc_header.detReso,
                               eds_dict['energy_resolution_MnKa'])
        nt.assert_almost_equal(spc_header.elevation,
                               eds_dict['elevation_angle'])
        nt.assert_almost_equal(spc_header.liveTime,
                               eds_dict['live_time'])
        nt.assert_almost_equal(spc_header.evPerChan,
                               self.spd.axes_manager[2].scale * 1000)
        nt.assert_almost_equal(spc_header.kV,
                               sem_dict['beam_energy'])
        nt.assert_almost_equal(spc_header.numElem,
                               len(elements))

