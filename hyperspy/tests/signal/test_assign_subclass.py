from nose.tools import assert_true

import numpy as np

import hyperspy.api as hs
from hyperspy.io import assign_signal_subclass


class TestSignalAssignSubclass:

    def test_signal(self):
        assert_true(assign_signal_subclass(
            dtype=np.dtype('float'),
            record_by="",
            signal_type="",
            signal_origin="") is hs.signals.BaseSignal)


    def test_sinal1d(self):
        assert_true(assign_signal_subclass(
            dtype=np.dtype('float'),
            record_by="spectrum",
            signal_type="",
            signal_origin="") is hs.signals.Signal1D)


    def test_signal2d(self):
        assert_true(assign_signal_subclass(
            dtype=np.dtype('float'),
            record_by="image",
            signal_type="",
            signal_origin="") is hs.signals.Signal2D)


    def test_image_simulation(self):
        assert_true(assign_signal_subclass(
            dtype=np.dtype('float'),
            record_by="image",
            signal_type="",
            signal_origin="simulation") is hs.signals.ImageSimulation)


    def test_eels_spectrum(self):
        assert_true(assign_signal_subclass(
            dtype=np.dtype('float'),
            record_by="spectrum",
            signal_type="EELS",
            signal_origin="") is hs.signals.EELSSpectrum)


    def test_eels_spectrum_simulation(self):
        assert_true(assign_signal_subclass(
            dtype=np.dtype('float'),
            record_by="spectrum",
            signal_type="EELS",
            signal_origin="simulation") is
            hs.signals.EELSSpectrumSimulation)


    def test_weird_spectrum(self):
        cls = assign_signal_subclass(
            dtype=np.dtype('float'),
            record_by="spectrum",
            signal_type="weird",
            signal_origin="")
        assert_true(cls is hs.signals.Signal1D)

    def test_dielectric_function(self):
        cls = assign_signal_subclass(
            dtype=np.dtype('complex'),
            record_by="spectrum",
            signal_type="DielectricFunction",
            signal_origin="")
        assert_true(cls is hs.signals.DielectricFunction)


    def test_complex(self):
        assert_true(assign_signal_subclass(
            dtype=np.dtype('complex'),
            record_by="",
            signal_type="",
            signal_origin="") is
                    hs.signals.ComplexSignal)


    def test_electron_wave_image(self):
        assert_true(assign_signal_subclass(
            dtype=np.dtype('complex'),
            record_by="image",
            signal_type="electron wave",
            signal_origin="") is
                    hs.signals.ElectronWaveImage)


    def test_complex_spectrum(self):
        assert_true(assign_signal_subclass(
            dtype=np.dtype('complex'),
            record_by="spectrum",
            signal_type="",
            signal_origin="") is
                    hs.signals.ComplexSignal)

    def test_complex_image(self):
        assert_true(assign_signal_subclass(
            dtype=np.dtype('complex'),
            record_by="image",
            signal_type="",
            signal_origin="") is
                    hs.signals.ComplexSignal)

    def test_weird_real(self):
        assert_true(assign_signal_subclass(
            dtype=np.dtype('float'),
            record_by="",
            signal_type="weird",
            signal_origin="") is
                    hs.signals.BaseSignal)

    def test_weird_complex(self):
        assert_true(assign_signal_subclass(
            dtype=np.dtype('complex'),
            record_by="",
            signal_type="weird",
            signal_origin="") is
                    hs.signals.ComplexSignal)


class TestConvertBaseSignal:

    def setUp(self):
        self.s = hs.signals.BaseSignal(np.zeros((3, 3)))

    def test_base_to_1d(self):
        self.s.metadata.Signal.record_by = "spectrum"
        self.s._assign_subclass()
        assert_true(isinstance(self.s, hs.signals.Signal1D))
        self.s.metadata.Signal.record_by = ''
        self.s._assign_subclass()
        assert_true(isinstance(self.s, hs.signals.BaseSignal))

    def test_base_to_2d(self):
        self.s.metadata.Signal.record_by = "image"
        self.s._assign_subclass()
        assert_true(isinstance(self.s, hs.signals.Signal2D))
        self.s.metadata.Signal.record_by = ""
        self.s._assign_subclass()
        assert_true(isinstance(self.s, hs.signals.BaseSignal))

    def test_base_to_complex(self):
        self.s.change_dtype(complex)
        assert_true(isinstance(self.s, hs.signals.ComplexSignal))
        # Going back from ComplexSignal to BaseSignal is not possible!
        # If real data is required use `real`, `imag`, `amplitude` or `phase`!

    def test_base_to_simulation(self):
        self.s.set_signal_origin("simulation")
        assert_true(isinstance(self.s, hs.signals.Simulation))
        self.s.set_signal_origin("")
        assert_true(isinstance(self.s, hs.signals.BaseSignal))


class TestConvertComplexSignal:

    def setUp(self):
        self.s = hs.signals.ComplexSignal(np.zeros((3, 3)))

    def test_complex_to_dielectric_function(self):
        self.s.metadata.Signal.record_by = "spectrum"
        self.s.set_signal_type("DielectricFunction")
        assert_true(isinstance(self.s, hs.signals.DielectricFunction))
        self.s.metadata.Signal.record_by = ""
        self.s.set_signal_type("")
        assert_true(isinstance(self.s, hs.signals.ComplexSignal))

    def test_complex_to_electron_wave_image(self):
        self.s.metadata.Signal.record_by = "image"
        self.s.set_signal_type("electron wave")
        assert_true(isinstance(self.s, hs.signals.ElectronWaveImage))
        self.s.metadata.Signal.record_by = ""
        self.s.set_signal_type("")
        assert_true(isinstance(self.s, hs.signals.ComplexSignal))


class TestConvertSignal1D:

    def setUp(self):
        self.s = hs.signals.Signal1D([0])

    def test_signal1d_to_spectrum_simulation(self):
        self.s.set_signal_origin("simulation")
        assert_true(isinstance(self.s, hs.signals.SpectrumSimulation))
        self.s.set_signal_origin("")
        assert_true(isinstance(self.s, hs.signals.Signal1D))

    def test_signal1d_to_eels(self):
        self.s.set_signal_type("EELS")
        assert_true(isinstance(self.s, hs.signals.EELSSpectrum))
        self.s.set_signal_type("")
        assert_true(isinstance(self.s, hs.signals.Signal1D))

    def test_signal1d_to_eels_simulation(self):
        self.s.set_signal_origin("simulation")
        self.s.set_signal_type("EELS")
        assert_true(isinstance(self.s, hs.signals.EELSSpectrumSimulation))
        self.s.set_signal_origin("")
        self.s.set_signal_type("")
        assert_true(isinstance(self.s, hs.signals.Signal1D))

    def test_signal1d_to_eds_tem(self):
        self.s.set_signal_type("EDS_TEM")
        assert_true(isinstance(self.s, hs.signals.EDSTEMSpectrum))
        self.s.set_signal_type("")
        assert_true(isinstance(self.s, hs.signals.Signal1D))

    def test_signal1d_to_eds_sem(self):
        self.s.set_signal_type("EDS_SEM")
        assert_true(isinstance(self.s, hs.signals.EDSSEMSpectrum))
        self.s.set_signal_type("")
        assert_true(isinstance(self.s, hs.signals.Signal1D))


class TestConvertSignal2D:

    def setUp(self):
        self.s = hs.signals.Signal2D(np.zeros((3, 3)))

    def test_signal2d_to_image_simulation(self):
        self.s.set_signal_origin("simulation")
        assert_true(isinstance(self.s, hs.signals.ImageSimulation))
        self.s.set_signal_origin("")
        assert_true(isinstance(self.s, hs.signals.Signal2D))


if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)
