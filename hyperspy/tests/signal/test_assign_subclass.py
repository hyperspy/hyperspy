import numpy as np

import hyperspy.api as hs
from hyperspy.io import assign_signal_subclass


class TestSignalAssignSubclass:

    def test_signal(self):
        assert assign_signal_subclass(
            dtype=np.dtype('float'),
            signal_dimension=1000,
            signal_type="") is hs.signals.BaseSignal

    def test_signal1d(self):
        assert assign_signal_subclass(
            dtype=np.dtype('float'),
            signal_dimension=1,
            signal_type="") is hs.signals.Signal1D

    def test_signal2d(self):
        assert assign_signal_subclass(
            dtype=np.dtype('float'),
            signal_dimension=2,
            signal_type="") is hs.signals.Signal2D

    def test_eels_spectrum(self):
        assert assign_signal_subclass(
            dtype=np.dtype('float'),
            signal_dimension=1,
            signal_type="EELS") is hs.signals.EELSSpectrum

    def test_eds_sem_spectrum(self):
        assert assign_signal_subclass(
            dtype=np.dtype('float'),
            signal_dimension=1,
            signal_type="EDS_SEM") is hs.signals.EDSSEMSpectrum

    def test_eds_tem_spectrum(self):
        assert assign_signal_subclass(
            dtype=np.dtype('float'),
            signal_dimension=1,
            signal_type="EDS_TEM") is hs.signals.EDSTEMSpectrum

    def test_dielectric_function(self):
        assert assign_signal_subclass(
            dtype=complex,
            signal_dimension=1,
            signal_type="DielectricFunction") is hs.signals.DielectricFunction

    def test_dielectric_function_alias(self):
        assert assign_signal_subclass(
            dtype=complex,
            signal_dimension=1,
            signal_type="dielectric function") is hs.signals.DielectricFunction

    def test_complex(self):
        assert assign_signal_subclass(
            dtype=complex,
            signal_dimension=1000,
            signal_type="") is hs.signals.ComplexSignal

    def test_complex_spectrum(self):
        assert assign_signal_subclass(
            dtype=complex,
            signal_dimension=1,
            signal_type="") is hs.signals.ComplexSignal1D

    def test_complex_image(self):
        assert assign_signal_subclass(
            dtype=complex,
            signal_dimension=2,
            signal_type="") is hs.signals.ComplexSignal2D

    def test_weird_real(self):
        assert assign_signal_subclass(
            dtype=np.dtype('float'),
            signal_dimension=1000,
            signal_type="weird") is hs.signals.BaseSignal

    def test_weird_spectrum(self):
        assert assign_signal_subclass(
            dtype=np.dtype('float'),
            signal_dimension=1,
            signal_type="weird") is hs.signals.Signal1D

    def test_weird_complex(self):
        assert assign_signal_subclass(
            dtype=complex,
            signal_dimension=1000,
            signal_type="weird") is hs.signals.ComplexSignal


class TestConvertBaseSignal:

    def setup_method(self, method):
        self.s = hs.signals.BaseSignal(np.zeros((3, 3)))

    def test_base_to_1d(self):
        self.s.axes_manager.set_signal_dimension(1)
        self.s._assign_subclass()
        assert isinstance(self.s, hs.signals.Signal1D)
        self.s.metadata.Signal.record_by = ''
        self.s._assign_subclass()
        assert isinstance(self.s, hs.signals.BaseSignal)

    def test_base_to_2d(self):
        self.s.axes_manager.set_signal_dimension(2)
        self.s._assign_subclass()
        assert isinstance(self.s, hs.signals.Signal2D)

    def test_base_to_complex(self):
        self.s.change_dtype(complex)
        assert isinstance(self.s, hs.signals.ComplexSignal)
        # Going back from ComplexSignal to BaseSignal is not possible!
        # If real data is required use `real`, `imag`, `amplitude` or `phase`!


class TestConvertSignal1D:

    def setup_method(self, method):
        self.s = hs.signals.Signal1D([0])

    def test_signal1d_to_eels(self):
        self.s.set_signal_type("EELS")
        assert isinstance(self.s, hs.signals.EELSSpectrum)
        self.s.set_signal_type("")
        assert isinstance(self.s, hs.signals.Signal1D)

    def test_signal1d_to_eds_tem(self):
        self.s.set_signal_type("EDS_TEM")
        assert isinstance(self.s, hs.signals.EDSTEMSpectrum)
        self.s.set_signal_type("")
        assert isinstance(self.s, hs.signals.Signal1D)

    def test_signal1d_to_eds_sem(self):
        self.s.set_signal_type("EDS_SEM")
        assert isinstance(self.s, hs.signals.EDSSEMSpectrum)
        self.s.set_signal_type("")
        assert isinstance(self.s, hs.signals.Signal1D)


class TestConvertComplexSignal:

    def setup_method(self, method):
        self.s = hs.signals.ComplexSignal(np.zeros((3, 3)))

    def test_complex_to_complex1d(self):
        self.s.axes_manager.set_signal_dimension(1)
        self.s._assign_subclass()
        assert isinstance(self.s, hs.signals.ComplexSignal1D)

    def test_complex_to_complex2d(self):
        self.s.axes_manager.set_signal_dimension(2)
        self.s._assign_subclass()
        assert isinstance(self.s, hs.signals.ComplexSignal2D)


class TestConvertComplexSignal1D:

    def setup_method(self, method):
        self.s = hs.signals.ComplexSignal1D([0])

    def test_complex_to_dielectric_function(self):
        self.s.set_signal_type("DielectricFunction")
        assert isinstance(self.s, hs.signals.DielectricFunction)
        self.s.set_signal_type("")
        assert isinstance(self.s, hs.signals.ComplexSignal1D)


if __name__ == '__main__':

    import pytest
    pytest.main(__name__)
