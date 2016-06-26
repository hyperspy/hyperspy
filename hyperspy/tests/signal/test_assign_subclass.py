from nose.tools import assert_true

import numpy as np

import hyperspy.api as hs
from hyperspy.io import assign_signal_subclass


def test_signal():
    assert_true(assign_signal_subclass(
        dtype=np.dtype('float'),
        record_by="",
        signal_type="",
        signal_origin="") is hs.signals.BaseSignal)


def test_spectrum():
    assert_true(assign_signal_subclass(
        dtype=np.dtype('float'),
        record_by="spectrum",
        signal_type="",
        signal_origin="") is hs.signals.Signal1D)


def test_image():
    assert_true(assign_signal_subclass(
        dtype=np.dtype('float'),
        record_by="image",
        signal_type="",
        signal_origin="") is hs.signals.Signal2D)


def test_image_simulation():
    assert_true(assign_signal_subclass(
        dtype=np.dtype('float'),
        record_by="image",
        signal_type="",
        signal_origin="simulation") is hs.signals.ImageSimulation)


def test_eels_spectrum():
    assert_true(assign_signal_subclass(
        dtype=np.dtype('float'),
        record_by="spectrum",
        signal_type="EELS",
        signal_origin="") is hs.signals.EELSSpectrum)


def test_eels_spectrum_simulation():
    assert_true(assign_signal_subclass(
        dtype=np.dtype('float'),
        record_by="spectrum",
        signal_type="EELS",
        signal_origin="simulation") is
        hs.signals.EELSSpectrumSimulation)


def test_weird_spectrum():
    cls = assign_signal_subclass(
        dtype=np.dtype('float'),
        record_by="spectrum",
        signal_type="weird",
        signal_origin="")
    assert_true(cls is hs.signals.Signal1D)

def test_dielectric_function():
    cls = assign_signal_subclass(
        dtype=np.dtype('complex'),
        record_by="spectrum",
        signal_type="dielectric_function",
        signal_origin="")
    assert_true(cls is hs.signals.DielectricFunction)


def test_complex():
    assert_true(assign_signal_subclass(
        dtype=np.dtype('complex'),
        record_by="",
        signal_type="",
        signal_origin="") is
                hs.signals.ComplexSignal)


def test_electron_wave_image():
    assert_true(assign_signal_subclass(
        dtype=np.dtype('complex'),
        record_by="image",
        signal_type="electron_wave",
        signal_origin="") is
                hs.signals.ElectronWaveImage)


def test_complex_spectrum():
    assert_true(assign_signal_subclass(
        dtype=np.dtype('complex'),
        record_by="spectrum",
        signal_type="",
        signal_origin="") is
                hs.signals.ComplexSignal)

def test_complex_image():
    assert_true(assign_signal_subclass(
        dtype=np.dtype('complex'),
        record_by="image",
        signal_type="",
        signal_origin="") is
                hs.signals.ComplexSignal)

def test_weird_real():
    assert_true(assign_signal_subclass(
        dtype=np.dtype('float'),
        record_by="",
        signal_type="weird",
        signal_origin="") is
                hs.signals.BaseSignal)

def test_weird_complex():
    assert_true(assign_signal_subclass(
        dtype=np.dtype('complex'),
        record_by="",
        signal_type="weird",
        signal_origin="") is
                hs.signals.ComplexSignal)


class TestSignalAssignSubclass:

    def setUp(self):
        self.s = hs.signals.Signal1D([0])

    def test_type_to_eels(self):
        self.s.set_signal_type("EELS")
        assert_true(isinstance(self.s, hs.signals.EELSSpectrum))

    def test_type_to_spectrumsimulation(self):
        self.s.set_signal_origin("simulation")
        assert_true(isinstance(self.s, hs.signals.SpectrumSimulation))


if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)
