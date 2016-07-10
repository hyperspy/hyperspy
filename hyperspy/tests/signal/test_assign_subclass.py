from nose.tools import assert_true

import hyperspy.api as hs
from hyperspy.io import assign_signal_subclass


def test_signal():
    assert_true(assign_signal_subclass(
        record_by="",
        signal_type="",) is hs.signals.BaseSignal)


def test_spectrum():
    assert_true(assign_signal_subclass(
        record_by="spectrum",
        signal_type="",) is hs.signals.Signal1D)


def test_image():
    assert_true(assign_signal_subclass(
        record_by="image",
        signal_type="",) is hs.signals.Signal2D)


def test_eels_spectrum():
    assert_true(assign_signal_subclass(
        record_by="spectrum",
        signal_type="EELS",) is hs.signals.EELSSpectrum)


def test_weird_spectrum():
    cls = assign_signal_subclass(
        record_by="spectrum",
        signal_type="weird",)
    assert_true(cls is hs.signals.Signal1D)


class TestSignalAssignSubclass:

    def setUp(self):
        self.s = hs.signals.Signal1D([0])

    def test_type_to_eels(self):
        self.s.set_signal_type("EELS")
        assert_true(isinstance(self.s, hs.signals.EELSSpectrum))
