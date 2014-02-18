from nose.tools import assert_true

from hyperspy.signals import *
from hyperspy.io import assign_signal_subclass


def test_signal():
    assert_true(assign_signal_subclass(
        record_by="",
        signal_type="",
        signal_origin="") is Signal)


def test_spectrum():
    assert_true(assign_signal_subclass(
        record_by="spectrum",
        signal_type="",
        signal_origin="") is Spectrum)


def test_image():
    assert_true(assign_signal_subclass(
        record_by="image",
        signal_type="",
        signal_origin="") is Image)


def test_image_simulation():
    assert_true(assign_signal_subclass(
        record_by="image",
        signal_type="",
        signal_origin="simulation") is ImageSimulation)


def test_eels_spectrum():
    assert_true(assign_signal_subclass(
        record_by="spectrum",
        signal_type="EELS",
        signal_origin="") is EELSSpectrum)


def test_eels_spectrum_simulation():
    assert_true(assign_signal_subclass(
        record_by="spectrum",
        signal_type="EELS",
        signal_origin="simulation") is
        EELSSpectrumSimulation)


def test_weird_spectrum():
    cls = assign_signal_subclass(
        record_by="spectrum",
        signal_type="weird",
        signal_origin="")
    assert_true(cls is Spectrum)
