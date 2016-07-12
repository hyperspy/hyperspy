# -*- coding: utf-8 -*-
from nose.tools import assert_equal

import hyperspy.signals
import hyperspy.signal


def test_signal_record_by():
    assert_equal(hyperspy.signal.BaseSignal._record_by, "")


def test_signal_signal_type():
    assert_equal(hyperspy.signal.BaseSignal._signal_type, "")


def test_spectrum_record_by():
    assert_equal(hyperspy.signals.Signal1D._record_by, "spectrum")


def test_spectrum_signal_type():
    assert_equal(hyperspy.signals.Signal1D._signal_type, "")


def test_image_record_by():
    assert_equal(hyperspy.signals.Signal2D._record_by, "image")


def test_image_signal_type():
    assert_equal(hyperspy.signals.Signal2D._signal_type, "")


def test_eels_record_by():
    assert_equal(hyperspy.signals.EELSSpectrum._record_by, "spectrum")


def test_eels_signal_type():
    assert_equal(hyperspy.signals.EELSSpectrum._signal_type, "EELS")
