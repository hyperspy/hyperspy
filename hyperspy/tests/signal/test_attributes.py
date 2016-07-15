# -*- coding: utf-8 -*-
from nose.tools import assert_equal

import hyperspy.signals
import hyperspy.signal


def test_basesignal_signal_dimension():
    assert_equal(hyperspy.signal.BaseSignal._signal_dimension, -1)


def test_signal_signal_type():
    assert_equal(hyperspy.signal.BaseSignal._signal_type, "")


def test_spectrum_signal_dimension():
    assert_equal(hyperspy.signals.Signal1D._signal_dimension, 1)


def test_spectrum_signal_type():
    assert_equal(hyperspy.signals.Signal1D._signal_type, "")


def test_image_signal_dimension():
    assert_equal(hyperspy.signals.Signal2D._signal_dimension, 2)


def test_image_signal_type():
    assert_equal(hyperspy.signals.Signal2D._signal_type, "")


def test_eels_signal_dimension():
    assert_equal(hyperspy.signals.EELSSpectrum._signal_dimension, 1)


def test_eels_signal_type():
    assert_equal(hyperspy.signals.EELSSpectrum._signal_type, "EELS")
