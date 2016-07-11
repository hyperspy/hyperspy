# -*- coding: utf-8 -*-
from nose.tools import assert_equal

import hyperspy.signals
import hyperspy.signal


def test_signal_signal_dimension():
    assert_equal(hyperspy.signal.BaseSignal._signal_dimension, -1)


def test_signal_signal_type():
    assert_equal(hyperspy.signal.BaseSignal._signal_type, "")


def test_signal_signal_origin():
    assert_equal(hyperspy.signal.BaseSignal._signal_origin, "")


def test_spectrum_signal_dimension():
    assert_equal(hyperspy.signals.Signal1D._signal_dimension, 1)


def test_spectrum_signal_type():
    assert_equal(hyperspy.signals.Signal1D._signal_type, "")


def test_spectrum_signal_origin():
    assert_equal(hyperspy.signals.Signal1D._signal_origin, "")


def test_image_signal_dimension():
    assert_equal(hyperspy.signals.Signal2D._signal_dimension, 2)


def test_image_signal_type():
    assert_equal(hyperspy.signals.Signal2D._signal_type, "")


def test_image_signal_origin():
    assert_equal(hyperspy.signals.Signal2D._signal_origin, "")


def test_simulation_signal_dimension():
    assert_equal(hyperspy.signals.Simulation._signal_dimension, -1)


def test_simulation_signal_type():
    assert_equal(hyperspy.signals.Simulation._signal_type, "")


def test_simulation_signal_origin():
    assert_equal(hyperspy.signals.Simulation._signal_origin, "simulation")


def test_spectrum_simulation_signal_dimension():
    assert_equal(hyperspy.signals.SpectrumSimulation._signal_dimension, 1)


def test_spectrum_simulation_signal_type():
    assert_equal(hyperspy.signals.SpectrumSimulation._signal_type, "")


def test_spectrum_simulation_signal_origin():
    assert_equal(hyperspy.signals.SpectrumSimulation._signal_origin,
                 "simulation")


def test_image_simulation_signal_dimension():
    assert_equal(hyperspy.signals.ImageSimulation._signal_dimension, 2)


def test_image_simulation_signal_type():
    assert_equal(hyperspy.signals.ImageSimulation._signal_type, "")


def test_image_simulation_signal_origin():
    assert_equal(hyperspy.signals.ImageSimulation._signal_origin,
                 "simulation")


def test_eels_signal_dimension():
    assert_equal(hyperspy.signals.EELSSpectrum._signal_dimension, 1)


def test_eels_signal_type():
    assert_equal(hyperspy.signals.EELSSpectrum._signal_type, "EELS")


def test_eels_signal_origin():
    assert_equal(hyperspy.signals.EELSSpectrum._signal_origin, "")
