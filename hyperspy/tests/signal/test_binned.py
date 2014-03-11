import numpy as np
import nose.tools

from hyperspy.hspy import *


def test_spectrum():
    s = signals.Spectrum([0])
    nose.tools.assert_false(s.metadata.Signal.binned)


def test_image():
    s = signals.Image(np.empty((2, 2)))
    nose.tools.assert_false(s.metadata.Signal.binned)


def test_image_simulation():
    s = signals.ImageSimulation(np.empty([2, 2]))
    nose.tools.assert_false(s.metadata.Signal.binned)


def test_dielectric_function():
    s = signals.DielectricFunction([0])
    nose.tools.assert_false(s.metadata.Signal.binned)


def test_signal():
    s = signals.Signal([0])
    nose.tools.assert_false(s.metadata.Signal.binned)


def test_simulation():
    s = signals.Simulation([0])
    nose.tools.assert_false(s.metadata.Signal.binned)


def test_spectrum_simulation():
    s = signals.SpectrumSimulation([0])
    nose.tools.assert_false(s.metadata.Signal.binned)


def test_eels_spectrum():
    s = signals.EELSSpectrum([0])
    nose.tools.assert_true(s.metadata.Signal.binned)


def test_eds_tem():
    s = signals.EDSTEMSpectrum([0])
    nose.tools.assert_true(s.metadata.Signal.binned)


def test_eds_sem():
    s = signals.EDSSEMSpectrum([0])
    nose.tools.assert_true(s.metadata.Signal.binned)
