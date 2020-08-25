import logging
import os
import tempfile

import hyperspy.api as hs
import pytest


def test_calibration_all(caplog):
    s1 = hs.signals.Signal1D([1.0, 2.0, 3.0])
    s2 = hs.signals.Signal1D([1.0, 2.0, 3.0])

    s1.axes_manager[0].scale = 1.0
    s1.axes_manager[0].offset = 1.0
    s1.axes_manager[0].units = "cm"

    s2.axes_manager[0].scale = 2.0
    s2.axes_manager[0].offset = 1.0 + 1e-5
    s2.axes_manager[0].units = "mm"

    with tempfile.TemporaryDirectory() as tmpdir:
        f1 = os.path.join(tmpdir, "s1.hspy")
        f2 = os.path.join(tmpdir, "s2.hspy")

        s1.save(f1, overwrite=True)
        s2.save(f2, overwrite=True)

        with caplog.at_level(logging.WARNING):
            _ = hs.load([f1, f2], stack=True)

    assert "Mismatched calibration when stacking signals" in caplog.text
    assert "scale" in caplog.text
    assert "offset" in caplog.text
    assert "units" in caplog.text


def test_calibration_units_only(caplog):
    s1 = hs.signals.Signal1D([1.0, 2.0, 3.0])
    s2 = hs.signals.Signal1D([1.0, 2.0, 3.0])

    s2.axes_manager[0].scale = 1.0
    s2.axes_manager[0].offset = 0.0
    s2.axes_manager[0].units = "mm"

    with tempfile.TemporaryDirectory() as tmpdir:
        f1 = os.path.join(tmpdir, "s1.hspy")
        f2 = os.path.join(tmpdir, "s2.hspy")

        s1.save(f1, overwrite=True)
        s2.save(f2, overwrite=True)

        with caplog.at_level(logging.WARNING):
            _ = hs.load([f1, f2], stack=True)

    assert "Mismatched calibration when stacking signals" in caplog.text
    assert "scale" not in caplog.text
    assert "offset" not in caplog.text
    assert "units" in caplog.text
