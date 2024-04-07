# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import importlib
import pathlib

import pytest

import hyperspy.api as hs
from hyperspy.exceptions import VisibleDeprecationWarning

DIRPATH = pathlib.Path(__file__).parent / "data"
EXSPY_SPEC = importlib.util.find_spec("exspy")

"""
The reference files from hyperspy v1.4 have been created using:

    import hyperspy
    import hyperspy.api as hs

    s = hs.signals.Signal1D(range(100))
    m = s.create_model()

    gaussian = hs.model.components1D.Gaussian(A=13, centre=9, sigma=2)
    arctan = hs.model.components1D.Arctan(minimum_at_zero=False, A=5, k=1.5, x0=75.5)
    arctan_eels = hs.model.components1D.Arctan(minimum_at_zero=True, A=5, k=1.5, x0=22.5)
    voigt = hs.model.components1D.Voigt()
    voigt.area.value = 100
    voigt.centre.value = 50
    voigt.FWHM.value = 1.5
    voigt.gamma.value = 2.5
    polynomial = hs.model.components1D.Polynomial()
    polynomial.coefficients.value = [0.01, -0.5, 25]


    m.extend([gaussian, arctan, arctan_eels, polynomial, voigt])
    m.plot()

    m.store('a')

    version = "".join(hyperspy.__version__.split('.')[:2])
    print("version:", version)
    s.save(f'hs{version}_model.hspy', overwrite=True)
"""


@pytest.mark.parametrize(
    ("versionfile"), ("hs14_model.hspy", "hs15_model.hspy", "hs16_model.hspy")
)
def test_model_backcompatibility(versionfile):
    if EXSPY_SPEC is not None:
        with pytest.warns(VisibleDeprecationWarning):
            # binned deprecated warning
            s = hs.load(DIRPATH / versionfile)

        m = s.models.restore("a")

        assert len(m) == 5

        g = m[0]
        assert g.name == "Gaussian"
        assert len(g.parameters) == 3
        assert g.A.value == 13
        assert g.centre.value == 9
        assert g.sigma.value == 2

        a = m[1]
        assert a.name == "Arctan"
        assert len(a.parameters) == 3
        assert a.A.value == 5
        assert a.k.value == 1.5
        assert a.x0.value == 75.5

        a_eels = m[2]
        assert len(a_eels.parameters) == 3
        assert a_eels.A.value == 5
        assert a_eels.k.value == 1.5
        assert a_eels.x0.value == 22.5

        p = m[3]
        assert p.name == "Polynomial"
        assert len(p.parameters) == 3
        assert p.a0.value == 25.0
        assert p.a1.value == -0.5
        assert p.a2.value == 0.01

        p = m[4]
        assert len(p.parameters) == 8
        assert p.area.value == 100.0
        assert p.centre.value == 50.0
        assert p.FWHM.value == 1.5
        assert p.resolution.value == 0.0
    else:
        with pytest.warns(VisibleDeprecationWarning):
            with pytest.raises(ImportError):
                s = hs.load(DIRPATH / versionfile)
                m = s.models.restore("a")


def test_loading_components_exspy_not_installed():
    with pytest.warns(VisibleDeprecationWarning):
        # warning is for old binning API
        s = hs.load(DIRPATH / "hs16_model.hspy")

    if EXSPY_SPEC is None:
        # This should raise an ImportError with
        # a suitable error message
        with pytest.raises(ImportError) as err:
            _ = s.models.restore("a")
            assert "exspy is not installed" in str(err.value)
    else:
        # The model contains components using numexpr
        pytest.importorskip("numexpr")
        # This should work fine
        _ = s.models.restore("a")
