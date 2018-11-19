# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

import matplotlib
from distutils.version import LooseVersion
import pytest

from hyperspy.misc.test_utils import check_running_tests_in_CI


def test_mlp_agg_for_CI_testing():
    if check_running_tests_in_CI():
        assert matplotlib.get_backend() == 'agg'


def test_mpl_version():
    # for simplicity, only matplotlib 2.x is supported for testing
    assert LooseVersion(matplotlib.__version__) >= LooseVersion('2.0.0')


# Skip if mpl plugin is not called, because it will not failed (no image
# comparison)
@pytest.mark.skipif(not pytest.config.getvalue("mpl"),
                    reason="'mpl' plugin not called.")
@pytest.mark.xfail(reason="Check if plotting tests are working: if this test passes,"
                   " it means that the image comparison of the plotting test are"
                   " not working.",
                   strict=True)
@pytest.mark.mpl_image_compare(baseline_dir='', tolerance=2)
def test_plotting_test_working(mpl_cleanup):
    # If this test passes, it means that the plotting tests are not working!
    # In this case, the test will be reported as failed because the xfail is
    # 'strict'.
    #
    # To check if the plotting test are working, we compare this plot with an
    # incorrect baseline image, so it would fail as expected.
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([1, 2, 2])
    # to generate a different plot uncomment the next line
    # ax.plot([1, 2, 3, 4]) # Uncomment this line to make sure the test is
    # properly failing
    return fig
