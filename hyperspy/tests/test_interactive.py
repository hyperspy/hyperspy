from __future__ import print_function
import nose.tools as nt
import numpy as np

import hyperspy.api as hs
from hyperspy.events import Event


class TestInteractive():

    def setUp(self):
        d = np.linspace(3, 10.5)
        d = np.tile(d, (3, 3, 1))
        # data shape (3, 3, 50)
        s = hs.signals.Spectrum(d)
        self.s = s

    def test_interactive_sum(self):
        s = self.s
        e = Event()
        ss = hs.interactive(s.sum, e, axis=0)
        np.testing.assert_allclose(ss.data, np.sum(s.data, axis=0))
        s.data += 3.2
        nt.assert_false(np.allclose(ss.data, np.sum(s.data, axis=0)))
        e.trigger()
        np.testing.assert_allclose(ss.data, np.sum(s.data, axis=0))

    def test_chained_interactive(self):
        s = self.s
        e1, e2 = Event(), Event()
        ss = hs.interactive(s.sum, e1, axis=0)
        sss = hs.interactive(ss.sum, e2, axis=0)
        np.testing.assert_allclose(sss.data, np.sum(s.data, axis=(0, 1)))
        s.data += 3.2
        nt.assert_false(np.allclose(ss.data, np.sum(s.data, axis=(1))))
        e1.trigger()
        np.testing.assert_allclose(ss.data, np.sum(s.data, axis=(1)))
        nt.assert_false(np.allclose(ss.data, np.sum(s.data, axis=(0, 1))))
        e2.trigger()
        np.testing.assert_allclose(sss.data, np.sum(s.data, axis=(0, 1)))

    def test_recompute(self):
        s = self.s
        e1 = Event()
        e2 = Event()
        ss = hs.interactive(s.sum, e1, recompute_out_event=e2, axis=0)
        # Check eveything as normal first
        np.testing.assert_equal(ss.data, np.sum(s.data, axis=1))
        # Modify axes and data in-place
        s.crop(1, 1)  # data shape (2, 3, 50)
        # Check that data is no longer comparable
        nt.assert_not_equal(ss.data.shape, np.sum(s.data, axis=1).shape)
        # Check that normal event raises an exception due to the invalid shape
        nt.assert_raises(ValueError, e1.trigger)
        # Check that recompute event fixes issue
        e2.trigger()
        np.testing.assert_equal(ss.data, np.sum(s.data, axis=1))
        # Finally, check that axes are updated as they should
        nt.assert_equal(ss.axes_manager.navigation_axes[0].offset, 1)
