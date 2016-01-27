from __future__ import print_function
import nose.tools as nt
import numpy as np

import hyperspy.api as hs
from hyperspy.events import Event


class TestInteractive():
    def setUp(self):
        d = np.linspace(3, 10.5)
        d = np.tile(d, (3, 3, 1))
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
        e1 = Event()
        e2 = Event()

        ss = hs.interactive(s.sum, e1, axis=0)
        ss.events.data_changed.connect(e2.trigger, 0)
        ss.events.data_changed.connect(lambda: print("Triggered!"), 0)

        sss = hs.interactive(ss.sum, e2, axis=0)

        np.testing.assert_allclose(sss.data, np.sum(s.data, axis=(0, 1)))
        s.data += 3.2
        nt.assert_false(np.allclose(sss.data, np.sum(s.data, axis=(0, 1))))
        e1.trigger()
        np.testing.assert_allclose(sss.data, np.sum(s.data, axis=(0, 1)))

    def _set_flag(self):
        self._triggered = True

    def test_recompute(self):
        s = self.s
        e1 = Event()
        e2 = Event()
        ss = hs.interactive(s.sum, e1, recompute_out_event=e2, axis=0)
        self._triggered = False
        ss.axes_manager.events.transformed.connect(self._set_flag, 0)
        # Check eveything as normal first
        np.testing.assert_allclose(ss.data, np.sum(s.data, axis=1))
        # Modify axes and data in-place
        s.crop(1, 1)
        # Check that data is no longer comparable
        nt.assert_false(np.allclose(ss.data.shape,
                                    np.sum(s.data, axis=1).shape))
        # Check that normal event doesn't fix problem
        e1.trigger()
        nt.assert_false(np.allclose(ss.data.shape,
                                    np.sum(s.data, axis=0).shape))
        # Check that recompute event fixes issue
        nt.assert_false(self._triggered)
        e2.trigger()
        nt.assert_true(self._triggered)
        np.testing.assert_allclose(ss.data, np.sum(s.data, axis=1))
        # Finally, check that axes are updated as they should
        nt.assert_equal(ss.axes_manager.navigation_axes[0].offset, 1)
