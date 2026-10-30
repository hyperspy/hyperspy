# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
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

from __future__ import print_function

from unittest import mock

import numpy as np
import pytest

import hyperspy.api as hs
from hyperspy.events import Event
from hyperspy.interactive import Interactive


class TestInteractive:
    def setup_method(self, method):
        d = np.linspace(3, 10.5)
        d = np.tile(d, (3, 3, 1))
        # data shape (3, 3, 50)
        s = hs.signals.Signal1D(d)
        self.s = s

    def test_interactive_sum(self):
        s = self.s
        e = Event()
        ss = hs.interactive(s.sum, e, axis=0)
        np.testing.assert_array_equal(ss.data, np.sum(s.data, axis=0))
        s.data += 3.2
        assert not np.allclose(ss.data, np.sum(s.data, axis=0))
        e.emit()
        np.testing.assert_array_equal(ss.data, np.sum(s.data, axis=0))

    def test_interactive_sum_no_out(self):
        s = self.s

        def sumf(axis):
            return s.sum(axis=axis)

        e = Event()
        ss = hs.interactive(sumf, e, axis=0)
        np.testing.assert_array_equal(ss.data, np.sum(s.data, axis=0))
        s.data += 3.2
        assert not np.allclose(ss.data, np.sum(s.data, axis=0))
        e.emit()
        np.testing.assert_array_equal(ss.data, np.sum(s.data, axis=0))

    def test_interactive_sum_auto_event(self):
        s = self.s
        ss = hs.interactive(s.sum, axis=0)
        np.testing.assert_equal(ss.data, np.sum(s.data, axis=0))
        s.data += 3.2
        assert not np.allclose(ss.data, np.sum(s.data, axis=0))
        s.events.data_changed.emit(s)
        np.testing.assert_array_equal(ss.data, np.sum(s.data, axis=0))

    def test_chained_interactive(self):
        s = self.s
        e1, e2 = Event(), Event()
        ss = hs.interactive(s.sum, e1, axis=0)
        sss = hs.interactive(ss.sum, e2, axis=0)
        np.testing.assert_allclose(sss.data, np.sum(s.data, axis=(0, 1)))
        s.data += 3.2
        assert not np.allclose(ss.data, np.sum(s.data, axis=(1)))
        e1.emit()
        np.testing.assert_allclose(ss.data, np.sum(s.data, axis=(1)))
        assert not np.allclose(sss.data, np.sum(s.data, axis=(0, 1)))
        e2.emit()
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
        assert ss.data.shape != np.sum(s.data, axis=1).shape
        # Check that normal event raises an exception due to the invalid shape
        with pytest.raises(ValueError):
            e1.emit()
        # Check that recompute event fixes issue
        e2.emit()
        np.testing.assert_equal(ss.data, np.sum(s.data, axis=1))
        # Finally, check that axes are updated as they should
        assert ss.axes_manager.navigation_axes[0].offset == 1

    def test_recompute_auto_recompute(self):
        s = self.s
        ss = hs.interactive(s.sum, axis=0)
        # Check eveything as normal first
        np.testing.assert_equal(ss.data, np.sum(s.data, axis=1))
        # Modify axes and data in-place
        m = mock.Mock()
        s.axes_manager.events.any_axis_changed.connect(m.changed)
        s.crop(1, 1)  # data shape (2, 3, 50)
        assert m.changed.called
        np.testing.assert_equal(ss.data, np.sum(s.data, axis=1))
        # Finally, check that axes are updated as they should
        assert ss.axes_manager.navigation_axes[0].offset == 1

    def test_two_update_events(self):
        s = self.s
        e1 = Event()
        e2 = Event()
        ss = hs.interactive(s.sum, event=(e1, e2), recompute_out_event=None, axis=0)
        s.data[:] = 0
        e1.emit()
        np.testing.assert_equal(ss.data, np.sum(s.data, axis=1))
        s.data[:] = 1
        e2.emit()
        np.testing.assert_equal(ss.data, np.sum(s.data, axis=1))

    def test_two_recompute_events(self):
        s = self.s
        e1 = Event()
        e2 = Event()
        ss = hs.interactive(s.sum, event=None, recompute_out_event=(e1, e2), axis=0)
        s.data[:] = 0
        e1.emit()
        np.testing.assert_equal(ss.data, np.sum(s.data, axis=1))
        s.data[:] = 1
        e2.emit()
        np.testing.assert_equal(ss.data, np.sum(s.data, axis=1))

    def test_interactive_function_return_None(self):
        e = Event()

        def function_return_None():
            print("function called")

        hs.interactive(function_return_None, e)
        e.emit()

    def test_close_disconnects_explicit_event(self):
        s = self.s
        e = Event()
        op = Interactive(s.sum, event=e, recompute_out_event=None, axis=0)
        initial_data = op.out.data.copy()
        s.data += 3.2
        op.close()
        e.emit()
        # After close, event should have no effect — data unchanged
        np.testing.assert_array_equal(op.out.data, initial_data)

    def test_close_disconnects_recompute_out_event(self):
        s = self.s
        e1 = Event()
        e2 = Event()
        op = Interactive(s.sum, event=e1, recompute_out_event=e2, axis=0)
        initial_data = op.out.data.copy()
        s.crop(1, 1)
        op.close()
        # Triggering should NOT update after close
        e1.emit()
        np.testing.assert_array_equal(op.out.data, initial_data)

    def test_close_keeps_out_accessible(self):
        s = self.s
        e = Event()
        op = Interactive(s.sum, event=e, recompute_out_event=None, axis=0)
        op.close()
        # out should still be readable
        assert op.out is not None
        np.testing.assert_array_equal(op.out.data, np.sum(s.data, axis=1))

    def test_close_idempotent(self):
        s = self.s
        e = Event()
        op = Interactive(s.sum, event=e, recompute_out_event=None, axis=0)
        op.close()
        # Second close should not raise
        op.close()
        assert op.out is not None

    def test_close_auto_event(self):
        s = self.s
        op = Interactive(s.sum, axis=0)
        initial_data = op.out.data.copy()
        s.data += 3.2
        op.close()
        s.events.data_changed.emit(s)
        # After close, data_changed should have no effect
        np.testing.assert_array_equal(op.out.data, initial_data)
