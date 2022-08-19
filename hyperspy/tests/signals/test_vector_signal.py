# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
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
# along with HyperSpy. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import pytest
import dask.array as da

from hyperspy.decorators import lazifyTestClass
from hyperspy.signals import BaseVectorSignal, BaseSignal


class TestVectorSignal:
    @pytest.fixture
    def two_d_vector(self):
        x = np.empty(shape=(4, 3), dtype=object)
        for i in np.ndindex(x.shape):
            x[i] = np.random.random((4, 2)) * 10
        s = BaseVectorSignal(x)
        s.vector = True
        for ax, name, scale in zip(s.axes_manager._axes, "abcd", [0.1, 0.1, 0.1, 0.1]):
            ax.name = name
            ax.scale = scale
        return s

    @pytest.fixture
    def lazy_four_d_vector(self):
        x = da.empty(shape=(4), dtype=object)
        for i in np.ndindex(x.shape):
            x[i] = da.random.random((6, 4))
        s = L(x).T
        s.vector = True
        s.set_signal_type("vector")
        for ax, name in zip(s.axes_manager._axes, "abcd"):
            ax.name = name
        return s

    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize("axis", [None, 0, (0, 1)])
    def test_all_to_vector(self, two_d_vector, inplace, axis):
        new = two_d_vector.nav_to_vector(inplace=inplace, axis=axis)
        if inplace:
            new = two_d_vector
        if axis is None or axis == (0, 1):
            assert len(new.axes_manager.signal_axes) == 4
            assert new.data.shape == (1,)
        elif axis == 0:
            assert len(new.axes_manager.signal_axes) == 3
            assert new.data.shape == (3,)

    @pytest.mark.parametrize("flatten", [True, False])
    @pytest.mark.parametrize("axis", [None, (0,), (0, 1), (2, 3),])
    @pytest.mark.parametrize("real_units", [True, False])
    def test_get_real_vectors(self,
                              two_d_vector,
                              axis,
                              real_units,
                              flatten,
                              ):
        new = two_d_vector.get_real_vectors(axis=axis,
                                            real_units=real_units,
                                            flatten=flatten,
                                            )
        if axis is None:
            axis = (0, 1, 2, 3)
        output_len = len(axis)
        if flatten:
            assert new.shape[1] == output_len
            if real_units:
                assert np.max(new) < 1
            else:
                assert np.max(new) < 10
        else:
            assert new[0, 0].shape[1] == output_len
            assert new.shape == two_d_vector.data.shape

    @pytest.mark.parametrize("top", [0.5, 5,])
    def test_slicing_vector(self, two_d_vector, top):
        sliced = two_d_vector.isig[0:top, :]
        is_between = np.all(
            [np.all(5 > sliced.data[i][:, 1]) for i in np.ndindex((4, 3))]
        )

        assert is_between
        assert isinstance(sliced, BaseVectorSignal)
        assert len(sliced.axes_manager.signal_axes) == 2

    def test_slicing_nav_vector(self, two_d_vector):
        sliced = two_d_vector.inav[0:2, :]
        assert isinstance(sliced, BaseVectorSignal)
        assert len(sliced.axes_manager.signal_axes) == 2
        assert sliced.axes_manager.navigation_shape == (2, 4)

    @pytest.mark.parametrize("inplace", [True, False, ])
    def test_flatten(self, two_d_vector, inplace):
        flat = two_d_vector.flatten(inplace=inplace)
        if inplace:
            flat = two_d_vector
        assert len(flat.axes_manager.signal_axes) == 4

    def test_deepcopy(self, two_d_vector):
        s = two_d_vector.deepcopy()
        assert isinstance(s, BaseVectorSignal)

    @pytest.mark.parametrize("axis", [0, 1, None])
    def test_all_to_vector_axis(self, two_d_vector, axis):
        new = two_d_vector.nav_to_vector(axis=axis, inplace=False)
        if axis is None:
            assert len(new.axes_manager.signal_axes) == 4
            assert new.data.shape == (1,)
        else:
            assert len(new.axes_manager.signal_axes) == 3
            assert new.data.shape == (two_d_vector.axes_manager.navigation_shape[axis],)

    @pytest.mark.parametrize("axis", [None, (0, 1), (1,), (0, 1, 3)])
    @pytest.mark.parametrize("real_units", [True, False])
    @pytest.mark.parametrize("inplace", [True, False])
    def test_cluster(self, two_d_vector, axis, real_units, inplace):
        from sklearn.cluster import KMeans
        clustering = KMeans(4)
        clustered = two_d_vector.cluster(clustering,
                                         axis=axis,
                                         real_units=real_units,
                                         inplace=inplace)
        assert len(clustered.data) == 3
        assert clustered.data[0].shape[1] == 4
        assert clustered.learning_results is not None





    def test_to_markers(self, two_d_vector):
        markers = two_d_vector.to_markers()
