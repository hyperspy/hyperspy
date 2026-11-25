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


import pytest

from hyperspy.components1d import Gaussian
from hyperspy.misc.export_dictionary import (
    load_from_dictionary,
    parse_flag_string,
    reconstruct_object,
)


class TestParseFlagString:
    def test_valid_flags_accepted(self):
        assert parse_flag_string("init, sig") == ["init", "sig"]
        assert parse_flag_string("") == [""]

    @pytest.mark.parametrize("flags", [None, b"", 1, []])
    def test_non_string_flags_rejected(self, flags):
        with pytest.raises(ValueError, match="flag"):
            parse_flag_string(flags)

    def test_unknown_flag_name_rejected(self):
        with pytest.raises(ValueError, match="unknown flag"):
            parse_flag_string("bogus,init")


class TestLoadFromDictionary:
    def test_valid_whitelist_accepted(self):
        component = Gaussian()
        load_from_dictionary(component, {"_whitelist": {"A.value": ""}, "A.value": 3.0})
        assert component.A.value == pytest.approx(3.0)

    @pytest.mark.parametrize("flags", [None, b"", 1, []])
    def test_non_string_whitelist_flags_rejected(self, flags):
        with pytest.raises(ValueError, match="flag"):
            load_from_dictionary(
                Gaussian(),
                {"_whitelist": {"A.value": flags}, "A.value": 3.0},
            )

    def test_unknown_flag_name_rejected(self):
        with pytest.raises(ValueError, match="unknown flag"):
            load_from_dictionary(
                Gaussian(),
                {"_whitelist": {"A.value": "bogus"}, "A.value": 3.0},
            )

    def test_incompatible_flag_pair_rejected(self):
        with pytest.raises(ValueError, match="not compatible"):
            load_from_dictionary(
                Gaussian(),
                {"_whitelist": {"A.value": "fn,id"}, "A.value": 3.0},
            )

    def test_non_dictionary_whitelist_rejected(self):
        with pytest.raises(ValueError, match="_whitelist"):
            load_from_dictionary(Gaussian(), {"_whitelist": None})

    def test_whitelist_key_missing_from_dictionary_rejected(self):
        with pytest.raises(ValueError, match="missing from the dictionary"):
            load_from_dictionary(Gaussian(), {"_whitelist": {"A.value": ""}})


class TestReconstructObject:
    @pytest.mark.parametrize("value", [None, 5, (True,), b""])
    def test_malformed_fn_container_rejected(self, value):
        with pytest.raises(ValueError, match='"fn" flag'):
            reconstruct_object("fn", value)

    def test_unpickled_fn_pair_accepted(self):
        assert reconstruct_object("fn", (None, "shape")) == "shape"
