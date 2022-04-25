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
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import numpy as np
import os.path
import tempfile
import pytest

from hyperspy.misc.utils import (
    DictionaryTreeBrowser,
    check_long_string,
    replace_html_symbols,
    nested_dictionary_merge,
)
from hyperspy.exceptions import VisibleDeprecationWarning
from hyperspy.signal import BaseSignal


@pytest.fixture(params=[True, False], ids=["lazy", "not_lazy"])
def tree(request):
    lazy = request.param
    tree = DictionaryTreeBrowser(
        {
            "Node1": {
                "leaf11": 11,
                "Node11": {"leaf111": 111},
            },
            "Node2": {
                "leaf21": 21,
                "Node21": {"leaf211": 211},
            },
        },
        lazy=lazy,
    )
    return tree


class TestDictionaryBrowser:
    def test_add_dictionary(self, tree):
        tree.add_dictionary(
            {
                "Node1": {
                    "leaf12": 12,
                    "Node11": {
                        "leaf111": 222,
                        "Node111": {"leaf1111": 1111},
                    },
                },
                "Node3": {"leaf31": 31},
            }
        )
        assert {
            "Node1": {
                "leaf11": 11,
                "leaf12": 12,
                "Node11": {
                    "leaf111": 222,
                    "Node111": {"leaf1111": 1111},
                },
            },
            "Node2": {
                "leaf21": 21,
                "Node21": {"leaf211": 211},
            },
            "Node3": {"leaf31": 31},
        } == tree.as_dictionary()
        tree.add_dictionary({"_double_lines": ""}, double_lines=True)
        assert tree._double_lines == True
        tree.add_dictionary({"_double_lines": ""}, double_lines=False)
        assert tree._double_lines == False

    def test_deepcopy(self, tree):
        a = tree.deepcopy()
        assert a.as_dictionary() == tree.as_dictionary()

    def test_add_dictionary_space_key(self, tree):
        tree.add_dictionary({"a key with a space": "a value"})
        assert tree.a_key_with_a_space == "a value"
        # All lazy attribute should have been processed
        assert len(tree._lazy_attributes) == 0

    def test_add_signal_in_dictionary(self, tree):
        s = BaseSignal([1.0, 2, 3])
        s.axes_manager[0].name = "x"
        s.axes_manager[0].units = "ly"
        tree.add_dictionary({"_sig_signal name": s._to_dictionary()})
        assert isinstance(tree.signal_name, BaseSignal)
        np.testing.assert_array_equal(tree.signal_name.data, s.data)
        assert tree.signal_name.metadata.as_dictionary() == s.metadata.as_dictionary()
        assert (
            tree.signal_name.axes_manager._get_axes_dicts()
            == s.axes_manager._get_axes_dicts()
        )

    def test_export(self, tree):
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, "testdict.txt")
            tree.export(fname)
            f = open(fname, "r", encoding="utf8")
            assert f.read(9) == "├── Node1"
            f.close()
            tree.export(fname, encoding="utf16")
            f = open(fname, "r", encoding="utf16")
            assert f.read(9) == "├── Node1"
            f.close()
            tree._double_lines = True
            tree.export(fname)
            f = open(fname, "r", encoding="utf8")
            assert f.read(9) == "╠══ Node1"
            f.close()

    def test_signal_to_dictionary(self, tree):
        s = BaseSignal([1.0, 2, 3])
        s.axes_manager[0].name = "x"
        s.axes_manager[0].units = "ly"
        tree.set_item("Some name", s)
        d = tree.as_dictionary()
        np.testing.assert_array_equal(d["_sig_Some name"]["data"], s.data)
        d["_sig_Some name"]["data"] = 0
        assert {
            "Node1": {
                "leaf11": 11,
                "Node11": {"leaf111": 111},
            },
            "Node2": {
                "leaf21": 21,
                "Node21": {"leaf211": 211},
            },
            "_sig_Some name": {
                "attributes": {"_lazy": False, "ragged": False},
                "axes": [
                    {
                        "_type": "UniformDataAxis",
                        "name": "x",
                        "navigate": False,
                        "is_binned": False,
                        "offset": 0.0,
                        "scale": 1.0,
                        "size": 3,
                        "units": "ly",
                    }
                ],
                "data": 0,
                "learning_results": {},
                "metadata": {
                    "General": {"title": ""},
                    "Signal": {"signal_type": ""},
                    "_HyperSpy": {
                        "Folding": {
                            "original_axes_manager": None,
                            "original_shape": None,
                            "unfolded": False,
                            "signal_unfolded": False,
                        }
                    },
                },
                "original_metadata": {},
                "tmp_parameters": {},
            },
        } == d

    def _test_date_time(self, tree, dt_str="now"):
        dt0 = np.datetime64(dt_str)
        data_str, time_str = np.datetime_as_string(dt0).split("T")
        tree.add_node("General")
        tree.General.date = data_str
        tree.General.time = time_str

        dt1 = np.datetime64(f"{tree.General.date}T{tree.General.time}")

        np.testing.assert_equal(dt0, dt1)
        return dt1

    def test_date_time_now(self, tree):
        # not really a test, more a demo to show how to set and use date and
        # time in the DictionaryBrowser
        self._test_date_time(tree)

    def test_date_time_nanosecond_precision(self, tree):
        # not really a test, more a demo to show how to set and use date and
        # time in the DictionaryBrowser
        dt_str = "2016-08-05T10:13:15.450580"
        self._test_date_time(tree, dt_str)

    def test_has_item(self, tree):
        # Check that it finds all actual items:
        assert tree.has_item("Node1")
        assert tree.has_item("Node1.leaf11")
        assert tree.has_item("Node1.Node11")
        assert tree.has_item("Node1.Node11.leaf111")
        assert tree.has_item("Node2")
        assert tree.has_item("Node2.leaf21")
        assert tree.has_item("Node2.Node21")
        assert tree.has_item("Node2.Node21.leaf211")

        # Check that it doesn't find non-existant ones
        assert not tree.has_item("Node3")
        assert not tree.has_item("General")
        assert not tree.has_item("Node1.leaf21")
        assert not tree.has_item("")
        assert not tree.has_item(".")
        assert not tree.has_item("..Node1")

    def test_get_item(self, tree):
        assert tree["Node1"]["leaf11"] == 11

        # Check that it gets all leaf nodes:
        assert tree.get_item("Node1.leaf11") == 11
        assert tree.get_item("Node1.Node11.leaf111") == 111
        assert tree.get_item("Node2.leaf21") == 21
        assert tree.get_item("Node2.Node21.leaf211") == 211

        # Check that it gets all leaf nodes, also with given default:
        assert tree.get_item("Node1.leaf11", 44) == 11
        assert tree.get_item("Node1.Node11.leaf111", 44) == 111
        assert tree.get_item("Node2.leaf21", 44) == 21
        assert tree.get_item("Node2.Node21.leaf211", 44) == 211

        # Check that it returns the default value for various incorrect paths:
        assert tree.get_item("Node1.leaf33", 44) == 44
        assert tree.get_item("Node1.leaf11.leaf111", 44) == 44
        assert tree.get_item("Node1.Node31.leaf311", 44) == 44
        assert tree.get_item("Node1.Node21.leaf311", 44) == 44
        assert tree.get_item(".Node1.Node21.leaf311", 44) == 44

    def test_str(self, tree):
        string = tree.__repr__()
        assert string == "".join(
            [
                "├── Node1\n",
                "│   ├── Node11\n",
                "│   │   └── leaf111 = 111\n",
                "│   └── leaf11 = 11\n",
                "└── Node2\n",
                "    ├── Node21\n",
                "    │   └── leaf211 = 211\n",
                "    └── leaf21 = 21\n",
            ]
        )

        assert tree.get_item("Node1.Node31.leaf311", 44) == 44
        assert tree.get_item("Node1.Node21.leaf311", 44) == 44
        assert tree.get_item(".Node1.Node21.leaf311", 44) == 44

    def test_has_nested_item(self, tree):
        assert tree.has_item("leaf11", full_path=False) == True
        assert tree.has_item("leaf111", full_path=False) == True
        assert tree.has_item("leaf211", full_path=False) == True
        assert tree.has_item("leaf333", full_path=False) == False
        assert tree.has_item("211", full_path=False, wild=True) == True
        assert tree.has_item("333", full_path=False, wild=True) == False
        tree.add_dictionary(
            {
                "_double_lines": False,
            }
        )
        assert tree.has_item("leaf211", full_path=False) == True
        assert tree.has_item("Node11.leaf111", full_path=False) == True
        assert tree.has_item("Node41.leaf111", full_path=False) == False

    def test_has_nested_item_path(self, tree):
        assert tree.has_item("leaf333", full_path=False, return_path=True) == None
        assert tree.has_item("leaf", full_path=False, return_path=True) == None
        assert (
            tree.has_item("leaf333", full_path=False, return_path=True, default=[])
            == []
        )
        assert (
            tree.has_item("333", full_path=False, return_path=True, wild=True) == None
        )
        assert (
            tree.has_item(
                "333", full_path=False, return_path=True, wild=True, default=[]
            )
            == []
        )
        assert (
            tree.has_item("leaf11", full_path=False, return_path=True, default=[])
            == "Node1.leaf11"
        )
        assert (
            tree.has_item("leaf111", full_path=False, return_path=True)
            == "Node1.Node11.leaf111"
        )
        assert (
            tree.has_item("leaf211", full_path=False, return_path=True)
            == "Node2.Node21.leaf211"
        )
        assert (
            tree.has_item("211", full_path=False, return_path=True, wild=True)
            == "Node2.Node21.leaf211"
        )
        assert tree.has_item("leaf", full_path=False, return_path=True, wild=True) == [
            "Node1.leaf11",
            "Node1.Node11.leaf111",
            "Node2.leaf21",
            "Node2.Node21.leaf211",
        ]
        assert (
            tree.has_item("Node11.leaf111", full_path=False, return_path=True)
            == "Node1.Node11.leaf111"
        )
        assert (
            tree.has_item("Node41.leaf111", full_path=False, return_path=True) == None
        )
        assert (
            tree.has_item(
                "Node41.leaf111", full_path=False, return_path=True, default=[]
            )
            == []
        )
        tree.add_dictionary(
            {
                "Node3": {"leaf211": 31},
            }
        )
        assert tree.has_item("leaf211", full_path=False, return_path=True) == [
            "Node2.Node21.leaf211",
            "Node3.leaf211",
        ]

    def test_get_nested_item(self, tree):
        assert tree.get_item("leaf333", full_path=False) == None
        assert tree.get_item("leaf", full_path=False) == None
        assert tree.get_item("leaf333", full_path=False, default=[]) == []
        assert tree.get_item("333", full_path=False, return_path=True, default=[]) == []
        assert tree.get_item("333", full_path=False, wild=True) == None
        assert tree.get_item("333", full_path=False, wild=True, default=[]) == []
        assert (
            tree.get_item("333", full_path=False, wild=True, return_path=True) == None
        )
        assert tree.get_item("333", full_path=False, return_path=True, default=[]) == []
        assert (
            tree.get_item(
                "333", full_path=False, wild=True, return_path=True, default=[]
            )
            == []
        )
        assert tree.get_item("leaf11", full_path=False) == 11
        assert tree.get_item("leaf111", full_path=False) == 111
        assert tree.get_item("leaf211", full_path=False, default=[]) == 211
        assert tree.get_item("211", full_path=False, wild=True) == 211
        assert tree.get_item("Node11.leaf111", full_path=False) == 111
        assert tree.get_item("Node41.leaf111", full_path=False) == None
        assert tree.get_item("Node41.leaf111", full_path=False, default=[]) == []
        assert tree.get_item("leaf211", full_path=False, return_path=True) == (
            211,
            "Node2.Node21.leaf211",
        )
        assert tree.get_item(
            "leaf211", full_path=False, return_path=True, wild=True
        ) == (211, "Node2.Node21.leaf211")
        assert tree.get_item("Node11.leaf111", full_path=False, return_path=True) == (
            111,
            "Node1.Node11.leaf111",
        )
        assert (
            tree.get_item("Node41.leaf111", full_path=False, return_path=True) == None
        )
        assert (
            tree.get_item(
                "Node41.leaf111", full_path=False, return_path=True, default=[]
            )
            == []
        )
        tree.add_dictionary(
            {
                "Node3": {"leaf211": 31},
            }
        )
        assert tree.get_item("leaf211", full_path=False) == [211, 31]
        assert tree.get_item("leaf211", full_path=False, default=[]) == [211, 31]
        assert tree.get_item("211", full_path=False, wild=True) == [211, 31]
        assert tree.get_item("leaf211", full_path=False, return_path=True) == (
            [211, 31],
            ["Node2.Node21.leaf211", "Node3.leaf211"],
        )
        assert tree.get_item("211", full_path=False, wild=True, return_path=True) == (
            [211, 31],
            ["Node2.Node21.leaf211", "Node3.leaf211"],
        )

    # Can be removed once metadata.Signal.binned is deprecated in v2.0
    def test_set_item_binned(self, tree):
        with pytest.warns(VisibleDeprecationWarning, match="Use of the `binned`"):
            tree.set_item("Signal.binned", True)

    def test_html(self, tree):
        "Test that the method actually runs"
        # We do not have a way to validate html
        # without relying on more dependencies
        tree["<myhtmltag>"] = "5 < 6"
        tree["<mybrokenhtmltag"] = "<hello>"
        tree["mybrokenhtmltag2>"] = ""
        tree._get_html_print_items()
        tree.add_dictionary(
            {
                "Node3": {"leaf31": (31, 32), "leaf32": [31, 32]},
                "Node4": {
                    "leaf41": "And now for something completely different. This "
                    "string is so long that it exceeds the max_len limit."
                },
            }
        )
        tree._get_html_print_items()

    def test_print_item_list(self, tree):
        tree.add_dictionary(
            {
                "Node3": {"leaf31": (31, 32), "leaf32": [31, 32]},
            }
        )
        tree.process_lazy_attributes()
        assert tree._get_print_items()[-35:-27] == "(31, 32)"
        assert tree._get_print_items()[-9:-1] == "[31, 32]"
        tree.add_dictionary(
            {
                "Node4": {
                    "leaf41": "And now for something completely different. This "
                    "string is so long that it exceeds the max_len limit."
                },
            }
        )
        assert (
            tree._get_print_items()[-102:-59]
            == "And now for something completely different."
        )

    def test_copy(self, tree):
        treecopy = tree.copy()
        assert treecopy.get_item("Node1.leaf11") == tree.get_item("Node1.leaf11")

    def test_length(self, tree):
        length = len(tree._lazy_attributes)
        assert len(tree) == 2
        assert len(tree._lazy_attributes) == length

    def test_iteration(self, tree):
        assert [key for key, value in tree] == ["Node1", "Node2"]


def test_check_long_string():
    max_len = 20
    value = "Hello everyone this is a long string"
    truth, shortened = check_long_string(value, max_len)
    assert truth == False
    assert shortened == "Hello everyone this is a long string"

    value = "No! It was not a long string! This is a long string!"
    truth, shortened = check_long_string(value, max_len)
    assert truth == True
    assert shortened == "No! It was not a lon ... is is a long string!"


def test_replace_html_symbols():
    assert "&lt;&gt;&amp" == replace_html_symbols("<>&")
    assert "no html symbols" == replace_html_symbols("no html symbols")
    assert "&lt;mix&gt;" == replace_html_symbols("<mix>")


def test_add_key_value():
    key = "<foo>"
    value = ">bar<"

    string = """<ul style="margin: 0px; list-style-position: outside;">
        <li style='margin-left:1em; padding-left: 0.5em'>{} = {}</li></ul>
        """.format(
        replace_html_symbols(key), replace_html_symbols(value)
    )

    assert (
        string
        == "<ul style=\"margin: 0px; list-style-position: outside;\">\n        <li style='margin-left:1em; padding-left: 0.5em'>&lt;foo&gt; = &gt;bar&lt;</li></ul>\n        "
    )


def test_nested_dictionary_merge():
    a = {
        "Node1": {
            "leaf11": 11,
            "Node11": {"leaf111": 111},
        },
        "Node2": {
            "leaf21": 21,
            "Node21": {"leaf211": 211},
        },
    }
    b = {
        "Node1": {
            "leaf12": 12,
            "Node11": {
                "leaf111": 222,
                "Node111": {"leaf1111": 1111},
            },
        },
        "Node3": {"leaf31": 31},
    }
    nested_dictionary_merge(a, b)
    merged_dict = {
        "Node1": {
            "leaf11": 11,
            "leaf12": 12,
            "Node11": {
                "leaf111": 222,
                "Node111": {"leaf1111": 1111},
            },
        },
        "Node2": {
            "leaf21": 21,
            "Node21": {"leaf211": 211},
        },
        "Node3": {"leaf31": 31},
    }
    assert a == merged_dict

    # Override b to a when conflicts
    a = {
        "Node1": {
            "leaf11": 11,
            "Node11": {"leaf111": 111},
        },
    }
    b = {
        "Node1": {
            "leaf11": 12,
            "Node11": {"leaf111": 111},
        },
        "Node2": {
            "leaf21": 21,
            "Node21": {"leaf211": 211},
        },
    }
    nested_dictionary_merge(a, b)
    assert a == b
