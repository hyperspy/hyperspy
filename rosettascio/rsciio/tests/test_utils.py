from dateutil import parser, tz

import dask.array as da
import numpy as np
import pytest

from rsciio.utils.tools import DTBox, dict2sarray
import rsciio.utils.date_time_tools as dtt

dt = [("x", np.uint8), ("y", np.uint16), ("text", (bytes, 6))]


def _get_example(date, time, time_zone=None):
    md = DTBox({"General": {"date": date, "time": time}}, box_dots=True)
    if time_zone:
        md.set_item("General.time_zone", time_zone)
        dt = parser.parse("%sT%s" % (date, time))
        dt = dt.replace(tzinfo=tz.gettz(time_zone))
        iso = dt.isoformat()
    else:
        iso = "%sT%s" % (date, time)
        dt = parser.parse(iso)
    return md, dt, iso


md1, dt1, iso1 = _get_example("2014-12-27", "00:00:00", "UTC")
serial1 = 42000.00

md2, dt2, iso2 = _get_example("2124-03-25", "10:04:48", "EST")
serial2 = 81900.62833333334

md3, dt3, iso3 = _get_example("2016-07-12", "22:57:32")
serial3 = 42563.95662037037


def test_d2s_fail():
    d = dict(x=5, y=10, text="abcdef")
    with pytest.raises(ValueError):
        dict2sarray(d)


def test_d2s_dtype():
    d = dict(x=5, y=10, text="abcdef")
    ref = np.zeros((1,), dtype=dt)
    ref["x"] = 5
    ref["y"] = 10
    ref["text"] = "abcdef"

    assert ref == dict2sarray(d, dtype=dt)


def test_d2s_extra_dict_ok():
    d = dict(x=5, y=10, text="abcdef", other=55)
    ref = np.zeros((1,), dtype=dt)
    ref["x"] = 5
    ref["y"] = 10
    ref["text"] = "abcdef"

    assert ref == dict2sarray(d, dtype=dt)


def test_d2s_sarray():
    d = dict(x=5, y=10, text="abcdef")

    base = np.zeros((1,), dtype=dt)
    base["x"] = 65
    base["text"] = "gg"

    ref = np.zeros((1,), dtype=dt)
    ref["x"] = 5
    ref["y"] = 10
    ref["text"] = "abcdef"

    assert ref == dict2sarray(d, sarray=base)


def test_d2s_partial_sarray():
    d = dict(text="abcdef")

    base = np.zeros((1,), dtype=dt)
    base["x"] = 65
    base["text"] = "gg"

    ref = np.zeros((1,), dtype=dt)
    ref["x"] = 65
    ref["y"] = 0
    ref["text"] = "abcdef"

    assert ref == dict2sarray(d, sarray=base)


def test_d2s_type_cast_ok():
    d = dict(x="34", text=55)

    ref = np.zeros((1,), dtype=dt)
    ref["x"] = 34
    ref["y"] = 0
    ref["text"] = "55"

    assert ref == dict2sarray(d, dtype=dt)


def test_d2s_type_cast_invalid():
    d = dict(x="Test")
    with pytest.raises(ValueError):
        dict2sarray(d, dtype=dt)


def test_d2s_string_cut():
    d = dict(text="Testerstring")
    sa = dict2sarray(d, dtype=dt)
    assert sa["text"][0] == b"Tester"


def test_d2s_array1():
    dt2 = dt + [("z", (np.uint8, 4)), ("u", (np.uint16, 4))]
    d = dict(z=2, u=[1, 2, 3, 4])
    sa = dict2sarray(d, dtype=dt2)
    np.testing.assert_array_equal(sa["z"][0], [2, 2, 2, 2])
    np.testing.assert_array_equal(sa["u"][0], [1, 2, 3, 4])


def test_d2s_array2():
    d = dict(x=2, y=[1, 2, 3, 4])
    sa = np.zeros((4,), dtype=dt)
    sa = dict2sarray(d, sarray=sa)
    np.testing.assert_array_equal(sa["x"], [2, 2, 2, 2])
    np.testing.assert_array_equal(sa["y"], [1, 2, 3, 4])


def test_d2s_arrayX():
    dt2 = dt + [("z", (np.uint8, 4)), ("u", (np.uint16, 4))]
    d = dict(z=2, u=[1, 2, 3, 4])
    sa = np.zeros((4,), dtype=dt2)
    sa = dict2sarray(d, sarray=sa)
    np.testing.assert_array_equal(
        sa["z"],
        [
            [2, 2, 2, 2],
        ]
        * 4,
    )
    np.testing.assert_array_equal(
        sa["u"],
        [
            [1, 2, 3, 4],
        ]
        * 4,
    )


def test_serial_date_to_ISO_format():
    iso_1 = dtt.serial_date_to_ISO_format(serial1)
    dt1_local = dt1.astimezone(tz.tzlocal())
    assert iso_1[0] == dt1_local.date().isoformat()
    assert iso_1[1] == dt1_local.time().isoformat()
    assert iso_1[2] == dt1_local.tzname()

    iso_2 = dtt.serial_date_to_ISO_format(serial2)
    dt2_local = dt2.astimezone(tz.tzlocal())
    assert iso_2[0] == dt2_local.date().isoformat()
    # The below line will/can fail due to accuracy loss when converting to serial date:
    # We therefore truncate milli/micro seconds
    assert iso_2[1][:8] == dt2_local.time().isoformat()
    assert iso_2[2] == dt2_local.tzname()

    iso_3 = dtt.serial_date_to_ISO_format(serial3)
    dt3_aware = dt3.replace(tzinfo=tz.tzutc())
    dt3_local = dt3_aware.astimezone(tz.tzlocal())
    assert iso_3[0] == dt3_local.date().isoformat()
    assert iso_3[1] == dt3_local.time().isoformat()
    assert iso_3[2] == dt3_local.tzname()


def test_ISO_format_to_serial_date():
    res1 = dtt.ISO_format_to_serial_date(
        dt1.date().isoformat(), dt1.time().isoformat(), timezone=dt1.tzname()
    )
    np.testing.assert_allclose(res1, serial1, atol=1e-5)
    dt = dt2.astimezone(tz.tzlocal())
    res2 = dtt.ISO_format_to_serial_date(
        dt.date().isoformat(), dt.time().isoformat(), timezone=dt.tzname()
    )
    np.testing.assert_allclose(res2, serial2, atol=1e-5)
    res3 = dtt.ISO_format_to_serial_date(
        dt3.date().isoformat(), dt3.time().isoformat(), timezone=dt3.tzname()
    )
    np.testing.assert_allclose(res3, serial3, atol=1e-5)


def test_datetime_to_serial_date():
    np.testing.assert_allclose(dtt.datetime_to_serial_date(dt1), serial1, atol=1e-5)
    np.testing.assert_allclose(dtt.datetime_to_serial_date(dt2), serial2, atol=1e-5)
    np.testing.assert_allclose(dtt.datetime_to_serial_date(dt3), serial3, atol=1e-5)
