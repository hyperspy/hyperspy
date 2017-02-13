# -*- coding: utf-8 -*-
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

import numpy as np
from dateutil import tz, parser

from numpy.testing import assert_allclose

import hyperspy.misc.date_time_tools as dtt
from hyperspy.misc.utils import DictionaryTreeBrowser
from hyperspy.misc.test_utils import assert_deep_almost_equal


def _get_example(date, time, time_zone=None):
    md = DictionaryTreeBrowser({'General': {'date': date,
                                            'time': time}})
    if time_zone:
        md.set_item('General.time_zone', time_zone)
        dt = parser.parse('%sT%s' % (date, time))
        dt = dt.replace(tzinfo=tz.gettz(time_zone))
        iso = dt.isoformat()
    else:
        iso = '%sT%s' % (date, time)
        dt = parser.parse(iso)
    return md, dt, iso

md1, dt1, iso1 = _get_example('2014-12-27', '00:00:00', 'UTC')
serial1 = 42000.00

md2, dt2, iso2 = _get_example('2124-03-25', '10:04:48', 'EST')
serial2 = 81900.62833333334

md3, dt3, iso3 = _get_example('2016-07-12', '22:57:32')
serial3 = 42563.95662037037


def test_get_date_time_from_metadata():
    assert (dtt.get_date_time_from_metadata(md1) ==
            '2014-12-27T00:00:00+00:00')
    assert (dtt.get_date_time_from_metadata(md1, formatting='ISO') ==
            '2014-12-27T00:00:00+00:00')
    assert (dtt.get_date_time_from_metadata(md1, formatting='datetime64') ==
            np.datetime64('2014-12-27T00:00:00.000000'))
    assert (dtt.get_date_time_from_metadata(md1, formatting='datetime') ==
            dt1)

    assert (dtt.get_date_time_from_metadata(md2) ==
            '2124-03-25T10:04:48-05:00')
    assert (dtt.get_date_time_from_metadata(md2, formatting='datetime') ==
            dt2)
    assert (dtt.get_date_time_from_metadata(md2, formatting='datetime64') ==
            np.datetime64('2124-03-25T10:04:48'))

    assert (dtt.get_date_time_from_metadata(md3) ==
            '2016-07-12T22:57:32')
    assert (dtt.get_date_time_from_metadata(md3, formatting='datetime') ==
            dt3)
    assert (dtt.get_date_time_from_metadata(md3, formatting='datetime64') ==
            np.datetime64('2016-07-12T22:57:32.000000'))

    assert (dtt.get_date_time_from_metadata(DictionaryTreeBrowser({'General': {}})) ==
            None)
    assert (dtt.get_date_time_from_metadata(DictionaryTreeBrowser({'General': {'date': '2016-07-12'}})) ==
            '2016-07-12')
    assert (dtt.get_date_time_from_metadata(DictionaryTreeBrowser({'General': {'time': '12:00'}})) ==
            '12:00:00')
    assert (dtt.get_date_time_from_metadata(DictionaryTreeBrowser({'General': {'time': '12:00',
                                                                               'time_zone': 'CET'}})) ==
            '12:00:00')


def _defaut_metadata():
    md = DictionaryTreeBrowser({'General': {'date': '2016-01-01',
                                            'time': '00:00:00',
                                            'time_zone': 'GMT'}})
    return md


def test_update_date_time_in_metadata():
    md = DictionaryTreeBrowser({'General': {}})
    # in case of iso, the exact time is lost, only the time offset is kept
    md11 = dtt.update_date_time_in_metadata(iso1, md.deepcopy())
    assert_deep_almost_equal(md11.General.date, md1.General.date)
    assert_deep_almost_equal(md11.General.time, md1.General.time)
    assert_deep_almost_equal(md11.General.time_zone, 'UTC')

    md12 = dtt.update_date_time_in_metadata(dt1, md.deepcopy())
    assert_deep_almost_equal(md12.General.date, md1.General.date)
    assert_deep_almost_equal(md12.General.time, md1.General.time)
    assert md12.General.time_zone in ('UTC', 'Coordinated Universal Time')

    md13 = dtt.update_date_time_in_metadata(iso2, md.deepcopy())
    assert_deep_almost_equal(md13.General.date, md2.General.date)
    assert_deep_almost_equal(md13.General.time, md2.General.time)
    assert_deep_almost_equal(md13.General.time_zone, '-05:00')
    assert_deep_almost_equal(dtt.update_date_time_in_metadata(dt2, md.deepcopy()).as_dictionary(),
                             md2.as_dictionary())

    assert_deep_almost_equal(dtt.update_date_time_in_metadata(iso3, md.deepcopy()).as_dictionary(),
                             md3.as_dictionary())
    assert_deep_almost_equal(dtt.update_date_time_in_metadata(dt3, md.deepcopy()).as_dictionary(),
                             md3.as_dictionary())


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
        dt1.date().isoformat(), dt1.time().isoformat(), timezone=dt1.tzname())
    assert_allclose(res1, serial1, atol=1E-5)
    dt = dt2.astimezone(tz.tzlocal())
    res2 = dtt.ISO_format_to_serial_date(
        dt.date().isoformat(), dt.time().isoformat(), timezone=dt.tzname())
    assert_allclose(res2, serial2, atol=1E-5)
    res3 = dtt.ISO_format_to_serial_date(
        dt3.date().isoformat(), dt3.time().isoformat(), timezone=dt3.tzname())
    assert_allclose(res3, serial3, atol=1E-5)


def test_datetime_to_serial_date():
    assert_allclose(dtt.datetime_to_serial_date(dt1), serial1, atol=1E-5)
    assert_allclose(dtt.datetime_to_serial_date(dt2), serial2, atol=1E-5)
    assert_allclose(dtt.datetime_to_serial_date(dt3), serial3, atol=1E-5)
