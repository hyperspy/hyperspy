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

import datetime
from dateutil import tz
import hyperspy.misc.date_time_tools as dtt

dt1 = datetime.datetime(2014, 12, 27, 00, 00, 00, tzinfo=tz.tzutc())
serial1 = 42000.00
dt2 = datetime.datetime(2124, 3, 25, 10, 4, 48, tzinfo=tz.tzutc())
serial2 = 81900.42
dt3 = datetime.datetime(2016, 7, 12, 22, 57, 32)
serial3 = 42563.95662037037


def test_serial_date_to_ISO_format():
    iso1 = dtt.serial_date_to_ISO_format(serial1)
    dt1_local = dt1.astimezone(tz.tzlocal())
    assert iso1[0] == dt1_local.date().isoformat()
    assert iso1[1] == dt1_local.time().isoformat()
    assert iso1[2] == dt1_local.tzname()

    iso2 = dtt.serial_date_to_ISO_format(serial2)
    dt2_local = dt2.astimezone(tz.tzlocal())
    assert iso2[0] == dt2_local.date().isoformat()
    assert iso2[1] == dt2_local.time().isoformat()
    assert iso2[2] == dt2_local.tzname()

    iso3 = dtt.serial_date_to_ISO_format(serial3)
    dt3_aware = dt3.replace(tzinfo=tz.tzutc())
    dt3_local = dt3_aware.astimezone(tz.tzlocal())
    assert iso3[0] == dt3_local.date().isoformat()
    assert iso3[1] == dt3_local.time().isoformat()
    assert iso3[2] == dt3_local.tzname()


def test_ISO_format_to_serial_date():
    res = dtt.ISO_format_to_serial_date(
        '2014-12-27', '00:00:00', timezone='UTC')
    assert res == serial1
    res2 = dtt.ISO_format_to_serial_date(
        '2124-03-25', '10:04:48', timezone='UTC')
    assert res2 == serial2
    res3 = dtt.ISO_format_to_serial_date(
        '2016-07-12', '22:57:32')
    assert res3 == serial3


def test_datetime_to_serial_date():
    assert dtt.datetime_to_serial_date(dt1) == serial1
    assert dtt.datetime_to_serial_date(dt2) == serial2
    assert dtt.datetime_to_serial_date(dt3) == serial3


def test_metadata_to_datetime():
    md1 = {'General': {'date': '2014-12-27',
                       'time': '00:00:00',
                       'time_zone': 'UTC'}}
    assert dtt.metadata_to_datetime(md1) == dt1

    md2 = {'General': {'date': '2124-03-25',
                       'time': '10:04:48',
                       'time_zone': 'UTC'}}
    assert dtt.metadata_to_datetime(md2) == dt2

    md3 = {'General': {'date': '2016-07-12',
                       'time': '22:57:32'}}
    assert dtt.metadata_to_datetime(md3) == dt3
