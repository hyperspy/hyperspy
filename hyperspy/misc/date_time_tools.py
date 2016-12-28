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
from dateutil import tz, parser


def metadata_to_datetime(metadata):
    date = metadata['General']['date']
    time = metadata['General']['time']
    if 'time_zone' in metadata['General']:
        return parser.parse('%sT%s %s' % (date, time, metadata['General']['time_zone']))
    else:
        return parser.parse('%sT%s' % (date, time))


def serial_date_to_ISO_format(serial):
    # Excel date&time format
    origin = datetime.datetime(1899, 12, 30, tzinfo=tz.tzutc())
    secs = (serial % 1.0) * 86400.0
    delta = datetime.timedelta(int(serial), secs, secs / 1E6)
    dt_utc = origin + delta
    dt_local = dt_utc.astimezone(tz.tzlocal())
    return dt_local.date().isoformat(), dt_local.time().isoformat(), dt_local.tzname()


def ISO_format_to_serial_date(date, time, timezone='UTC'):
    dt = parser.parse('%sT%s %s' % (date, time, timezone))
    return datetime_to_serial_date(dt)


def datetime_to_serial_date(dt):
    if dt.tzname() is None:
        dt = dt.replace(tzinfo=tz.tzutc())
    origin = datetime.datetime(1899, 12, 30, tzinfo=tz.tzutc())
    delta = dt - origin
    return float(delta.days) + (float(delta.seconds) / 86400.0)
