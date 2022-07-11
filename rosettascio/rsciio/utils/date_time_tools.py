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

import datetime
from dateutil import tz, parser
import logging

import numpy as np

_logger = logging.getLogger(__name__)



def serial_date_to_ISO_format(serial):
    """
    Convert serial_date to a tuple of string (date, time, time_zone) in ISO
    format. By default, the serial date is converted in local time zone.
    """
    dt_utc = serial_date_to_datetime(serial)
    dt_local = dt_utc.astimezone(tz.tzlocal())
    return dt_local.date().isoformat(), dt_local.time().isoformat(), dt_local.tzname()


def ISO_format_to_serial_date(date, time, timezone='UTC'):
    """ Convert ISO format to a serial date. """
    if timezone is None or timezone == 'Coordinated Universal Time':
        timezone = 'UTC'
    dt = parser.parse(
        '%sT%s' %
        (date, time)).replace(
        tzinfo=tz.gettz(timezone))
    return datetime_to_serial_date(dt)


def datetime_to_serial_date(dt):
    """ Convert datetime.datetime object to a serial date. """
    if dt.tzname() is None:
        dt = dt.replace(tzinfo=tz.tzutc())
    origin = datetime.datetime(1899, 12, 30, tzinfo=tz.tzutc())
    delta = dt - origin
    return float(delta.days) + (float(delta.seconds) / 86400.0)


def serial_date_to_datetime(serial):
    """ Convert serial date to a datetime.datetime object. """
    # Excel date&time format
    origin = datetime.datetime(1899, 12, 30, tzinfo=tz.tzutc())
    secs = (serial % 1.0) * 86400
    delta = datetime.timedelta(int(serial), secs)
    return origin + delta


def get_date_time_from_metadata(metadata, formatting='ISO'):
    """
    Get the date and time from a metadata tree.

    Parameters
    ----------
    metadata : metadata dict
    formatting : string, ('ISO', 'datetime', 'datetime64')
        Default: 'ISO'. This parameter set the formatting of the date,
        and the time, it can be ISO 8601 string, datetime.datetime
        or a numpy.datetime64 object. In the later case, the time zone
        is not supported.

    Returns
    -------
        string, datetime.datetime or numpy.datetime64 object

    """
    md_gen = metadata['General']
    date, time = md_gen.get('date'), md_gen.get('time')
    if date and time:
        dt = parser.parse(f'{date}T{time}')
        time_zone = md_gen.get('time_zone')
        if time_zone:
            dt = dt.replace(tzinfo=tz.gettz(time_zone))
            if dt.tzinfo is None:
                # time_zone metadata must be offset string
                dt = parser.parse(f'{date}T{time}{time_zone}')

    elif not date and time:
        dt = parser.parse(f'{time}').time()
    elif date and not time:
        dt = parser.parse(f'{date}').date()
    else:
        return

    if formatting == 'ISO':
        res = dt.isoformat()
    if formatting == 'datetime':
        res = dt
    # numpy.datetime64 doesn't support time zone
    if formatting == 'datetime64':
        res = np.datetime64(f'{date}T{time}')

    return res
