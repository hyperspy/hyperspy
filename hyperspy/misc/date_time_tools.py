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
import datetime
from dateutil import tz, parser
import logging

_logger = logging.getLogger(__name__)


def metadata_to_datetime(metadata):
    date = metadata['General']['date']
    time = metadata['General']['time']
    if 'time_zone' in metadata['General']:
        return parser.parse('%sT%s %s' % (date, time, metadata['General']['time_zone']))
    else:
        return parser.parse('%sT%s' % (date, time))
        

def get_date_time_from_metadata(metadata, formatting='ISO'):
    """ Get the date and time from a metadata tree.
        
        Parameters
        ----------
            metadata : metadata object
            formatting : string, ('ISO', 'datetime', 'datetime64')
                Default: 'ISO'. This parameter set the formatting of the date,
                and the time, it can be ISO 8601 string, datetime.datetime  
                or a numpy.datetime64 object. In the later case, the time zone
                is not supported.

        Return
        ----------
            string, datetime.datetime or numpy.datetime64 object
                
        Example
        -------
        >>> s = hs.load("example1.msa")
        >>> s.metadata
            ├── General
            │   ├── date = 1991-10-01
            │   ├── original_filename = example1.msa
            │   ├── time = 12:00:00
            │   └── title = NIO EELS OK SHELL
        
        >>> s = get_date_time_from_metadata(s.metadata)
        '1991-10-01T12:00:00'
        >>> s = get_date_time_from_metadata(s.metadata, format='ISO')
        '1991-10-01T12:00:00'
        >>> s = get_date_time_from_metadata(s.metadata, format='datetime')
        
        >>> s = get_date_time_from_metadata(s.metadata, format='datetime64')
        
    """
    date = metadata['General']['date']
    time = metadata['General']['time']

    if 'time_zone' in metadata['General']:
        dt = parser.parse('%sT%s%s' % (date, time, metadata['General']['time_zone']))
    else:
        dt = parser.parse('%sT%s' % (date, time))
        
    if formatting == 'ISO':
        res = dt.isoformat()
    if formatting == 'datetime':
        res = dt
    # numpy.datetime64 doesn't support time zone
    if formatting == 'datetime64':
        res = np.datetime64('%sT%s' % (date, time))
        
    return res

        
def update_date_time_in_metadata(dt, metadata):
    """
    TODO: documentation
    """
    time_zone = None
    if isinstance(dt, str):
        split = datetime.split('T')
        date = split[0]
        time = split[1]
        if len(split) == 2:
            time_zone = datetime.split('T')[2]
    if isinstance(dt, datetime.datetime):
        date = dt.date().isoformat()
        time = dt.time().isoformat()
        if dt.tzname() is not None:
            time_zone = dt.tzname()
            metadata.set_item('General.time_zone', time_zone)
    
    metadata.set_item('General.date', date)
    metadata.set_item('General.time', time)
    if metadata.has_item('General.time_zone') and not time_zone:
        pass
#        TODO: remove time_zone when necessary
    return metadata

        
def serial_date_to_ISO_format(serial):
    # Excel date&time format
    origin = datetime.datetime(1899, 12, 30, tzinfo=tz.tzutc())
    secs = (serial % 1.0) * 86400.0
    delta = datetime.timedelta(int(serial), secs, secs / 1E6)
    dt_utc = origin + delta
    dt_local = dt_utc.astimezone(tz.tzlocal())
    return dt_local.date().isoformat(), dt_local.time().isoformat(), dt_local.tzname()


def ISO_format_to_serial_date(date, time, timezone='UTC'):
    dt = parser.parse('%sT%s%s' % (date, time, timezone))
    return datetime_to_serial_date(dt)


def datetime_to_serial_date(dt):
    if dt.tzname() is None:
        dt = dt.replace(tzinfo=tz.tzutc())
    origin = datetime.datetime(1899, 12, 30, tzinfo=tz.tzutc())
    delta = dt - origin
    return float(delta.days) + (float(delta.seconds) / 86400.0)
