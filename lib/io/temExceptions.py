#!/usr/bin/env python
# -*- coding: latin-1 -*-

# custom exceptions
#
# Copyright (c) 2010 Stefano Mazzucco.
# All rights reserved.
#
# This program is still at an early stage to be released, so the use of this
# code must be explicitly authorized by its author and cannot be shared for any reason.
#
# Once the program will be mature, it will be released under a GNU GPL license

class ByteOrderError(Exception):
    def __init__(self, order=''):
	self.byte_order = order

    def __str__(self):
	return repr(self.byte_order)
