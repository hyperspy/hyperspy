
class MountainsMapFileError(Exception):

    def __init__(self, msg = "Corrupt Mountainsmap file"):
        self.error =  msg

    def __str__(self):
        return repr(self.error)

class ByteOrderError(Exception):

    def __init__(self, order=''):
        self.byte_order = order

    def __str__(self):
        return repr(self.byte_order)


class DM3FileVersionError(Exception):

    def __init__(self, value=''):
        self.dm3_version = value

    def __str__(self):
        return repr(self.dm3_version)


class DM3TagError(Exception):

    def __init__(self, value=''):
        self.dm3_tag = value

    def __str__(self):
        return repr(self.dm3_tag)


class DM3DataTypeError(Exception):

    def __init__(self, value=''):
        self.dm3_dtype = value

    def __str__(self):
        return repr(self.dm3_dtype)


class DM3TagTypeError(Exception):

    def __init__(self, value=''):
        self.dm3_tagtype = value

    def __str__(self):
        return repr(self.dm3_tagtype)


class DM3TagIDError(Exception):

    def __init__(self, value=''):
        self.dm3_tagID = value

    def __str__(self):
        return repr(self.dm3_tagID)

class VisibleDeprecationWarning(UserWarning):

    """Visible deprecation warning.
    By default, python will not show deprecation warnings, so this class
    provides a visible one.

    """
    pass

class LazyCupyConversion(Exception):

    def __init__(self):
        self.error = (
            "Automatically converting data to cupy array is not supported "
            "for lazy signals. Read the corresponding section in the user "
            "guide for more information on how to use GPU with lazy signals."
            )

    def __str__(self):
        return repr(self.error)
