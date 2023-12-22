import inspect


def nargs_error(name, takes, given):
    """Generate a TypeError to be raised by function calls with wrong arity."""
    return TypeError(f"{name}() takes {takes} positional arguments but "
                     f"{given} were given")


def check_in_list(values, /, *, _print_supported_values=True, **kwargs):
    """
    For each *key, value* pair in *kwargs*, check that *value* is in *values*;
    if not, raise an appropriate ValueError.

    Parameters
    ----------
    values : iterable
        Sequence of values to check on.
    _print_supported_values : bool, default: True
        Whether to print *values* when raising ValueError.
    **kwargs : dict
        *key, value* pairs as keyword arguments to find in *values*.

    Raises
    ------
    ValueError
        If any *value* in *kwargs* is not found in *values*.

    Examples
    --------
    >>> _api.check_in_list(["foo", "bar"], arg=arg, other_arg=other_arg)
    """
    if not kwargs:
        raise TypeError("No argument to check!")
    for key, val in kwargs.items():
        if val not in values:
            msg = f"{val!r} is not a valid value for {key}"
            if _print_supported_values:
                msg += f"; supported values are {', '.join(map(repr, values))}"
            raise ValueError(msg)


def check_getitem(mapping, /, **kwargs):
    """
    *kwargs* must consist of a single *key, value* pair.  If *key* is in
    *mapping*, return ``mapping[value]``; else, raise an appropriate
    ValueError.

    Examples
    --------
    >>> _api.check_getitem({"foo": "bar"}, arg=arg)
    """
    if len(kwargs) != 1:
        raise ValueError("check_getitem takes a single keyword argument")
    (k, v), = kwargs.items()
    try:
        return mapping[v]
    except KeyError:
        raise ValueError(
            f"{v!r} is not a valid value for {k}; supported values are "
            f"{', '.join(map(repr, mapping))}") from None
