import inspect
from contextlib import contextmanager


class Events(object):

    """
    Events container.

    All available events are attributes of this class.

    """
    def __init__(self):
        self._events = {}

    @contextmanager
    def suppress(self):
        """
        Use this function with a 'with' statement to temporarily suppress
        all events in the container. When the 'with' lock completes, the old
        suppression values will be restored.

        Example usage
        -------------
        with obj.events.suppress:
            # Any events triggered by assignments are prevented:
            obj.val_a = a
            obj.val_b = b
        # Trigger one event instead:
        obj.events.values_changed.trigger()

        See also
        --------
        Event.suppress
        Event.suppress_callback
        """

        old = {}

        try:
            for e in self._events.itervalues():
                old[e] = e._suppress
                e._suppress = True
            yield
        finally:
            for e, oldval in old.iteritems():
                e._suppress = oldval

    def update(self, other, prefix=''):
        if prefix:
            other = [(prefix + '_' + k, e)
                     for (k, e) in other._events.iteritems()]
        self._events.update(other)

    def __setattr__(self, name, value):
        """
        Magic to enable having `Event`s as attributes, and keeping them
        separate from other attributes.

        If it's an `Event`, store it in self._events, otherwise set attribute
        in normal way.
        """
        if isinstance(value, Event):
            self._events[name] = value
        else:
            super(Events, self).__setattr__(name, value)

    def __getattr__(self, name):
        """
        Magic to enable having `Event`s as attributes, and keeping them
        separate from other attributes.

        Returns Event attribute `name` (__getattr__ is only called if attribute
        could not be found in the normal way).
        """
        return self._events[name]

    def __delattr__(self, name):
        """
        Magic to enable having `Event`s as attributes, and keeping them
        separate from other attributes.

        Deletes attribute from self._events if present, otherwise delete
        attribute in normal way.
        """
        if name in self._events:
            del self._events[name]
        else:
            super(Events, self).__delattr__(name)

    def __dir__(self):
        """
        Magic to enable having `Event`s as attributes, and keeping them
        separate from other attributes.

        Makes sure tab-completion works in IPython etc.
        """
        d = dir(type(self))
        d.extend(self.__dict__.iterkeys())
        d.extend(self._events.iterkeys())
        return sorted(set(d))

    def __repr__(self):
        text = "<hyperspy.events.Events: " + repr(self._events) + ">"
        return text.encode('utf8')


class Event(object):

    def __init__(self, doc=''):
        self.__doc__ = doc
        self._connected = {}
        self._suppress = False

    @contextmanager
    def suppress(self):
        """
        Use this function with a 'with' statement to temporarily suppress
        all events in the container. When the 'with' lock completes, the old
        suppression values will be restored.

        Example usage
        -------------
        with obj.events.myevent.suppress():
            # These would normally both trigger myevent:
            obj.val_a = a
            obj.val_b = b
        # Trigger manually once:
        obj.events.myevent.trigger()

        See also
        --------
        suppress_callback
        Events.suppress
        """
        old = self._suppress
        self._suppress = True
        try:
            yield
        finally:
            self._suppress = old

    @contextmanager
    def suppress_callback(self, function):
        """
        Use this function with a 'with' statement to temporarily suppress
        a single callback from being called. All other connected callbacks
        will trigger. When the 'with' lock completes, the old suppression value
        will be restored.

        Example usage
        -------------
        with obj.events.myevent.suppress_callback(f):
            # Events will trigger as normal, but `f` will not be called
            obj.val_a = a
            obj.val_b = b
        # Here, `f` will be called as before:
        obj.events.myevent.trigger()

        See also
        --------
        suppress
        Events.suppress
        """
        nargs = None
        found = False
        for nargs, c in self._connected.iteritems():
            for f in c:
                if f == function:
                    found = True
                    break
        if found:
            self.disconnect(function)
            try:
                yield
            finally:
                self.connect(function, nargs)
        else:
            yield   # Do nothing

    def connected(self, nargs=None):
        """
        Connected functions. The default behavior is to include all
        functions, but by using the 'nargs' argument, it can be filtered by
        function signature.
        """
        if nargs is None:
            ret = set()
            ret.update(*self._connected.values())
            return ret
        else:
            if nargs in self._connected:
                return self._connected[nargs]
            else:
                return set()

    def connect(self, function, nargs='all'):
        """
        Connects a function to the event.
        Arguments:
        ----------
        function : callable
            The function to call when the event triggers.
        nargs : int, 'all' (default), or 'auto'
            The number of arguments to supply to the function. If 'all', it
            will be called with all arguments passed to trigger(). If 'auto'
            inspect.getargspec() will be used to determine the number of
            arguments the function accepts (arguments with default values will
            be included in the count).

        See also
        --------
        disconnect
        """
        if not callable(function):
            raise TypeError("Only callables can be registered")
        if nargs == 'auto':
            spec = inspect.getargspec(function)[0]
            if spec is None:
                nargs = 0
            else:
                nargs = len(spec)
        elif nargs is None:
            nargs = 0
        if nargs not in self._connected:
            self._connected[nargs] = set()
        self._connected[nargs].add(function)

    def disconnect(self, function):
        """
        Disconnects a function from the event. The passed function will be
        disconnected irregardless of which 'nargs' argument was passed to
        connect().

        If you only need to temporarily prevent a function from being called,
        single callback suppression is supported by the `suppress_callback`
        context manager.

        See also
        --------
        connect
        suppress_callback
        """
        for c in self._connected.itervalues():
            if function in c:
                c.remove(function)

    @staticmethod
    def _trigger_nargs(f, args, nargs):
        """
        Basic trigger resolution.
        """
        return f(*args[0:nargs])

    def trigger(self, *args, **kwargs):
        """
        Triggers the event. If the event is suppressed, this does nothing.
        Otherwise it calls all the connected functions with the arguments as
        specified when connected.

        See also
        --------
        suppress
        suppress_callback
        Events.suppress
        """
        if not self._suppress:
            # Loop on copy to deal with callbacks which change connections
            for nargs, c in self._connected.copy().iteritems():
                if nargs is 'all':
                    for f in c:
                        f(*args, **kwargs)
                else:
                    if len(args) < nargs:
                        raise ValueError(
                            ("Tried to call %s which require %d args " +
                             "with only %d.") % (str(c), nargs, len(args)))
                    for f in c.copy():
                        self._trigger_nargs(f, args, nargs)

    def __deepcopy__(self, memo):
        dc = type(self)()
        memo[id(self)] = dc
        return dc

    def __repr__(self):
        text = "<hyperspy.events.Event: " + repr(self._connected) + ">"
        return text.encode('utf8')
