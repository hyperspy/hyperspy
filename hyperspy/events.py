import sys
import inspect


class EventSuppressionContext(object):

    """
    Context manager for event suppression. When passed an Events class,
    it will suppress all the events in that container when activated by
    using it in a 'with' statement. The previous suppression state will be
    restored when the 'with' block completes.
    """

    def __init__(self, events):
        self.events = events
        self.old = {}

    def __enter__(self):
        self.old = {}
        try:
            for e in self.events.__dict__.itervalues():
                self.old[e] = e.suppress
                e.suppress = True
        except e:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, type, value, tb):
        for e, oldval in self.old.iteritems():
            e.suppress = oldval
        # Never suppress events


class Events(object):

    """
    Events container.

    All available events are attributes of this class.

    """

    @property
    def suppress(self):
        """
        Use this property with a 'with' statement to temporarily suppress all
        events in the container. When the 'with' vlock completes, the old
        suppression values will be restored.

        Example usage pattern:
        with obj.events.suppress:
            obj.val_a = a
            obj.val_b = b
        obj.events.values_changed.trigger()
        """
        return EventSuppressionContext(self)


class Event(object):

    def __init__(self):
        self._connected = {0: set()}
        self.suppress = False

    def connected(self, nargs='all'):
        if nargs == 'all':
            ret = set()
            ret.update(*[v for v in self._connected.itervalues()])
            return ret
        else:
            if nargs in self._connected:
                return self._connected[nargs]
            else:
                return set()

    def connect(self, function, nargs='all'):
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
        for c in self._connected.itervalues():
            if function in c:
                c.remove(function)

    def trigger(self, *args, **kwargs):
        if not self.suppress:
            for nargs, c in self._connected.iteritems():
                if nargs is 'all':
                    for f in c:
                        f(*args, **kwargs)
                else:
                    if len(args) < nargs:
                        raise ValueError(
                            ("Tried to call %s which require %d args " +
                             "with only %d.") % (str(c), nargs, len(args)))
                    for f in c:
                        f(*args[0:nargs])

    def __deepcopy__(self, memo):
        dc = type(self)()
        memo[id(self)] = dc
        return dc
