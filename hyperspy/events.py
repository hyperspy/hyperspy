import sys, copy

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
        self._connected = {0: set()} # Shared
        self._suppress = [False]    # Keep in list so shallow copies share
        self._nargs = 0
        self._signatures = {tuple([]): self}    # Shared
    
    @property
    def connected(self):
        return self._connected[self._nargs]
    
    @connected.setter
    def connected(self, value):
        self._connected[self._nargs] = value
    
    @property
    def suppress(self):
        return self._suppress[0]
    
    @suppress.setter
    def suppress(self, value):
        self._suppress[0] = value

    def connect(self, function):
        if not callable(function):
            raise TypeError("Only callables can be registered")
        self.connected.add(function)

    def disconnect(self, function):
        for i in xrange(len(self._connected)):
            if function in self._connected[i]:
                self._connected[i].remove(function)
    
    def _do_trigger(self, *args):
        if len(args) < self._nargs:
            raise ValueError(("Tried to call %s which require %d args " + \
                "with only %d.") % (str(self.connected), self._nargs, len(args)))
        for f in self.connected:
            f(*args[0:self._nargs])

    def trigger(self, *args):
        if not self.suppress:
            self._do_trigger(*args)
            for s in self._signatures.values():
                s._do_trigger(*args)

    def __deepcopy__(self, memo):
        dc = type(self)()
        memo[id(self)] = dc
        return dc
    
    def __getitem__(self, nargs):
        if nargs is None:
            nargs = 0
        if nargs in self._signatures:
            return self._signatures[nargs]
        else:
            r = copy.copy(self)
            r._nargs = nargs
            r._connected[nargs] = set()
            self._signatures[nargs] = r
            return r
            
