import copy

import nose.tools as nt

import hyperspy.events as he


class EventsBase():

    def on_trigger(self, **kwargs):
        self.triggered = True

    def on_trigger2(self, **kwargs):
        self.triggered2 = True

    def trigger_check(self, trigger, should_trigger, **kwargs):
        self.triggered = False
        trigger(**kwargs)
        nt.assert_equal(self.triggered, should_trigger)

    def trigger_check2(self, trigger, should_trigger, **kwargs):
        self.triggered2 = False
        trigger(**kwargs)
        nt.assert_equal(self.triggered2, should_trigger)


class TestEventsSuppression(EventsBase):

    def setUp(self):
        self.events = he.Events()

        self.events.a = he.Event()
        self.events.b = he.Event()
        self.events.c = he.Event()

        self.events.a.connect(self.on_trigger)
        self.events.a.connect(self.on_trigger2)
        self.events.b.connect(self.on_trigger)
        self.events.c.connect(self.on_trigger)

    def test_simple_suppression(self):
        with self.events.a.suppress():
            self.trigger_check(self.events.a.trigger, False)
            self.trigger_check(self.events.b.trigger, True)

        with self.events.suppress():
            self.trigger_check(self.events.a.trigger, False)
            self.trigger_check(self.events.b.trigger, False)
            self.trigger_check(self.events.c.trigger, False)

        self.trigger_check(self.events.a.trigger, True)
        self.trigger_check(self.events.b.trigger, True)
        self.trigger_check(self.events.c.trigger, True)

    def test_suppression_restore(self):
        with self.events.a.suppress():
            with self.events.suppress():
                self.trigger_check(self.events.a.trigger, False)
                self.trigger_check(self.events.b.trigger, False)
                self.trigger_check(self.events.c.trigger, False)

            self.trigger_check(self.events.a.trigger, False)
            self.trigger_check(self.events.b.trigger, True)
            self.trigger_check(self.events.c.trigger, True)

    def test_suppresion_nesting(self):
        with self.events.a.suppress():
            with self.events.suppress():
                self.events.c._suppress = False
                self.trigger_check(self.events.a.trigger, False)
                self.trigger_check(self.events.b.trigger, False)
                self.trigger_check(self.events.c.trigger, True)

                with self.events.suppress():
                    self.trigger_check(self.events.a.trigger, False)
                    self.trigger_check(self.events.b.trigger, False)
                    self.trigger_check(self.events.c.trigger, False)

                self.trigger_check(self.events.a.trigger, False)
                self.trigger_check(self.events.b.trigger, False)
                self.trigger_check(self.events.c.trigger, True)

            self.trigger_check(self.events.a.trigger, False)
            self.trigger_check(self.events.b.trigger, True)
            self.trigger_check(self.events.c.trigger, True)

    def test_suppression_single(self):
        with self.events.b.suppress():
            with self.events.a.suppress_callback(self.on_trigger):
                self.trigger_check(self.events.a.trigger, False)
                self.trigger_check2(self.events.a.trigger, True)
                self.trigger_check(self.events.b.trigger, False)
                self.trigger_check(self.events.c.trigger, True)

            self.trigger_check(self.events.a.trigger, True)
            self.trigger_check2(self.events.a.trigger, True)
            self.trigger_check(self.events.b.trigger, False)
            self.trigger_check(self.events.c.trigger, True)

        # Reverse order:
        with self.events.a.suppress_callback(self.on_trigger):
            with self.events.b.suppress():
                self.trigger_check(self.events.a.trigger, False)
                self.trigger_check2(self.events.a.trigger, True)
                self.trigger_check(self.events.b.trigger, False)
                self.trigger_check(self.events.c.trigger, True)

            self.trigger_check(self.events.a.trigger, False)
            self.trigger_check2(self.events.a.trigger, True)
            self.trigger_check(self.events.b.trigger, True)
            self.trigger_check(self.events.c.trigger, True)

    @nt.raises(ValueError)
    def test_exception_event(self):
        try:
            with self.events.a.suppress():
                self.trigger_check(self.events.a.trigger, False)
                self.trigger_check(self.events.b.trigger, True)
                self.trigger_check(self.events.c.trigger, True)
                raise ValueError()
        finally:
            self.trigger_check(self.events.a.trigger, True)
            self.trigger_check(self.events.b.trigger, True)
            self.trigger_check(self.events.c.trigger, True)

    @nt.raises(ValueError)
    def test_exception_events(self):
        try:
            with self.events.suppress():
                self.trigger_check(self.events.a.trigger, False)
                self.trigger_check(self.events.b.trigger, False)
                self.trigger_check(self.events.c.trigger, False)
                raise ValueError()
        finally:
            self.trigger_check(self.events.a.trigger, True)
            self.trigger_check(self.events.b.trigger, True)
            self.trigger_check(self.events.c.trigger, True)

    @nt.raises(ValueError)
    def test_exception_single(self):
        try:
            with self.events.a.suppress_callback(self.on_trigger):
                self.trigger_check(self.events.a.trigger, False)
                self.trigger_check2(self.events.a.trigger, True)
                self.trigger_check(self.events.b.trigger, True)
                self.trigger_check(self.events.c.trigger, True)
                raise ValueError()
        finally:
            self.trigger_check(self.events.a.trigger, True)
            self.trigger_check2(self.events.a.trigger, True)
            self.trigger_check(self.events.b.trigger, True)
            self.trigger_check(self.events.c.trigger, True)

    @nt.raises(ValueError)
    def test_exception_nested(self):
        try:
            with self.events.a.suppress_callback(self.on_trigger):
                try:
                    with self.events.a.suppress():
                        try:
                            with self.events.suppress():
                                self.trigger_check(self.events.a.trigger,
                                                   False)
                                self.trigger_check2(self.events.a.trigger,
                                                    False)
                                self.trigger_check(self.events.b.trigger,
                                                   False)
                                self.trigger_check(self.events.c.trigger,
                                                   False)
                                raise ValueError()
                        finally:
                            self.trigger_check(self.events.a.trigger, False)
                            self.trigger_check2(self.events.a.trigger, False)
                            self.trigger_check(self.events.b.trigger, True)
                            self.trigger_check(self.events.c.trigger, True)
                finally:
                    self.trigger_check(self.events.a.trigger, False)
                    self.trigger_check2(self.events.a.trigger, True)
                    self.trigger_check(self.events.b.trigger, True)
                    self.trigger_check(self.events.c.trigger, True)
        finally:
            self.trigger_check(self.events.a.trigger, True)
            self.trigger_check2(self.events.a.trigger, True)
            self.trigger_check(self.events.b.trigger, True)
            self.trigger_check(self.events.c.trigger, True)

    def test_suppress_wrong(self):
        with self.events.a.suppress_callback(f_a):
            self.trigger_check(self.events.a.trigger, True)
            self.trigger_check2(self.events.a.trigger, True)

    def test_suppressor_init_args(self):
        with self.events.b.suppress():
            es = he.EventSupressor((self.events.a, self.on_trigger),
                                   self.events.c)
            with es.suppress():
                self.trigger_check(self.events.a.trigger, False)
                self.trigger_check2(self.events.a.trigger, True)
                self.trigger_check(self.events.b.trigger, False)
                self.trigger_check(self.events.c.trigger, False)
                with self.events.a.suppress_callback(self.on_trigger2):
                    self.trigger_check2(self.events.a.trigger, False)
                self.trigger_check2(self.events.a.trigger, True)

            self.trigger_check(self.events.a.trigger, True)
            self.trigger_check2(self.events.a.trigger, True)
            self.trigger_check(self.events.b.trigger, False)
            self.trigger_check(self.events.c.trigger, True)

        self.trigger_check(self.events.a.trigger, True)
        self.trigger_check2(self.events.a.trigger, True)
        self.trigger_check(self.events.b.trigger, True)
        self.trigger_check(self.events.c.trigger, True)

    def test_suppressor_add_args(self):
        with self.events.b.suppress():
            es = he.EventSupressor()
            es.add((self.events.a, self.on_trigger), self.events.c)
            with es.suppress():
                self.trigger_check(self.events.a.trigger, False)
                self.trigger_check2(self.events.a.trigger, True)
                self.trigger_check(self.events.b.trigger, False)
                self.trigger_check(self.events.c.trigger, False)
                with self.events.a.suppress_callback(self.on_trigger2):
                    self.trigger_check2(self.events.a.trigger, False)
                self.trigger_check2(self.events.a.trigger, True)

            self.trigger_check(self.events.a.trigger, True)
            self.trigger_check2(self.events.a.trigger, True)
            self.trigger_check(self.events.b.trigger, False)
            self.trigger_check(self.events.c.trigger, True)

        self.trigger_check(self.events.a.trigger, True)
        self.trigger_check2(self.events.a.trigger, True)
        self.trigger_check(self.events.b.trigger, True)
        self.trigger_check(self.events.c.trigger, True)

    def test_suppressor_all_callback_in_events(self):
        with self.events.b.suppress():
            es = he.EventSupressor()
            es.add((self.events, self.on_trigger),)
            with es.suppress():
                self.trigger_check(self.events.a.trigger, False)
                self.trigger_check2(self.events.a.trigger, True)
                self.trigger_check(self.events.b.trigger, False)
                self.trigger_check(self.events.c.trigger, False)
                with self.events.a.suppress_callback(self.on_trigger2):
                    self.trigger_check2(self.events.a.trigger, False)
                self.trigger_check2(self.events.a.trigger, True)

            self.trigger_check(self.events.a.trigger, True)
            self.trigger_check2(self.events.a.trigger, True)
            self.trigger_check(self.events.b.trigger, False)
            self.trigger_check(self.events.c.trigger, True)

        self.trigger_check(self.events.a.trigger, True)
        self.trigger_check2(self.events.a.trigger, True)
        self.trigger_check(self.events.b.trigger, True)
        self.trigger_check(self.events.c.trigger, True)


def f_a(**kwargs): pass


def f_b(**kwargs): pass


def f_c(**kwargs): pass


def f_d(a, b, c): pass


class TestEventsSignatures(EventsBase):

    def setUp(self):
        self.events = he.Events()
        self.events.a = he.Event()

    def test_trigger_kwarg_validity(self):
        self.events.a.connect(lambda **kwargs: 0)
        self.events.a.connect(lambda: 0, [])
        self.events.a.connect(lambda one: 0, ["one"])
        self.events.a.connect(lambda one, two: 0, ["one", "two"])
        self.events.a.connect(lambda one, two=988:
                              nt.assert_equal(two, 988), ["one"])
        self.events.a.connect(lambda one, two=988:
                              nt.assert_not_equal(two, 988), ["one", "two"])
        self.events.a.connect(lambda A, B=988:
                              nt.assert_not_equal(A, 988),
                              {"one": "A", "two": "B"})
        self.events.a.trigger(one=2, two=5)
        self.events.a.trigger(one=2, two=5, three=8)
        self.events.a.connect(lambda one, two: 0, )
        nt.assert_raises(TypeError, self.events.a.trigger, three=None)
        nt.assert_raises(TypeError, self.events.a.trigger, one=2)

    def test_connected_and_disconnect(self):
        self.events.a.connect(f_a)
        self.events.a.connect(f_b, ["A", "B"])
        self.events.a.connect(f_c, {"a": "A", "b": "B"})
        self.events.a.connect(f_d, 'auto')
        nt.assert_equal(self.events.a.connected, set([f_a, f_b, f_c, f_d]))
        self.events.a.disconnect(f_a)
        self.events.a.disconnect(f_b)
        self.events.a.disconnect(f_c)
        self.events.a.disconnect(f_d)
        nt.assert_equal(self.events.a.connected, set([]))

    @nt.raises(TypeError)
    def test_type(self):
        self.events.a.connect('f_a')


def test_events_container_magic_attributes():
    events = he.Events()
    event = he.Event()
    events.event = event
    events.a = 3
    nt.assert_in("event", events.__dir__())
    nt.assert_in("a", events.__dir__())
    nt.assert_equal(repr(events),
                    "<hyperspy.events.Events: "
                    "{'event': <hyperspy.events.Event: set()>}>")
    del events.event
    del events.a
    nt.assert_not_in("event", events.__dir__())
    nt.assert_not_in("a", events.__dir__())


class TestTriggerArgResolution(EventsBase):

    def setup(self):
        self.events = he.Events()
        self.events.a = he.Event(arguments=['A', 'B'])
        self.events.b = he.Event(arguments=['A', 'B', ('C', "vC")])
        self.events.c = he.Event()

    @nt.raises(SyntaxError)
    def test_wrong_default_order(self):
        self.events.d = he.Event(arguments=['A', ('C', "vC"), "B"])

    @nt.raises(ValueError)
    def test_wrong_kwarg_name(self):
        self.events.d = he.Event(arguments=['A', "B+"])

    def test_arguments(self):
        nt.assert_equal(self.events.a.arguments, ("A", "B"))
        nt.assert_equal(self.events.b.arguments, ("A", "B", ("C", "vC")))
        nt.assert_equal(self.events.c.arguments, None)

    def test_some_kwargs_resolution(self):
        self.events.a.connect(lambda x=None: nt.assert_equal(x, None), [])
        self.events.a.connect(lambda A: nt.assert_equal(A, 'vA'), ["A"])
        self.events.a.connect(lambda A, B: nt.assert_equal((A, B),
                                                           ('vA', 'vB')),
                              ["A", "B"])
        self.events.a.connect(lambda A, B:
                              nt.assert_equal((A, B), ('vA', 'vB')), "auto")
        nt.assert_raises(NotImplementedError,
                         self.events.a.connect, function=lambda *args: 0,
                         kwargs="auto")

        self.events.a.connect(lambda **kwargs:
                              nt.assert_equal((kwargs["A"], kwargs["B"]),
                                              ('vA', 'vB')), "auto")
        self.events.a.connect(lambda A, B=None, C=None:
                              nt.assert_equal((A, B, C),
                                              ('vA', 'vB', None)), ["A", "B"])
        # Test default argument
        self.events.b.connect(lambda A, B=None, C=None:
                              nt.assert_equal((A, B, C),
                                              ('vA', 'vB', "vC")))
        self.events.a.trigger(A='vA', B='vB')
        self.events.b.trigger(A='vA', B='vB')
        nt.assert_raises(TypeError, self.events.a.trigger, A='vA', B='vB',
                         C='vC')
        self.events.a.trigger(A='vA', B='vB')
        self.events.a.trigger(B='vB', A='vA')
        nt.assert_raises(TypeError, self.events.a.trigger,
                         A='vA', C='vC', B='vB', D='vD')

    @nt.raises(ValueError)
    def test_not_connected(self):
        self.events.a.disconnect(lambda: 0)

    @nt.raises(ValueError)
    def test_already_connected(self):
        def f(): pass
        self.events.a.connect(f)
        self.events.a.connect(f)

    def test_deepcopy(self):
        def f(): pass
        self.events.a.connect(f)
        nt.assert_not_in(f, copy.deepcopy(self.events.a).connected)

    def test_all_kwargs_resolution(self):
        self.events.a.connect(lambda A, B:
                              nt.assert_equal((A, B), ('vA', 'vB')), )
        self.events.a.connect(lambda x=None, y=None, A=None, B=None:
                              nt.assert_equal((x, y, A, B),
                                              (None, None, 'vA', 'vB')))

        self.events.a.trigger(A='vA', B='vB')
