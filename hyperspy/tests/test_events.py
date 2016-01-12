import nose.tools
import hyperspy.events as he

class TriggeredCorrectly: pass

class EventsBase():
    def on_trigger(self, *args, **kwargs):
        self.triggered = True

    def trigger_check(self, trigger, should_trigger, *args):
        self.triggered = False
        trigger(*args)
        nose.tools.assert_equal(self.triggered, should_trigger)

class TestEventsSuppression(EventsBase):

    def setUp(self):
        self.events = he.Events()
        
        self.events.a = he.Event()
        self.events.b = he.Event()
        self.events.c = he.Event()
        
        self.events.a.connect(self.on_trigger)
        self.events.b.connect(self.on_trigger)
        self.events.c.connect(self.on_trigger)

    def test_simple_suppression(self):
        self.events.a.suppress = True
        self.trigger_check(self.events.a.trigger, False)
        self.trigger_check(self.events.b.trigger, True)
        self.events.a.suppress = False
        
        with self.events.suppress:
            self.trigger_check(self.events.a.trigger, False)
            self.trigger_check(self.events.b.trigger, False)
            self.trigger_check(self.events.c.trigger, False)
        
        self.trigger_check(self.events.a.trigger, True)
        self.trigger_check(self.events.b.trigger, True)
        self.trigger_check(self.events.c.trigger, True)

    def test_suppression_restore(self):
        self.events.a.suppress = True
        self.events.b.suppress = False
        self.events.c.suppress = False
        
        with self.events.suppress:
            self.trigger_check(self.events.a.trigger, False)
            self.trigger_check(self.events.b.trigger, False)
            self.trigger_check(self.events.c.trigger, False)
            
        self.trigger_check(self.events.a.trigger, False)
        self.trigger_check(self.events.b.trigger, True)
        self.trigger_check(self.events.c.trigger, True)

    def test_suppresion_nesting(self):
        self.events.a.suppress = True
        self.events.b.suppress = False
        self.events.c.suppress = False
        
        with self.events.suppress:
            self.events.c.suppress = False
            self.trigger_check(self.events.a.trigger, False)
            self.trigger_check(self.events.b.trigger, False)
            self.trigger_check(self.events.c.trigger, True)
            
            with self.events.suppress:
                self.trigger_check(self.events.a.trigger, False)
                self.trigger_check(self.events.b.trigger, False)
                self.trigger_check(self.events.c.trigger, False)
                
            self.trigger_check(self.events.a.trigger, False)
            self.trigger_check(self.events.b.trigger, False)
            self.trigger_check(self.events.c.trigger, True)
            
        self.trigger_check(self.events.a.trigger, False)
        self.trigger_check(self.events.b.trigger, True)
        self.trigger_check(self.events.c.trigger, True)


class TestEventsSignatures(EventsBase):
    
    def setUp(self):
        self.events = he.Events()
        
        self.events.a = he.Event()
        
        
    def test_basic_triggers(self):
        self.events.a.connect(lambda *args, **kwargs: 0)
        self.events.a.connect(lambda: 0, None)
        self.events.a.connect(lambda x: 0, 1)
        self.events.a.connect(lambda x, y: 0, 2)
        self.events.a.connect(lambda x, y=988: \
                                 nose.tools.assert_equal(y, 988), 1)
        self.events.a.connect(lambda x, y=988: \
                                 nose.tools.assert_not_equal(y, 988), 2)
        self.events.a.trigger(2, 5)
        
        nose.tools.assert_raises(ValueError, self.events.a.trigger)
        nose.tools.assert_raises(ValueError, self.events.a.trigger, 2)
        self.events.a.trigger(2,5,8)