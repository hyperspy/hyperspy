import nose.tools
import hyperspy.events as he

class TriggeredCorrectly: pass

class TestEventsSuppression:
        
    def on_trigger(self, should_trigger):
        if should_trigger:
            raise TriggeredCorrectly()
        else:
            nose.tools.assert_true(False, "This event should not trigger")
            
    def trigger_check(self, trigger, should_trigger):
        if should_trigger:
            nose.tools.assert_raises(TriggeredCorrectly, trigger, True)
        else:
            trigger(False)
        
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
        
        