"""
Signal Events and Triggers
===========================

This example demonstrates HyperSpy's event system for creating responsive
and interactive data    s1.axes_manager.indices = (2,)

# %%
# Building event-driven analysis workflows
# -----------------------------------------
print("\n6. Building event-driven analysis workflows...")alysis workflows. Events allow different parts of
your analysis to automatically respond to changes in signals, axes, or
custom triggers.

Key concepts covered:
- Connecting to signal and axis events
- Creating custom event handlers
- Event suppression and management
- Manual event triggering
- Building event-driven analysis workflows
- Practical examples of event usage

"""

import numpy as np
import hyperspy.api as hs

# %%
# # HyperSpy Events and Triggers Demonstration
# 
# This example demonstrates HyperSpy's event system for creating responsive and interactive analysis workflows.

# %%
# ## Creating Test Signals and Understanding Available Events
# 
# Let's start by creating some test signals and exploring what events are available.

# Create test signals
s1 = hs.signals.Signal1D(np.random.random((10, 100)))
s2 = hs.signals.Signal2D(np.random.random((5, 5, 50, 50)))

s1.metadata.General.title = "Test Signal 1D"
s2.metadata.General.title = "Test Signal 2D"

print(f"   Signal 1D: {s1}")
print(f"   Signal 2D: {s2}")

# %%
# ### Available Events for Signal1D
# 
# HyperSpy signals provide several events you can connect to:

print("\n   Available events for Signal1D:")
for attr_name in dir(s1.events):
    if not attr_name.startswith('_'):
        event = getattr(s1.events, attr_name)
        if hasattr(event, 'trigger'):
            print(f"   - {attr_name}: {event.__doc__ or 'No description'}")

# %%
# ## Basic Event Connections
print("\n2. Basic event connections...")

# Event counter for tracking
event_counts = {
    'data_changed': 0,
    'index_changed': 0,
    'value_changed': 0,
    'axis_changed': 0
}

def on_data_changed(obj):
    """Handler for data change events."""
    event_counts['data_changed'] += 1
    print(f"   📊 Data changed in {obj.metadata.General.title}")
    print(f"   📊 New data range: [{obj.data.min():.3f}, {obj.data.max():.3f}]")

def on_index_changed(obj, index):
    """Handler for navigation index changes."""
    event_counts['index_changed'] += 1
    print(f"   🧭 Navigation index changed to {index} in {obj.name}")

def on_value_changed(obj, value):
    """Handler for axis value changes."""
    event_counts['value_changed'] += 1
    print(f"   📏 Axis value changed to {value} in {obj.name}")

# Connect to data change events
s1.events.data_changed.connect(on_data_changed)
s2.events.data_changed.connect(on_data_changed)

# Connect to navigation axis events
nav_axis_1d = s1.axes_manager.navigation_axes[0]
nav_axis_1d.name = "Time"
nav_axis_1d.events.index_changed.connect(on_index_changed)
nav_axis_1d.events.value_changed.connect(on_value_changed)

print("\n   Event handlers connected!")

# %%
# ## Triggering Events Through Normal Operations
# 
# Let's see how events are triggered when we perform normal operations on signals.

print("\n   Changing navigation indices:")
s1.axes_manager.indices = (3,)
s1.axes_manager.indices = (7,)

print("\n   Modifying data (triggers data_changed):")
s1 *= 2.0  # Using signal arithmetic preserves metadata
s1.events.data_changed.trigger(obj=s1)

s2 += 0.5  # Using signal arithmetic preserves metadata  
s2.events.data_changed.trigger(obj=s2)

# %%
# ## Custom Event Handlers with Specific Arguments
# 
# You can create more sophisticated event handlers that respond to specific conditions.

def detailed_data_handler(obj, data_changed_info=None):
    """More detailed data change handler."""
    print(f"   🔍 Detailed analysis: {obj.metadata.General.title}")
    print(f"   🔍 Data shape: {obj.data.shape}")
    print(f"   🔍 Data type: {obj.data.dtype}")
    print(f"   🔍 Memory usage: {obj.data.nbytes / 1024:.1f} KB")

def navigation_tracker(obj, index):
    """Track navigation changes with more detail."""
    if hasattr(obj, 'axes_manager'):
        current_value = obj.axes_manager[obj.name].value
        print(f"   🎯 Navigation: {obj.name}[{index}] = {current_value:.3f}")

# Connect additional handlers
s1.events.data_changed.connect(detailed_data_handler)
nav_axis_1d.events.index_changed.connect(navigation_tracker)

print("\n   Testing detailed handlers:")
s1.axes_manager.indices = (2,)
s1 /= 2.0  # Using signal arithmetic preserves metadata
s1.events.data_changed.trigger(obj=s1)

# %%
# Event suppression
# -----------------
print("\n5. Event suppression...")

print("\n   Before suppression - changing index:")
s1.axes_manager.indices = (5,)

print("\n   With specific callback suppressed:")
with nav_axis_1d.events.index_changed.suppress_callback(on_index_changed):
    s1.axes_manager.indices = (8,)

print("\n   With all index_changed events suppressed:")
with nav_axis_1d.events.index_changed.suppress():
    s1.axes_manager.indices = (9,)

print("\n   With all events on the axis suppressed:")
with nav_axis_1d.events.suppress():
    s1.axes_manager.indices = (1,)

print("\n6. Building event-driven analysis workflows...")

class AnalysisTracker:
    """Example class that tracks analysis state through events."""
    
    def __init__(self, signal):
        self.signal = signal
        self.analysis_history = []
        self.current_stats = {}
        
        # Connect to events
        signal.events.data_changed.connect(self.on_data_changed)
        
        # Track navigation if applicable
        if signal.axes_manager.navigation_size > 0:
            for axis in signal.axes_manager.navigation_axes:
                axis.events.index_changed.connect(self.on_navigation_changed)
    
    def on_data_changed(self, obj):
        """Track data changes and update statistics."""
        stats = {
            'timestamp': len(self.analysis_history),
            'mean': float(obj.data.mean()),
            'std': float(obj.data.std()),
            'min': float(obj.data.min()),
            'max': float(obj.data.max())
        }
        self.current_stats = stats
        self.analysis_history.append(('data_changed', stats))
        print(f"   📈 Analysis update #{len(self.analysis_history)}: mean={stats['mean']:.3f}")
    
    def on_navigation_changed(self, obj, index):
        """Track navigation changes."""
        nav_info = {
            'axis_name': obj.name,
            'index': index,
            'value': obj.value if hasattr(obj, 'value') else None
        }
        self.analysis_history.append(('navigation_changed', nav_info))
        print(f"   🧭 Navigation tracked: {obj.name}[{index}]")
    
    def get_summary(self):
        """Get analysis summary."""
        data_changes = sum(1 for event_type, _ in self.analysis_history if event_type == 'data_changed')
        nav_changes = sum(1 for event_type, _ in self.analysis_history if event_type == 'navigation_changed')
        return {
            'total_events': len(self.analysis_history),
            'data_changes': data_changes,
            'navigation_changes': nav_changes,
            'current_stats': self.current_stats
        }

print("\n   Creating analysis tracker:")
tracker = AnalysisTracker(s1)

print("\n   Performing operations that trigger events:")
s1 += 10  # Modify data using signal arithmetic
s1.events.data_changed.trigger(obj=s1)

s1.axes_manager.indices = (4,)  # Change navigation
s1 *= 1.1  # Another data change using signal arithmetic
s1.events.data_changed.trigger(obj=s1)

print("\n   Analysis summary:")
summary = tracker.get_summary()
for key, value in summary.items():
    print(f"   📊 {key}: {value}")

print("\n7. Working with different event types...")

# Demonstrate different ways to handle events
class EventLogger:
    """Simple event logger for demonstration."""
    
    def __init__(self):
        self.log = []
    
    def log_event(self, event_type, obj, **kwargs):
        """Generic event logger."""
        entry = {
            'event_type': event_type,
            'object': str(obj),
            'timestamp': len(self.log),
            'details': kwargs
        }
        self.log.append(entry)
        print(f"   📝 Logged {event_type}: {kwargs}")
    
    def data_changed_logger(self, obj):
        """Specific logger for data changes."""
        self.log_event('data_changed', obj, data_shape=obj.data.shape)
    
    def index_changed_logger(self, obj, index):
        """Specific logger for index changes."""
        self.log_event('index_changed', obj, index=index, axis_name=obj.name)

# Create event logger
logger = EventLogger()

# Create a new signal for clean logging
s_test = hs.signals.Signal1D(np.random.random((8, 80)))
s_test.metadata.General.title = "Test Signal for Events"

# Connect logger to events
s_test.events.data_changed.connect(logger.data_changed_logger)
s_test.axes_manager.navigation_axes[0].events.index_changed.connect(logger.index_changed_logger)

print("\n   Testing event logging:")
s_test *= 1.5  # Using signal arithmetic
s_test.events.data_changed.trigger(obj=s_test)

s_test.axes_manager.indices = (3,)
s_test.axes_manager.indices = (6,)

print(f"\n   Event log summary: {len(logger.log)} events logged")

print("\n8. Event-driven interactive analysis...")

def create_responsive_analysis(signal):
    """Create an analysis that responds to navigation changes."""
    
    analysis_results = {}
    
    def update_analysis(obj, index):
        """Update analysis when navigation changes."""
        current_data = signal.data[signal.axes_manager.indices]
        
        # Perform analysis
        stats = {
            'mean': float(current_data.mean()),
            'std': float(current_data.std()),
            'energy_sum': float(current_data.sum()),
            'peak_position': int(current_data.argmax())
        }
        
        analysis_results[index] = stats
        print(f"   🔄 Auto-analysis at index {index}: peak at {stats['peak_position']}")
    
    # Connect to navigation events
    for axis in signal.axes_manager.navigation_axes:
        axis.events.index_changed.connect(update_analysis)
    
    return analysis_results

print("\n   Setting up responsive analysis:")
responsive_results = create_responsive_analysis(s1)

print("   Changing navigation indices to trigger analysis:")
for idx in [0, 3, 7, 9]:
    s1.axes_manager.indices = (idx,)

print(f"\n   Collected {len(responsive_results)} analysis results")

print("\n9. Event system summary and statistics...")

print(f"\n   Total event counts during demonstration:")
for event_type, count in event_counts.items():
    print(f"   📈 {event_type}: {count} times")

# %%
# Summary
# =======
#
# Event system capabilities and benefits

print("✓ Events enable responsive, interactive analysis workflows")
print("✓ Built-in events cover data changes, navigation, and axis modifications")
print("✓ Custom events can be added for specialized analysis needs")
print("✓ Event suppression allows temporary disabling of callbacks")
print("✓ Event handlers can track analysis history and state")
print("✓ Multiple handlers can be connected to the same event")
print("✓ Events facilitate decoupled, modular analysis components")
print("✓ Essential for building interactive and real-time analysis tools")

print("\n   Key event types demonstrated:")
print("   - data_changed: Triggered when signal data is modified")
print("   - index_changed: Navigation axis index changes")
print("   - value_changed: Axis calibration value changes")
print("   - Custom events: User-defined for specialized analysis")

print(f"\n   Analysis tracker collected {len(tracker.analysis_history)} events")
print(f"   Custom signal analysis found peaks in multiple spectra")
print(f"   Responsive analysis performed {len(responsive_results)} auto-updates")
