import traits.api as t
from hyperspy.link_traits.link_traits import link_directional, link_bidirectional


class TestLinkBidirectional:

    def test_connect_same(self):
        """Verify two traitlets of the same type can be linked together using link."""

        # Create two simple classes with Int traitlets.
        class A(t.HasTraits):
            value = t.Int()
        a = A(value=9)
        b = A(value=8)

        # Conenct the two classes.
        c = link_bidirectional((a, 'value'), (b, 'value'))

        # Make sure the values are the same at the point of linking.
        assert a.value == b.value

        # Change one of the values to make sure they stay in sync.
        a.value = 5
        assert a.value == b.value
        b.value = 6
        assert a.value == b.value

    def test_link_different(self):
        """Verify two traitlets of different types can be linked together using link."""

        # Create two simple classes with Int traitlets.
        class A(t.HasTraits):
            value = t.Int()

        class B(t.HasTraits):
            count = t.Int()
        a = A(value=9)
        b = B(count=8)

        # Conenct the two classes.
        c = link_bidirectional((a, 'value'), (b, 'count'))

        # Make sure the values are the same at the point of linking.
        assert a.value == b.count

        # Change one of the values to make sure they stay in sync.
        a.value = 5
        assert a.value == b.count
        b.count = 4
        assert a.value == b.count

    def test_unlink(self):
        """Verify two linked traitlets can be unlinked."""

        # Create two simple classes with Int traitlets.
        class A(t.HasTraits):
            value = t.Int()
        a = A(value=9)
        b = A(value=8)

        # Connect the two classes.
        c = link_bidirectional((a, 'value'), (b, 'value'))
        a.value = 4
        c.unlink()

        # Change one of the values to make sure they don't stay in sync.
        a.value = 5
        assert a.value != b.value

    def test_callbacks(self):
        """Verify two linked traitlets have their callbacks called once."""

        # Create two simple classes with Int traitlets.
        class A(t.HasTraits):
            value = t.Int()

        class B(t.HasTraits):
            count = t.Int()
        a = A(value=9)
        b = B(count=8)

        # Register callbacks that count.
        callback_count = []

        def a_callback(name, old, new):
            callback_count.append('a')
        a.on_trait_change(a_callback, 'value')

        def b_callback(name, old, new):
            callback_count.append('b')
        b.on_trait_change(b_callback, 'count')

        # Connect the two classes.
        c = link_bidirectional((a, 'value'), (b, 'count'))

        # Make sure b's count was set to a's value once.
        assert ''.join(callback_count) == 'b'
        del callback_count[:]

        # Make sure a's value was set to b's count once.
        b.count = 5
        assert ''.join(callback_count) == 'ba'
        del callback_count[:]

        # Make sure b's count was set to a's value once.
        a.value = 4
        assert ''.join(callback_count) == 'ab'
        del callback_count[:]


class TestDirectionalLink:

    def test_connect_same(self):
        """Verify two traitlets of the same type can be linked together using link_directional."""

        # Create two simple classes with Int traitlets.
        class A(t.HasTraits):
            value = t.Int()
        a = A(value=9)
        b = A(value=8)

        # Conenct the two classes.
        c = link_directional((a, 'value'), (b, 'value'))

        # Make sure the values are the same at the point of linking.
        assert a.value == b.value

        # Change one the value of the source and check that it synchronizes the
        # target.
        a.value = 5
        assert b.value == 5
        # Change one the value of the target and check that it has no impact on
        # the source
        b.value = 6
        assert a.value == 5

    def test_tranform(self):
        """Test transform link."""

        # Create two simple classes with Int traitlets.
        class A(t.HasTraits):
            value = t.Int()
        a = A(value=9)
        b = A(value=8)

        # Conenct the two classes.
        c = link_directional((a, 'value'), (b, 'value'), lambda x: 2 * x)

        # Make sure the values are correct at the point of linking.
        assert b.value == 2 * a.value

        # Change one the value of the source and check that it modifies the
        # target.
        a.value = 5
        assert b.value == 10
        # Change one the value of the target and check that it has no impact on
        # the source
        b.value = 6
        assert a.value == 5

    def test_link_different(self):
        """Verify two traitlets of different types can be linked together using link."""

        # Create two simple classes with Int traitlets.
        class A(t.HasTraits):
            value = t.Int()

        class B(t.HasTraits):
            count = t.Int()
        a = A(value=9)
        b = B(count=8)

        # Conenct the two classes.
        c = link_directional((a, 'value'), (b, 'count'))

        # Make sure the values are the same at the point of linking.
        assert a.value == b.count

        # Change one the value of the source and check that it synchronizes the
        # target.
        a.value = 5
        assert b.count == 5
        # Change one the value of the target and check that it has no impact on
        # the source
        b.value = 6
        assert a.value == 5

    def test_unlink(self):
        """Verify two linked traitlets can be unlinked."""

        # Create two simple classes with Int traitlets.
        class A(t.HasTraits):
            value = t.Int()
        a = A(value=9)
        b = A(value=8)

        # Connect the two classes.
        c = link_directional((a, 'value'), (b, 'value'))
        a.value = 4
        c.unlink()

        # Change one of the values to make sure they don't stay in sync.
        a.value = 5
        assert a.value != b.value
